"""Trainer for TRM-LLM with deep supervision

Implements training loop with:
- Deep supervision (multi-step training)
- Curriculum learning (gradually increase supervision steps)
- Adaptive computation time (ACT) for efficient training
- Gradient clipping and EMA (future)
- Muon optimizer support for faster convergence
- DDP (Distributed Data Parallel) for multi-GPU training
"""

from typing import Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Literal
import os

from trm_llm.utils.config import TRMLLMConfig
from trm_llm.data.tokenizer import ToolCallTokenizer
from trm_llm.utils.logger import log, log_warning, reset_main_process_cache
from trm_llm.training.loss import (
    compute_trm_loss,
    compute_action_accuracy,
    compute_per_step_accuracy,
    compute_valid_tool_call_format_accuracy,
)


def setup_distributed(rank: int = None, world_size: int = None, backend: str = "nccl"):
    """Initialize distributed training environment

    Args:
        rank: Process rank (if None, read from environment)
        world_size: Total number of processes (if None, read from environment)
        backend: "nccl" for GPU, "gloo" for CPU

    Returns:
        rank, world_size, is_distributed
    """
    # Check if already initialized
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), True

    # Check environment variables (set by torchrun/torch.distributed.launch)
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Only initialize if world_size > 1 or explicitly requested
    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        # Reset logger cache after DDP init
        reset_main_process_cache()
        return rank, world_size, True

    return 0, 1, False


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _init_distributed_for_muon():
    """Initialize distributed environment for Muon (required even for single GPU)"""
    if dist.is_initialized():
        return  # Already initialized (DDP mode)

    # Use a random port to avoid conflicts
    port = random.randint(12355, 12455)
    try:
        # Initialize with single process for non-distributed training
        dist.init_process_group(
            backend="gloo",  # Use gloo for CPU compatibility
            init_method=f"tcp://localhost:{port}",
            rank=0,
            world_size=1,
        )
        log("Initialized distributed environment for Muon", port=port)
    except Exception as e:
        log_warning(
            "Could not initialize distributed environment",
            error=str(e),
            note="Muon may not work correctly",
        )
        raise


def create_muon_optimizer(
    model: nn.Module,
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
    weight_decay: float = 0.01,
    muon_momentum: float = 0.95,
) -> torch.optim.Optimizer:
    """Create Muon optimizer with AdamW for non-hidden params

    Muon is used for hidden weights (ndim >= 2) of:
    - Encoder layers
    - Reasoning module
    - Action module

    AdamW is used for:
    - Embeddings (token, position)
    - Output heads
    - Generation head
    - All biases and gains (ndim < 2)

    Args:
        model: TRMLLM model
        muon_lr: Learning rate for Muon (hidden weights)
        adam_lr: Learning rate for AdamW (embeddings, heads, biases)
        weight_decay: Weight decay for both optimizers
        muon_momentum: Momentum for Muon

    Returns:
        MuonWithAuxAdam optimizer
    """
    # Initialize distributed environment (required by Muon)
    _init_distributed_for_muon()

    from muon import MuonWithAuxAdam

    # Categorize parameters
    hidden_weights = []  # For Muon (ndim >= 2)
    other_params = []  # For AdamW (embeddings, heads, biases)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Use Muon for hidden weights in encoder, reasoning, and action modules
        is_hidden_module = any(
            mod in name
            for mod in [
                "encoder",
                "reasoning_module",
                "action_module",
            ]
        )

        if is_hidden_module and param.ndim >= 2:
            hidden_weights.append(param)
        else:
            other_params.append(param)

    log(
        "Muon optimizer parameter groups",
        hidden_weights_muon=f"{sum(p.numel() for p in hidden_weights) / 1e6:.2f}M params",
        other_params_adamw=f"{sum(p.numel() for p in other_params) / 1e6:.2f}M params",
    )

    param_groups = [
        dict(
            params=hidden_weights,
            use_muon=True,
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
        ),
        dict(
            params=other_params,
            use_muon=False,
            lr=adam_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        ),
    ]

    return MuonWithAuxAdam(param_groups)


class TRMTrainer:
    """Trainer for TRM-LLM with deep supervision

    Key TRM training techniques:
    1. Deep supervision: Provide loss at each refinement step
    2. Curriculum learning: Start with few steps, gradually increase
    3. State detaching: Gradients only flow through last step
    4. Muon optimizer support for faster convergence on hidden layers
    5. DDP (Distributed Data Parallel) for multi-GPU training
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TRMLLMConfig,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        tokenizer: Optional[ToolCallTokenizer] = None,
        tool_id_to_name: Optional[Dict[int, str]] = None,
        save_interval: int = 10,
        log_sample_interval: int = 0,
        optimizer_type: Literal["adamw", "muon"] = "adamw",
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        use_ddp: bool = False,
        local_rank: int = 0,
    ):
        """
        Args:
            model: TRMLLM model
            train_loader: Training data loader
            config: TRMLLMConfig
            val_loader: Optional validation data loader
            device: Device to train on
            tokenizer: Optional tokenizer for logging sample predictions
            tool_id_to_name: Optional mapping from tool IDs to names
            save_interval: Save checkpoint every N epochs (default: 10)
            log_sample_interval: Log sample prediction every N steps (0 = only at end of epoch)
            optimizer_type: "adamw" or "muon" (Muon uses AdamW for embeddings/heads)
            muon_lr: Learning rate for Muon hidden weights (default: 0.02)
            muon_momentum: Momentum for Muon (default: 0.95)
            use_ddp: Whether to use Distributed Data Parallel
            local_rank: Local rank for DDP (GPU device index)
        """
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.tool_id_to_name = tool_id_to_name or {}
        self.save_interval = save_interval
        self.log_sample_interval = log_sample_interval
        self.optimizer_type = optimizer_type
        self.use_ddp = use_ddp
        self.local_rank = local_rank

        # Get special token IDs for weighted loss (e.g., <tool_call>, </tool_call>)
        self.special_token_ids = []
        self.tool_call_token_id = None
        if tokenizer is not None:
            special_tokens = [
                tokenizer.TOOL_CALL_START,  # <tool_call>
                tokenizer.TOOL_CALL_END,  # </tool_call>
            ]
            for token in special_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    self.special_token_ids.append(token_id)

            # Get tool_call start token ID for consistency loss
            self.tool_call_token_id = tokenizer.convert_tokens_to_ids(tokenizer.TOOL_CALL_START)
            if self.tool_call_token_id == tokenizer.unk_token_id:
                self.tool_call_token_id = None
        self._log_struct(
            "Special token IDs for loss weighting",
            special_token_ids=self.special_token_ids,
            tool_call_token_id=self.tool_call_token_id,
        )
        # DDP setup
        if use_ddp:
            self.rank = dist.get_rank() if dist.is_initialized() else 0
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            self.is_main = self.rank == 0
            # Move model to correct device and wrap with DDP
            model = model.to(device)
            # find_unused_parameters=True required because generation head is only used
            # on the last supervision step, so its params don't participate every iteration
            self.model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
            self._log_struct("DDP initialized", rank=self.rank, world_size=self.world_size)
        else:
            self.rank = 0
            self.world_size = 1
            self.is_main = True
            self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Get the underlying model (unwrap DDP if needed)
        self.raw_model = self.model.module if use_ddp else self.model

        # Optimizer (use raw model for parameter groups)
        if optimizer_type == "muon":
            self._log_struct(
                "Using Muon optimizer",
                muon_lr=muon_lr,
                adam_lr=config.learning_rate,
                momentum=muon_momentum,
            )
            self.optimizer = create_muon_optimizer(
                self.raw_model,
                muon_lr=muon_lr,
                adam_lr=config.learning_rate,
                weight_decay=config.weight_decay,
                muon_momentum=muon_momentum,
            )
        else:
            self._log_struct(
                "Using AdamW optimizer", lr=config.learning_rate, weight_decay=config.weight_decay
            )
            self.optimizer = torch.optim.AdamW(
                self.raw_model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),  # Standard for transformers
            )

        # Learning rate scheduler (warmup + cosine decay)
        self.scheduler = self._create_scheduler()

        # Curriculum learning: Start with fewer supervision steps, gradually increase
        self.current_max_steps = 2  # Start with 2 steps
        self.step_increase_interval = 5  # Increase every 5 epochs

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0

    def _log(self, msg: str):
        """Print only from main process (uses global logger)"""
        log(msg)

    def _log_struct(self, msg: str, **kwargs):
        """Structured log only from main process"""
        log(msg, **kwargs)

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            # Warmup
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))

            # Cosine decay after warmup
            progress = float(current_step - self.config.warmup_steps)
            total_steps = len(self.train_loader) * self.config.max_epochs - self.config.warmup_steps
            return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress / total_steps * 3.14159))))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Single training step

        Args:
            batch: Batch dict with input_ids, target_action, target_tool_id

        Returns:
            loss: Scalar loss value
            metrics: Dict with loss components and accuracies
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with deep supervision
        outputs_per_step = self.model(
            batch["input_ids"],
            max_supervision_steps=self.current_max_steps,
            training=True,
            target_generation_ids=batch.get("target_generation_ids"),
        )

        # Compute loss (with special token weighting and action-generation consistency)
        loss, loss_dict = compute_trm_loss(
            outputs_per_step,
            batch,
            self.config,
            special_token_ids=self.special_token_ids if self.special_token_ids else None,
            tool_call_token_id=self.tool_call_token_id,
        )

        # Compute accuracies
        acc_dict = compute_action_accuracy(outputs_per_step, batch)

        # Compute strict tool call format accuracy: <tool_call>{...}</tool_call>
        # Return counts for proper aggregation (avoids dilution from batches without tool_calls)
        if self.tokenizer is not None:
            valid_count, total_count, sample_correct = compute_valid_tool_call_format_accuracy(
                outputs_per_step, batch, self.tokenizer, return_counts=True, return_sample=True
            )
            acc_dict["valid_tool_format_correct"] = valid_count
            acc_dict["valid_tool_format_total"] = total_count
            # Log sample correct prediction if any valid
            if valid_count > 0 and sample_correct is not None and self.is_main:
                self._log_struct(
                    "Valid tool call prediction",
                    step=self.global_step,
                    prediction=sample_correct,
                )

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        self.scheduler.step()

        self.global_step += 1

        # Combine metrics
        metrics = {**loss_dict, **acc_dict}
        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        return loss.item(), metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            avg_metrics: Dict with averaged metrics
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = {}

        # Set epoch for distributed sampler
        if self.use_ddp and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        # Only show progress bar on main process
        if self.is_main:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
        else:
            pbar = self.train_loader

        for batch_idx, batch in enumerate(pbar):
            loss, metrics = self.train_step(batch)

            total_loss += loss

            # Accumulate metrics
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value

            # Update progress bar with running averages (only main process)
            if self.is_main and batch_idx % 10 == 0:
                n = batch_idx + 1
                avg_loss = total_loss / n
                avg_act = total_metrics.get("action_accuracy", 0.0) / n
                avg_n_calls = total_metrics.get("num_calls_accuracy", 0.0) / n
                avg_tool_gen = total_metrics.get("tool_gen_accuracy", 0.0) / n
                avg_direct_gen = total_metrics.get("direct_gen_accuracy", 0.0) / n
                # Compute tool_fmt from accumulated counts (not averaged ratios)
                tool_fmt_correct = total_metrics.get("valid_tool_format_correct", 0)
                tool_fmt_total = total_metrics.get("valid_tool_format_total", 0)
                avg_tool_fmt = tool_fmt_correct / tool_fmt_total if tool_fmt_total > 0 else 0.0
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "act": f"{avg_act:.3f}",
                        "n_calls": f"{avg_n_calls:.3f}",
                        "tool_call": f"{avg_tool_gen:.3f}",
                        "direct_answer": f"{avg_direct_gen:.3f}",
                        "tool_fmt": f"{avg_tool_fmt:.3f}",
                    }
                )

            # Log sample prediction at intervals (if enabled)
            if self.log_sample_interval > 0 and self.global_step % self.log_sample_interval == 0:
                sample_idx = random.randint(0, len(self.train_loader.dataset) - 1)
                self.log_sample_prediction(
                    sample_idx=sample_idx, context=f"step {self.global_step}"
                )

        # Average metrics
        num_batches = len(self.train_loader)
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()
                       if key not in ("valid_tool_format_correct", "valid_tool_format_total")}
        avg_metrics["loss"] = total_loss / num_batches
        # Compute valid_tool_format from accumulated counts
        tool_fmt_correct = total_metrics.get("valid_tool_format_correct", 0)
        tool_fmt_total = total_metrics.get("valid_tool_format_total", 0)
        avg_metrics["valid_tool_format"] = tool_fmt_correct / tool_fmt_total if tool_fmt_total > 0 else 0.0

        return avg_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set

        Returns:
            val_metrics: Dict with validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_metrics = {}

        # Only show progress bar on main process
        if self.is_main:
            loader = tqdm(self.val_loader, desc="Validating")
        else:
            loader = self.val_loader

        for batch in loader:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs_per_step = self.model(
                batch["input_ids"],
                max_supervision_steps=self.config.max_supervision_steps,  # Use full steps for val
                training=False,
                target_generation_ids=batch.get("target_generation_ids"),
            )

            # Compute loss (with special token weighting and action-generation consistency)
            loss, loss_dict = compute_trm_loss(
                outputs_per_step,
                batch,
                self.config,
                special_token_ids=self.special_token_ids if self.special_token_ids else None,
                tool_call_token_id=self.tool_call_token_id,
            )

            # Compute accuracies
            acc_dict = compute_action_accuracy(outputs_per_step, batch)

            # Compute strict tool call format accuracy: <tool_call>{...}</tool_call>
            # Return counts for proper aggregation
            if self.tokenizer is not None:
                valid_count, total_count = compute_valid_tool_call_format_accuracy(
                    outputs_per_step, batch, self.tokenizer, return_counts=True
                )
                acc_dict["valid_tool_format_correct"] = valid_count
                acc_dict["valid_tool_format_total"] = total_count

            total_loss += loss.item()
            for key, value in {**loss_dict, **acc_dict}.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value

        # Average (exclude count metrics from averaging)
        num_batches = len(self.val_loader)
        val_metrics = {f"val_{key}": value / num_batches for key, value in total_metrics.items()
                       if key not in ("valid_tool_format_correct", "valid_tool_format_total")}
        val_metrics["val_loss"] = total_loss / num_batches
        # Compute valid_tool_format from accumulated counts
        tool_fmt_correct = total_metrics.get("valid_tool_format_correct", 0)
        tool_fmt_total = total_metrics.get("valid_tool_format_total", 0)
        val_metrics["val_valid_tool_format"] = tool_fmt_correct / tool_fmt_total if tool_fmt_total > 0 else 0.0

        return val_metrics

    @torch.no_grad()
    def log_sample_prediction(self, sample_idx: int = 0, context: str = ""):
        """Log a sample prediction to monitor training progress

        Args:
            sample_idx: Index of sample to log from training set
            context: Optional context string (e.g., "step 100", "end of epoch")
        """
        # Only log on main process
        if not self.is_main or self.tokenizer is None:
            return

        self.model.eval()

        # Get a sample from dataset
        dataset = self.train_loader.dataset
        # Handle DistributedSampler wrapper
        if hasattr(dataset, "dataset"):
            base_dataset = dataset.dataset
            actual_idx = dataset.indices[sample_idx] if hasattr(dataset, "indices") else sample_idx
            sample = base_dataset[actual_idx]
        else:
            sample = dataset[sample_idx]

        # Prepare batch (single sample)
        input_ids = torch.tensor([sample["input_ids"]], device=self.device)
        target_action = sample["target_action"]
        target_num_calls = sample.get("target_num_calls", 0)
        target_generation_ids = sample.get("target_generation_ids", [])

        # Forward pass (use raw model for inference)
        outputs_per_step = self.raw_model(
            input_ids,
            max_supervision_steps=self.config.max_supervision_steps,
            training=False,
        )

        final_output = outputs_per_step[-1]

        # Decode predictions
        action_probs = F.softmax(final_output["action_logits"][0], dim=-1)
        pred_action = action_probs.argmax().item()
        action_conf = action_probs[pred_action].item()

        pred_num_calls = 1
        if "num_calls_logits" in final_output:
            num_calls_probs = F.softmax(final_output["num_calls_logits"][0], dim=-1)
            pred_num_calls = num_calls_probs.argmax().item() + 1

        # Build sample prediction info
        input_text = self.tokenizer.decode(sample["input_ids"][:200])
        target_action_str = "tool_call" if target_action == 1 else "direct_answer"
        pred_action_str = "tool_call" if pred_action == 1 else "direct_answer"
        action_match = "match" if pred_action == target_action else "mismatch"

        sample_pred = {
            "input_tokens": len(sample["input_ids"]),
            "input_preview": input_text + "...",
            "target_action": target_action_str,
            "pred_action": pred_action_str,
            "pred_confidence": f"{action_conf:.3f}",
            "action_match": action_match,
        }

        if target_action == 1:
            sample_pred["target_num_calls"] = target_num_calls
        if pred_action == 1:
            sample_pred["pred_num_calls"] = pred_num_calls

        if target_generation_ids:
            gen_text = self.tokenizer.decode(target_generation_ids)
            if len(gen_text) > 300:
                gen_text = gen_text[:300] + "..."
            sample_pred["target_generation"] = f"({len(target_generation_ids)} tokens) {gen_text}"

        # Generate output
        if hasattr(self.raw_model, "generate"):
            y_state = final_output["y_state"]
            gen_ids = self.raw_model.generate(y_state, max_length=128)
            gen_text = self.tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
            if len(gen_text) > 300:
                gen_text = gen_text[:300] + "..."
            sample_pred["generated"] = gen_text

        log_msg = f"Sample prediction ({context})" if context else "Sample prediction"
        log(log_msg, **sample_pred)

        self.model.train()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filepath: str):
        """Save model checkpoint (only on main process)

        Args:
            epoch: Current epoch
            metrics: Current metrics
            filepath: Path to save checkpoint
        """
        # Only save on main process
        if not self.is_main:
            return

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Save raw model state (unwrap DDP)
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "current_max_steps": self.current_max_steps,
        }

        torch.save(checkpoint, filepath)
        log("Checkpoint saved", path=filepath, epoch=epoch + 1)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Load into raw model (handles both DDP and non-DDP)
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.current_max_steps = checkpoint.get("current_max_steps", 2)

        self._log_struct(
            "Checkpoint loaded", path=filepath, epoch=self.current_epoch, step=self.global_step
        )

    def train(self, save_dir: str = "checkpoints"):
        """Full training loop

        Args:
            save_dir: Directory to save checkpoints
        """
        training_info = {
            "model_params": f"{self.raw_model.get_num_trainable_params() / 1e6:.1f}M",
            "train_examples": len(self.train_loader.dataset),
            "batch_size": self.config.batch_size,
            "max_epochs": self.config.max_epochs,
            "initial_supervision_steps": self.current_max_steps,
        }
        if self.val_loader:
            training_info["val_examples"] = len(self.val_loader.dataset)
        if self.use_ddp:
            training_info["world_size"] = f"{self.world_size} GPUs"
            training_info["effective_batch_size"] = self.config.batch_size * self.world_size
        self._log_struct("Starting TRM-LLM Training", **training_info)

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            # Curriculum: Gradually increase max supervision steps
            self.update_curriculum(epoch)

            # Train epoch
            self._log_struct(
                "Epoch started",
                epoch=f"{epoch+1}/{self.config.max_epochs}",
                max_supervision_steps=self.current_max_steps,
                learning_rate=f"{self.scheduler.get_last_lr()[0]:.6f}",
            )

            train_metrics = self.train_epoch(epoch)

            # Synchronize metrics across processes (for DDP)
            if self.use_ddp:
                dist.barrier()

            # Print training metrics (main process only)
            self._log_struct(
                "Training results",
                loss=f"{train_metrics['loss']:.4f}",
                action_acc=f"{train_metrics['action_accuracy']:.3f}",
                num_calls_acc=f"{train_metrics.get('num_calls_accuracy', 0.0):.3f}",
                tool_gen_acc=f"{train_metrics.get('tool_gen_accuracy', 0.0):.3f}",
                direct_gen_acc=f"{train_metrics.get('direct_gen_accuracy', 0.0):.3f}",
                valid_tool_format=f"{train_metrics.get('valid_tool_format', 0.0):.3f}",
                overall_acc=f"{train_metrics['overall_accuracy']:.3f}",
            )

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                self._log_struct(
                    "Validation results",
                    loss=f"{val_metrics['val_loss']:.4f}",
                    action_acc=f"{val_metrics['val_action_accuracy']:.3f}",
                    num_calls_acc=f"{val_metrics.get('val_num_calls_accuracy', 0.0):.3f}",
                    tool_gen_acc=f"{val_metrics.get('val_tool_gen_accuracy', 0.0):.3f}",
                    direct_gen_acc=f"{val_metrics.get('val_direct_gen_accuracy', 0.0):.3f}",
                    valid_tool_format=f"{val_metrics.get('val_valid_tool_format', 0.0):.3f}",
                    overall_acc=f"{val_metrics['val_overall_accuracy']:.3f}",
                )

                # Save best model (main process only via save_checkpoint)
                if val_metrics["val_overall_accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["val_overall_accuracy"]
                    self.save_checkpoint(
                        epoch, {**train_metrics, **val_metrics}, f"{save_dir}/best_model.pt"
                    )
                    self._log_struct("New best model saved", accuracy=f"{self.best_val_acc:.3f}")

            # Log sample prediction at end of epoch (random sample, main process only)
            sample_idx = random.randint(0, len(self.train_loader.dataset) - 1)
            self.log_sample_prediction(sample_idx=sample_idx, context=f"epoch {epoch+1} end")

            # Save periodic checkpoint (main process only via save_checkpoint)
            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(
                    epoch, train_metrics, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt"
                )

            # Synchronize before next epoch
            if self.use_ddp:
                dist.barrier()

        self._log_struct("Training completed", best_val_accuracy=f"{self.best_val_acc:.3f}")

    def update_curriculum(self, epoch: int):
        """Update curriculum (gradually increase supervision steps)

        Args:
            epoch: Current epoch
        """
        # Increase by 1 step every step_increase_interval epochs
        # Start at 2, max out at config.max_supervision_steps
        new_max_steps = min(
            2 + epoch // self.step_increase_interval, self.config.max_supervision_steps
        )

        if new_max_steps != self.current_max_steps:
            self.current_max_steps = new_max_steps
            self._log_struct("Curriculum update", max_supervision_steps=self.current_max_steps)
