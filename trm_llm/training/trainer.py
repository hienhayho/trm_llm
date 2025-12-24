"""Trainer for TRM-LLM with deep supervision

Implements training loop with:
- Deep supervision (multi-step training)
- Curriculum learning (gradually increase supervision steps)
- Adaptive computation time (ACT) for efficient training
- Gradient clipping and EMA for stable recursive depth
- Muon optimizer support for faster convergence
- DDP (Distributed Data Parallel) for multi-GPU training
- DeepSpeed support for memory-efficient training with ZeRO optimization
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
from typing import Optional, Dict, Literal, Any
import os
import json

from trm_llm.utils.config import TRMLLMConfig
from trm_llm.data.tokenizer import ToolCallTokenizer
from trm_llm.utils.logger import log, log_warning, reset_main_process_cache
from trm_llm.training.loss import (
    compute_trm_loss,
    compute_action_accuracy,
    compute_per_step_accuracy,
    compute_valid_tool_call_format_accuracy,
)


def get_adamw_param_groups(model: nn.Module, weight_decay: float, lr: float):
    """Create AdamW parameter groups with proper weight decay handling.

    Following best practices from Hugging Face Transformers and GPT-2:
    - Apply weight decay to most parameters
    - Do NOT apply weight decay to bias and LayerNorm parameters

    Args:
        model: The model to create parameter groups for
        weight_decay: Weight decay value for eligible parameters
        lr: Learning rate

    Returns:
        List of parameter group dicts for AdamW optimizer
    """
    # Parameters that should NOT have weight decay
    no_decay = ["bias", "LayerNorm", "layer_norm", "layernorm", "ln_"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this parameter should skip weight decay
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
    ]

    # Log the split
    decay_count = sum(p.numel() for p in decay_params)
    no_decay_count = sum(p.numel() for p in no_decay_params)
    log(
        "AdamW parameter groups created",
        with_weight_decay=f"{decay_count / 1e6:.2f}M params",
        without_weight_decay=f"{no_decay_count / 1e6:.2f}M params (bias/LayerNorm)",
    )

    return param_groups


class EMAModel:
    """Exponential Moving Average of model parameters

    Maintains a shadow copy of model parameters that is updated as:
        ema_param = decay * ema_param + (1 - decay) * param

    Benefits for TRM-LLM:
    - Stabilizes recursive reasoning by smoothing parameter updates
    - The same reasoning network f(x,y,z) is applied n times recursively
    - EMA ensures consistent behavior across recursion depths
    - Improves generalization on validation/test sets

    Usage:
        ema = EMAModel(model, decay=0.9999)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update()  # Update EMA after each step

        # For evaluation, use EMA weights
        ema.apply_shadow()
        evaluate(model)
        ema.restore()  # Restore original weights for training
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: Model to track (can be DDP-wrapped or raw)
            decay: EMA decay rate (higher = slower updates, more smoothing)
                   Typical values: 0.999, 0.9999, 0.99999
            device: Device for EMA parameters (default: same as model)
        """
        self.decay = decay
        self.device = device

        # Get raw model (unwrap DDP if needed)
        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model

        # Create shadow parameters (EMA copy)
        self.shadow = {}
        self.backup = {}  # For storing original params during eval

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device:
                    self.shadow[name] = self.shadow[name].to(device)

    @torch.no_grad()
    def update(self):
        """Update EMA parameters after a training step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA update: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)

        Saves original weights to backup so they can be restored.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state dict for checkpointing"""
        return {
            'shadow': self.shadow.copy(),
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load EMA state from checkpoint"""
        self.decay = state_dict.get('decay', self.decay)

        # Load shadow tensors and move to correct device
        loaded_shadow = state_dict['shadow']
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in loaded_shadow:
                # Move shadow tensor to same device as model parameter
                self.shadow[name] = loaded_shadow[name].to(param.device)

    def copy_to_model(self):
        """Permanently copy EMA weights to model (for final model saving)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


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


def create_pytorch_muon_optimizer(
    model: nn.Module,
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
    weight_decay: float = 0.01,
    muon_momentum: float = 0.95,
) -> "MuonAdamW":
    """Create unified Muon+AdamW optimizer for DeepSpeed compatibility

    Uses PyTorch's torch.optim.Muon algorithm for 2D hidden layer weights and
    AdamW algorithm for all other parameters, implemented in a single optimizer
    for DeepSpeed ZeRO-2 compatibility.

    Args:
        model: TRMLLM model
        muon_lr: Learning rate for Muon (hidden weights)
        adam_lr: Learning rate for AdamW (embeddings, heads, biases)
        weight_decay: Weight decay for both optimizers
        muon_momentum: Momentum for Muon

    Returns:
        MuonAdamW optimizer compatible with DeepSpeed
    """
    # Categorize parameters: Muon only supports 2D params
    muon_params = []  # For Muon (2D hidden weights only)
    adamw_params = []  # For AdamW (everything else)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Use Muon for 2D weights in encoder, reasoning, and action modules
        is_hidden_module = any(
            mod in name
            for mod in [
                "encoder",
                "reasoning_module",
                "action_module",
            ]
        )

        # Muon requires exactly 2D params
        if is_hidden_module and param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    log(
        "PyTorch Muon optimizer parameter groups",
        muon_2d_weights=f"{sum(p.numel() for p in muon_params) / 1e6:.2f}M params",
        adamw_other=f"{sum(p.numel() for p in adamw_params) / 1e6:.2f}M params",
    )

    # Create unified optimizer with both param groups
    param_groups = [
        {
            "params": muon_params,
            "lr": muon_lr,
            "weight_decay": weight_decay,
            "momentum": muon_momentum,
            "use_muon": True,
        },
        {
            "params": adamw_params,
            "lr": adam_lr,
            "betas": (0.9, 0.95),
            "weight_decay": weight_decay,
            "use_muon": False,
        },
    ]

    return MuonAdamW(param_groups)


class MuonAdamW(torch.optim.Optimizer):
    """Unified optimizer that applies Muon to 2D weights and AdamW to others

    This is a single optimizer class that implements both Muon and AdamW
    update rules based on the `use_muon` flag in param groups.
    Compatible with DeepSpeed ZeRO optimization.

    Muon algorithm (for 2D params):
    - Uses momentum with Newton-Schulz orthogonalization
    - Specifically designed for hidden layer weights

    AdamW algorithm (for other params):
    - Standard Adam with decoupled weight decay
    - Used for embeddings, biases, and output heads
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01,
                 momentum=0.95, nesterov=True, ns_steps=5, eps=1e-8):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            use_muon=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step

        Applies Muon update to param groups with use_muon=True,
        AdamW update to others.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        """Muon update step for 2D parameters"""
        momentum = group["momentum"]
        nesterov = group.get("nesterov", True)
        ns_steps = group.get("ns_steps", 5)
        lr = group["lr"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            state = self.state[p]

            # Initialize state
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]

            # Momentum update
            buf.lerp_(grad, 1 - momentum)

            # Apply Nesterov momentum if enabled
            if nesterov:
                g = grad.lerp(buf, momentum)
            else:
                g = buf

            # Newton-Schulz orthogonalization (core Muon innovation)
            # Only for 2D tensors
            if p.ndim == 2:
                g = self._newton_schulz(g, ns_steps)

                # Scale learning rate by matrix dimensions for consistent RMS
                # This ensures similar update magnitudes regardless of layer size
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                adjusted_lr = lr * scale
            else:
                adjusted_lr = lr

            # Decoupled weight decay
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            # Parameter update
            p.add_(g, alpha=-adjusted_lr)

    def _newton_schulz(self, G, steps=5):
        """Newton-Schulz iteration for approximate matrix orthogonalization

        This is the key innovation of Muon - it orthogonalizes the gradient
        update direction, which helps with optimization landscape navigation.
        """
        # Coefficients from Muon paper
        a, b, c = 3.4445, -4.7750, 2.0315

        # Work in bfloat16 for efficiency
        orig_dtype = G.dtype
        G = G.to(torch.bfloat16)

        # Normalize
        G = G / (G.norm() + 1e-7)

        # Newton-Schulz iterations
        for _ in range(steps):
            A = G @ G.T
            B = b * A + c * A @ A
            G = a * G + B @ G

        return G.to(orig_dtype)

    def _adamw_step(self, group):
        """AdamW update step for non-2D parameters"""
        lr = group["lr"]
        betas = group.get("betas", (0.9, 0.95))
        weight_decay = group["weight_decay"]
        eps = group.get("eps", 1e-8)

        beta1, beta2 = betas

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")

            state = self.state[p]

            # Initialize state
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            step = state["step"]

            # Decoupled weight decay
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            # Update biased first moment estimate
            exp_avg.lerp_(grad, 1 - beta1)

            # Update biased second raw moment estimate
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Compute step size
            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2 ** 0.5

            # Update parameters
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)


class TrainingStage:
    """Training stage constants for staged/curriculum training

    Staged training pipeline:
    - Stage 0: Train backbone (encoder, reasoning, action) + output heads (freeze generation_head)
    - Stage 1: Train generation_head only (freeze everything else)
    - Stage 2: Fine-tune all parameters (typically with smaller dataset)
    - Stage -1: Standard training (all params, all losses)
    """
    STANDARD = -1       # Normal training (all params, all losses)
    BACKBONE = 0        # Encoder + reasoning + action + output_heads (freeze generation)
    GENERATION = 1      # Generation head only
    FULL_FINETUNE = 2   # All params, all losses (fine-tune with smaller dataset)


class TRMTrainer:
    """Trainer for TRM-LLM with deep supervision

    Key TRM training techniques:
    1. Deep supervision: Provide loss at each refinement step
    2. Curriculum learning: Start with few steps, gradually increase
    3. State detaching: Gradients only flow through last step
    4. Muon optimizer support for faster convergence on hidden layers
    5. DDP (Distributed Data Parallel) for multi-GPU training
    6. DeepSpeed support for memory-efficient training with ZeRO optimization
    7. EMA (Exponential Moving Average) for stable recursive depth
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
        use_deepspeed: bool = False,
        ds_config: Optional[Dict[str, Any]] = None,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        training_stage: int = -1,
        early_stopping_patience: int = 0,
        use_original_trm_training: bool = False,
        wandb_run: Optional[Any] = None,
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
            use_deepspeed: Whether to use DeepSpeed for training
            ds_config: DeepSpeed configuration dict (or path to config file)
            use_ema: Whether to use EMA for stable recursive depth (default: False)
            ema_decay: EMA decay rate (default: 0.9999, higher = more smoothing)
            training_stage: Training stage (-1=standard, 0=backbone, 1=generation, 2=finetune)
            early_stopping_patience: Stop if val F1 doesn't improve for N epochs (0 = disabled)
            use_original_trm_training: If True, use original TRM training flow where
                backward/step happens after EACH supervision step (not accumulated)
            wandb_run: Optional wandb run object for logging metrics
        """
        self.config = config
        self.use_original_trm_training = use_original_trm_training
        self.wandb_run = wandb_run
        self.training_stage = training_stage
        self.device = device
        self.tokenizer = tokenizer
        self.tool_id_to_name = tool_id_to_name or {}
        self.save_interval = save_interval
        self.log_sample_interval = log_sample_interval
        self.optimizer_type = optimizer_type
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.use_ddp = use_ddp
        self.local_rank = local_rank
        self.use_deepspeed = use_deepspeed
        self.ds_config = ds_config
        self.use_ema = use_ema
        self.ema_decay = ema_decay

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

        self.train_loader = train_loader
        self.val_loader = val_loader

        # DeepSpeed setup (takes priority over DDP)
        if use_deepspeed:
            import deepspeed

            self.rank = int(os.environ.get("LOCAL_RANK", local_rank))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.is_main = self.rank == 0

            # Prepare DeepSpeed config
            if ds_config is None:
                ds_config = self._get_default_ds_config()
            elif isinstance(ds_config, str):
                # Load from file
                with open(ds_config, "r") as f:
                    ds_config = json.load(f)

            # Set auto values in config (resolve "auto" strings that DeepSpeed can't handle)
            ds_config["train_micro_batch_size_per_gpu"] = config.batch_size
            ds_config["gradient_clipping"] = config.gradient_clip_norm
            # DeepSpeed requires these to be integers, not "auto"
            if ds_config.get("gradient_accumulation_steps") == "auto":
                ds_config["gradient_accumulation_steps"] = 1
            if ds_config.get("train_batch_size") == "auto":
                # Let DeepSpeed calculate from micro_batch * grad_accum * world_size
                del ds_config["train_batch_size"]

            # Get ZeRO stage from config
            zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)

            # Apply parameter freezing for staged training (before creating optimizer)
            if training_stage >= 0:
                # Freeze parameters directly on model (before deepspeed.initialize)
                if training_stage == TrainingStage.STANDARD or training_stage == TrainingStage.FULL_FINETUNE:
                    for param in model.parameters():
                        param.requires_grad = True
                else:
                    for param in model.parameters():
                        param.requires_grad = False
                    # Use prefixes to match top-level modules only (avoid matching submodules)
                    if training_stage == TrainingStage.BACKBONE:
                        unfreeze_prefixes = [
                            "token_embedding.", "position_embedding.", "embedding_dropout.",
                            "encoder.", "reasoning_module.", "action_module.", "output_heads.", "init_y"
                        ]
                    elif training_stage == TrainingStage.GENERATION:
                        unfreeze_prefixes = ["generation_head."]
                    else:
                        unfreeze_prefixes = []
                    for name, param in model.named_parameters():
                        if any(name.startswith(prefix) or name == prefix.rstrip('.') for prefix in unfreeze_prefixes):
                            param.requires_grad = True
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                self._log_struct(
                    "DeepSpeed staged training: parameters frozen",
                    stage=training_stage,
                    trainable_params=f"{sum(p.numel() for p in trainable_params) / 1e6:.2f}M",
                )
            else:
                trainable_params = list(model.parameters())

            # Create optimizer for DeepSpeed
            # Muon is now supported with ZeRO-2 (PyTorch 2.9+)
            if training_stage >= 0:
                # Staged training: use AdamW with only trainable params
                # Note: For staged training, we use simpler param groups since params are already filtered
                self._log_struct(
                    "Using AdamW optimizer with DeepSpeed (staged training)",
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    zero_stage=zero_stage,
                    stage=training_stage,
                )
                # Create proper param groups (no weight decay for bias/LayerNorm)
                param_groups = get_adamw_param_groups(model, config.weight_decay, config.learning_rate)
                optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
            elif optimizer_type == "muon" and zero_stage == 2:
                self._log_struct(
                    "Using PyTorch Muon optimizer with DeepSpeed ZeRO-2",
                    muon_lr=muon_lr,
                    adam_lr=config.learning_rate,
                    momentum=muon_momentum,
                    zero_stage=zero_stage,
                )
                optimizer = create_pytorch_muon_optimizer(
                    model,
                    muon_lr=muon_lr,
                    adam_lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    muon_momentum=muon_momentum,
                )
            elif optimizer_type == "muon":
                # Muon only supported with ZeRO-2
                log_warning(
                    "Muon optimizer is only compatible with DeepSpeed ZeRO-2. Falling back to AdamW.",
                    requested_optimizer=optimizer_type,
                    using_optimizer="adamw",
                    zero_stage=zero_stage,
                )
                # Create proper param groups (no weight decay for bias/LayerNorm)
                param_groups = get_adamw_param_groups(model, config.weight_decay, config.learning_rate)
                optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
            else:
                self._log_struct(
                    "Using AdamW optimizer with DeepSpeed",
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    zero_stage=zero_stage,
                )
                # Create proper param groups (no weight decay for bias/LayerNorm)
                param_groups = get_adamw_param_groups(model, config.weight_decay, config.learning_rate)
                optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

            # Update total_num_steps in ds_config scheduler if present
            if isinstance(ds_config, dict) and "scheduler" in ds_config:
                total_steps = len(train_loader) * config.max_epochs
                ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
                cos_min_ratio = ds_config["scheduler"]["params"].get("cos_min_ratio", 0.0001)
                self._log_struct(
                    "DeepSpeed scheduler configured",
                    type=ds_config["scheduler"]["type"],
                    warmup_steps=ds_config["scheduler"]["params"]["warmup_num_steps"],
                    total_steps=total_steps,
                    min_lr_ratio=f"{cos_min_ratio:.1%}",
                )

            # Initialize DeepSpeed
            self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=ds_config,
                model_parameters=trainable_params,
            )
            self.raw_model = self.model.module

            self._log_struct(
                "DeepSpeed initialized",
                rank=self.rank,
                world_size=self.world_size,
                zero_stage=zero_stage,
                optimizer=optimizer_type if optimizer_type == "muon" and zero_stage == 2 else "adamw",
                scheduler="WarmupDecayLR" if self.scheduler else "none",
            )

        # DDP setup (if not using DeepSpeed)
        elif use_ddp:
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
            self.raw_model = self.model.module
            self._log_struct("DDP initialized", rank=self.rank, world_size=self.world_size)

            # Apply parameter freezing for staged training
            if training_stage >= 0:
                self._freeze_parameters(training_stage)

            # Optimizer (use raw model for parameter groups)
            if training_stage >= 0:
                # Staged training: use stage-specific optimizer with only trainable params
                self.optimizer = self._create_stage_optimizer(training_stage, muon_lr, muon_momentum)
            elif optimizer_type == "muon":
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
                # Create proper param groups (no weight decay for bias/LayerNorm)
                param_groups = get_adamw_param_groups(self.raw_model, config.weight_decay, config.learning_rate)
                self.optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

            # Learning rate scheduler (warmup + cosine decay)
            self.scheduler = self._create_scheduler()

        # Single GPU / CPU setup
        else:
            self.rank = 0
            self.world_size = 1
            self.is_main = True
            self.model = model.to(device)
            self.raw_model = self.model

            # Apply parameter freezing for staged training
            if training_stage >= 0:
                self._freeze_parameters(training_stage)

            # Optimizer (use raw model for parameter groups)
            if training_stage >= 0:
                # Staged training: use stage-specific optimizer with only trainable params
                self.optimizer = self._create_stage_optimizer(training_stage, muon_lr, muon_momentum)
            elif optimizer_type == "muon":
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
                # Create proper param groups (no weight decay for bias/LayerNorm)
                param_groups = get_adamw_param_groups(self.raw_model, config.weight_decay, config.learning_rate)
                self.optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

            # Learning rate scheduler (warmup + cosine decay)
            self.scheduler = self._create_scheduler()

        # Curriculum learning: Start with fewer supervision steps, gradually increase
        # For staged training (stage 0/1), disable curriculum to avoid destabilization
        if training_stage == TrainingStage.BACKBONE:
            # Stage 0: Use fixed max_steps to avoid curriculum destabilization
            self.current_max_steps = config.max_supervision_steps
            self.step_increase_interval = 999999  # Effectively disabled
            self._log_struct(
                "Curriculum disabled for Stage 0",
                fixed_max_steps=self.current_max_steps,
                reason="Avoids destabilization from changing supervision depth",
            )
        elif training_stage == TrainingStage.GENERATION:
            # Stage 1: Use full supervision steps (generation needs full context)
            self.current_max_steps = config.max_supervision_steps
            self.step_increase_interval = 999999  # Effectively disabled
        else:
            # Standard training: use curriculum
            self.current_max_steps = 2  # Start with 2 steps
            self.step_increase_interval = 5  # Increase every 5 epochs

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_f1 = 0.0  # Use macro_f1 for best model selection (better for imbalanced datasets)
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0  # Counter for early stopping

        # Initialize EMA for stable recursive depth
        self.ema = None
        if use_ema:
            self.ema = EMAModel(self.model, decay=ema_decay, device=device)
            self._log_struct(
                "EMA initialized",
                decay=ema_decay,
                note="Stabilizes recursive reasoning depth",
            )

        # Log original TRM training mode
        if use_original_trm_training:
            self._log_struct(
                "Original TRM training mode enabled",
                note="Backward/step after EACH supervision step",
                deep_recursion_steps=getattr(config, 'deep_recursion_steps', 1),
                use_original_trm_grad=getattr(config, 'use_original_trm_grad', False),
            )

    def _get_default_ds_config(self) -> Dict[str, Any]:
        """Get default DeepSpeed configuration"""
        return {
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8
            },
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False
        }

    def _log(self, msg: str):
        """Print only from main process (uses global logger)"""
        log(msg)

    def _log_struct(self, msg: str, **kwargs):
        """Structured log only from main process"""
        log(msg, **kwargs)

    def _log_wandb(self, metrics: Dict[str, float], step: Optional[int] = None, commit: bool = True):
        """Log metrics to wandb (only from main process)

        Args:
            metrics: Dict of metric names to values
            step: Optional step number (uses global_step if not provided)
            commit: Whether to commit the log (default: True)
        """
        if self.wandb_run is not None and self.is_main:
            if step is None:
                step = self.global_step
            self.wandb_run.log(metrics, step=step, commit=commit)

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine decay

        Uses cosine annealing with a minimum LR floor of 10% to prevent
        the learning rate from decaying too low, which can cause training instability.
        """
        from torch.optim.lr_scheduler import LambdaLR
        import math

        # Minimum LR ratio (10% of initial LR)
        min_lr_ratio = 0.1

        def lr_lambda(current_step):
            # Warmup phase: linear increase from 0 to 1
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))

            # Cosine decay phase after warmup
            # Decays from 1.0 to min_lr_ratio using cosine annealing
            progress = float(current_step - self.config.warmup_steps)
            total_decay_steps = len(self.train_loader) * self.config.max_epochs - self.config.warmup_steps
            total_decay_steps = max(1, total_decay_steps)  # Avoid division by zero

            # Cosine annealing: (1 + cos(pi * progress / total)) / 2
            # This gives smooth decay from 1.0 to 0.0
            # We scale it to decay from 1.0 to min_lr_ratio instead
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress / total_decay_steps))

            # Scale to range [min_lr_ratio, 1.0]
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(self.optimizer, lr_lambda)

    def _freeze_parameters(self, stage: int):
        """Freeze/unfreeze parameters based on training stage

        Args:
            stage: Training stage (from TrainingStage constants)
        """
        if stage == TrainingStage.STANDARD or stage == TrainingStage.FULL_FINETUNE:
            # Unfreeze all parameters
            for param in self.raw_model.parameters():
                param.requires_grad = True
            self._log_struct("All parameters unfrozen", stage=stage)
            return

        # First freeze all parameters
        for param in self.raw_model.parameters():
            param.requires_grad = False

        # Define which modules to unfreeze for each stage
        # Use startswith to match top-level modules only (avoid matching submodules)
        if stage == TrainingStage.BACKBONE:
            # Stage 0: Unfreeze everything except generation_head
            unfreeze_prefixes = [
                "token_embedding.", "position_embedding.", "embedding_dropout.",
                "encoder.", "reasoning_module.", "action_module.", "output_heads.", "init_y"
            ]
        elif stage == TrainingStage.GENERATION:
            # Stage 1: Unfreeze only generation_head
            unfreeze_prefixes = ["generation_head."]
        else:
            unfreeze_prefixes = []

        # Unfreeze matching parameters
        unfrozen_count = 0
        frozen_count = 0
        for name, param in self.raw_model.named_parameters():
            # Check if name starts with any of the prefixes (top-level module match)
            should_unfreeze = any(name.startswith(prefix) or name == prefix.rstrip('.') for prefix in unfreeze_prefixes)
            if should_unfreeze:
                param.requires_grad = True
                unfrozen_count += param.numel()
            else:
                frozen_count += param.numel()

        self._log_struct(
            "Parameter freezing applied",
            stage=stage,
            trainable_params=f"{unfrozen_count / 1e6:.2f}M",
            frozen_params=f"{frozen_count / 1e6:.2f}M",
            unfreeze_prefixes=unfreeze_prefixes,
        )

    def _create_stage_optimizer(self, stage: int, muon_lr: float, muon_momentum: float):
        """Create optimizer for the current training stage

        Only includes trainable parameters (unfrozen).

        Args:
            stage: Training stage
            muon_lr: Learning rate for Muon
            muon_momentum: Momentum for Muon

        Returns:
            Optimizer configured for the stage
        """
        trainable_params = [p for p in self.raw_model.parameters() if p.requires_grad]

        if not trainable_params:
            raise ValueError(f"No trainable parameters for stage {stage}")

        trainable_count = sum(p.numel() for p in trainable_params)
        self._log_struct(
            "Creating stage optimizer",
            stage=stage,
            trainable_params=f"{trainable_count / 1e6:.2f}M",
        )

        # Use Muon for backbone training (stage 0, 2, -1), AdamW for generation (stage 1)
        if self.optimizer_type == "muon" and stage != TrainingStage.GENERATION:
            # For Muon, we need to categorize params
            muon_params = []
            adamw_params = []

            for name, param in self.raw_model.named_parameters():
                if not param.requires_grad:
                    continue

                is_hidden_module = any(
                    mod in name for mod in ["encoder", "reasoning_module", "action_module"]
                )

                if is_hidden_module and param.ndim == 2:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            if muon_params:
                self._log_struct(
                    "Stage optimizer using Muon",
                    muon_params=f"{sum(p.numel() for p in muon_params) / 1e6:.2f}M",
                    adamw_params=f"{sum(p.numel() for p in adamw_params) / 1e6:.2f}M",
                )
                param_groups = [
                    {
                        "params": muon_params,
                        "lr": muon_lr,
                        "weight_decay": self.config.weight_decay,
                        "momentum": muon_momentum,
                        "use_muon": True,
                    },
                    {
                        "params": adamw_params,
                        "lr": self.config.learning_rate,
                        "betas": (0.9, 0.95),
                        "weight_decay": self.config.weight_decay,
                        "use_muon": False,
                    },
                ]
                return MuonAdamW(param_groups)

        # Use AdamW for generation head (simpler) or fallback
        # Create proper param groups (no weight decay for bias/LayerNorm)
        no_decay = ["bias", "LayerNorm", "layer_norm", "layernorm", "ln_"]
        decay_params = []
        no_decay_params = []

        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self._log_struct(
            "Stage optimizer using AdamW",
            lr=self.config.learning_rate,
            with_weight_decay=f"{sum(p.numel() for p in decay_params) / 1e6:.2f}M",
            without_weight_decay=f"{sum(p.numel() for p in no_decay_params) / 1e6:.2f}M",
        )

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay, "lr": self.config.learning_rate},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": self.config.learning_rate},
        ]
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    def _compute_stage_loss(
        self,
        outputs_per_step,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss appropriate for training stage

        Stage 0: action + num_calls + Q losses (no generation)
        Stage 1: generation loss only
        Stage -1/2: Full multi-component loss

        Args:
            outputs_per_step: Model outputs per supervision step
            batch: Batch dict with targets

        Returns:
            loss: Tensor loss value
            loss_dict: Dict with loss components
        """
        stage = self.training_stage

        if stage == TrainingStage.STANDARD or stage == TrainingStage.FULL_FINETUNE:
            # Full loss for standard training and fine-tuning
            return compute_trm_loss(
                outputs_per_step,
                batch,
                self.config,
                special_token_ids=self.special_token_ids if self.special_token_ids else None,
                tool_call_token_id=self.tool_call_token_id,
            )

        device = outputs_per_step[-1]['action_logits'].device
        final_output = outputs_per_step[-1]

        if stage == TrainingStage.BACKBONE:
            # Stage 0: action + num_calls + Q losses (no generation)

            # Get class weights for action loss (handle imbalanced datasets)
            action_class_weights = getattr(self.config, 'action_class_weights', None)
            if action_class_weights is not None:
                action_weight_tensor = torch.tensor(action_class_weights, device=device)
            else:
                # Auto-compute from batch
                num_direct = (batch['target_action'] == 0).sum().float()
                num_tool = (batch['target_action'] == 1).sum().float()
                total = num_direct + num_tool
                if num_direct > 0 and num_tool > 0:
                    weight_direct = total / (2 * num_direct)
                    weight_tool = total / (2 * num_tool)
                    action_weight_tensor = torch.tensor([weight_direct, weight_tool], device=device)
                else:
                    action_weight_tensor = None

            # Action loss with Focal Loss support
            use_focal_loss = getattr(self.config, 'use_focal_loss', True)
            focal_gamma = getattr(self.config, 'focal_gamma', 2.0)

            if use_focal_loss:
                from trm_llm.training.loss import FocalLoss
                action_loss_fn = FocalLoss(gamma=focal_gamma, alpha=action_weight_tensor)
                action_loss = action_loss_fn(
                    final_output['action_logits'],
                    batch['target_action']
                )
            else:
                action_loss = F.cross_entropy(
                    final_output['action_logits'],
                    batch['target_action'],
                    weight=action_weight_tensor,
                    label_smoothing=self.config.label_smoothing,
                )

            # Num calls loss (only for tool_call samples)
            tool_mask = (batch['target_action'] == 1)
            if tool_mask.any():
                target_num_calls_idx = batch['target_num_calls'][tool_mask] - 1
                max_classes = final_output['num_calls_logits'].size(-1)
                target_num_calls_idx = target_num_calls_idx.clamp(0, max_classes - 1)
                num_calls_loss = F.cross_entropy(
                    final_output['num_calls_logits'][tool_mask],
                    target_num_calls_idx,
                )
            else:
                num_calls_loss = torch.tensor(0.0, device=device)

            # Q loss (correctness prediction, TRM paper)
            # Target: 1.0 if prediction matches ground truth, else 0.0
            # Q-head learns to predict if the action prediction is correct
            num_steps = len(outputs_per_step)
            q_loss = torch.tensor(0.0, device=device)
            for step_idx, step_output in enumerate(outputs_per_step):
                pred_action = step_output['action_logits'].argmax(dim=-1)
                is_correct = (pred_action == batch['target_action']).float()
                q_loss = q_loss + F.binary_cross_entropy_with_logits(
                    step_output['q_logit'].squeeze(-1), is_correct
                )
            q_loss = q_loss / num_steps

            num_calls_loss_weight = getattr(self.config, 'num_calls_loss_weight', 1.0)
            total_loss = (
                self.config.action_loss_weight * action_loss +
                num_calls_loss_weight * num_calls_loss +
                self.config.q_loss_weight * q_loss
            )

            return total_loss, {
                'action': action_loss.item(),
                'num_calls': num_calls_loss.item(),
                'q': q_loss.item(),
                'total': total_loss.item(),
            }

        elif stage == TrainingStage.GENERATION:
            # Stage 1: generation loss only
            if 'generation_logits' not in final_output:
                # No generation target provided
                return torch.tensor(0.0, device=device, requires_grad=True), {'generation': 0.0}

            generation_logits = final_output['generation_logits']
            target_ids = batch.get('target_generation_ids')

            if target_ids is None:
                return torch.tensor(0.0, device=device, requires_grad=True), {'generation': 0.0}

            # Shift for autoregressive loss
            shift_logits = generation_logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            # Cross entropy loss
            gen_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )

            return gen_loss, {'generation': gen_loss.item(), 'total': gen_loss.item()}

        # Fallback to full loss
        return compute_trm_loss(
            outputs_per_step,
            batch,
            self.config,
            special_token_ids=self.special_token_ids if self.special_token_ids else None,
            tool_call_token_id=self.tool_call_token_id,
        )

    def _compute_single_step_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        step_idx: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Compute loss for a single supervision step (for original TRM training)

        Args:
            outputs: Single step outputs from forward_single_step
            batch: Batch dict with targets
            step_idx: Current step index
            num_steps: Total number of supervision steps

        Returns:
            loss: Scalar loss value for this step
        """
        device = outputs['action_logits'].device

        # Get class weights for action loss
        action_class_weights = getattr(self.config, 'action_class_weights', None)
        if action_class_weights is not None:
            action_weight_tensor = torch.tensor(action_class_weights, device=device)
        else:
            # Auto-compute from batch
            num_direct = (batch['target_action'] == 0).sum().float()
            num_tool = (batch['target_action'] == 1).sum().float()
            total = num_direct + num_tool
            if num_direct > 0 and num_tool > 0:
                weight_direct = total / (2 * num_direct)
                weight_tool = total / (2 * num_tool)
                action_weight_tensor = torch.tensor([weight_direct, weight_tool], device=device)
            else:
                action_weight_tensor = None

        # Action loss with Focal Loss support
        use_focal_loss = getattr(self.config, 'use_focal_loss', True)
        focal_gamma = getattr(self.config, 'focal_gamma', 2.0)

        if use_focal_loss:
            from trm_llm.training.loss import FocalLoss
            action_loss_fn = FocalLoss(gamma=focal_gamma, alpha=action_weight_tensor)
            action_loss = action_loss_fn(outputs['action_logits'], batch['target_action'])
        else:
            action_loss = F.cross_entropy(
                outputs['action_logits'],
                batch['target_action'],
                weight=action_weight_tensor,
                label_smoothing=self.config.label_smoothing,
            )

        # Num calls loss (only for tool_call samples)
        tool_mask = (batch['target_action'] == 1)
        if tool_mask.any():
            target_num_calls_idx = batch['target_num_calls'][tool_mask] - 1
            max_classes = outputs['num_calls_logits'].size(-1)
            target_num_calls_idx = target_num_calls_idx.clamp(0, max_classes - 1)
            num_calls_loss = F.cross_entropy(
                outputs['num_calls_logits'][tool_mask],
                target_num_calls_idx,
            )
        else:
            num_calls_loss = torch.tensor(0.0, device=device)

        # Q loss (correctness prediction)
        pred_action = outputs['action_logits'].argmax(dim=-1)
        is_correct = (pred_action == batch['target_action']).float()
        q_loss = F.binary_cross_entropy_with_logits(
            outputs['q_logit'].squeeze(-1), is_correct
        )

        # Combine losses
        num_calls_loss_weight = getattr(self.config, 'num_calls_loss_weight', 1.0)
        total_loss = (
            self.config.action_loss_weight * action_loss +
            num_calls_loss_weight * num_calls_loss +
            self.config.q_loss_weight * q_loss
        )

        # Generation loss only on last step
        is_last_step = (step_idx == num_steps - 1)
        if is_last_step and 'generation_logits' in outputs:
            generation_logits = outputs['generation_logits']
            target_ids = batch.get('target_generation_ids')

            if target_ids is not None:
                # Shift for autoregressive loss
                shift_logits = generation_logits[:, :-1, :].contiguous()
                shift_labels = target_ids[:, 1:].contiguous()

                gen_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=self.config.label_smoothing,
                )

                # Apply action-type weighting for generation
                # Average weights across tool_call and direct_answer
                tool_call_gen_weight = getattr(self.config, 'tool_call_gen_weight', 2.0)
                direct_answer_gen_weight = getattr(self.config, 'direct_answer_gen_weight', 1.0)
                avg_gen_weight = (tool_call_gen_weight + direct_answer_gen_weight) / 2
                total_loss = total_loss + avg_gen_weight * gen_loss

        return total_loss

    def train_step_original_trm(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, Dict[str, float]]:
        """Training step with original TRM gradient flow

        Backward/step happens after EACH supervision step, not accumulated.
        This matches the original TRM paper's training procedure.

        Args:
            batch: Batch dict with input_ids, target_action, target_tool_id

        Returns:
            loss: Average loss across all supervision steps
            metrics: Dict with averaged loss components and accuracies
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        batch_size = batch["input_ids"].size(0)
        max_steps = self.current_max_steps

        # Encode input once (shared across all supervision steps)
        # Note: We detach x_encoded because multiple backward() calls on the same graph
        # would cause issues. The encoder is trained through standard training mode.
        # Original TRM training focuses on the reasoning/action modules.
        if self.use_deepspeed:
            with torch.cuda.amp.autocast(enabled=True):
                x_encoded = self.raw_model.encode(batch["input_ids"]).detach()
        else:
            x_encoded = self.raw_model.encode(batch["input_ids"]).detach()

        # Initialize states
        y, z = self.raw_model.init_states(batch_size, self.device)

        # Track metrics across steps
        total_loss = 0.0
        outputs_per_step = []

        # === Original TRM training loop ===
        # Backward/step after EACH supervision step
        for step in range(max_steps):
            is_last_step = (step == max_steps - 1)

            # Forward single step
            if self.use_deepspeed:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.raw_model.forward_single_step(
                        x_encoded=x_encoded,
                        y=y,
                        z=z,
                        step=step,
                        target_generation_ids=batch.get("target_generation_ids") if is_last_step else None,
                        is_last_step=is_last_step,
                    )
            else:
                outputs = self.raw_model.forward_single_step(
                    x_encoded=x_encoded,
                    y=y,
                    z=z,
                    step=step,
                    target_generation_ids=batch.get("target_generation_ids") if is_last_step else None,
                    is_last_step=is_last_step,
                )

            outputs_per_step.append(outputs)

            # Compute loss for this step
            loss = self._compute_single_step_loss(outputs, batch, step, max_steps)

            # Check for inf/nan loss
            if torch.isnan(loss) or torch.isinf(loss):
                self._log_struct(
                    "Warning: inf/nan loss detected in original TRM training",
                    step=step,
                    loss=loss.item(),
                    batch_size=batch_size,
                )
                # Skip this step to avoid corrupting model
                continue

            total_loss += loss.item()

            # Backward and step for THIS supervision step
            if self.use_deepspeed:
                self.model.backward(loss)
                self.model.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update EMA after each step
            if self.ema is not None:
                self.ema.update()

            # Get updated states for next step (already detached by _deep_recursion if use_original_trm_grad)
            y = outputs["y_state"]
            z = outputs["z_state"]

            # For original TRM training, states are already detached in _deep_recursion
            # But we need to handle the case where use_original_trm_grad=False
            use_original_grad = getattr(self.config, 'use_original_trm_grad', False)
            if not use_original_grad:
                y = y.detach()
                z = z.detach()

        self.global_step += 1

        # Step scheduler ONCE per batch (not per supervision step)
        if not self.use_deepspeed and self.scheduler is not None:
            self.scheduler.step()

        # Compute accuracies from final outputs
        acc_dict = compute_action_accuracy(outputs_per_step, batch)

        # Compute valid tool call format accuracy
        if self.tokenizer is not None:
            (valid_count, total_count, sample_correct, sample_target,
             sample_decoded_tokens, target_decoded_tokens) = compute_valid_tool_call_format_accuracy(
                outputs_per_step, batch, self.tokenizer, return_counts=True, return_sample=True, return_tokens=True
            )
            acc_dict["valid_tool_format_correct"] = valid_count
            acc_dict["valid_tool_format_total"] = total_count
            if valid_count > 0 and sample_correct is not None and self.is_main:
                self._log_struct(
                    "Valid tool call prediction",
                    step=self.global_step,
                    prediction=sample_correct,
                    target=sample_target,
                )

        # Get learning rate
        if self.use_deepspeed:
            lr = self.model.get_lr()[0] if hasattr(self.model, 'get_lr') else self.config.learning_rate
        elif self.scheduler is not None:
            lr = self.scheduler.get_last_lr()[0]
        else:
            lr = self.config.learning_rate

        # Average loss across steps
        avg_loss = total_loss / max_steps

        # Build metrics dict
        metrics = {
            "total": avg_loss,
            "learning_rate": lr,
            **acc_dict,
        }

        # Log to wandb (every step)
        if self.wandb_run is not None and self.is_main:
            wandb_metrics = {
                "train/loss": avg_loss,
                "train/learning_rate": lr,
                "train/action_accuracy": acc_dict.get("action_accuracy", 0.0),
                "train/macro_f1": acc_dict.get("macro_f1", 0.0),
                "train/tool_call_f1": acc_dict.get("tool_call_f1", 0.0),
                "train/direct_answer_f1": acc_dict.get("direct_answer_f1", 0.0),
                "train/tool_gen_accuracy": acc_dict.get("tool_gen_accuracy", 0.0),
                "train/direct_gen_accuracy": acc_dict.get("direct_gen_accuracy", 0.0),
            }
            self._log_wandb(wandb_metrics)

        return avg_loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Single training step

        Args:
            batch: Batch dict with input_ids, target_action, target_tool_id

        Returns:
            loss: Scalar loss value
            metrics: Dict with loss components and accuracies
        """
        # Use original TRM training if enabled
        if self.use_original_trm_training:
            return self.train_step_original_trm(batch)

        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with deep supervision
        # Use autocast for DeepSpeed FP16 to handle dtype consistency
        if self.use_deepspeed:
            with torch.cuda.amp.autocast(enabled=True):
                outputs_per_step = self.model(
                    batch["input_ids"],
                    max_supervision_steps=self.current_max_steps,
                    training=True,
                    target_generation_ids=batch.get("target_generation_ids"),
                )

                # Compute loss (stage-aware for staged training)
                loss, loss_dict = self._compute_stage_loss(outputs_per_step, batch)
        else:
            outputs_per_step = self.model(
                batch["input_ids"],
                max_supervision_steps=self.current_max_steps,
                training=True,
                target_generation_ids=batch.get("target_generation_ids"),
            )

            # Compute loss (stage-aware for staged training)
            loss, loss_dict = self._compute_stage_loss(outputs_per_step, batch)

        # Compute accuracies
        acc_dict = compute_action_accuracy(outputs_per_step, batch)

        # Compute strict tool call format accuracy: <tool_call>{...}</tool_call>
        # Return counts for proper aggregation (avoids dilution from batches without tool_calls)
        if self.tokenizer is not None:
            (valid_count, total_count, sample_correct, sample_target,
             sample_decoded_tokens, target_decoded_tokens) = compute_valid_tool_call_format_accuracy(
                outputs_per_step, batch, self.tokenizer, return_counts=True, return_sample=True, return_tokens=True
            )
            acc_dict["valid_tool_format_correct"] = valid_count
            acc_dict["valid_tool_format_total"] = total_count
            # Log sample correct prediction if any valid
            if valid_count > 0 and sample_correct is not None and self.is_main:
                self._log_struct(
                    "Valid tool call prediction",
                    step=self.global_step,
                    prediction=sample_correct,
                    target=sample_target,
                )
                # Log decoded tokens for debugging
                if sample_decoded_tokens is not None:
                    self._log_struct(
                        "Decoded tokens",
                        step=self.global_step,
                        prediction_tokens=sample_decoded_tokens,
                        target_tokens=target_decoded_tokens,
                    )

        # Backward pass and optimizer step
        if self.use_deepspeed:
            # DeepSpeed handles backward, gradient clipping, and optimizer step
            self.model.backward(loss)
            self.model.step()
            # Get learning rate from DeepSpeed
            lr = self.model.get_lr()[0] if hasattr(self.model, 'get_lr') else self.config.learning_rate
        else:
            # Standard PyTorch backward
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.config.learning_rate

        self.global_step += 1

        # Update EMA after each step (stabilizes recursive reasoning)
        if self.ema is not None:
            self.ema.update()

        # Combine metrics
        metrics = {**loss_dict, **acc_dict}
        metrics["learning_rate"] = lr

        # Log to wandb (every step)
        if self.wandb_run is not None and self.is_main:
            wandb_metrics = {
                "train/loss": loss.item(),
                "train/learning_rate": lr,
                "train/action_accuracy": acc_dict.get("action_accuracy", 0.0),
                "train/macro_f1": acc_dict.get("macro_f1", 0.0),
                "train/tool_call_f1": acc_dict.get("tool_call_f1", 0.0),
                "train/direct_answer_f1": acc_dict.get("direct_answer_f1", 0.0),
                "train/tool_gen_accuracy": acc_dict.get("tool_gen_accuracy", 0.0),
                "train/direct_gen_accuracy": acc_dict.get("direct_gen_accuracy", 0.0),
            }
            # Add loss components
            for key, value in loss_dict.items():
                wandb_metrics[f"train/loss_{key}"] = value
            self._log_wandb(wandb_metrics)

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
        if (self.use_ddp or self.use_deepspeed) and hasattr(self.train_loader.sampler, "set_epoch"):
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
                avg_macro_f1 = total_metrics.get("macro_f1", 0.0) / n
                avg_tool_call_f1 = total_metrics.get("tool_call_f1", 0.0) / n
                avg_direct_f1 = total_metrics.get("direct_answer_f1", 0.0) / n
                avg_tool_gen = total_metrics.get("tool_gen_accuracy", 0.0) / n
                avg_direct_gen = total_metrics.get("direct_gen_accuracy", 0.0) / n
                current_lr = metrics.get("learning_rate", self.config.learning_rate)
                # Compute tool_fmt from accumulated counts (not averaged ratios)
                tool_fmt_correct = total_metrics.get("valid_tool_format_correct", 0)
                tool_fmt_total = total_metrics.get("valid_tool_format_total", 0)
                avg_tool_fmt = tool_fmt_correct / tool_fmt_total if tool_fmt_total > 0 else 0.0
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "F1": f"{avg_macro_f1:.3f}",
                        "tc_F1": f"{avg_tool_call_f1:.3f}",
                        "da_F1": f"{avg_direct_f1:.3f}",
                        "tc_gen": f"{avg_tool_gen:.3f}",
                        "da_gen": f"{avg_direct_gen:.3f}",
                        "tc_fmt": f"{avg_tool_fmt:.3f}",
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

        # Log epoch summary to wandb
        if self.wandb_run is not None and self.is_main:
            epoch_metrics = {
                "epoch": epoch + 1,
                "epoch/train_loss": avg_metrics["loss"],
                "epoch/train_action_accuracy": avg_metrics.get("action_accuracy", 0.0),
                "epoch/train_macro_f1": avg_metrics.get("macro_f1", 0.0),
                "epoch/train_tool_call_f1": avg_metrics.get("tool_call_f1", 0.0),
                "epoch/train_direct_answer_f1": avg_metrics.get("direct_answer_f1", 0.0),
                "epoch/train_tool_gen_accuracy": avg_metrics.get("tool_gen_accuracy", 0.0),
                "epoch/train_direct_gen_accuracy": avg_metrics.get("direct_gen_accuracy", 0.0),
                "epoch/train_valid_tool_format": avg_metrics.get("valid_tool_format", 0.0),
                "epoch/max_supervision_steps": self.current_max_steps,
            }
            self._log_wandb(epoch_metrics)

        return avg_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set

        Uses EMA weights if enabled for more stable evaluation.

        Returns:
            val_metrics: Dict with validation metrics
        """
        if self.val_loader is None:
            return {}

        # Apply EMA weights for validation (if enabled)
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        total_loss = 0.0
        total_metrics = {}

        # Only show progress bar on main process
        if self.is_main:
            loader = tqdm(self.val_loader, desc="Validating (EMA)" if self.ema else "Validating")
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

        # Log validation metrics to wandb
        if self.wandb_run is not None and self.is_main:
            wandb_val_metrics = {
                "val/loss": val_metrics["val_loss"],
                "val/action_accuracy": val_metrics.get("val_action_accuracy", 0.0),
                "val/macro_f1": val_metrics.get("val_macro_f1", 0.0),
                "val/tool_call_f1": val_metrics.get("val_tool_call_f1", 0.0),
                "val/direct_answer_f1": val_metrics.get("val_direct_answer_f1", 0.0),
                "val/tool_gen_accuracy": val_metrics.get("val_tool_gen_accuracy", 0.0),
                "val/direct_gen_accuracy": val_metrics.get("val_direct_gen_accuracy", 0.0),
                "val/valid_tool_format": val_metrics.get("val_valid_tool_format", 0.0),
            }
            self._log_wandb(wandb_val_metrics)

        # Restore original weights after validation (if using EMA)
        if self.ema is not None:
            self.ema.restore()

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
        """Save model checkpoint (only on main process, or all ranks for DeepSpeed)

        Args:
            epoch: Current epoch
            metrics: Current metrics
            filepath: Path to save checkpoint
        """
        if self.use_deepspeed:
            # DeepSpeed saves checkpoints to a directory
            save_dir = filepath.replace(".pt", "_ds")
            client_state = {
                "epoch": epoch,
                "global_step": self.global_step,
                "config": self.config,
                "metrics": metrics,
                "current_max_steps": self.current_max_steps,
                "training_stage": self.training_stage,
            }
            # Include EMA state if enabled
            if self.ema is not None:
                client_state["ema_state_dict"] = self.ema.state_dict()

            # Pre-save barrier to ensure all ranks start checkpoint save together
            dist.barrier()

            self.model.save_checkpoint(save_dir, client_state=client_state)
            if self.is_main:
                log("DeepSpeed checkpoint saved", path=save_dir, epoch=epoch + 1, ema=self.ema is not None)

            # Barrier to ensure all ranks have finished saving before extracting FP32 weights
            dist.barrier()

            # Small delay to ensure filesystem sync (NFS/shared storage may have lag)
            import time
            time.sleep(1.0)

            # Also save a regular checkpoint for inference (main process only)
            # IMPORTANT: Extract FP32 weights from DeepSpeed checkpoint, NOT from raw_model.state_dict()
            # With ZeRO optimization, raw_model.state_dict() returns FP16/BF16 weights, not FP32 master weights
            if self.is_main:
                os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
                try:
                    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                    log("Extracting FP32 weights from DeepSpeed checkpoint for inference...")

                    # Retry logic for FP32 extraction (in case of filesystem lag)
                    fp32_state_dict = None
                    max_retries = 3
                    last_error = None
                    for retry in range(max_retries):
                        try:
                            fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(save_dir)
                            break  # Success
                        except Exception as e:
                            last_error = e
                            if retry < max_retries - 1:
                                log_warning(f"FP32 extraction attempt {retry + 1} failed, retrying...", error=str(e))
                                time.sleep(2.0)  # Wait longer before retry

                    if fp32_state_dict is None:
                        raise last_error if last_error else RuntimeError("Failed to extract FP32 weights")

                    # Note: EMA is not applied here since EMA shadow weights are also in FP16/BF16
                    # The FP32 extraction from DeepSpeed checkpoint gives us the proper training weights
                    inference_checkpoint = {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "model_state_dict": fp32_state_dict,
                        "config": self.config,
                        "metrics": metrics,
                        "current_max_steps": self.current_max_steps,
                        "training_stage": self.training_stage,
                        "used_ema": False,  # EMA not applied when using FP32 extraction
                    }
                    torch.save(inference_checkpoint, filepath)
                    log("Inference checkpoint saved (FP32)", path=filepath, epoch=epoch + 1, params=len(fp32_state_dict))
                except Exception as e:
                    log_warning(
                        "Could not extract FP32 weights from DeepSpeed checkpoint. "
                        "Use --ds_checkpoint for inference instead of .pt file.",
                        error=str(e),
                    )
                    # Fallback: save FP16/BF16 weights with warning
                    inference_checkpoint = {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "model_state_dict": self.raw_model.state_dict(),
                        "config": self.config,
                        "metrics": metrics,
                        "current_max_steps": self.current_max_steps,
                        "training_stage": self.training_stage,
                        "used_ema": False,
                        "fp16_weights": True,  # Flag to indicate these are FP16 weights
                    }
                    torch.save(inference_checkpoint, filepath)
                    log_warning("Inference checkpoint saved with FP16/BF16 weights (may have precision issues)", path=filepath)

            # Final barrier to sync all ranks before continuing
            # This ensures rank 0 finishes its extra work before other ranks proceed
            dist.barrier()
        else:
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
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.config,
                "metrics": metrics,
                "current_max_steps": self.current_max_steps,
                "training_stage": self.training_stage,
            }
            # Include EMA state if enabled
            if self.ema is not None:
                checkpoint["ema_state_dict"] = self.ema.state_dict()

            torch.save(checkpoint, filepath)
            log("Checkpoint saved", path=filepath, epoch=epoch + 1, ema=self.ema is not None)

    def load_checkpoint(self, filepath: str, ds_checkpoint_dir: str = None):
        """Load model checkpoint

        Args:
            filepath: Path to checkpoint (or directory for DeepSpeed)
            ds_checkpoint_dir: Explicit DeepSpeed checkpoint directory (use --ds_checkpoint).
                               If provided, this takes priority over filepath for DeepSpeed.
        """
        if self.use_deepspeed:
            # Determine DeepSpeed checkpoint path
            if ds_checkpoint_dir and os.path.isdir(ds_checkpoint_dir):
                ds_path = ds_checkpoint_dir
            else:
                # Try auto-detecting from filepath
                ds_path = filepath.replace(".pt", "_ds")

            if os.path.isdir(ds_path):
                _, client_state = self.model.load_checkpoint(ds_path)
                self.current_epoch = client_state.get("epoch", 0)
                self.global_step = client_state.get("global_step", 0)
                self.current_max_steps = client_state.get("current_max_steps", 2)

                # Get current LR to verify scheduler was restored
                current_lr = self.model.get_lr()[0] if hasattr(self.model, 'get_lr') else None

                # Load EMA state if available
                if self.ema is not None and "ema_state_dict" in client_state:
                    self.ema.load_state_dict(client_state["ema_state_dict"])
                    self._log_struct(
                        "DeepSpeed checkpoint loaded with EMA",
                        path=ds_path,
                        epoch=self.current_epoch + 1,
                        step=self.global_step,
                        current_lr=f"{current_lr:.2e}" if current_lr else "N/A",
                        ema_decay=self.ema.decay,
                        scheduler_restored=True,
                    )
                else:
                    self._log_struct(
                        "DeepSpeed checkpoint loaded",
                        path=ds_path,
                        epoch=self.current_epoch + 1,
                        step=self.global_step,
                        current_lr=f"{current_lr:.2e}" if current_lr else "N/A",
                        scheduler_restored=True,
                    )
            else:
                # Fall back to regular checkpoint (for inference weights only)
                log_warning(
                    "DeepSpeed checkpoint directory not found. Loading from .pt file. "
                    "Optimizer and scheduler states will NOT be restored! "
                    "Use --ds_checkpoint to specify the DeepSpeed checkpoint directory.",
                    expected_ds_path=ds_path,
                    fallback_path=filepath,
                )
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
                self.raw_model.load_state_dict(checkpoint["model_state_dict"])
                self.current_epoch = checkpoint.get("epoch", 0)
                self.global_step = checkpoint.get("global_step", 0)
                self.current_max_steps = checkpoint.get("current_max_steps", 2)
                self._log_struct(
                    "Regular checkpoint loaded into DeepSpeed (NO optimizer/scheduler)",
                    path=filepath,
                    epoch=self.current_epoch + 1,
                    scheduler_restored=False,
                )
        else:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

            # Load into raw model (handles both DDP and non-DDP)
            self.raw_model.load_state_dict(checkpoint["model_state_dict"])

            optimizer_restored = False
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                optimizer_restored = True

            scheduler_restored = False
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                scheduler_restored = True

            self.current_epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            self.current_max_steps = checkpoint.get("current_max_steps", 2)

            # Load EMA state if available and EMA is enabled
            ema_loaded = False
            if self.ema is not None and "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
                ema_loaded = True

            # Get current LR after restoration
            current_lr = None
            if self.scheduler:
                current_lr = self.scheduler.get_last_lr()[0]

            self._log_struct(
                "Checkpoint loaded",
                path=filepath,
                epoch=self.current_epoch + 1,
                step=self.global_step,
                current_lr=f"{current_lr:.2e}" if current_lr else "N/A",
                optimizer_restored=optimizer_restored,
                scheduler_restored=scheduler_restored,
                ema=ema_loaded,
            )

    def load_checkpoint_for_stage(self, filepath: str, new_stage: int):
        """Load checkpoint from previous stage and prepare for new stage

        This method:
        1. Loads model weights from checkpoint (ignores optimizer state)
        2. Applies new freezing scheme for the new stage
        3. Recreates optimizer for trainable params only
        4. Resets training state (epoch, global_step)

        Args:
            filepath: Path to checkpoint from previous stage
            new_stage: The new training stage to transition to
        """
        # Load checkpoint
        if self.use_deepspeed:
            # For DeepSpeed, load from regular checkpoint (not DeepSpeed checkpoint)
            # because we're changing the optimizer
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        else:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Load only model weights (not optimizer state)
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])

        prev_stage = checkpoint.get("training_stage", -1)
        self._log_struct(
            "Stage transition: loaded model weights",
            path=filepath,
            previous_stage=prev_stage,
            new_stage=new_stage,
        )

        # Update training stage
        self.training_stage = new_stage

        # Apply new freezing scheme
        self._freeze_parameters(new_stage)

        # Recreate optimizer for new trainable params (for non-DeepSpeed)
        if not self.use_deepspeed:
            self.optimizer = self._create_stage_optimizer(
                new_stage, self.muon_lr, self.muon_momentum
            )
            # Recreate scheduler with new optimizer
            self.scheduler = self._create_scheduler()

        # Reset training state
        self.current_epoch = 0
        self.global_step = 0
        self.current_max_steps = 2  # Reset curriculum

        # Reinitialize EMA with new trainable params
        if self.ema is not None:
            self.ema = EMAModel(self.model, decay=self.ema_decay, device=self.device)
            self._log_struct(
                "EMA reinitialized for new stage",
                decay=self.ema_decay,
            )

        self._log_struct(
            "Stage transition complete",
            new_stage=new_stage,
            trainable_params=f"{sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad) / 1e6:.2f}M",
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
        if self.use_ddp or self.use_deepspeed:
            training_info["world_size"] = f"{self.world_size} GPUs"
            training_info["effective_batch_size"] = self.config.batch_size * self.world_size
        self._log_struct("Starting TRM-LLM Training", **training_info)

        # current_epoch is the last completed epoch from checkpoint
        # Resume from the next epoch (current_epoch + 1)
        # If starting fresh (current_epoch=0), start from epoch 0
        start_epoch = self.current_epoch + 1 if self.current_epoch > 0 else 0
        if start_epoch > 0:
            self._log_struct("Resuming training", from_epoch=start_epoch + 1, total_epochs=self.config.max_epochs)
        for epoch in range(start_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            # Curriculum: Gradually increase max supervision steps
            self.update_curriculum(epoch)

            # Train epoch
            # Get learning rate
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
            elif self.use_deepspeed and hasattr(self.model, 'get_lr'):
                current_lr = self.model.get_lr()[0]
            else:
                current_lr = self.config.learning_rate

            self._log_struct(
                "Epoch started",
                epoch=f"{epoch+1}/{self.config.max_epochs}",
                max_supervision_steps=self.current_max_steps,
                learning_rate=f"{current_lr:.6f}",
            )

            train_metrics = self.train_epoch(epoch)

            # Synchronize metrics across processes (for DDP/DeepSpeed)
            if self.use_ddp or self.use_deepspeed:
                dist.barrier()

            # Print training metrics (main process only)
            self._log_struct(
                "Training results",
                loss=f"{train_metrics['loss']:.4f}",
                action_acc=f"{train_metrics['action_accuracy']:.3f}",
                macro_f1=f"{train_metrics.get('macro_f1', 0.0):.3f}",
                valid_tool_format=f"{train_metrics.get('valid_tool_format', 0.0):.3f}",
            )
            # Per-class metrics (important for imbalanced datasets)
            self._log_struct(
                "Per-class metrics",
                direct_answer_acc=f"{train_metrics.get('direct_answer_acc', 0.0):.3f}",
                direct_answer_f1=f"{train_metrics.get('direct_answer_f1', 0.0):.3f}",
                tool_call_acc=f"{train_metrics.get('tool_call_acc', 0.0):.3f}",
                tool_call_f1=f"{train_metrics.get('tool_call_f1', 0.0):.3f}",
            )
            # Generation metrics
            self._log_struct(
                "Generation metrics",
                tool_gen_acc=f"{train_metrics.get('tool_gen_accuracy', 0.0):.3f}",
                direct_gen_acc=f"{train_metrics.get('direct_gen_accuracy', 0.0):.3f}",
            )

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                self._log_struct(
                    "Validation results",
                    loss=f"{val_metrics['val_loss']:.4f}",
                    action_acc=f"{val_metrics['val_action_accuracy']:.3f}",
                    macro_f1=f"{val_metrics.get('val_macro_f1', 0.0):.3f}",
                    valid_tool_format=f"{val_metrics.get('val_valid_tool_format', 0.0):.3f}",
                )
                # Per-class validation metrics
                self._log_struct(
                    "Val per-class metrics",
                    direct_answer_acc=f"{val_metrics.get('val_direct_answer_acc', 0.0):.3f}",
                    direct_answer_f1=f"{val_metrics.get('val_direct_answer_f1', 0.0):.3f}",
                    tool_call_acc=f"{val_metrics.get('val_tool_call_acc', 0.0):.3f}",
                    tool_call_f1=f"{val_metrics.get('val_tool_call_f1', 0.0):.3f}",
                )
                # Validation generation metrics
                self._log_struct(
                    "Val generation metrics",
                    tool_gen_acc=f"{val_metrics.get('val_tool_gen_accuracy', 0.0):.3f}",
                    direct_gen_acc=f"{val_metrics.get('val_direct_gen_accuracy', 0.0):.3f}",
                )

                # Save best model based on macro_f1 (better for imbalanced datasets)
                val_f1 = val_metrics.get("val_macro_f1", 0.0)

                # For distributed training, broadcast the save decision from rank 0
                # to ensure all ranks agree (prevents deadlock from conditional barriers)
                should_save_best = val_f1 > self.best_val_f1
                if self.use_deepspeed or self.use_ddp:
                    should_save_tensor = torch.tensor([1 if should_save_best else 0], device=self.device)
                    dist.broadcast(should_save_tensor, src=0)
                    should_save_best = should_save_tensor.item() == 1

                if should_save_best:
                    self.best_val_f1 = val_f1
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(
                        epoch, {**train_metrics, **val_metrics}, f"{save_dir}/best_model.pt"
                    )
                    self._log_struct("New best model saved", macro_f1=f"{self.best_val_f1:.3f}")
                    # Log best model to wandb
                    if self.wandb_run is not None and self.is_main:
                        self._log_wandb({
                            "best/val_macro_f1": self.best_val_f1,
                            "best/epoch": epoch + 1,
                        })
                else:
                    self.epochs_without_improvement += 1
                    if self.early_stopping_patience > 0:
                        self._log_struct(
                            "No improvement in validation F1",
                            epochs_without_improvement=self.epochs_without_improvement,
                            patience=self.early_stopping_patience,
                        )

                # Early stopping check - broadcast decision to all ranks
                should_early_stop = self.early_stopping_patience > 0 and self.epochs_without_improvement >= self.early_stopping_patience
                if self.use_deepspeed or self.use_ddp:
                    stop_tensor = torch.tensor([1 if should_early_stop else 0], device=self.device)
                    dist.broadcast(stop_tensor, src=0)
                    should_early_stop = stop_tensor.item() == 1

                if should_early_stop:
                    self._log_struct(
                        "Early stopping triggered",
                        epochs_without_improvement=self.epochs_without_improvement,
                        best_val_f1=f"{self.best_val_f1:.3f}",
                    )
                    break

            # Log sample prediction at end of epoch (random sample, main process only)
            sample_idx = random.randint(0, len(self.train_loader.dataset) - 1)
            self.log_sample_prediction(sample_idx=sample_idx, context=f"epoch {epoch+1} end")

            # Save periodic checkpoint - broadcast decision to ensure all ranks agree
            should_save_periodic = self.save_interval > 0 and (epoch + 1) % self.save_interval == 0
            if self.use_deepspeed or self.use_ddp:
                save_periodic_tensor = torch.tensor([1 if should_save_periodic else 0], device=self.device)
                dist.broadcast(save_periodic_tensor, src=0)
                should_save_periodic = save_periodic_tensor.item() == 1

            if should_save_periodic:
                self.save_checkpoint(
                    epoch, train_metrics, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt"
                )

            # Synchronize before next epoch
            if self.use_ddp or self.use_deepspeed:
                dist.barrier()

        self._log_struct("Training completed", best_val_macro_f1=f"{self.best_val_f1:.3f}")

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
