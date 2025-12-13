"""Trainer for TRM-LLM with deep supervision

Implements training loop with:
- Deep supervision (multi-step training)
- Curriculum learning (gradually increase supervision steps)
- Adaptive computation time (ACT) for efficient training
- Gradient clipping and EMA (future)
"""

from typing import Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict
import os

from ..utils.config import TRMLLMConfig
from ..data.tokenizer import ToolCallTokenizer
from .loss import compute_trm_loss, compute_action_accuracy, compute_per_step_accuracy, compute_valid_json_accuracy


class TRMTrainer:
    """Trainer for TRM-LLM with deep supervision

    Key TRM training techniques:
    1. Deep supervision: Provide loss at each refinement step
    2. Curriculum learning: Start with few steps, gradually increase
    3. State detaching: Gradients only flow through last step
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
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.tool_id_to_name = tool_id_to_name or {}
        self.save_interval = save_interval

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
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
            target_param_ids=batch.get("target_param_ids"),
            target_response_ids=batch.get("target_response_ids"),
        )

        # Compute loss
        loss, loss_dict = compute_trm_loss(outputs_per_step, batch, self.config)

        # Compute accuracies
        acc_dict = compute_action_accuracy(outputs_per_step, batch)

        # Compute valid JSON accuracy for tool params
        if self.tokenizer is not None:
            valid_json_acc = compute_valid_json_accuracy(outputs_per_step, batch, self.tokenizer)
            acc_dict['valid_json'] = valid_json_acc

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

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")

        for batch_idx, batch in enumerate(pbar):
            loss, metrics = self.train_step(batch)

            total_loss += loss

            # Accumulate metrics
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value

            # Update progress bar with running averages
            if batch_idx % 10 == 0:
                n = batch_idx + 1
                avg_loss = total_loss / n
                avg_act = total_metrics.get('action_accuracy', 0.0) / n
                avg_tool = total_metrics.get('tool_accuracy', 0.0) / n
                avg_n_calls = total_metrics.get('num_calls_accuracy', 0.0) / n
                avg_param = total_metrics.get('param_accuracy', 0.0) / n
                avg_resp = total_metrics.get('response_accuracy', 0.0) / n
                avg_json = total_metrics.get('valid_json', 0.0) / n
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "act": f"{avg_act:.3f}",
                        "tool": f"{avg_tool:.3f}",
                        "n_calls": f"{avg_n_calls:.3f}",
                        "param": f"{avg_param:.3f}",
                        "resp": f"{avg_resp:.3f}",
                        "json": f"{avg_json:.3f}",
                    }
                )

        # Average metrics
        num_batches = len(self.train_loader)
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        avg_metrics["loss"] = total_loss / num_batches

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

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs_per_step = self.model(
                batch["input_ids"],
                max_supervision_steps=self.config.max_supervision_steps,  # Use full steps for val
                training=False,
                target_param_ids=batch.get("target_param_ids"),
                target_response_ids=batch.get("target_response_ids"),
            )

            # Compute loss
            loss, loss_dict = compute_trm_loss(outputs_per_step, batch, self.config)

            # Compute accuracies
            acc_dict = compute_action_accuracy(outputs_per_step, batch)

            # Compute valid JSON accuracy
            if self.tokenizer is not None:
                valid_json_acc = compute_valid_json_accuracy(outputs_per_step, batch, self.tokenizer)
                acc_dict['valid_json'] = valid_json_acc

            total_loss += loss.item()
            for key, value in {**loss_dict, **acc_dict}.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value

        # Average
        num_batches = len(self.val_loader)
        val_metrics = {f"val_{key}": value / num_batches for key, value in total_metrics.items()}
        val_metrics["val_loss"] = total_loss / num_batches

        return val_metrics

    @torch.no_grad()
    def log_sample_prediction(self, sample_idx: int = 0):
        """Log a sample prediction to monitor training progress

        Args:
            sample_idx: Index of sample to log from training set
        """
        if self.tokenizer is None:
            return

        self.model.eval()

        # Get a sample from dataset
        dataset = self.train_loader.dataset
        # Handle Subset (from random_split)
        if hasattr(dataset, 'dataset'):
            base_dataset = dataset.dataset
            actual_idx = dataset.indices[sample_idx] if hasattr(dataset, 'indices') else sample_idx
            sample = base_dataset[actual_idx]
        else:
            sample = dataset[sample_idx]

        # Prepare batch (single sample)
        input_ids = torch.tensor([sample['input_ids']], device=self.device)
        target_action = sample['target_action']
        target_tool_id = sample['target_tool_id']
        target_num_calls = sample.get('target_num_calls', 0)
        target_param_ids = sample.get('target_param_ids', [])
        target_response_ids = sample.get('target_response_ids', [])

        # Forward pass
        outputs_per_step = self.model(
            input_ids,
            max_supervision_steps=self.config.max_supervision_steps,
            training=False,
        )

        final_output = outputs_per_step[-1]

        # Decode predictions
        action_probs = F.softmax(final_output['action_logits'][0], dim=-1)
        pred_action = action_probs.argmax().item()
        action_conf = action_probs[pred_action].item()

        tool_probs = F.softmax(final_output['tool_logits'][0], dim=-1)
        pred_tool_id = tool_probs.argmax().item()
        tool_conf = tool_probs[pred_tool_id].item()

        pred_num_calls = 1
        if 'num_calls_logits' in final_output:
            num_calls_probs = F.softmax(final_output['num_calls_logits'][0], dim=-1)
            pred_num_calls = num_calls_probs.argmax().item() + 1

        # Print sample prediction
        print("\n" + "-" * 60)
        print("Sample Prediction (End of Epoch)")
        print("-" * 60)

        # Input (truncated)
        input_text = self.tokenizer.decode(sample['input_ids'][:200])
        print(f"Input ({len(sample['input_ids'])} tokens): {input_text}...")

        # Target
        target_action_str = "tool_call" if target_action == 1 else "direct_answer"
        target_tool_name = self.tool_id_to_name.get(target_tool_id, f"tool_{target_tool_id}")
        print(f"\nTarget:")
        print(f"  Action: {target_action_str}")
        if target_action == 1:
            print(f"  Tool: {target_tool_name} (id={target_tool_id})")
            print(f"  Num calls: {target_num_calls}")
            if target_param_ids:
                param_text = self.tokenizer.decode(target_param_ids)
                if len(param_text) > 200:
                    param_text = param_text[:200] + "..."
                print(f"  Params ({len(target_param_ids)} tokens): {param_text}")
        else:
            if target_response_ids:
                resp_text = self.tokenizer.decode(target_response_ids)
                if len(resp_text) > 200:
                    resp_text = resp_text[:200] + "..."
                print(f"  Response ({len(target_response_ids)} tokens): {resp_text}")

        # Prediction
        pred_action_str = "tool_call" if pred_action == 1 else "direct_answer"
        pred_tool_name = self.tool_id_to_name.get(pred_tool_id, f"tool_{pred_tool_id}")
        print(f"\nPrediction:")
        print(f"  Action: {pred_action_str} (conf={action_conf:.3f})")
        if pred_action == 1:
            print(f"  Tool: {pred_tool_name} (id={pred_tool_id}, conf={tool_conf:.3f})")
            print(f"  Num calls: {pred_num_calls}")

            # Generate params
            if hasattr(self.model, 'generate_params'):
                y_state = final_output['y_state']
                param_ids = self.model.generate_params(y_state, max_length=64)
                gen_param_text = self.tokenizer.decode(param_ids[0].tolist(), skip_special_tokens=True)
                if len(gen_param_text) > 200:
                    gen_param_text = gen_param_text[:200] + "..."
                print(f"  Generated params: {gen_param_text}")
        else:
            # Generate response
            if hasattr(self.model, 'generate_response'):
                y_state = final_output['y_state']
                response_ids = self.model.generate_response(y_state, max_length=128)
                gen_resp_text = self.tokenizer.decode(response_ids[0].tolist(), skip_special_tokens=True)
                if len(gen_resp_text) > 200:
                    gen_resp_text = gen_resp_text[:200] + "..."
                print(f"  Generated response: {gen_resp_text}")

        # Match indicators
        action_match = "✓" if pred_action == target_action else "✗"
        tool_match = "✓" if pred_tool_id == target_tool_id else "✗"
        print(f"\nMatch: Action {action_match}, Tool {tool_match}")
        print("-" * 60)

        self.model.train()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filepath: str):
        """Save model checkpoint

        Args:
            epoch: Current epoch
            metrics: Current metrics
            filepath: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "current_max_steps": self.current_max_steps,
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.current_max_steps = checkpoint.get("current_max_steps", 2)

        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def train(self, save_dir: str = "checkpoints"):
        """Full training loop

        Args:
            save_dir: Directory to save checkpoints
        """
        print("=" * 80)
        print("Starting TRM-LLM Training")
        print("=" * 80)
        print(f"Model parameters: {self.model.get_num_trainable_params() / 1e6:.1f}M")
        print(f"Training examples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation examples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max epochs: {self.config.max_epochs}")
        print(f"Initial supervision steps: {self.current_max_steps}")
        print("=" * 80)

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            # Curriculum: Gradually increase max supervision steps
            self.update_curriculum(epoch)

            # Train epoch
            print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
            print(f"Max supervision steps: {self.current_max_steps}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

            train_metrics = self.train_epoch(epoch)

            # Print training metrics
            print(f"\nTraining Results:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Action Accuracy: {train_metrics['action_accuracy']:.3f}")
            print(f"  Tool Accuracy: {train_metrics['tool_accuracy']:.3f}")
            print(f"  Num Calls Accuracy: {train_metrics.get('num_calls_accuracy', 0.0):.3f}")
            print(f"  Param Accuracy: {train_metrics.get('param_accuracy', 0.0):.3f}")
            print(f"  Valid JSON: {train_metrics.get('valid_json', 0.0):.3f}")
            print(f"  Response Accuracy: {train_metrics.get('response_accuracy', 0.0):.3f}")
            print(f"  Overall Accuracy: {train_metrics['overall_accuracy']:.3f}")

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                print(f"\nValidation Results:")
                print(f"  Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Action Accuracy: {val_metrics['val_action_accuracy']:.3f}")
                print(f"  Tool Accuracy: {val_metrics.get('val_tool_accuracy', 0.0):.3f}")
                print(f"  Num Calls Accuracy: {val_metrics.get('val_num_calls_accuracy', 0.0):.3f}")
                print(f"  Param Accuracy: {val_metrics.get('val_param_accuracy', 0.0):.3f}")
                print(f"  Valid JSON: {val_metrics.get('val_valid_json', 0.0):.3f}")
                print(f"  Response Accuracy: {val_metrics.get('val_response_accuracy', 0.0):.3f}")
                print(f"  Overall Accuracy: {val_metrics['val_overall_accuracy']:.3f}")

                # Save best model
                if val_metrics["val_overall_accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["val_overall_accuracy"]
                    self.save_checkpoint(
                        epoch, {**train_metrics, **val_metrics}, f"{save_dir}/best_model.pt"
                    )
                    print(f"  New best model! Accuracy: {self.best_val_acc:.3f}")

            # Log sample prediction at end of epoch (random sample)
            sample_idx = random.randint(0, len(self.train_loader.dataset) - 1)
            self.log_sample_prediction(sample_idx=sample_idx)

            # Save periodic checkpoint
            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(
                    epoch, train_metrics, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt"
                )

        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.3f}")
        print("=" * 80)

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
            print(
                f"Curriculum update: Increasing max supervision steps to {self.current_max_steps}"
            )
