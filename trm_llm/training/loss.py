"""Loss functions for TRM-LLM training

Implements multi-step supervision losses for deep supervision training
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from trm_llm.utils.config import TRMLLMConfig


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (FP16-safe implementation)

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
    - p_t is the probability of the correct class
    - γ (gamma) focuses on hard examples (higher = more focus on hard)
    - α_t is the class weight for balancing

    This implementation is numerically stable for FP16 mixed precision training:
    - Clamps pt to prevent underflow in (1-pt)^gamma
    - Uses stable log computation
    - Adds epsilon for numerical safety

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = 'mean',
        eps: float = 1e-6,
        pt_clamp_min: float = 1e-4,
        pt_clamp_max: float = 1.0 - 1e-4,
    ):
        """
        Args:
            gamma: Focusing parameter. Higher values focus more on hard examples.
                   gamma=0 is equivalent to CrossEntropyLoss. Default: 2.0
            alpha: Class weights tensor of shape (num_classes,). Default: None (uniform)
            reduction: 'none', 'mean', or 'sum'. Default: 'mean'
            eps: Small epsilon for numerical stability. Default: 1e-6
            pt_clamp_min: Minimum value for pt clamping (prevents log(0)). Default: 1e-4
            pt_clamp_max: Maximum value for pt clamping (prevents (1-pt)^gamma underflow). Default: 0.9999
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps
        self.pt_clamp_min = pt_clamp_min
        self.pt_clamp_max = pt_clamp_max

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C) where C is number of classes
            targets: Ground truth labels of shape (N,)

        Returns:
            Focal loss value
        """
        # Compute log softmax for numerical stability (better than softmax + log)
        log_probs = F.log_softmax(inputs, dim=-1)

        # Get log probability of correct class: log(pt)
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Compute pt = exp(log_pt) with clamping for FP16 stability
        # Clamp to prevent:
        # - pt too close to 0: log(pt) -> -inf
        # - pt too close to 1: (1-pt)^gamma underflows in FP16
        pt = torch.exp(log_pt).clamp(min=self.pt_clamp_min, max=self.pt_clamp_max)

        # Compute focal weight: (1 - pt)^gamma
        # The clamping ensures (1-pt) >= pt_clamp_min, so this won't underflow
        focal_weight = (1.0 - pt) ** self.gamma

        # Compute cross entropy: -log(pt)
        # Clamp log_pt to prevent -inf
        ce_loss = -log_pt.clamp(min=-100.0)

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Add small epsilon to prevent zero gradients
        focal_loss = focal_loss + self.eps

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_trm_loss(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    config: TRMLLMConfig,
    special_token_ids: Optional[List[int]] = None,
    tool_call_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute TRM loss across all supervision steps

    Key insight from TRM: Provide supervision at EACH refinement step,
    not just the final output. This teaches the model to progressively improve.

    Args:
        outputs_per_step: List of output dicts from model, one per supervision step
            Each dict contains:
                - action_logits: (batch_size, num_action_types)
                - num_calls_logits: (batch_size, max_parallel_calls)
                - q_logit: (batch_size, 1) - correctness prediction (TRM paper)
                - generation_logits: (batch_size, gen_seq_len, vocab_size) [optional, last step only]
        targets: Ground truth dict with:
            - target_action: (batch_size,) - 0 for direct_answer, 1 for tool_call
            - target_num_calls: (batch_size,) - number of parallel calls (1-indexed)
            - target_generation_ids: (batch_size, gen_seq_len) - target generation tokens
            - generation_mask: (batch_size, gen_seq_len) - mask for valid generation tokens
        config: TRMLLMConfig
        special_token_ids: Optional list of token IDs for special tokens (e.g., <tool_call>, </tool_call>)
            that should receive higher weight in generation loss
        tool_call_token_id: Token ID for <tool_call> token (for consistency loss)

    Returns:
        total_loss: Averaged loss across all supervision steps
        loss_dict: Dict with individual loss components (for logging)
    """
    total_loss = 0.0
    losses = {
        'action': 0.0,
        'num_calls': 0.0,
        'q': 0.0,  # Q loss (correctness prediction, TRM paper)
        'tool_call_gen': 0.0,
        'direct_answer_gen': 0.0,
        'consistency': 0.0,
    }

    num_steps = len(outputs_per_step)
    tool_mask = (targets['target_action'] == 1)  # tool_call examples

    # Get class weights for action loss (handle imbalanced datasets)
    # If action_class_weights is provided in config, use it
    # Otherwise compute from batch or use uniform weights
    action_class_weights = getattr(config, 'action_class_weights', None)
    if action_class_weights is not None:
        # Use provided weights: [direct_answer_weight, tool_call_weight]
        action_weight_tensor = torch.tensor(action_class_weights, device=targets['target_action'].device)
    else:
        # Compute weights from batch to handle imbalance
        # More weight to minority class
        num_direct = (targets['target_action'] == 0).sum().float()
        num_tool = (targets['target_action'] == 1).sum().float()
        total = num_direct + num_tool
        if num_direct > 0 and num_tool > 0:
            # Inverse frequency weighting
            weight_direct = total / (2 * num_direct)
            weight_tool = total / (2 * num_tool)
            action_weight_tensor = torch.tensor([weight_direct, weight_tool], device=targets['target_action'].device)
        else:
            action_weight_tensor = None

    # Check if num_calls loss should be computed (disabled if all samples have same num_calls)
    num_calls_loss_weight = getattr(config, 'num_calls_loss_weight', 1.0)

    # Use Focal Loss for action classification (better for class imbalance)
    use_focal_loss = getattr(config, 'use_focal_loss', True)
    focal_gamma = getattr(config, 'focal_gamma', 2.0)

    if use_focal_loss:
        action_loss_fn = FocalLoss(gamma=focal_gamma, alpha=action_weight_tensor)
    else:
        action_loss_fn = None  # Use standard cross entropy

    for step_idx, outputs in enumerate(outputs_per_step):
        # ===== 1. Action Classification Loss =====
        # Should the model answer directly or call a tool?
        # Uses Focal Loss (default) or CrossEntropy with class weights
        if use_focal_loss:
            action_loss = action_loss_fn(
                outputs['action_logits'],
                targets['target_action']
            )
        else:
            action_loss = F.cross_entropy(
                outputs['action_logits'],
                targets['target_action'],
                weight=action_weight_tensor
            )

        # ===== 2. Number of Parallel Calls Loss =====
        # How many tools to call in parallel? (only for tool_call examples)
        # Can be disabled via num_calls_loss_weight=0 if dataset has no parallel calls
        num_calls_loss = torch.tensor(0.0, device=action_loss.device)
        if num_calls_loss_weight > 0 and tool_mask.any() and 'num_calls_logits' in outputs and 'target_num_calls' in targets:
            # target_num_calls is 1-indexed (1, 2, 3, ...), logits are 0-indexed
            # So we subtract 1 from target to get the class index
            target_num_calls_idx = targets['target_num_calls'][tool_mask] - 1
            # Clamp to valid range [0, max_parallel_calls-1]
            max_classes = outputs['num_calls_logits'].size(-1)
            target_num_calls_idx = target_num_calls_idx.clamp(0, max_classes - 1)
            num_calls_loss = F.cross_entropy(
                outputs['num_calls_logits'][tool_mask],
                target_num_calls_idx
            )

        # ===== 3. Q Loss (Correctness Prediction, TRM paper) =====
        # Q-head learns to predict if the current action prediction is correct
        # Target: 1.0 if prediction matches ground truth, else 0.0
        # Used for early stopping: if Q > threshold, model thinks it's correct

        # Check if action prediction is correct
        pred_action = outputs['action_logits'].argmax(dim=-1)
        is_correct = (pred_action == targets['target_action']).float()

        q_loss = F.binary_cross_entropy_with_logits(
            outputs['q_logit'].squeeze(-1),
            is_correct
        )

        # ===== 4. Unified Generation Loss (only on last step) =====
        # Generates either tool call JSON or direct answer text
        # Separate losses for tool_call and direct_answer to allow different weights
        tool_call_gen_loss = torch.tensor(0.0, device=action_loss.device)
        direct_answer_gen_loss = torch.tensor(0.0, device=action_loss.device)
        consistency_loss = torch.tensor(0.0, device=action_loss.device)

        if 'generation_logits' in outputs:
            gen_logits = outputs['generation_logits']  # (batch_size, seq_len, vocab_size)
            target_gen_ids = targets['target_generation_ids']  # (batch_size, seq_len)
            gen_mask = targets['generation_mask']  # (batch_size, seq_len)

            # Shift targets for next-token prediction: predict token[i+1] from logits[i]
            if gen_logits.size(1) > 1 and target_gen_ids.size(1) > 1:
                shift_logits = gen_logits[:, :-1, :].contiguous()  # (batch, seq-1, vocab)
                shift_targets = target_gen_ids[:, 1:].contiguous()  # (batch, seq-1)
                shift_mask = gen_mask[:, 1:].contiguous()  # (batch, seq-1)

                # Flatten for cross entropy
                batch_size, seq_len, vocab_size = shift_logits.shape
                flat_logits = shift_logits.view(-1, vocab_size)  # (batch*seq, vocab)
                flat_targets = shift_targets.view(-1)  # (batch*seq,)
                flat_mask = shift_mask.view(-1).float()  # (batch*seq,)

                # Compute per-token loss with label smoothing
                label_smoothing = getattr(config, 'label_smoothing', 0.1)

                # Clamp targets to valid range to prevent cross_entropy errors
                flat_targets = flat_targets.clamp(0, vocab_size - 1)

                per_token_loss = F.cross_entropy(
                    flat_logits, flat_targets,
                    reduction='none',
                    label_smoothing=label_smoothing
                )

                # NaN protection for per-token loss
                per_token_loss = torch.where(
                    torch.isnan(per_token_loss) | torch.isinf(per_token_loss),
                    torch.zeros_like(per_token_loss),
                    per_token_loss
                )

                # Create special token weight mask (higher weight for <tool_call>, </tool_call>, etc.)
                special_token_weight = getattr(config, 'special_token_weight', 5.0)
                token_weights = torch.ones_like(flat_targets, dtype=torch.float)
                if special_token_ids is not None and len(special_token_ids) > 0:
                    for token_id in special_token_ids:
                        token_weights = torch.where(
                            flat_targets == token_id,
                            torch.full_like(token_weights, special_token_weight),
                            token_weights
                        )

                # Tool call generation loss (weighted separately)
                if tool_mask.any():
                    tool_mask_expanded = tool_mask.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1).float()
                    tool_combined_mask = flat_mask * tool_mask_expanded
                    weight_sum = (tool_combined_mask * token_weights).sum()
                    if weight_sum > 0:
                        # Apply both mask and special token weights
                        weighted_loss = per_token_loss * tool_combined_mask * token_weights
                        tool_call_gen_loss = weighted_loss.sum() / weight_sum
                        # NaN protection
                        if torch.isnan(tool_call_gen_loss) or torch.isinf(tool_call_gen_loss):
                            tool_call_gen_loss = torch.tensor(0.0, device=action_loss.device)

                # Direct answer generation loss
                direct_mask = (targets['target_action'] == 0)
                if direct_mask.any():
                    direct_mask_expanded = direct_mask.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1).float()
                    direct_combined_mask = flat_mask * direct_mask_expanded
                    mask_sum = direct_combined_mask.sum()
                    if mask_sum > 0:
                        direct_answer_gen_loss = (per_token_loss * direct_combined_mask).sum() / mask_sum
                        # NaN protection
                        if torch.isnan(direct_answer_gen_loss) or torch.isinf(direct_answer_gen_loss):
                            direct_answer_gen_loss = torch.tensor(0.0, device=action_loss.device)

            losses['tool_call_gen'] += tool_call_gen_loss.item()
            losses['direct_answer_gen'] += direct_answer_gen_loss.item()

            # ===== 5. Action-Generation Consistency Loss =====
            # Aligns action prediction with generation format
            if tool_call_token_id is not None:
                consistency_loss = compute_action_generation_consistency_loss(
                    [outputs], targets, tool_call_token_id
                )
                losses['consistency'] += consistency_loss.item()

        # ===== Combine Losses =====
        # Weights for different loss components
        action_loss_weight = getattr(config, 'action_loss_weight', 2.0)  # Higher weight for action classification
        tool_call_gen_weight = getattr(config, 'tool_call_gen_weight', 2.0)  # Higher weight for tool calls
        direct_answer_gen_weight = getattr(config, 'direct_answer_gen_weight', 1.0)
        consistency_loss_weight = getattr(config, 'consistency_loss_weight', 1.0)  # Weight for action-generation alignment

        step_loss = (action_loss_weight * action_loss +
                     num_calls_loss_weight * num_calls_loss +
                     config.q_loss_weight * q_loss +
                     tool_call_gen_weight * tool_call_gen_loss +
                     direct_answer_gen_weight * direct_answer_gen_loss +
                     consistency_loss_weight * consistency_loss)
        total_loss += step_loss

        # Accumulate for logging
        losses['action'] += action_loss.item()
        losses['num_calls'] += num_calls_loss.item() if isinstance(num_calls_loss, torch.Tensor) else 0.0
        losses['q'] += q_loss.item()

    # Average across supervision steps
    total_loss = total_loss / num_steps

    # Final NaN protection for total loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        # Return a small valid loss to avoid breaking training
        total_loss = torch.tensor(0.1, device=total_loss.device, requires_grad=True)

    for key in losses:
        losses[key] /= num_steps
        # Replace NaN/Inf in losses dict with 0 for logging
        if not isinstance(losses[key], (int, float)):
            losses[key] = 0.0
        elif losses[key] != losses[key]:  # NaN check
            losses[key] = 0.0

    return total_loss, losses


def compute_action_accuracy(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute accuracy and F1 metrics for imbalanced datasets

    Args:
        outputs_per_step: List of model outputs
        targets: Ground truth targets

    Returns:
        metrics: Dict with accuracy and F1 metrics including:
            - action_accuracy: Overall accuracy
            - direct_answer_acc: Accuracy for direct_answer class (class 0)
            - tool_call_acc: Accuracy for tool_call class (class 1)
            - direct_answer_precision, direct_answer_recall, direct_answer_f1
            - tool_call_precision, tool_call_recall, tool_call_f1
            - macro_f1: Average F1 across classes
    """
    # Use final step for evaluation
    final_outputs = outputs_per_step[-1]

    # Action predictions and targets
    pred_action = final_outputs['action_logits'].argmax(dim=-1)
    target_action = targets['target_action']

    # Overall action accuracy
    action_acc = (pred_action == target_action).float().mean().item()

    # Per-class metrics
    # Class 0: direct_answer, Class 1: tool_call
    direct_mask = (target_action == 0)  # True labels for direct_answer
    tool_mask = (target_action == 1)    # True labels for tool_call

    pred_direct = (pred_action == 0)    # Predicted as direct_answer
    pred_tool = (pred_action == 1)      # Predicted as tool_call

    # Per-class accuracy (recall per class)
    direct_answer_acc = 0.0
    tool_call_acc = 0.0

    if direct_mask.sum() > 0:
        direct_answer_acc = (pred_action[direct_mask] == 0).float().mean().item()
    if tool_mask.sum() > 0:
        tool_call_acc = (pred_action[tool_mask] == 1).float().mean().item()

    # Precision, Recall, F1 for each class
    # direct_answer (class 0)
    tp_direct = (pred_direct & direct_mask).sum().float()
    fp_direct = (pred_direct & tool_mask).sum().float()  # Predicted direct but was tool
    fn_direct = (pred_tool & direct_mask).sum().float()  # Predicted tool but was direct

    direct_precision = (tp_direct / (tp_direct + fp_direct)).item() if (tp_direct + fp_direct) > 0 else 0.0
    direct_recall = (tp_direct / (tp_direct + fn_direct)).item() if (tp_direct + fn_direct) > 0 else 0.0
    direct_f1 = (2 * direct_precision * direct_recall / (direct_precision + direct_recall)) if (direct_precision + direct_recall) > 0 else 0.0

    # tool_call (class 1)
    tp_tool = (pred_tool & tool_mask).sum().float()
    fp_tool = (pred_tool & direct_mask).sum().float()  # Predicted tool but was direct
    fn_tool = (pred_direct & tool_mask).sum().float()  # Predicted direct but was tool

    tool_precision = (tp_tool / (tp_tool + fp_tool)).item() if (tp_tool + fp_tool) > 0 else 0.0
    tool_recall = (tp_tool / (tp_tool + fn_tool)).item() if (tp_tool + fn_tool) > 0 else 0.0
    tool_f1 = (2 * tool_precision * tool_recall / (tool_precision + tool_recall)) if (tool_precision + tool_recall) > 0 else 0.0

    # Macro F1 (average of per-class F1)
    macro_f1 = (direct_f1 + tool_f1) / 2

    # Num calls accuracy (only for tool_call examples)
    num_calls_acc = 0.0
    if tool_mask.any() and 'num_calls_logits' in final_outputs and 'target_num_calls' in targets:
        pred_num_calls = final_outputs['num_calls_logits'].argmax(dim=-1) + 1  # 0-indexed to 1-indexed
        target_num_calls = targets['target_num_calls']
        num_calls_acc = (pred_num_calls[tool_mask] == target_num_calls[tool_mask]).float().mean().item()

    # Overall accuracy is just action accuracy now (tool selection is via generation)
    overall_acc = action_acc

    # Generation accuracy split by action type
    tool_gen_acc = 0.0  # For tool_call samples (generates JSON)
    direct_gen_acc = 0.0  # For direct_answer samples

    if 'generation_logits' in final_outputs:
        gen_logits = final_outputs['generation_logits']  # (batch_size, seq_len, vocab_size)
        target_gen_ids = targets['target_generation_ids']  # (batch_size, seq_len)
        gen_mask = targets['generation_mask']  # (batch_size, seq_len)

        if gen_logits.size(1) > 1 and target_gen_ids.size(1) > 1:
            # Shift for next-token prediction
            shift_preds = gen_logits[:, :-1, :].argmax(dim=-1)  # (batch, seq-1)
            shift_targets = target_gen_ids[:, 1:]  # (batch, seq-1)
            shift_mask = gen_mask[:, 1:]  # (batch, seq-1)
            seq_len = shift_mask.size(1)

            # Tool call samples accuracy
            if tool_mask.any():
                tool_mask_expanded = tool_mask.unsqueeze(1).expand(-1, seq_len)
                tool_combined_mask = shift_mask.bool() & tool_mask_expanded
                if tool_combined_mask.sum() > 0:
                    tool_correct = (shift_preds == shift_targets) & tool_combined_mask
                    tool_gen_acc = tool_correct.sum().float() / tool_combined_mask.sum().float()
                    tool_gen_acc = tool_gen_acc.item()

            # Direct answer samples accuracy
            if direct_mask.any():
                direct_mask_expanded = direct_mask.unsqueeze(1).expand(-1, seq_len)
                direct_combined_mask = shift_mask.bool() & direct_mask_expanded
                if direct_combined_mask.sum() > 0:
                    direct_correct = (shift_preds == shift_targets) & direct_combined_mask
                    direct_gen_acc = direct_correct.sum().float() / direct_combined_mask.sum().float()
                    direct_gen_acc = direct_gen_acc.item()

    return {
        # Overall metrics
        'action_accuracy': action_acc,
        'overall_accuracy': overall_acc,
        'macro_f1': macro_f1,
        # Per-class accuracy
        'direct_answer_acc': direct_answer_acc,
        'tool_call_acc': tool_call_acc,
        # direct_answer metrics
        'direct_answer_precision': direct_precision,
        'direct_answer_recall': direct_recall,
        'direct_answer_f1': direct_f1,
        # tool_call metrics
        'tool_call_precision': tool_precision,
        'tool_call_recall': tool_recall,
        'tool_call_f1': tool_f1,
        # Other metrics
        'num_calls_accuracy': num_calls_acc,
        'tool_gen_accuracy': tool_gen_acc,
        'direct_gen_accuracy': direct_gen_acc,
    }


def compute_per_step_accuracy(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor]
) -> List[float]:
    """Compute accuracy at each supervision step

    Useful for analyzing how accuracy improves across steps

    Args:
        outputs_per_step: List of model outputs
        targets: Ground truth

    Returns:
        accuracies: List of accuracy values, one per step
    """
    accuracies = []

    for outputs in outputs_per_step:
        pred_action = outputs['action_logits'].argmax(dim=-1)
        acc = (pred_action == targets['target_action']).float().mean().item()
        accuracies.append(acc)

    return accuracies


def compute_action_generation_consistency_loss(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    tool_call_token_id: int,
) -> torch.Tensor:
    """Compute consistency loss between action prediction and generation output

    This loss penalizes mismatches where:
    - Action head predicts tool_call but generation doesn't start with <tool_call>
    - Action head predicts direct_answer but generation starts with <tool_call>

    This helps align the action prediction head with the generation head.

    Args:
        outputs_per_step: List of model outputs
        targets: Ground truth targets
        tool_call_token_id: Token ID for <tool_call> token

    Returns:
        consistency_loss: Scalar loss tensor
    """
    final_outputs = outputs_per_step[-1]

    if 'generation_logits' not in final_outputs:
        return torch.tensor(0.0, device=final_outputs['action_logits'].device)

    gen_logits = final_outputs['generation_logits']  # (batch_size, seq_len, vocab_size)
    action_logits = final_outputs['action_logits']  # (batch_size, num_action_types)

    if gen_logits.size(1) < 2:
        return torch.tensor(0.0, device=action_logits.device)

    # Get the first generated token (after BOS/start)
    # Use position 0 or 1 depending on whether BOS is included
    first_token_logits = gen_logits[:, 0, :]  # (batch_size, vocab_size)

    # Probability that first token is <tool_call>
    first_token_probs = F.softmax(first_token_logits, dim=-1)
    prob_tool_call_token = first_token_probs[:, tool_call_token_id]  # (batch_size,)

    # Probability that action is tool_call (index 1)
    action_probs = F.softmax(action_logits, dim=-1)
    prob_action_tool_call = action_probs[:, 1]  # (batch_size,)

    # Consistency loss: action prediction should match generation format
    # If action predicts tool_call (prob high), generation should start with <tool_call> (prob high)
    # Use MSE or BCE to align these probabilities
    consistency_loss = F.mse_loss(prob_action_tool_call, prob_tool_call_token)

    return consistency_loss


def compute_valid_tool_call_format_accuracy(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    tokenizer,
    return_counts: bool = False,
    return_sample: bool = False,
    return_tokens: bool = False
):
    """Compute accuracy of generating tool calls with exact format

    Checks if generation follows EXACTLY: <tool_call>{...}</tool_call>
    - Must start with <tool_call>
    - Must end with </tool_call>
    - Must have valid JSON in between
    - No content before or after the tags

    Args:
        outputs_per_step: List of model outputs
        targets: Ground truth targets
        tokenizer: Tokenizer for decoding tokens
        return_counts: If True, return (valid_count, total_count) for proper aggregation
        return_sample: If True, also return sample_correct_prediction
        return_tokens: If True, also return decoded tokens list (token-by-token)

    Returns:
        Base returns depend on return_counts and return_sample flags.
        If return_tokens=True, additionally appends (sample_decoded_tokens, target_decoded_tokens)
    """
    import re

    def _make_return(accuracy_or_counts, sample=None, target=None, sample_tokens=None, target_tokens=None):
        """Helper to construct return value based on flags"""
        result = accuracy_or_counts if isinstance(accuracy_or_counts, tuple) else (accuracy_or_counts,)
        if return_sample:
            result = result + (sample, target)
        if return_tokens:
            result = result + (sample_tokens, target_tokens)
        # Flatten single value
        if len(result) == 1:
            return result[0]
        return result

    if tokenizer is None:
        if return_counts:
            return _make_return((0, 0))
        return _make_return(0.0)

    final_outputs = outputs_per_step[-1]

    # Only check tool_call examples
    tool_mask = (targets['target_action'] == 1)
    if not tool_mask.any():
        if return_counts:
            return _make_return((0, 0))
        return _make_return(0.0)

    if 'generation_logits' not in final_outputs:
        if return_counts:
            return _make_return((0, 0))
        return _make_return(0.0)

    gen_logits = final_outputs['generation_logits']  # (batch_size, seq_len, vocab_size)
    gen_mask = targets['generation_mask']  # (batch_size, seq_len)

    # Get predicted tokens
    pred_tokens = gen_logits.argmax(dim=-1)  # (batch_size, seq_len)

    valid_count = 0
    total_count = 0
    sample_correct = None  # Store first correct prediction
    sample_target = None   # Store corresponding target
    sample_decoded_tokens = None  # Store decoded tokens list
    target_decoded_tokens = None  # Store target decoded tokens list

    # Get target generation tokens for comparison
    target_gen_ids = targets.get('target_generation_ids', None)

    # Pattern: exactly <tool_call>{...}</tool_call> with optional whitespace
    # Allows single object {...} or array [{...}, {...}]
    pattern = r'^\s*<tool_call>\s*(\{.*\}|\[.*\])\s*</tool_call>\s*$'

    for i in range(pred_tokens.size(0)):
        if not tool_mask[i]:
            continue

        total_count += 1

        # Get valid tokens (where mask is 1)
        mask = gen_mask[i].bool()
        tokens = pred_tokens[i][mask].tolist()

        if not tokens:
            continue

        try:
            # Decode tokens to text
            text = tokenizer.decode(tokens, skip_special_tokens=False)

            # Check if matches exact pattern
            match = re.match(pattern, text, re.DOTALL)
            if match:
                # Also verify the JSON inside is valid
                json_content = match.group(1)
                json.loads(json_content)
                valid_count += 1
                # Store first correct prediction and its target
                if sample_correct is None:
                    sample_correct = text.strip()
                    # Decode each token individually for debugging
                    sample_decoded_tokens = [tokenizer.decode([t]) for t in tokens]
                    # Decode target for this sample
                    if target_gen_ids is not None:
                        target_mask = gen_mask[i].bool()
                        target_tokens_list = target_gen_ids[i][target_mask].tolist()
                        if target_tokens_list:
                            sample_target = tokenizer.decode(target_tokens_list, skip_special_tokens=False).strip()
                            target_decoded_tokens = [tokenizer.decode([t]) for t in target_tokens_list]

        except (json.JSONDecodeError, Exception):
            pass

    accuracy = valid_count / total_count if total_count > 0 else 0.0

    if return_counts:
        return _make_return((valid_count, total_count), sample_correct, sample_target, sample_decoded_tokens, target_decoded_tokens)
    return _make_return(accuracy, sample_correct, sample_target, sample_decoded_tokens, target_decoded_tokens)
