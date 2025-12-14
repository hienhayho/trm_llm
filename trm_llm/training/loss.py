"""Loss functions for TRM-LLM training

Implements multi-step supervision losses for deep supervision training
"""

import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from trm_llm.utils.config import TRMLLMConfig


def compute_trm_loss(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    config: TRMLLMConfig,
    special_token_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute TRM loss across all supervision steps

    Key insight from TRM: Provide supervision at EACH refinement step,
    not just the final output. This teaches the model to progressively improve.

    Args:
        outputs_per_step: List of output dicts from model, one per supervision step
            Each dict contains:
                - action_logits: (batch_size, num_action_types)
                - num_calls_logits: (batch_size, max_parallel_calls)
                - halt_logit: (batch_size, 1)
                - generation_logits: (batch_size, gen_seq_len, vocab_size) [optional, last step only]
        targets: Ground truth dict with:
            - target_action: (batch_size,) - 0 for direct_answer, 1 for tool_call
            - target_num_calls: (batch_size,) - number of parallel calls (1-indexed)
            - target_generation_ids: (batch_size, gen_seq_len) - target generation tokens
            - generation_mask: (batch_size, gen_seq_len) - mask for valid generation tokens
        config: TRMLLMConfig
        special_token_ids: Optional list of token IDs for special tokens (e.g., <tool_call>, </tool_call>)
            that should receive higher weight in generation loss

    Returns:
        total_loss: Averaged loss across all supervision steps
        loss_dict: Dict with individual loss components (for logging)
    """
    total_loss = 0.0
    losses = {
        'action': 0.0,
        'num_calls': 0.0,
        'halt': 0.0,
        'tool_call_gen': 0.0,
        'direct_answer_gen': 0.0,
    }

    num_steps = len(outputs_per_step)
    tool_mask = (targets['target_action'] == 1)  # tool_call examples

    for step_idx, outputs in enumerate(outputs_per_step):
        # ===== 1. Action Classification Loss =====
        # Should the model answer directly or call a tool?
        action_loss = F.cross_entropy(
            outputs['action_logits'],
            targets['target_action']
        )

        # ===== 2. Number of Parallel Calls Loss =====
        # How many tools to call in parallel? (only for tool_call examples)
        num_calls_loss = torch.tensor(0.0, device=action_loss.device)
        if tool_mask.any() and 'num_calls_logits' in outputs and 'target_num_calls' in targets:
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

        # ===== 3. Adaptive Computation Time (ACT) Halting Loss =====
        # Learn to halt when the prediction is correct
        # Halt target: 1.0 if prediction matches ground truth, else 0.0

        # Check if action prediction is correct
        pred_action = outputs['action_logits'].argmax(dim=-1)
        is_correct = (pred_action == targets['target_action']).float()

        halt_loss = F.binary_cross_entropy_with_logits(
            outputs['halt_logit'].squeeze(-1),
            is_correct
        )

        # ===== 4. Unified Generation Loss (only on last step) =====
        # Generates either tool call JSON or direct answer text
        # Separate losses for tool_call and direct_answer to allow different weights
        tool_call_gen_loss = torch.tensor(0.0, device=action_loss.device)
        direct_answer_gen_loss = torch.tensor(0.0, device=action_loss.device)

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
                per_token_loss = F.cross_entropy(
                    flat_logits, flat_targets,
                    reduction='none',
                    label_smoothing=label_smoothing
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
                    if tool_combined_mask.sum() > 0:
                        # Apply both mask and special token weights
                        weighted_loss = per_token_loss * tool_combined_mask * token_weights
                        weight_sum = (tool_combined_mask * token_weights).sum()
                        tool_call_gen_loss = weighted_loss.sum() / weight_sum

                # Direct answer generation loss
                direct_mask = (targets['target_action'] == 0)
                if direct_mask.any():
                    direct_mask_expanded = direct_mask.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1).float()
                    direct_combined_mask = flat_mask * direct_mask_expanded
                    if direct_combined_mask.sum() > 0:
                        direct_answer_gen_loss = (per_token_loss * direct_combined_mask).sum() / direct_combined_mask.sum()

            losses['tool_call_gen'] += tool_call_gen_loss.item()
            losses['direct_answer_gen'] += direct_answer_gen_loss.item()

        # ===== Combine Losses =====
        # Weights for different loss components
        action_loss_weight = getattr(config, 'action_loss_weight', 2.0)  # Higher weight for action classification
        tool_call_gen_weight = getattr(config, 'tool_call_gen_weight', 2.0)  # Higher weight for tool calls
        direct_answer_gen_weight = getattr(config, 'direct_answer_gen_weight', 1.0)

        step_loss = (action_loss_weight * action_loss + num_calls_loss +
                     config.halt_loss_weight * halt_loss +
                     tool_call_gen_weight * tool_call_gen_loss +
                     direct_answer_gen_weight * direct_answer_gen_loss)
        total_loss += step_loss

        # Accumulate for logging
        losses['action'] += action_loss.item()
        losses['num_calls'] += num_calls_loss.item() if isinstance(num_calls_loss, torch.Tensor) else 0.0
        losses['halt'] += halt_loss.item()

    # Average across supervision steps
    total_loss = total_loss / num_steps
    for key in losses:
        losses[key] /= num_steps

    return total_loss, losses


def compute_action_accuracy(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute accuracy metrics

    Args:
        outputs_per_step: List of model outputs
        targets: Ground truth targets

    Returns:
        metrics: Dict with accuracy metrics
    """
    # Use final step for evaluation
    final_outputs = outputs_per_step[-1]

    # Action accuracy
    pred_action = final_outputs['action_logits'].argmax(dim=-1)
    action_acc = (pred_action == targets['target_action']).float().mean().item()

    # Num calls accuracy (only for tool_call examples)
    tool_mask = (targets['target_action'] == 1)
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
            direct_mask = (targets['target_action'] == 0)
            if direct_mask.any():
                direct_mask_expanded = direct_mask.unsqueeze(1).expand(-1, seq_len)
                direct_combined_mask = shift_mask.bool() & direct_mask_expanded
                if direct_combined_mask.sum() > 0:
                    direct_correct = (shift_preds == shift_targets) & direct_combined_mask
                    direct_gen_acc = direct_correct.sum().float() / direct_combined_mask.sum().float()
                    direct_gen_acc = direct_gen_acc.item()

    return {
        'action_accuracy': action_acc,
        'num_calls_accuracy': num_calls_acc,
        'overall_accuracy': overall_acc,
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


def compute_valid_json_accuracy(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    tokenizer
) -> float:
    """Compute accuracy of generating valid JSON for tool call examples

    Handles Hermes format where JSON is wrapped in <tool_call>...</tool_call> tags.

    Args:
        outputs_per_step: List of model outputs
        targets: Ground truth targets
        tokenizer: Tokenizer for decoding tokens

    Returns:
        valid_json_ratio: Ratio of tool_call examples with valid JSON generation
    """
    if tokenizer is None:
        return 0.0

    final_outputs = outputs_per_step[-1]

    # Only check tool_call examples
    tool_mask = (targets['target_action'] == 1)
    if not tool_mask.any():
        return 0.0

    if 'generation_logits' not in final_outputs:
        return 0.0

    gen_logits = final_outputs['generation_logits']  # (batch_size, seq_len, vocab_size)
    gen_mask = targets['generation_mask']  # (batch_size, seq_len)

    # Get predicted tokens
    pred_tokens = gen_logits.argmax(dim=-1)  # (batch_size, seq_len)

    valid_count = 0
    total_count = 0

    for i in range(pred_tokens.size(0)):
        if not tool_mask[i]:
            continue

        total_count += 1

        # Get valid tokens (where mask is 1)
        mask = gen_mask[i].bool()
        tokens = pred_tokens[i][mask].tolist()

        if not tokens:
            continue

        # Decode tokens to text
        try:
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            text = text.strip()

            # Remove Hermes tool_call tags if present
            # Look for content between <tool_call> and </tool_call>
            tool_call_start = '<tool_call>'
            tool_call_end = '</tool_call>'

            if tool_call_start in text:
                start_idx = text.find(tool_call_start) + len(tool_call_start)
                end_idx = text.find(tool_call_end)
                if end_idx > start_idx:
                    text = text[start_idx:end_idx].strip()

            # Try to find and parse JSON (could be single object or array)
            # First try array format
            start_idx = text.find('[')
            if start_idx != -1:
                end_idx = text.rfind(']')
                if end_idx != -1 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx + 1]
                    json.loads(json_str)  # Will raise if invalid
                    valid_count += 1
                    continue

            # Try single object format
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx + 1]
                json.loads(json_str)  # Will raise if invalid
                valid_count += 1
        except (json.JSONDecodeError, Exception):
            pass

    return valid_count / total_count if total_count > 0 else 0.0
