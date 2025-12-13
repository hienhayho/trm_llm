"""Loss functions for TRM-LLM training

Implements multi-step supervision losses for deep supervision training
"""

import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from ..utils.config import TRMLLMConfig


def compute_trm_loss(
    outputs_per_step: List[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    config: TRMLLMConfig
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute TRM loss across all supervision steps

    Key insight from TRM: Provide supervision at EACH refinement step,
    not just the final output. This teaches the model to progressively improve.

    Args:
        outputs_per_step: List of output dicts from model, one per supervision step
            Each dict contains:
                - action_logits: (batch_size, num_action_types)
                - tool_logits: (batch_size, max_tools)
                - halt_logit: (batch_size, 1)
                - param_logits: (batch_size, param_seq_len, vocab_size) [optional, last step only]
                - response_logits: (batch_size, response_seq_len, vocab_size) [optional, last step only]
        targets: Ground truth dict with:
            - target_action: (batch_size,) - 0 for direct_answer, 1 for tool_call
            - target_tool_id: (batch_size,) - tool ID or -1
            - target_param_ids: (batch_size, param_seq_len) - target param tokens
            - param_mask: (batch_size, param_seq_len) - mask for valid param tokens
            - target_response_ids: (batch_size, response_seq_len) - target response tokens
            - response_mask: (batch_size, response_seq_len) - mask for valid response tokens
        config: TRMLLMConfig

    Returns:
        total_loss: Averaged loss across all supervision steps
        loss_dict: Dict with individual loss components (for logging)
    """
    total_loss = 0.0
    losses = {
        'action': 0.0,
        'tool': 0.0,
        'num_calls': 0.0,
        'halt': 0.0,
        'param': 0.0,
        'response': 0.0,
    }

    num_steps = len(outputs_per_step)

    for step_idx, outputs in enumerate(outputs_per_step):
        # ===== 1. Action Classification Loss =====
        # Should the model answer directly or call a tool?
        action_loss = F.cross_entropy(
            outputs['action_logits'],
            targets['target_action']
        )

        # ===== 2. Tool Selection Loss =====
        # Which tool should be called? (only for tool_call examples)
        tool_loss = torch.tensor(0.0, device=action_loss.device)
        tool_mask = (targets['target_action'] == 1)  # Only examples with tool_call

        if tool_mask.any():
            tool_loss = F.cross_entropy(
                outputs['tool_logits'][tool_mask],
                targets['target_tool_id'][tool_mask]
            )

        # ===== 3. Number of Parallel Calls Loss =====
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

        # ===== 4. Adaptive Computation Time (ACT) Halting Loss =====
        # Learn to halt when the prediction is correct
        # Halt target: 1.0 if prediction matches ground truth, else 0.0

        # Check if action prediction is correct
        pred_action = outputs['action_logits'].argmax(dim=-1)
        is_action_correct = (pred_action == targets['target_action']).float()

        # For tool_call examples, also check if tool is correct
        if tool_mask.any():
            pred_tool = outputs['tool_logits'].argmax(dim=-1)
            is_tool_correct = (pred_tool == targets['target_tool_id']).float()
            # Only count as correct if both action AND tool are correct
            is_correct = is_action_correct * torch.where(
                tool_mask,
                is_tool_correct,
                torch.ones_like(is_tool_correct)
            )
        else:
            is_correct = is_action_correct

        halt_loss = F.binary_cross_entropy_with_logits(
            outputs['halt_logit'].squeeze(-1),
            is_correct
        )

        # ===== 4. Parameter Generation Loss (only on last step) =====
        param_loss = torch.tensor(0.0, device=action_loss.device)
        if 'param_logits' in outputs and tool_mask.any():
            param_logits = outputs['param_logits']  # (batch_size, seq_len, vocab_size)
            target_param_ids = targets['target_param_ids']  # (batch_size, seq_len)
            param_mask = targets['param_mask']  # (batch_size, seq_len)

            # Only compute loss for tool_call examples with valid params
            # Shift targets for next-token prediction: predict token[i+1] from logits[i]
            # logits[:, :-1] predicts target[:, 1:]
            if param_logits.size(1) > 1 and target_param_ids.size(1) > 1:
                shift_logits = param_logits[:, :-1, :].contiguous()  # (batch, seq-1, vocab)
                shift_targets = target_param_ids[:, 1:].contiguous()  # (batch, seq-1)
                shift_mask = param_mask[:, 1:].contiguous()  # (batch, seq-1)

                # Flatten for cross entropy
                batch_size, seq_len, vocab_size = shift_logits.shape
                flat_logits = shift_logits.view(-1, vocab_size)  # (batch*seq, vocab)
                flat_targets = shift_targets.view(-1)  # (batch*seq,)
                flat_mask = shift_mask.view(-1).float()  # (batch*seq,)

                # Also mask out non-tool_call examples
                tool_mask_expanded = tool_mask.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1).float()
                combined_mask = flat_mask * tool_mask_expanded

                if combined_mask.sum() > 0:
                    # Compute per-token loss
                    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
                    # Apply mask and average
                    param_loss = (per_token_loss * combined_mask).sum() / combined_mask.sum()

            losses['param'] += param_loss.item()

        # ===== 5. Response Generation Loss (only on last step, for direct_answer samples) =====
        response_loss = torch.tensor(0.0, device=action_loss.device)
        direct_answer_mask = (targets['target_action'] == 0)  # direct_answer examples

        if 'response_logits' in outputs and direct_answer_mask.any():
            response_logits = outputs['response_logits']  # (batch_size, seq_len, vocab_size)
            target_response_ids = targets['target_response_ids']  # (batch_size, seq_len)
            response_mask = targets['response_mask']  # (batch_size, seq_len)

            # Only compute loss for direct_answer examples with valid responses
            # Shift targets for next-token prediction: predict token[i+1] from logits[i]
            if response_logits.size(1) > 1 and target_response_ids.size(1) > 1:
                shift_logits = response_logits[:, :-1, :].contiguous()  # (batch, seq-1, vocab)
                shift_targets = target_response_ids[:, 1:].contiguous()  # (batch, seq-1)
                shift_mask = response_mask[:, 1:].contiguous()  # (batch, seq-1)

                # Flatten for cross entropy
                batch_size, seq_len, vocab_size = shift_logits.shape
                flat_logits = shift_logits.view(-1, vocab_size)  # (batch*seq, vocab)
                flat_targets = shift_targets.view(-1)  # (batch*seq,)
                flat_mask = shift_mask.view(-1).float()  # (batch*seq,)

                # Also mask out non-direct_answer examples
                direct_mask_expanded = direct_answer_mask.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1).float()
                combined_mask = flat_mask * direct_mask_expanded

                if combined_mask.sum() > 0:
                    # Compute per-token loss
                    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
                    # Apply mask and average
                    response_loss = (per_token_loss * combined_mask).sum() / combined_mask.sum()

            losses['response'] += response_loss.item()

        # ===== Combine Losses =====
        param_loss_weight = getattr(config, 'param_loss_weight', 1.0)
        response_loss_weight = getattr(config, 'response_loss_weight', 1.0)
        step_loss = (action_loss + tool_loss + num_calls_loss +
                     config.halt_loss_weight * halt_loss +
                     param_loss_weight * param_loss +
                     response_loss_weight * response_loss)
        total_loss += step_loss

        # Accumulate for logging
        losses['action'] += action_loss.item()
        losses['tool'] += tool_loss.item() if isinstance(tool_loss, torch.Tensor) else 0.0
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

    # Tool accuracy (only for tool_call examples)
    tool_mask = (targets['target_action'] == 1)
    if tool_mask.any():
        pred_tool = final_outputs['tool_logits'].argmax(dim=-1)
        tool_acc = (pred_tool[tool_mask] == targets['target_tool_id'][tool_mask]).float().mean().item()
    else:
        tool_acc = 0.0

    # Num calls accuracy (only for tool_call examples)
    num_calls_acc = 0.0
    if tool_mask.any() and 'num_calls_logits' in final_outputs and 'target_num_calls' in targets:
        pred_num_calls = final_outputs['num_calls_logits'].argmax(dim=-1) + 1  # 0-indexed to 1-indexed
        target_num_calls = targets['target_num_calls']
        num_calls_acc = (pred_num_calls[tool_mask] == target_num_calls[tool_mask]).float().mean().item()

    # Overall accuracy (action + tool both correct)
    if tool_mask.any():
        pred_tool = final_outputs['tool_logits'].argmax(dim=-1)
        tool_correct = (pred_tool == targets['target_tool_id']) | (~tool_mask)
        overall_acc = ((pred_action == targets['target_action']) & tool_correct).float().mean().item()
    else:
        overall_acc = action_acc

    # Parameter accuracy (token-level accuracy for tool_call examples)
    param_acc = 0.0
    if 'param_logits' in final_outputs and tool_mask.any():
        param_logits = final_outputs['param_logits']  # (batch_size, seq_len, vocab_size)
        target_param_ids = targets['target_param_ids']  # (batch_size, seq_len)
        param_mask = targets['param_mask']  # (batch_size, seq_len)

        if param_logits.size(1) > 1 and target_param_ids.size(1) > 1:
            # Shift for next-token prediction
            shift_preds = param_logits[:, :-1, :].argmax(dim=-1)  # (batch, seq-1)
            shift_targets = target_param_ids[:, 1:]  # (batch, seq-1)
            shift_mask = param_mask[:, 1:]  # (batch, seq-1)

            # Only count tool_call examples
            tool_mask_expanded = tool_mask.unsqueeze(1).expand_as(shift_mask)
            combined_mask = shift_mask & tool_mask_expanded

            if combined_mask.sum() > 0:
                correct = (shift_preds == shift_targets) & combined_mask
                param_acc = correct.sum().float() / combined_mask.sum().float()
                param_acc = param_acc.item()

    # Response accuracy (token-level accuracy for direct_answer examples)
    response_acc = 0.0
    direct_answer_mask = (targets['target_action'] == 0)
    if 'response_logits' in final_outputs and direct_answer_mask.any():
        response_logits = final_outputs['response_logits']  # (batch_size, seq_len, vocab_size)
        target_response_ids = targets['target_response_ids']  # (batch_size, seq_len)
        response_mask = targets['response_mask']  # (batch_size, seq_len)

        if response_logits.size(1) > 1 and target_response_ids.size(1) > 1:
            # Shift for next-token prediction
            shift_preds = response_logits[:, :-1, :].argmax(dim=-1)  # (batch, seq-1)
            shift_targets = target_response_ids[:, 1:]  # (batch, seq-1)
            shift_mask = response_mask[:, 1:]  # (batch, seq-1)

            # Only count direct_answer examples
            direct_mask_expanded = direct_answer_mask.unsqueeze(1).expand_as(shift_mask)
            combined_mask = shift_mask & direct_mask_expanded

            if combined_mask.sum() > 0:
                correct = (shift_preds == shift_targets) & combined_mask
                response_acc = correct.sum().float() / combined_mask.sum().float()
                response_acc = response_acc.item()

    return {
        'action_accuracy': action_acc,
        'tool_accuracy': tool_acc,
        'num_calls_accuracy': num_calls_acc,
        'overall_accuracy': overall_acc,
        'param_accuracy': param_acc,
        'response_accuracy': response_acc,
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
    """Compute accuracy of generating valid JSON for tool parameters

    Args:
        outputs_per_step: List of model outputs
        targets: Ground truth targets
        tokenizer: Tokenizer for decoding tokens

    Returns:
        valid_json_ratio: Ratio of tool_call examples with valid JSON params
    """
    if tokenizer is None:
        return 0.0

    final_outputs = outputs_per_step[-1]

    # Only check tool_call examples
    tool_mask = (targets['target_action'] == 1)
    if not tool_mask.any():
        return 0.0

    if 'param_logits' not in final_outputs:
        return 0.0

    param_logits = final_outputs['param_logits']  # (batch_size, seq_len, vocab_size)
    param_mask = targets['param_mask']  # (batch_size, seq_len)

    # Get predicted tokens
    pred_tokens = param_logits.argmax(dim=-1)  # (batch_size, seq_len)

    valid_count = 0
    total_count = 0

    for i in range(pred_tokens.size(0)):
        if not tool_mask[i]:
            continue

        total_count += 1

        # Get valid tokens (where mask is 1)
        mask = param_mask[i].bool()
        tokens = pred_tokens[i][mask].tolist()

        if not tokens:
            continue

        # Decode tokens to text
        try:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            text = text.strip()

            # Try to find and parse JSON
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx + 1]
                json.loads(json_str)  # Will raise if invalid
                valid_count += 1
        except (json.JSONDecodeError, Exception):
            pass

    return valid_count / total_count if total_count > 0 else 0.0
