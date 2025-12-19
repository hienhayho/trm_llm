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
        tool_call_token_id: Token ID for <tool_call> token (for consistency loss)

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
        'consistency': 0.0,
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

        step_loss = (action_loss_weight * action_loss + num_calls_loss +
                     config.halt_loss_weight * halt_loss +
                     tool_call_gen_weight * tool_call_gen_loss +
                     direct_answer_gen_weight * direct_answer_gen_loss +
                     consistency_loss_weight * consistency_loss)
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
