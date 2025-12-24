"""Tool call evaluation for TRM-LLM

Evaluates the model's ability to generate correct tool calls.
"""

import torch
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

from trm_llm.inference.generator import TRMInference
from trm_llm.utils.config import TRMLLMConfig
from trm_llm.utils.logger import log, log_warning


@dataclass
class EvalSample:
    """A single evaluation sample"""
    tools_json: str
    messages_before: List[Dict[str, str]]  # Conversation history before tool_call
    expected_tool_name: str
    expected_tool_call: Dict  # Full expected tool call


@dataclass
class EvalResult:
    """Evaluation result for a single sample"""
    expected_tool_name: str
    predicted_tool_name: Optional[str]
    is_correct: bool
    predicted_action: str  # 'tool_call' or 'direct_answer'
    confidence: float
    raw_output: Optional[str] = None  # Raw model output text
    token_ids: Optional[List[int]] = None  # Generated token IDs


def extract_eval_samples(
    data_path: str,
    tools_json: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> List[EvalSample]:
    """Extract evaluation samples from dataset

    Splits conversations at points where the next message is a tool_call.

    Args:
        data_path: Path to eval dataset. Supports formats:
            - JSON file with list of conversations: [[msg1, msg2, ...], [msg1, msg2, ...]]
            - JSONL file with one conversation per line
        tools_json: Tools definition JSON string (used if not in dataset)
        system_prompt: System prompt to prepend (used if not in dataset)

    Returns:
        List of EvalSample objects
    """
    samples = []

    # Load data based on file extension
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Try to parse as JSON first (list of conversations)
    try:
        all_data = json.loads(content)
        if isinstance(all_data, list) and len(all_data) > 0:
            # Check if it's a list of lists (conversations) or list of dicts
            if isinstance(all_data[0], list):
                # Format: [[msg1, msg2, ...], [msg1, msg2, ...]]
                conversations = all_data
            elif isinstance(all_data[0], dict):
                if 'role' in all_data[0]:
                    # Single conversation: [msg1, msg2, ...]
                    conversations = [all_data]
                else:
                    # List of {"tools": ..., "messages": ...} dicts
                    conversations = all_data
            else:
                conversations = []
        else:
            conversations = []
    except json.JSONDecodeError:
        # Fall back to JSONL format
        conversations = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                conversations.append(data)
            except json.JSONDecodeError:
                continue

    # Process each conversation
    for conv in conversations:
        # Support formats:
        # 1. {"tools": "...", "messages": [...]} - full format
        # 2. [...] - just messages list
        if isinstance(conv, list):
            # Data is just a list of messages
            messages = conv
            sample_tools_json = tools_json or '[]'
        else:
            # Data is a dict with tools and messages
            messages = conv.get('messages', [])
            sample_tools_json = conv.get('tools', tools_json or '[]')

        # Add system prompt if provided and not already present
        if system_prompt and messages:
            if messages[0].get('role') != 'system':
                messages = [{'role': 'system', 'content': system_prompt}] + messages

        # Find all positions where next message is a tool_call
        for i, msg in enumerate(messages):
            if msg.get('role') == 'tool_call':
                # Extract conversation history before this tool_call
                messages_before = messages[:i]

                # Skip if no messages before (need at least user query)
                if not messages_before:
                    continue

                # Parse expected tool call
                try:
                    tool_call_content = msg.get('content', {})
                    # Handle both dict and JSON string formats
                    if isinstance(tool_call_content, dict):
                        expected_tool_call = tool_call_content
                    else:
                        expected_tool_call = json.loads(tool_call_content)

                    expected_tool_name = expected_tool_call.get('name', '')

                    if expected_tool_name:
                        samples.append(EvalSample(
                            tools_json=sample_tools_json,
                            messages_before=messages_before,
                            expected_tool_name=expected_tool_name,
                            expected_tool_call=expected_tool_call
                        ))
                except (json.JSONDecodeError, AttributeError):
                    continue

    return samples


def parse_tool_call_from_text(text: str) -> Optional[Dict]:
    """Parse tool call from raw text output

    Handles cases where model generates tool call tags even when
    action head predicts direct_answer.

    Args:
        text: Raw model output text

    Returns:
        Parsed tool call dict with 'name' and 'arguments', or None
    """
    import re

    if not text:
        return None

    text = text.strip()

    # Method 1: Check for <tool_call> tags
    tool_call_pattern = r'<tool_call>\s*([\s\S]*?)\s*</tool_call>'
    match = re.search(tool_call_pattern, text)

    if match:
        content = match.group(1).strip()
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                parsed = json.loads(json_str)
                if 'name' in parsed:
                    return parsed
        except json.JSONDecodeError:
            pass

    # Method 2: Check if text contains <tool_call> without closing tag
    if '<tool_call>' in text:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                parsed = json.loads(json_str)
                if 'name' in parsed:
                    return parsed
        except json.JSONDecodeError:
            pass

    # Method 3: Check for raw JSON with "name" field
    if '{"name"' in text or '{"name":' in text:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                parsed = json.loads(json_str)
                if 'name' in parsed:
                    return parsed
        except json.JSONDecodeError:
            pass

    return None


def evaluate_sample(
    inference: TRMInference,
    sample: EvalSample,
    max_steps: Optional[int] = None,
    parse_raw_output: bool = False,
    max_gen_len: int = 64
) -> EvalResult:
    """Evaluate a single sample

    Args:
        inference: TRMInference instance
        sample: EvalSample to evaluate
        max_steps: Max supervision steps
        parse_raw_output: Parse raw output for tool calls
        max_gen_len: Max generation length for tool params

    Returns:
        EvalResult with prediction details
    """
    # Generate prediction
    result = inference.generate_with_history(
        messages=sample.messages_before,
        tools_json=sample.tools_json,
        max_steps=max_steps,
        generate_params=True,
        generate_response=True,
        max_param_len=max_gen_len
    )

    predicted_action = result['action']
    predicted_tool_name = None
    is_correct = False
    raw_output = None
    token_ids = result.get('token_ids')

    if predicted_action == 'tool_call':
        # Get tool name from result
        tool_call = result.get('tool_call', {})
        if tool_call:
            predicted_tool_name = tool_call.get('name')
            raw_output = tool_call.get('raw_output')

        # Check if tool name matches
        if predicted_tool_name == sample.expected_tool_name:
            is_correct = True
    else:
        # Direct answer - get response text
        raw_output = result.get('response')

        # Optionally check if raw output actually contains a tool call
        if parse_raw_output:
            parsed_tool_call = parse_tool_call_from_text(raw_output)
            if parsed_tool_call:
                # Override action to tool_call since output contains tool call
                predicted_action = 'tool_call'
                predicted_tool_name = parsed_tool_call.get('name')

                # Check if tool name matches
                if predicted_tool_name == sample.expected_tool_name:
                    is_correct = True

    return EvalResult(
        expected_tool_name=sample.expected_tool_name,
        predicted_tool_name=predicted_tool_name,
        is_correct=is_correct,
        predicted_action=predicted_action,
        confidence=result.get('confidence', 0.0),
        raw_output=raw_output,
        token_ids=token_ids
    )


def evaluate_tool_call_accuracy(
    model: torch.nn.Module,
    tokenizer,
    config: TRMLLMConfig,
    eval_data_path: str,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
    max_steps: Optional[int] = None,
    tools_json: Optional[str] = None,
    system_prompt: Optional[str] = None,
    batch_size: int = 1,
    ddp: bool = False,
    local_rank: int = -1,
    world_size: int = 1,
    parse_raw_output: bool = False,
    max_gen_len: int = 64,
    verbose: bool = False,
    tool_name_to_id: Optional[Dict[str, int]] = None,
) -> Dict:
    """Evaluate tool call generation accuracy

    This function can be called from training script for periodic evaluation.

    Args:
        model: TRMLLM model (will be set to eval mode)
        tokenizer: SentencePieceTokenizer
        config: TRMLLMConfig
        eval_data_path: Path to eval JSONL file
        device: Device for inference
        max_samples: Maximum samples to evaluate (None for all)
        max_steps: Max supervision steps for inference
        tools_json: Tools definition JSON string (if not in dataset)
        system_prompt: System prompt to prepend (if not in dataset)
        batch_size: Batch size for evaluation
        ddp: Whether DDP is enabled
        local_rank: Local rank for DDP
        world_size: World size for DDP
        parse_raw_output: Parse raw output for tool calls even when action head predicts direct_answer
        verbose: Print detailed results

    Returns:
        Dict with:
            - tool_call_accuracy: Accuracy of predicting correct tool name
            - action_accuracy: Accuracy of predicting tool_call action (vs direct_answer)
            - total_samples: Number of samples evaluated
            - correct_tool_calls: Number of correct tool name predictions
            - correct_actions: Number of correct action predictions
            - per_tool_accuracy: Dict of accuracy per tool name
    """
    # Extract eval samples
    samples = extract_eval_samples(eval_data_path, tools_json, system_prompt)

    if not samples:
        log_warning("No valid eval samples found", path=eval_data_path)
        return {
            'tool_call_accuracy': 0.0,
            'action_accuracy': 0.0,
            'total_samples': 0,
            'correct_tool_calls': 0,
            'correct_actions': 0,
            'per_tool_accuracy': {}
        }

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    total_samples_all = len(samples)

    # DDP: split samples across GPUs
    if ddp and world_size > 1:
        # Each rank processes a subset of samples
        samples_per_rank = len(samples) // world_size
        start_idx = local_rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if local_rank < world_size - 1 else len(samples)
        samples = samples[start_idx:end_idx]
        if verbose:
            log(f"DDP: Processing samples {start_idx}-{end_idx} on rank {local_rank}")

    # Initialize inference
    was_training = model.training
    model.eval()
    inference = TRMInference(model, tokenizer, config, device=device)

    # Set tool name mapping
    # IMPORTANT: Must use the same mapping from training, otherwise tool IDs won't match!
    if tool_name_to_id:
        inference.set_tool_mapping(tool_name_to_id)
        if verbose:
            log(f"Tool mapping loaded from checkpoint", num_tools=len(tool_name_to_id), tools=list(tool_name_to_id.keys()))
    else:
        # Fallback: Build tool name mapping from samples (WARNING: may not match training!)
        log_warning("No tool_name_to_id provided! Building from samples - this may not match training mapping!")
        tool_names = set()
        for sample in samples:
            tool_names.add(sample.expected_tool_name)
            if sample.tools_json:
                try:
                    tools = json.loads(sample.tools_json)
                    for tool in tools:
                        if isinstance(tool, dict) and 'name' in tool:
                            tool_names.add(tool['name'])
                except json.JSONDecodeError:
                    pass
        fallback_mapping = {name: idx for idx, name in enumerate(sorted(tool_names))}
        inference.set_tool_mapping(fallback_mapping)
        if verbose:
            log(f"Tool mapping built from samples", num_tools=len(fallback_mapping), tools=list(fallback_mapping.keys()))

    # Evaluate samples
    results: List[EvalResult] = []
    tool_counts: Dict[str, Dict[str, int]] = {}  # tool_name -> {correct, total}

    # Log first sample to verify preprocessing
    if verbose and samples:
        first_sample = samples[0]
        log("=" * 60)
        log("Sample preview (first sample)")
        log("=" * 60)
        log(f"Expected tool: {first_sample.expected_tool_name}")
        log(f"Expected tool call: {json.dumps(first_sample.expected_tool_call, ensure_ascii=False)}")
        log(f"Tools JSON: {first_sample.tools_json[:200]}..." if len(first_sample.tools_json) > 200 else f"Tools JSON: {first_sample.tools_json}")
        log("-" * 60)
        log(f"Messages before ({len(first_sample.messages_before)}):")
        for i, msg in enumerate(first_sample.messages_before):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            log(f"  [{i}] {role}:")
            log(f"      {content}")
        log("=" * 60)

    correct_so_far = 0
    with torch.no_grad():
        # Process in batches
        num_batches = (len(samples) + batch_size - 1) // batch_size
        pbar = tqdm(range(num_batches), desc="Evaluating", disable=not verbose)

        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(samples))
            batch_samples = samples[start:end]

            # Evaluate each sample in batch (sequential for now, as generation is autoregressive)
            for sample in batch_samples:
                result = evaluate_sample(inference, sample, max_steps, parse_raw_output, max_gen_len)
                results.append(result)

                # Track per-tool accuracy
                tool_name = sample.expected_tool_name
                if tool_name not in tool_counts:
                    tool_counts[tool_name] = {'correct': 0, 'total': 0}
                tool_counts[tool_name]['total'] += 1
                if result.is_correct:
                    correct_so_far += 1
                    tool_counts[tool_name]['correct'] += 1

            # Update progress bar with current accuracy
            current_acc = correct_so_far / len(results) if results else 0
            pbar.set_postfix(acc=f"{current_acc:.2%}", correct=correct_so_far, total=len(results))

    # Restore training mode if needed
    if was_training:
        model.train()

    # DDP: gather results from all ranks
    local_correct = correct_so_far
    local_total = len(results)
    local_action_correct = sum(1 for r in results if r.predicted_action == 'tool_call')

    if ddp and world_size > 1:
        import torch.distributed as dist
        # Gather counts from all ranks
        correct_tensor = torch.tensor([local_correct], device=device)
        total_tensor = torch.tensor([local_total], device=device)
        action_tensor = torch.tensor([local_action_correct], device=device)

        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(action_tensor, op=dist.ReduceOp.SUM)

        correct_so_far = correct_tensor.item()
        total_samples_evaluated = total_tensor.item()
        action_correct_total = action_tensor.item()
    else:
        total_samples_evaluated = local_total
        action_correct_total = local_action_correct

    # Compute metrics (use aggregated values for DDP)
    if ddp and world_size > 1:
        total_samples = int(total_samples_evaluated)
        correct_tool_calls = int(correct_so_far)
        correct_actions = int(action_correct_total)
    else:
        total_samples = len(results)
        correct_tool_calls = sum(1 for r in results if r.is_correct)
        correct_actions = sum(1 for r in results if r.predicted_action == 'tool_call')

    tool_call_accuracy = correct_tool_calls / total_samples if total_samples > 0 else 0.0
    action_accuracy = correct_actions / total_samples if total_samples > 0 else 0.0

    # Per-tool accuracy (local only, not aggregated across ranks for simplicity)
    per_tool_accuracy = {}
    for tool_name, counts in tool_counts.items():
        if counts['total'] > 0:
            per_tool_accuracy[tool_name] = {
                'accuracy': counts['correct'] / counts['total'],
                'correct': counts['correct'],
                'total': counts['total']
            }

    if verbose:
        log("Evaluation Results",
            total_samples=total_samples,
            tool_call_accuracy=f"{tool_call_accuracy:.4f}",
            action_accuracy=f"{action_accuracy:.4f}")

        log("Per-tool accuracy (local rank):")
        for tool_name, stats in per_tool_accuracy.items():
            log(f"  {tool_name}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    # Build detailed results for each sample
    detailed_results = []
    for i, (sample, result) in enumerate(zip(samples, results)):
        detailed_results.append({
            'index': i,
            'expected_tool_name': result.expected_tool_name,
            'predicted_tool_name': result.predicted_tool_name,
            'predicted_action': result.predicted_action,
            'is_correct': result.is_correct,
            'confidence': result.confidence,
            'raw_output': result.raw_output,
            'messages_before': sample.messages_before,
            'expected_tool_call': sample.expected_tool_call,
        })

    return {
        'tool_call_accuracy': tool_call_accuracy,
        'action_accuracy': action_accuracy,
        'total_samples': total_samples,
        'correct_tool_calls': correct_tool_calls,
        'correct_actions': correct_actions,
        'per_tool_accuracy': per_tool_accuracy,
        'detailed_results': detailed_results
    }
