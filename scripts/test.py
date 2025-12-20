#!/usr/bin/env python3
"""Evaluation script for TRM-LLM tool call generation

Evaluates the model's ability to generate correct tool calls by:
1. Loading eval dataset (same format as training)
2. Splitting conversations where next message is a tool_call
3. Checking if model generates correct tool name
4. Computing accuracy metrics

Usage:
    uv run scripts/test.py --checkpoint checkpoints/best_model.pt --eval_data data/eval.jsonl
"""

import argparse
import torch
import json
import sys
import os
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trm_llm.models.trm_llm import TRMLLM
from trm_llm.data.sp_tokenizer import SentencePieceTokenizer
from trm_llm.utils.config import TRMLLMConfig
from trm_llm.utils.logger import log, log_error
from trm_llm.evaluation import evaluate_tool_call_accuracy


def load_model_for_eval(checkpoint_path: str, device: str, config_path: Optional[str] = None):
    """Load model from checkpoint for evaluation

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        config_path: Optional config path

    Returns:
        Tuple of (model, config, tokenizer_path, special_tokens_path)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Load config
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TRMLLMConfig(**config_dict)
    else:
        default_config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as f:
                config_dict = json.load(f)
            config = TRMLLMConfig(**config_dict)
        elif "config" in checkpoint:
            config = checkpoint["config"]
        else:
            raise ValueError("No config found")

    # Load training args for tokenizer paths
    training_args_path = os.path.join(checkpoint_dir, "training_args.json")
    training_args = {}
    if os.path.exists(training_args_path):
        with open(training_args_path, 'r') as f:
            training_args = json.load(f)

    sp_model_path = training_args.get("sp_model")
    if not sp_model_path or not os.path.exists(sp_model_path):
        sp_model_path = os.path.join(checkpoint_dir, "sp_tokenizer.model")

    special_tokens_path = training_args.get("special_tokens")

    # Create model and load weights
    model = TRMLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    return model, config, sp_model_path, special_tokens_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TRM-LLM tool call generation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to eval JSONL file")
    parser.add_argument("--tools", type=str, default=None,
                        help="Path to tools definition JSON file (if not in dataset)")
    parser.add_argument("--system", type=str, default=None,
                        help="Path to system prompt text file (if not in dataset)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON")
    parser.add_argument("--sp_model", type=str, default=None,
                        help="Path to SentencePiece model")
    parser.add_argument("--special_tokens", type=str, default=None,
                        help="Path to special tokens file")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max supervision steps")
    parser.add_argument("--parse_raw_output", action="store_true",
                        help="Parse raw output for tool calls even when action head predicts direct_answer")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save results (summary.json + batch_*.json)")
    parser.add_argument("--samples_per_file", type=int, default=100,
                        help="Number of samples per output file (default: 100)")

    # DDP arguments
    parser.add_argument("--ddp", action="store_true",
                        help="Use DistributedDataParallel for multi-GPU evaluation")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for DDP (set by torchrun)")

    return parser.parse_args()


def setup_ddp(local_rank):
    """Setup DDP environment"""
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup DDP"""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()

    # Setup DDP if enabled
    local_rank = -1
    world_size = 1
    is_main = True

    if args.ddp:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if local_rank == -1:
            log_error("DDP enabled but LOCAL_RANK not set. Use torchrun to launch.")
            sys.exit(1)
        setup_ddp(local_rank)
        world_size = dist.get_world_size()
        is_main = (local_rank == 0)
        args.device = f"cuda:{local_rank}"
        if is_main:
            log(f"DDP enabled", world_size=world_size, local_rank=local_rank)

    if is_main:
        log("Loading model", checkpoint=args.checkpoint)
    model, config, sp_model_path, special_tokens_path = load_model_for_eval(
        args.checkpoint, args.device, args.config
    )

    # Use provided paths or defaults from checkpoint
    sp_model_path = args.sp_model or sp_model_path
    special_tokens_path = args.special_tokens or special_tokens_path

    if not os.path.exists(sp_model_path):
        log_error("SentencePiece model not found", path=sp_model_path)
        sys.exit(1)

    if is_main:
        log("Loading tokenizer", sp_model=sp_model_path)
    tokenizer = SentencePieceTokenizer(
        model_path=sp_model_path,
        special_tokens_file=special_tokens_path
    )
    config.vocab_size = tokenizer.vocab_size

    # Load tools definition if provided
    tools_json = None
    if args.tools:
        if not os.path.exists(args.tools):
            log_error("Tools file not found", path=args.tools)
            sys.exit(1)
        with open(args.tools, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        tools_json = json.dumps(tools)
        if is_main:
            log("Loaded tools definition", path=args.tools, num_tools=len(tools))

    # Load system prompt if provided
    system_prompt = None
    if args.system:
        if not os.path.exists(args.system):
            log_error("System prompt file not found", path=args.system)
            sys.exit(1)
        with open(args.system, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
        if is_main:
            log("Loaded system prompt", path=args.system, length=len(system_prompt))

    if is_main:
        log("Starting evaluation", eval_data=args.eval_data, batch_size=args.batch_size)

    results = evaluate_tool_call_accuracy(
        model=model,
        tokenizer=tokenizer,
        config=config,
        eval_data_path=args.eval_data,
        device=args.device,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        tools_json=tools_json,
        system_prompt=system_prompt,
        batch_size=args.batch_size,
        ddp=args.ddp,
        local_rank=local_rank,
        world_size=world_size,
        parse_raw_output=args.parse_raw_output,
        verbose=is_main
    )

    # Print summary (only on main process)
    if is_main:
        log("=" * 50)
        log("EVALUATION SUMMARY")
        log("=" * 50)
        log(f"Total Samples: {results['total_samples']}")
        log(f"Tool Call Accuracy: {results['tool_call_accuracy']:.4f} ({results['correct_tool_calls']}/{results['total_samples']})")
        log(f"Action Accuracy: {results['action_accuracy']:.4f} ({results['correct_actions']}/{results['total_samples']})")

        # Save results if output path provided
        if args.output:
            # Treat output as directory
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)

            # Save summary file
            summary = {
                'tool_call_accuracy': results['tool_call_accuracy'],
                'action_accuracy': results['action_accuracy'],
                'total_samples': results['total_samples'],
                'correct_tool_calls': results['correct_tool_calls'],
                'correct_actions': results['correct_actions'],
                'per_tool_accuracy': results['per_tool_accuracy'],
            }
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            log(f"Summary saved to {summary_path}")

            # Save detailed results in chunks
            detailed = results.get('detailed_results', [])
            if detailed:
                samples_per_file = args.samples_per_file
                num_files = (len(detailed) + samples_per_file - 1) // samples_per_file

                for file_idx in range(num_files):
                    start = file_idx * samples_per_file
                    end = min(start + samples_per_file, len(detailed))
                    chunk = detailed[start:end]

                    chunk_path = os.path.join(output_dir, f"batch_{file_idx:04d}.json")
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'file_index': file_idx,
                            'start_sample': start,
                            'end_sample': end,
                            'num_samples': len(chunk),
                            'results': chunk
                        }, f, indent=2, ensure_ascii=False)

                log(f"Detailed results saved to {output_dir}/ ({num_files} files)")

    # Cleanup DDP
    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
