#!/usr/bin/env python3
"""Evaluation script for TRM-LLM tool call generation

Evaluates the model's ability to generate correct tool calls by:
1. Loading eval dataset (same format as training)
2. Splitting conversations where next message is a tool_call
3. Checking if model generates correct tool name
4. Computing accuracy metrics

Usage:
    uv run scripts/test.py --checkpoint checkpoints/best_model.pt --eval_data data/eval.jsonl

Usage (with DeepSpeed checkpoint):
    uv run scripts/test.py --ds_checkpoint checkpoints/checkpoint_epoch_10_ds --eval_data data/eval.jsonl
"""

import argparse
import torch
import json
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from trm_llm.models.trm_llm import TRMLLM
from trm_llm.data.sp_tokenizer import SentencePieceTokenizer
from trm_llm.utils.config import TRMLLMConfig
from trm_llm.utils.logger import log, log_error
from trm_llm.evaluation import evaluate_tool_call_accuracy


def load_model_for_eval(
    checkpoint_path: str,
    device: str,
    config_path: Optional[str] = None,
    ds_checkpoint_dir: Optional[str] = None,
):
    """Load model from checkpoint for evaluation

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on
        config_path: Optional config path
        ds_checkpoint_dir: Optional DeepSpeed checkpoint directory. If provided,
                          extracts model weights from DeepSpeed checkpoint.

    Returns:
        Tuple of (model, config, tokenizer_path, special_tokens_path, current_max_steps)
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    ds_checkpoint_path = Path(ds_checkpoint_dir) if ds_checkpoint_dir else None

    # Determine checkpoint directory for loading config/training_args
    if ds_checkpoint_path and ds_checkpoint_path.is_dir():
        checkpoint_dir = ds_checkpoint_path.parent
        log("Loading from DeepSpeed checkpoint", path=str(ds_checkpoint_path))
    elif checkpoint_path:
        checkpoint_dir = checkpoint_path.parent
    else:
        raise ValueError("Either --checkpoint or --ds_checkpoint must be provided")

    # Load config
    config = None
    config_file = Path(config_path) if config_path else None
    if config_file and config_file.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TRMLLMConfig(**config_dict)
    else:
        default_config_path = checkpoint_dir / "config.json"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                config_dict = json.load(f)
            config = TRMLLMConfig(**config_dict)
        else:
            raise ValueError(f"No config found at {default_config_path}")

    # Load training args for tokenizer paths
    training_args_path = checkpoint_dir / "training_args.json"
    training_args = {}
    if training_args_path.exists():
        with open(training_args_path, 'r') as f:
            training_args = json.load(f)

    sp_model_path = training_args.get("sp_model")
    if not sp_model_path or not Path(sp_model_path).exists():
        sp_model_path = str(checkpoint_dir / "sp_tokenizer.model")

    special_tokens_path = training_args.get("special_tokens")

    # Get vocab_size from training_args if available (more reliable)
    if "vocab_size" in training_args:
        config.vocab_size = training_args["vocab_size"]

    # Load state_dict based on checkpoint type
    current_max_steps = config.max_supervision_steps

    if ds_checkpoint_path and ds_checkpoint_path.is_dir():
        # Load from DeepSpeed checkpoint directory
        state_dict, current_max_steps = load_deepspeed_checkpoint(
            ds_checkpoint_path, device, config.max_supervision_steps
        )
    elif checkpoint_path and checkpoint_path.exists():
        # Load from regular .pt checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        current_max_steps = checkpoint.get("current_max_steps", config.max_supervision_steps)

        # Warn if checkpoint has FP16 weights (saved before the fix)
        if checkpoint.get("fp16_weights", False):
            from trm_llm.utils.logger import log_warning
            log_warning(
                "This checkpoint contains FP16/BF16 weights which may cause incorrect predictions! "
                "Use --ds_checkpoint to load from DeepSpeed checkpoint instead."
            )
    else:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Create model and load weights
    model = TRMLLM(config)
    model_state = model.state_dict()

    # Check embedding size match
    if "embedding.weight" in state_dict and "embedding.weight" in model_state:
        saved_vocab = state_dict["embedding.weight"].shape[0]
        model_vocab = model_state["embedding.weight"].shape[0]
        if saved_vocab != model_vocab:
            log(f"WARNING: vocab_size mismatch! Checkpoint: {saved_vocab}, Model: {model_vocab}")
            log(f"Updating config.vocab_size to {saved_vocab}")
            config.vocab_size = saved_vocab
            # Recreate model with correct vocab_size
            model = TRMLLM(config)

    model.load_state_dict(state_dict)
    model = model.to(device)

    return model, config, sp_model_path, special_tokens_path, current_max_steps


def load_deepspeed_checkpoint(ds_checkpoint_path: Path, device: str, default_max_steps: int):
    """Load model weights from DeepSpeed checkpoint directory

    Args:
        ds_checkpoint_path: Path to DeepSpeed checkpoint directory
        device: Device to load model on
        default_max_steps: Default max_supervision_steps if not found in checkpoint

    Returns:
        Tuple of (state_dict, current_max_steps)
    """
    current_max_steps = default_max_steps

    # Find the latest checkpoint tag
    latest_path = ds_checkpoint_path / "latest"
    if latest_path.exists():
        with open(latest_path, 'r') as f:
            tag = f.read().strip()
    else:
        # Try to find any subdirectory starting with "global_step"
        subdirs = [d for d in ds_checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("global_step")]
        if subdirs:
            tag = subdirs[0].name
        else:
            tag = None

    # Try to use DeepSpeed's utility to convert ZeRO checkpoint to FP32
    try:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        log("Extracting FP32 weights from DeepSpeed ZeRO checkpoint...")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(ds_checkpoint_path))
        log("Successfully extracted FP32 weights from DeepSpeed checkpoint")
    except ImportError:
        log_error("DeepSpeed not installed. Cannot load DeepSpeed checkpoint.")
        log_error("Install with: pip install deepspeed")
        sys.exit(1)
    except Exception as e:
        # Fallback: try loading the model state directly if not a ZeRO checkpoint
        log(f"Could not use ZeRO extraction ({e}), trying direct load...")
        if tag:
            mp_rank_file = ds_checkpoint_path / tag / "mp_rank_00_model_states.pt"
        else:
            mp_rank_file = ds_checkpoint_path / "mp_rank_00_model_states.pt"

        if mp_rank_file.exists():
            checkpoint = torch.load(mp_rank_file, map_location=device, weights_only=False)
            if "module" in checkpoint:
                state_dict = checkpoint["module"]
            else:
                state_dict = checkpoint
        else:
            log_error(f"Could not load DeepSpeed checkpoint from {ds_checkpoint_path}")
            sys.exit(1)

    # Load client state (epoch, current_max_steps, etc.) from mp_rank_00_model_states.pt
    # DeepSpeed stores client_state values directly at the top level of this file
    if tag:
        mp_rank_file = ds_checkpoint_path / tag / "mp_rank_00_model_states.pt"
    else:
        mp_rank_file = ds_checkpoint_path / "mp_rank_00_model_states.pt"

    if mp_rank_file.exists():
        try:
            checkpoint = torch.load(mp_rank_file, map_location="cpu", weights_only=False)
            # Client state values are stored at top level, not nested
            current_max_steps = checkpoint.get("current_max_steps", default_max_steps)
            epoch = checkpoint.get("epoch", "N/A")
            global_step = checkpoint.get("global_step", "N/A")
            log("Loaded checkpoint metadata", epoch=epoch, global_step=global_step, current_max_steps=current_max_steps)
        except Exception as e:
            log(f"Could not load checkpoint metadata: {e}")

    return state_dict, current_max_steps


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TRM-LLM tool call generation")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--ds_checkpoint", type=str, default=None,
                        help="Path to DeepSpeed checkpoint directory (e.g., checkpoint_epoch_10_ds). "
                        "Use this to load directly from DeepSpeed checkpoints without converting to .pt first.")
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
    parser.add_argument("--max_gen_len", type=int, default=64,
                        help="Max generation length for tool params (lower = faster, default: 64)")
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

    # Validate checkpoint arguments
    if not args.checkpoint and not args.ds_checkpoint:
        log_error("Either --checkpoint or --ds_checkpoint must be provided")
        sys.exit(1)

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
        checkpoint_info = args.ds_checkpoint if args.ds_checkpoint else args.checkpoint
        log("Loading model", checkpoint=checkpoint_info)
    model, config, sp_model_path, special_tokens_path, current_max_steps = load_model_for_eval(
        checkpoint_path=args.checkpoint,
        device=args.device,
        config_path=args.config,
        ds_checkpoint_dir=args.ds_checkpoint,
    )

    # Use current_max_steps from checkpoint if --max_steps not provided
    if args.max_steps is None:
        args.max_steps = current_max_steps
        if is_main:
            log("Using max_steps from checkpoint (curriculum learning)", max_steps=args.max_steps)

    # Use provided paths or defaults from checkpoint
    sp_model_path = Path(args.sp_model) if args.sp_model else Path(sp_model_path)
    special_tokens_path = args.special_tokens or special_tokens_path

    if not sp_model_path.exists():
        log_error("SentencePiece model not found", path=str(sp_model_path))
        sys.exit(1)

    if is_main:
        log("Loading tokenizer", sp_model=str(sp_model_path))
    tokenizer = SentencePieceTokenizer(
        model_path=str(sp_model_path),
        special_tokens_file=special_tokens_path
    )

    # Verify tokenizer vocab matches model vocab
    if tokenizer.vocab_size != config.vocab_size:
        log_error(f"Tokenizer vocab_size ({tokenizer.vocab_size}) != model vocab_size ({config.vocab_size})")
        log_error("This will cause incorrect predictions! Check if you're using the correct tokenizer.")
        sys.exit(1)

    # Load tool mapping from checkpoint directory
    # IMPORTANT: Must use the same mapping from training!
    checkpoint_dir = None
    if args.ds_checkpoint:
        checkpoint_dir = Path(args.ds_checkpoint).parent
    elif args.checkpoint:
        checkpoint_dir = Path(args.checkpoint).parent

    tool_name_to_id = None
    if checkpoint_dir:
        tool_mapping_path = checkpoint_dir / "tool_mapping.json"
        if tool_mapping_path.exists():
            with open(tool_mapping_path, 'r', encoding='utf-8') as f:
                tool_name_to_id = json.load(f)
            if is_main:
                log("Loaded tool mapping from checkpoint", path=str(tool_mapping_path), num_tools=len(tool_name_to_id))
        else:
            if is_main:
                log("WARNING: tool_mapping.json not found in checkpoint directory. Tool IDs may not match!")

    # Load tools definition if provided
    tools_json = None
    if args.tools:
        tools_path = Path(args.tools)
        if not tools_path.exists():
            log_error("Tools file not found", path=str(tools_path))
            sys.exit(1)
        with open(tools_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        tools_json = json.dumps(tools)
        if is_main:
            log("Loaded tools definition", path=str(tools_path), num_tools=len(tools))

    # Load system prompt if provided
    system_prompt = None
    if args.system:
        system_path = Path(args.system)
        if not system_path.exists():
            log_error("System prompt file not found", path=str(system_path))
            sys.exit(1)
        with open(system_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
        if is_main:
            log("Loaded system prompt", path=str(system_path), length=len(system_prompt))

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
        max_gen_len=args.max_gen_len,
        verbose=is_main,
        tool_name_to_id=tool_name_to_id,
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
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save summary file
            summary = {
                'tool_call_accuracy': results['tool_call_accuracy'],
                'action_accuracy': results['action_accuracy'],
                'total_samples': results['total_samples'],
                'correct_tool_calls': results['correct_tool_calls'],
                'correct_actions': results['correct_actions'],
                'per_tool_accuracy': results['per_tool_accuracy'],
            }
            summary_path = output_dir / "summary.json"
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

                    chunk_path = output_dir / f"batch_{file_idx:04d}.json"
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
