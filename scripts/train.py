#!/usr/bin/env python3
"""Training script for TRM-LLM

Usage (single GPU):
    uv run scripts/train.py --data_path data/train.jsonl

Usage (with pre-trained SentencePiece model):
    uv run scripts/train.py --data_path data/train.jsonl --sp_model checkpoints/sp_tokenizer.model

Usage (multi-GPU with DDP via torchrun):
    torchrun --nproc_per_node=4 scripts/train.py --data_path data/train.jsonl --ddp

Usage (multi-GPU with uv and torchrun):
    uv run torchrun --nproc_per_node=4 scripts/train.py --data_path data/train.jsonl --ddp

Usage (multi-GPU with DeepSpeed):
    deepspeed --num_gpus=4 scripts/train.py --data_path data/train.jsonl --deepspeed

Usage (DeepSpeed with ZeRO-3 for larger models):
    deepspeed --num_gpus=4 scripts/train.py --data_path data/train.jsonl --deepspeed --zero_stage 3

Usage (DeepSpeed with custom config):
    deepspeed --num_gpus=4 scripts/train.py --data_path data/train.jsonl --deepspeed --ds_config configs/ds_config.json
"""

import argparse
import shutil
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import os

from trm_llm.models.trm_llm import TRMLLM
from trm_llm.data.dataset import ToolCallDataset
from trm_llm.data.sp_tokenizer import SentencePieceTokenizer
from trm_llm.data.collator import DataCollator
from trm_llm.training.trainer import TRMTrainer, setup_distributed, cleanup_distributed
from trm_llm.utils.config import TRMLLMConfig
from trm_llm.utils.logger import log, log_warning, reset_main_process_cache, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRM-LLM")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL training data")
    parser.add_argument(
        "--eval_path", type=str, default=None, help="Path to JSONL validation data (optional)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Validation split ratio (default: 0.0, only used if eval_path not provided)",
    )

    # Tokenizer
    parser.add_argument(
        "--sp_model",
        type=str,
        default=None,
        help="Path to pre-trained SentencePiece model (.model file). If not provided, trains from dataset.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="Vocabulary size for SentencePiece training (default: 8000, ignored if --sp_model provided)",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        default=None,
        help="Path to special tokens file (one token per line). If not provided, uses defaults.",
    )

    # Model
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=768,
        help="Hidden dimension (default: 768 for ~150M params)",
    )
    parser.add_argument(
        "--num_layers", type=int, default=12, help="Number of encoder layers (default: 12)"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Number of attention heads (default: 12)"
    )
    parser.add_argument(
        "--reasoning_dim", type=int, default=512, help="Reasoning state dimension (default: 512)"
    )
    parser.add_argument(
        "--action_dim", type=int, default=256, help="Action state dimension (default: 256)"
    )
    parser.add_argument(
        "--num_recursions",
        type=int,
        default=3,
        help="Number of recursive refinements per step (default: 3)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum input sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max_generation_len",
        type=int,
        default=512,
        help="Maximum generation sequence length for tool call JSON or direct answers (default: 512)",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs (default: 50)")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for AdamW (default: 1e-4)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for LR scheduler (default: 1000)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "muon"],
        help="Optimizer to use: 'adamw' or 'muon' (default: adamw)",
    )
    parser.add_argument(
        "--muon_lr",
        type=float,
        default=0.02,
        help="Learning rate for Muon hidden weights (default: 0.02)",
    )
    parser.add_argument(
        "--muon_momentum",
        type=float,
        default=0.95,
        help="Momentum for Muon (default: 0.95)",
    )
    parser.add_argument(
        "--max_supervision_steps",
        type=int,
        default=8,
        help="Maximum supervision steps (default: 8)",
    )
    parser.add_argument(
        "--action_loss_weight",
        type=float,
        default=2.0,
        help="Weight for action classification loss (default: 2.0, higher = focus more on action decision)",
    )
    parser.add_argument(
        "--tool_call_gen_weight",
        type=float,
        default=2.0,
        help="Weight for tool call generation loss (default: 2.0, higher = focus more on tool calls)",
    )
    parser.add_argument(
        "--direct_answer_gen_weight",
        type=float,
        default=1.0,
        help="Weight for direct answer generation loss (default: 1.0)",
    )
    parser.add_argument(
        "--special_token_weight",
        type=float,
        default=5.0,
        help="Extra weight for special tokens like <tool_call>, </tool_call> (default: 5.0, improves structure)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for generation loss (default: 0.1, 0.0 = no smoothing)",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        default=True,
        help="Use Focal Loss for action classification (default: True, better for class imbalance)",
    )
    parser.add_argument(
        "--no_focal_loss",
        action="store_true",
        help="Disable Focal Loss, use CrossEntropy instead",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter (default: 2.0, higher = more focus on hard examples). "
             "For FP16, use 1.0-1.5 for better numerical stability.",
    )
    parser.add_argument(
        "--action_class_weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("DIRECT", "TOOL"),
        help="Manual class weights for action loss: [direct_answer_weight, tool_call_weight]. "
             "Example: --action_class_weights 0.3 0.7 (give 70%% weight to tool_call). "
             "If not set, weights are auto-computed from batch frequencies.",
    )
    parser.add_argument(
        "--num_calls_loss_weight",
        type=float,
        default=1.0,
        help="Weight for num_calls loss (default: 1.0, set to 0 to disable for single-call datasets)",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Enable EMA (Exponential Moving Average) for stable recursive depth",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate (default: 0.9999, higher = more smoothing)",
    )

    # Architecture options
    parser.add_argument(
        "--use_causal_encoder",
        action="store_true",
        help="Use causal attention in encoder (default: False, bidirectional)",
    )
    parser.add_argument(
        "--no_detach",
        action="store_true",
        help="Disable state detaching between supervision steps (allows gradients through all steps)",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable Flash Attention (use standard nn.MultiheadAttention)",
    )

    # Staged training
    parser.add_argument(
        "--training_stage",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2],
        help="Training stage: -1=standard, 0=backbone+heads, 1=generation, 2=finetune (default: -1)",
    )
    parser.add_argument(
        "--stage_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint from previous stage (required for stages 1-2)",
    )
    parser.add_argument(
        "--finetune_data_path",
        type=str,
        default=None,
        help="Path to smaller dataset for stage 2 fine-tuning (uses --data_path if not provided)",
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10, 0 to disable)",
    )
    parser.add_argument(
        "--log_sample_interval",
        type=int,
        default=0,
        help="Log sample prediction every N steps (default: 0 = only at end of epoch)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--skip_stats",
        action="store_true",
        help="Skip computing dataset statistics (faster startup for large datasets)",
    )
    parser.add_argument(
        "--stats_num_workers",
        type=int,
        default=0,
        help="Number of workers for parallel tokenization in stats computation (default: 0 = sequential)",
    )

    # Distributed training (DDP)
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable Distributed Data Parallel (DDP) for multi-GPU training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for DDP (automatically set by torchrun, do not set manually)",
    )

    # DeepSpeed
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Enable DeepSpeed for memory-efficient training with ZeRO optimization",
    )
    parser.add_argument(
        "--ds_config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON file (default: use built-in config)",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="DeepSpeed ZeRO stage (default: 2). 0=disabled, 1=optimizer, 2=optimizer+gradients, 3=full",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True,
        help="Use BF16 mixed precision (default: True, more stable than FP16)",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 mixed precision instead of BF16 (less stable but wider GPU support)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed training if requested
    rank, world_size, use_ddp = 0, 1, False
    use_deepspeed = args.deepspeed
    local_rank = 0

    # DeepSpeed takes priority over DDP
    if use_deepspeed:
        # Get local_rank from environment (set by deepspeed launcher)
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if local_rank < 0:
            local_rank = 0
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Update device to use local GPU
        args.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        # Reset logger cache
        reset_main_process_cache()

    elif args.ddp:
        # Get local_rank from environment (set by torchrun) or argument
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if local_rank < 0:
            local_rank = 0

        rank, world_size, use_ddp = setup_distributed(backend="nccl")

        # Reset logger cache after DDP init
        reset_main_process_cache()

        # Update device to use local GPU
        args.device = f"cuda:{local_rank}"

    # Validate staged training arguments
    if args.training_stage in [1, 2] and not args.stage_checkpoint:
        raise ValueError(f"--stage_checkpoint is required for training stage {args.training_stage}")

    # Use finetune data path for stage 2 if provided
    data_path = args.data_path
    if args.training_stage == 2 and args.finetune_data_path:
        data_path = args.finetune_data_path
        log("Using fine-tune dataset for stage 2", path=data_path)

    log("TRM-LLM Training started")

    # Prepare DeepSpeed config if using DeepSpeed
    ds_config = None
    if use_deepspeed:
        if args.ds_config:
            ds_config = args.ds_config  # Path to config file
        else:
            # Build config from arguments
            # Use BF16 by default (more stable than FP16, especially with Focal Loss)
            # --use_fp16 overrides --use_bf16
            use_bf16 = args.use_bf16 and not args.use_fp16

            # Warn if using FP16 with high focal_gamma
            if not use_bf16 and args.use_focal_loss and not args.no_focal_loss:
                if args.focal_gamma > 1.5:
                    log_warning(
                        "FP16 with high focal_gamma may cause loss scale issues. "
                        "Consider using --focal_gamma 1.0 for better stability.",
                        focal_gamma=args.focal_gamma,
                    )

            ds_config = {
                "train_micro_batch_size_per_gpu": args.batch_size,
                "gradient_accumulation_steps": 1,
                "gradient_clipping": 1.0,
                "bf16": {
                    "enabled": use_bf16
                },
                "fp16": {
                    "enabled": not use_bf16,
                    "loss_scale": 0,  # Dynamic loss scaling
                    "loss_scale_window": 2000,  # More steps before scaling adjustment (stability)
                    "initial_scale_power": 16,  # Higher initial scale (65536)
                    "hysteresis": 4,  # More stable scaling decisions
                    "min_loss_scale": 1  # Minimum scale before failing
                },
                "scheduler": {
                    "type": "WarmupDecayLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": args.learning_rate,
                        "warmup_num_steps": args.warmup_steps,
                        "total_num_steps": args.warmup_steps + 100000  # Will be updated in trainer
                    }
                },
                "zero_optimization": {
                    "stage": args.zero_stage,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 5e8,
                    "allgather_bucket_size": 5e8
                },
                "zero_allow_untested_optimizer": True,
                "wall_clock_breakdown": False
            }

    # Log training configuration
    train_config = {
        "data_path": args.data_path,
        "device": args.device,
        "save_dir": args.save_dir,
        "optimizer": args.optimizer,
        "vocab_size": args.vocab_size,
    }
    if args.sp_model:
        train_config["sp_model"] = args.sp_model
    if args.special_tokens:
        train_config["special_tokens"] = args.special_tokens
    if use_ddp:
        train_config.update({"rank": rank, "world_size": world_size, "local_rank": local_rank})
    if use_deepspeed:
        train_config.update({
            "deepspeed": True,
            "zero_stage": args.zero_stage,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
        })
    if args.optimizer == "muon":
        train_config.update(
            {
                "muon_lr": args.muon_lr,
                "muon_momentum": args.muon_momentum,
                "adam_lr": args.learning_rate,
            }
        )
    else:
        train_config["learning_rate"] = args.learning_rate
    if args.use_ema:
        train_config.update({"use_ema": True, "ema_decay": args.ema_decay})
    if args.training_stage >= 0:
        train_config.update({
            "training_stage": args.training_stage,
            "stage_checkpoint": args.stage_checkpoint,
        })
    if args.use_focal_loss and not args.no_focal_loss:
        train_config.update({
            "focal_loss": True,
            "focal_gamma": args.focal_gamma,
            "action_class_weights": args.action_class_weights if args.action_class_weights else "auto",
        })

    log("Training configuration", **train_config)

    # Create config
    config = TRMLLMConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        reasoning_dim=args.reasoning_dim,
        action_dim=args.action_dim,
        num_recursions=args.num_recursions,
        max_generation_len=args.max_generation_len,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        max_supervision_steps=args.max_supervision_steps,
        action_loss_weight=args.action_loss_weight,
        tool_call_gen_weight=args.tool_call_gen_weight,
        direct_answer_gen_weight=args.direct_answer_gen_weight,
        special_token_weight=args.special_token_weight,
        label_smoothing=args.label_smoothing,
        # Focal Loss options
        use_focal_loss=args.use_focal_loss and not args.no_focal_loss,
        focal_gamma=args.focal_gamma,
        action_class_weights=tuple(args.action_class_weights) if args.action_class_weights else None,
        num_calls_loss_weight=args.num_calls_loss_weight,
        # Architecture options
        use_causal_encoder=args.use_causal_encoder,
        detach_between_steps=not args.no_detach,
        use_flash_attention=not args.no_flash_attention,
        # Staged training
        training_stage=args.training_stage,
    )

    # Print estimated parameters
    params = config.estimate_parameters()
    log(
        "Model configuration",
        config=str(config),
        **{f"est_{k}": f"{v:.1f}M" for k, v in params.items()},
    )

    # Initialize SentencePiece tokenizer
    # Only main process should train tokenizer in distributed mode (DDP or DeepSpeed)
    tokenizer = SentencePieceTokenizer(
        vocab_size=args.vocab_size,
        special_tokens_file=args.special_tokens,
    )
    is_distributed = use_ddp or use_deepspeed
    is_main = (rank == 0)

    if args.sp_model and os.path.exists(args.sp_model):
        # Load pre-trained SentencePiece model (all processes load, only main logs)
        tokenizer.load(args.sp_model)
        if is_main:
            log("SentencePiece model loaded", path=args.sp_model, vocab_size=tokenizer.vocab_size)
    else:
        # Train SentencePiece model from dataset (only on rank 0)
        sp_model_path = os.path.join(args.save_dir, "sp_tokenizer.model")

        if is_main:
            os.makedirs(args.save_dir, exist_ok=True)

            # Collect all data paths for training
            data_paths = [args.data_path]
            if args.eval_path:
                data_paths.append(args.eval_path)

            log("Training SentencePiece model from dataset...", data_paths=data_paths)
            sp_model_path = tokenizer.train(
                data_paths=data_paths,
                output_dir=args.save_dir,
                model_prefix="sp_tokenizer",
                vocab_size=args.vocab_size,
            )
            log("SentencePiece model trained", path=sp_model_path, vocab_size=tokenizer.vocab_size)

        # Synchronize in distributed mode - other processes wait and then load
        if is_distributed:
            import torch.distributed as dist
            # Initialize process group for DeepSpeed if not already done
            if use_deepspeed and not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            dist.barrier()
            if not is_main:
                tokenizer.load(sp_model_path)
                log("SentencePiece model loaded (synced)", path=sp_model_path, vocab_size=tokenizer.vocab_size)

    config.vocab_size = tokenizer.vocab_size

    # Load dataset (only compute stats on main process)
    train_dataset = ToolCallDataset(
        data_path,
        tokenizer,
        max_length=args.max_seq_len,
        max_generation_len=args.max_generation_len,
        compute_stats=not args.skip_stats,
        is_main_process=is_main,
        num_workers=args.stats_num_workers,
    )
    log("Training dataset loaded", path=data_path, examples=len(train_dataset))

    # Log dataset statistics
    if train_dataset.stats:
        stats = train_dataset.stats

        # Basic counts
        log(
            "Dataset statistics",
            raw_conversations=stats["raw_conversations"],
            total_samples=stats["total_samples"],
            tool_call_samples=stats["tool_call_samples"],
            direct_answer_samples=stats["direct_answer_samples"],
            unique_tools=stats["num_unique_tools"],
        )

        # Parallel calls distribution
        if stats["num_calls_distribution"]:
            log(
                "Parallel calls distribution",
                **{
                    f"{n}_calls": count
                    for n, count in sorted(stats["num_calls_distribution"].items())
                },
            )

        # Character-level length stats
        ctx_stats = stats["context_char_lengths"]
        log(
            "Context lengths (chars)",
            min=ctx_stats["min"],
            max=ctx_stats["max"],
            avg=int(ctx_stats["avg"]),
        )

        gen_char_stats = stats["generation_char_lengths"]
        if gen_char_stats["count"] > 0:
            log(
                "Generation lengths (chars)",
                count=gen_char_stats["count"],
                min=gen_char_stats["min"],
                max=gen_char_stats["max"],
                avg=int(gen_char_stats["avg"]),
            )

        # Token-level stats (full dataset)
        inp_tok_stats = stats["input_token_lengths"]
        inp_tok_orig = stats["input_token_lengths_orig"]
        log(
            "Input token lengths",
            total=inp_tok_stats["total"],
            max_length=inp_tok_stats["max_length"],
            orig_min=inp_tok_orig["min"],
            orig_max=inp_tok_orig["max"],
            orig_avg=int(inp_tok_orig["avg"]),
            trunc_min=inp_tok_stats["min"],
            trunc_max=inp_tok_stats["max"],
            trunc_avg=int(inp_tok_stats["avg"]),
        )
        if inp_tok_stats["truncated_count"] > 0:
            pct = inp_tok_stats["truncated_count"] / inp_tok_stats["total"] * 100
            log_warning(
                "Input samples truncated",
                count=inp_tok_stats["truncated_count"],
                percent=f"{pct:.1f}%",
            )

        gen_tok_stats = stats["generation_token_lengths"]
        gen_tok_orig = stats["generation_token_lengths_orig"]
        if gen_tok_stats["count"] > 0:
            log(
                "Generation token lengths",
                count=gen_tok_stats["count"],
                max_length=gen_tok_stats["max_length"],
                orig_min=gen_tok_orig["min"],
                orig_max=gen_tok_orig["max"],
                orig_avg=int(gen_tok_orig["avg"]),
                trunc_min=gen_tok_stats["min"],
                trunc_max=gen_tok_stats["max"],
                trunc_avg=int(gen_tok_stats["avg"]),
            )
            if gen_tok_stats["truncated_count"] > 0:
                pct = gen_tok_stats["truncated_count"] / gen_tok_stats["count"] * 100
                log_warning(
                    "Generation samples truncated",
                    count=gen_tok_stats["truncated_count"],
                    percent=f"{pct:.1f}%",
                )
    else:
        log("Dataset statistics skipped (use without --skip_stats to compute)")

    # Log 1 sample from training dataset
    sample = train_dataset[0]
    action_type = "tool_call" if sample["target_action"] == 1 else "direct_answer"
    sample_info = {
        "input_ids_length": len(sample["input_ids"]),
        "input_preview": tokenizer.decode(sample["input_ids"][:200]) + "...",
        "target_action": f"{sample['target_action']} ({action_type})",
        "target_tool_id": sample["target_tool_id"],
        "target_num_calls": sample["target_num_calls"],
    }
    if sample.get("target_generation_ids"):
        gen_text = tokenizer.decode(sample["target_generation_ids"])
        if len(gen_text) > 300:
            gen_text = gen_text[:300] + "..."
        sample_info["target_generation"] = (
            f"({len(sample['target_generation_ids'])} tokens) {gen_text}"
        )
    log("Training sample", **sample_info)

    # Load validation dataset (skip stats for validation)
    val_dataset = None
    if args.eval_path:
        val_dataset = ToolCallDataset(
            args.eval_path,
            tokenizer,
            max_length=args.max_seq_len,
            max_generation_len=args.max_generation_len,
            compute_stats=False,  # Skip stats for validation dataset
            is_main_process=is_main,
        )
        log("Validation dataset loaded", path=args.eval_path, examples=len(val_dataset))
    elif args.val_split > 0:
        val_size = int(len(train_dataset) * args.val_split)
        train_size = len(train_dataset) - val_size
        if val_size > 0:
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            log("Dataset split", train_examples=len(train_dataset), val_examples=len(val_dataset))

    # Create dataloaders
    collator = DataCollator(tokenizer.pad_token_id)

    # Use DistributedSampler for DDP or DeepSpeed
    train_sampler = None
    val_sampler = None

    if use_ddp or use_deepspeed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        log("Using DistributedSampler", world_size=world_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True if "cuda" in args.device else False,
    )

    val_loader = None
    if val_dataset is not None:
        if use_ddp or use_deepspeed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True if "cuda" in args.device else False,
        )

    # Initialize model (train from scratch, no pretrained weights)
    model = TRMLLM(config)

    # Log parameter breakdown (only on main process)
    if is_main_process():
        model.log_param_breakdown()

    # Get tool mapping from dataset (handle Subset case from random_split)
    base_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
    tool_id_to_name = {v: k for k, v in base_dataset.tool_name_to_id.items()}

    # Initialize trainer
    trainer = TRMTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        device=args.device,
        tokenizer=tokenizer,
        tool_id_to_name=tool_id_to_name,
        save_interval=args.save_interval,
        log_sample_interval=args.log_sample_interval,
        optimizer_type=args.optimizer,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        use_ddp=use_ddp,
        local_rank=local_rank,
        use_deepspeed=use_deepspeed,
        ds_config=ds_config,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        training_stage=args.training_stage,
    )

    # Load checkpoint from previous stage or resume
    if args.stage_checkpoint:
        log("Loading checkpoint from previous stage", path=args.stage_checkpoint, stage=args.training_stage)
        trainer.load_checkpoint_for_stage(args.stage_checkpoint, args.training_stage)
    elif args.resume:
        log("Resuming from checkpoint", path=args.resume)
        trainer.load_checkpoint(args.resume)

    # Log param breakdown after freezing is applied (for staged training)
    if args.training_stage >= 0 and is_main_process():
        log(f"Parameter breakdown after stage {args.training_stage} freezing:")
        trainer.raw_model.log_param_breakdown()

    # Save tool mapping and config for inference (only on main process)
    import json
    from dataclasses import asdict

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

        # Save tool mapping
        tool_mapping_path = os.path.join(args.save_dir, "tool_mapping.json")
        with open(tool_mapping_path, "w") as f:
            json.dump(base_dataset.tool_name_to_id, f, indent=2)

        # Save config
        config_path = os.path.join(args.save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Save training args (for inference)
        training_args = {
            "sp_model": os.path.join(args.save_dir, "sp_tokenizer.model"),
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": args.special_tokens,
            "optimizer": args.optimizer,
            "muon_lr": args.muon_lr if args.optimizer == "muon" else None,
            "muon_momentum": args.muon_momentum if args.optimizer == "muon" else None,
            "ddp": use_ddp,
            "world_size": world_size,
            "use_ema": args.use_ema,
            "ema_decay": args.ema_decay if args.use_ema else None,
            "training_stage": args.training_stage,
        }
        training_args_path = os.path.join(args.save_dir, "training_args.json")
        with open(training_args_path, "w") as f:
            json.dump(training_args, f, indent=2)

        # Copy SentencePiece model to save_dir if it was provided externally
        if args.sp_model and os.path.exists(args.sp_model):
            sp_dest = os.path.join(args.save_dir, "sp_tokenizer.model")
            if os.path.abspath(args.sp_model) != os.path.abspath(sp_dest):
                shutil.copy(args.sp_model, sp_dest)
                # Also copy vocab file if exists
                vocab_src = args.sp_model.replace(".model", ".vocab")
                if os.path.exists(vocab_src):
                    shutil.copy(vocab_src, os.path.join(args.save_dir, "sp_tokenizer.vocab"))

        log(
            "Training artifacts saved",
            tool_mapping=tool_mapping_path,
            config=config_path,
            training_args=training_args_path,
        )

    # Train
    log("Starting training")
    try:
        trainer.train(save_dir=args.save_dir)
    finally:
        # Clean up distributed environment
        if use_ddp:
            cleanup_distributed()


if __name__ == "__main__":
    main()
