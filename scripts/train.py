#!/usr/bin/env python3
"""Training script for TRM-LLM

Usage (single GPU):
    uv run scripts/train.py --data_path data/train.jsonl

Usage (multi-GPU with DDP via torchrun):
    torchrun --nproc_per_node=4 scripts/train.py --data_path data/train.jsonl --ddp

Usage (multi-GPU with uv and torchrun):
    uv run torchrun --nproc_per_node=4 scripts/train.py --data_path data/train.jsonl --ddp
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import os

from trm_llm.models.trm_llm import TRMLLM
from trm_llm.data.dataset import ToolCallDataset
from trm_llm.data.tokenizer import ToolCallTokenizer
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
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--skip_stats",
        action="store_true",
        help="Skip computing dataset statistics (faster startup for large datasets)",
    )

    # Pretrained model loading
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="HuggingFace model name or path to load pretrained embeddings and LM head from (e.g., Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--freeze_pretrained",
        action="store_true",
        default=True,
        help="Freeze pretrained embeddings and LM head (default: True)",
    )
    parser.add_argument(
        "--no_freeze_pretrained",
        action="store_true",
        help="Don't freeze pretrained weights (fine-tune everything)",
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

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed training if requested
    rank, world_size, use_ddp = 0, 1, False
    local_rank = 0

    if args.ddp:
        # Get local_rank from environment (set by torchrun) or argument
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if local_rank < 0:
            local_rank = 0

        rank, world_size, use_ddp = setup_distributed(backend="nccl")

        # Reset logger cache after DDP init
        reset_main_process_cache()

        # Update device to use local GPU
        args.device = f"cuda:{local_rank}"

    log("TRM-LLM Training started")

    # Log training configuration
    train_config = {
        "data_path": args.data_path,
        "device": args.device,
        "save_dir": args.save_dir,
        "optimizer": args.optimizer,
    }
    if use_ddp:
        train_config.update({"rank": rank, "world_size": world_size, "local_rank": local_rank})
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
    if args.pretrained_model:
        train_config.update(
            {
                "pretrained_model": args.pretrained_model,
                "freeze_pretrained": args.freeze_pretrained and not args.no_freeze_pretrained,
            }
        )

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
    )

    # Print estimated parameters
    params = config.estimate_parameters()
    log(
        "Model configuration",
        config=str(config),
        **{f"est_{k}": f"{v:.1f}M" for k, v in params.items()},
    )

    # Initialize tokenizer (use pretrained model's tokenizer if specified)
    base_tokenizer_model = args.pretrained_model if args.pretrained_model else "gpt2"
    tokenizer = ToolCallTokenizer(base_model=base_tokenizer_model)
    config.vocab_size = tokenizer.vocab_size  # Update vocab size
    log("Tokenizer initialized", base_model=base_tokenizer_model, vocab_size=tokenizer.vocab_size)

    # Load dataset
    train_dataset = ToolCallDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        max_generation_len=args.max_generation_len,
        compute_stats=not args.skip_stats,
    )
    log("Training dataset loaded", path=args.data_path, examples=len(train_dataset))

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

    # Load validation dataset
    val_dataset = None
    if args.eval_path:
        val_dataset = ToolCallDataset(
            args.eval_path,
            tokenizer,
            max_length=args.max_seq_len,
            max_generation_len=args.max_generation_len,
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

    # Use DistributedSampler for DDP
    train_sampler = None
    val_sampler = None

    if use_ddp:
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
        if use_ddp:
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

    # Initialize model
    model = TRMLLM(config)

    # Load pretrained embeddings if specified
    if args.pretrained_model:
        freeze = args.freeze_pretrained and not args.no_freeze_pretrained
        vocab_size, pretrained_hidden_dim = model.load_pretrained_embeddings(
            args.pretrained_model,
            freeze=freeze,
            device=args.device,
            tokenizer_vocab_size=tokenizer.vocab_size,  # Pass tokenizer vocab size to handle special tokens
        )
        # Update config vocab size to match
        config.vocab_size = vocab_size
        log(
            "Pretrained embeddings loaded",
            model=args.pretrained_model,
            pretrained_hidden_dim=pretrained_hidden_dim,
            vocab_size=vocab_size,
            freeze=freeze,
        )

    actual_params = model.get_num_trainable_params()
    log("Model initialized", trainable_params=f"{actual_params / 1e6:.1f}M")

    if is_main_process():
        print(model)

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
        optimizer_type=args.optimizer,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        use_ddp=use_ddp,
        local_rank=local_rank,
    )

    # Resume from checkpoint if provided
    if args.resume:
        log("Resuming from checkpoint", path=args.resume)
        trainer.load_checkpoint(args.resume)

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

        # Save training args (for inference to know pretrained model, etc.)
        training_args = {
            "pretrained_model": args.pretrained_model,
            "freeze_pretrained": args.freeze_pretrained and not args.no_freeze_pretrained,
            "tokenizer_base_model": base_tokenizer_model,
            "optimizer": args.optimizer,
            "muon_lr": args.muon_lr if args.optimizer == "muon" else None,
            "muon_momentum": args.muon_momentum if args.optimizer == "muon" else None,
            "ddp": use_ddp,
            "world_size": world_size,
        }
        training_args_path = os.path.join(args.save_dir, "training_args.json")
        with open(training_args_path, "w") as f:
            json.dump(training_args, f, indent=2)

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
