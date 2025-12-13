#!/usr/bin/env python3
"""Training script for TRM-LLM

Usage:
    uv run scripts/train.py --data_path data/train.jsonl --config configs/model_150m.yaml

Or with default config:
    uv run scripts/train.py --data_path data/train.jsonl
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trm_llm.models.trm_llm import TRMLLM
from trm_llm.data.dataset import ToolCallDataset
from trm_llm.data.tokenizer import ToolCallTokenizer
from trm_llm.data.collator import DataCollator
from trm_llm.training.trainer import TRMTrainer
from trm_llm.utils.config import TRMLLMConfig


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
        "--max_param_len",
        type=int,
        default=128,
        help="Maximum parameter sequence length (default: 128)",
    )
    parser.add_argument(
        "--max_response_len",
        type=int,
        default=512,
        help="Maximum response sequence length (default: 512)",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs (default: 50)")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--max_supervision_steps",
        type=int,
        default=8,
        help="Maximum supervision steps (default: 8)",
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

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("TRM-LLM Training")
    print("=" * 80)
    print(f"Data: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)

    # Create config
    config = TRMLLMConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        reasoning_dim=args.reasoning_dim,
        action_dim=args.action_dim,
        num_recursions=args.num_recursions,
        max_param_len=args.max_param_len,
        max_response_len=args.max_response_len,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        max_supervision_steps=args.max_supervision_steps,
    )

    # Print estimated parameters
    print("\nModel Configuration:")
    print(config)
    params = config.estimate_parameters()
    print(f"\nEstimated Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:.1f}M")

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = ToolCallTokenizer()
    config.vocab_size = tokenizer.vocab_size  # Update vocab size

    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    train_dataset = ToolCallDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        max_param_len=args.max_param_len,
        max_response_len=args.max_response_len,
        compute_stats=not args.skip_stats,
    )
    print(f"Train examples: {len(train_dataset)}")

    # Log dataset statistics
    if train_dataset.stats:
        print("\n" + "=" * 60)
        print("Dataset Statistics:")
        print("=" * 60)
        stats = train_dataset.stats

        # Basic counts
        print(f"  Raw conversations: {stats['raw_conversations']}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Tool call samples: {stats['tool_call_samples']}")
        print(f"  Direct answer samples: {stats['direct_answer_samples']}")
        print(f"  Unique tools: {stats['num_unique_tools']}")

        # Parallel calls distribution
        if stats["num_calls_distribution"]:
            print(f"\n  Parallel calls distribution:")
            for n, count in sorted(stats["num_calls_distribution"].items()):
                print(f"    {n} call(s): {count}")

        # Character-level length stats
        ctx_stats = stats["context_char_lengths"]
        print(f"\n  Context lengths (chars):")
        print(
            f"    Min: {ctx_stats['min']:,}, Max: {ctx_stats['max']:,}, Avg: {ctx_stats['avg']:,.0f}"
        )

        param_stats = stats["param_char_lengths"]
        if param_stats["count"] > 0:
            print(f"\n  Param lengths (chars) [{param_stats['count']} samples]:")
            print(
                f"    Min: {param_stats['min']:,}, Max: {param_stats['max']:,}, Avg: {param_stats['avg']:,.0f}"
            )

        resp_stats = stats["response_char_lengths"]
        if resp_stats["count"] > 0:
            print(f"\n  Response lengths (chars) [{resp_stats['count']} samples]:")
            print(
                f"    Min: {resp_stats['min']:,}, Max: {resp_stats['max']:,}, Avg: {resp_stats['avg']:,.0f}"
            )

        # Token-level stats (full dataset)
        inp_tok_stats = stats["input_token_lengths"]
        inp_tok_orig = stats["input_token_lengths_orig"]
        print(
            f"\n  Input token lengths ({inp_tok_stats['total']} samples, max_length={inp_tok_stats['max_length']}):"
        )
        print(
            f"    Original:  Min: {inp_tok_orig['min']}, Max: {inp_tok_orig['max']}, Avg: {inp_tok_orig['avg']:.0f}"
        )
        print(
            f"    Truncated: Min: {inp_tok_stats['min']}, Max: {inp_tok_stats['max']}, Avg: {inp_tok_stats['avg']:.0f}"
        )
        if inp_tok_stats["truncated_count"] > 0:
            pct = inp_tok_stats["truncated_count"] / inp_tok_stats["total"] * 100
            print(
                f"    WARNING: {inp_tok_stats['truncated_count']} samples ({pct:.1f}%) truncated!"
            )

        param_tok_stats = stats["param_token_lengths"]
        param_tok_orig = stats["param_token_lengths_orig"]
        if param_tok_stats["count"] > 0:
            print(
                f"\n  Param token lengths [{param_tok_stats['count']} samples, max_length={param_tok_stats['max_length']}]:"
            )
            print(
                f"    Original:  Min: {param_tok_orig['min']}, Max: {param_tok_orig['max']}, Avg: {param_tok_orig['avg']:.0f}"
            )
            print(
                f"    Truncated: Min: {param_tok_stats['min']}, Max: {param_tok_stats['max']}, Avg: {param_tok_stats['avg']:.0f}"
            )
            if param_tok_stats["truncated_count"] > 0:
                pct = param_tok_stats["truncated_count"] / param_tok_stats["count"] * 100
                print(
                    f"    WARNING: {param_tok_stats['truncated_count']} samples ({pct:.1f}%) truncated!"
                )

        resp_tok_stats = stats["response_token_lengths"]
        resp_tok_orig = stats["response_token_lengths_orig"]
        if resp_tok_stats["count"] > 0:
            print(
                f"\n  Response token lengths [{resp_tok_stats['count']} samples, max_length={resp_tok_stats['max_length']}]:"
            )
            print(
                f"    Original:  Min: {resp_tok_orig['min']}, Max: {resp_tok_orig['max']}, Avg: {resp_tok_orig['avg']:.0f}"
            )
            print(
                f"    Truncated: Min: {resp_tok_stats['min']}, Max: {resp_tok_stats['max']}, Avg: {resp_tok_stats['avg']:.0f}"
            )
            if resp_tok_stats["truncated_count"] > 0:
                pct = resp_tok_stats["truncated_count"] / resp_tok_stats["count"] * 100
                print(
                    f"    WARNING: {resp_tok_stats['truncated_count']} samples ({pct:.1f}%) truncated!"
                )

        print("=" * 60)
    else:
        print("  (Dataset statistics skipped, use without --skip_stats to compute)")

    # Log 1 sample from training dataset
    print("\nSample from training dataset:")
    print("-" * 40)
    sample = train_dataset[0]
    print(f"Input IDs length: {len(sample['input_ids'])}")
    print(f"Input text (first 200 tokens): {tokenizer.decode(sample['input_ids'][:200])}...")
    print(
        f"Target action: {sample['target_action']} ({'tool_call' if sample['target_action'] == 1 else 'direct_answer'})"
    )
    print(f"Target tool ID: {sample['target_tool_id']}")
    print(f"Target num calls: {sample['target_num_calls']}")
    if sample.get("target_param_ids"):
        print(
            f"Target params ({len(sample['target_param_ids'])} tokens): {tokenizer.decode(sample['target_param_ids'])}"
        )
    if sample.get("target_response_ids"):
        resp_text = tokenizer.decode(sample["target_response_ids"])
        if len(resp_text) > 200:
            resp_text = resp_text[:200] + "..."
        print(f"Target response ({len(sample['target_response_ids'])} tokens): {resp_text}")
    print("-" * 40 + "\n")

    # Load validation dataset
    val_dataset = None
    if args.eval_path:
        print(f"Loading validation dataset from {args.eval_path}...")
        val_dataset = ToolCallDataset(
            args.eval_path,
            tokenizer,
            max_length=args.max_seq_len,
            max_param_len=args.max_param_len,
            max_response_len=args.max_response_len,
        )
        print(f"Validation examples: {len(val_dataset)}")
    elif args.val_split > 0:
        val_size = int(len(train_dataset) * args.val_split)
        train_size = len(train_dataset) - val_size
        if val_size > 0:
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            print(f"Train examples (after split): {len(train_dataset)}")
            print(f"Validation examples: {len(val_dataset)}")

    # Create dataloaders
    collator = DataCollator(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True if args.device == "cuda" else False,
        )

    # Initialize model
    print("\nInitializing model...")
    model = TRMLLM(config)
    print(model)
    actual_params = model.get_num_trainable_params()
    print(f"Actual trainable parameters: {actual_params / 1e6:.1f}M")

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
    )

    # Resume from checkpoint if provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Save tool mapping and config for inference
    import json
    from dataclasses import asdict

    os.makedirs(args.save_dir, exist_ok=True)

    # Save tool mapping
    tool_mapping_path = os.path.join(args.save_dir, "tool_mapping.json")
    with open(tool_mapping_path, "w") as f:
        json.dump(base_dataset.tool_name_to_id, f, indent=2)
    print(f"Tool mapping saved to {tool_mapping_path}")

    # Save config
    config_path = os.path.join(args.save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")

    # Train
    print("\n")
    trainer.train(save_dir=args.save_dir)


if __name__ == "__main__":
    main()
