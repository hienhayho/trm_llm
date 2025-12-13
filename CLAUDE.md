# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRM-LLM is a parameter-efficient model for LLM tool-calling using Tiny Recursive Models (TRM) concepts. It applies recursive reasoning and deep supervision to decide when/which tools to call with ~150M parameters.

## Common Commands

```bash
# Install dependencies
uv sync

# Train model
uv run scripts/train.py --data_path data/train.jsonl --batch_size 8 --max_epochs 50 --save_dir checkpoints

# Train small test model
uv run scripts/train.py --data_path data/example_train.jsonl --batch_size 2 --max_epochs 10 --hidden_dim 256 --num_layers 4 --num_heads 8 --save_dir checkpoints_test

# Inference (interactive)
uv run scripts/inference.py --checkpoint checkpoints/best_model.pt --interactive

# Inference with analysis
uv run scripts/inference.py --checkpoint checkpoints/best_model.pt --query "Calculate 25 * 47" --analyze

# Resume training from checkpoint
uv run scripts/train.py --data_path data/train.jsonl --resume checkpoints/checkpoint_epoch_30.pt

# Format code
uv run black trm_llm scripts --line-length 100
```

## Architecture

The model uses a deep supervision loop where each iteration refines the decision:

```
Input → Encoder → [Recursive Reasoning (z) → Action Update (y)]×T → Output Heads
```

**Key modules:**
- `models/trm_llm.py` - Main model orchestrating the deep supervision loop
- `models/reasoning_module.py` - Recursive refinement of reasoning state z (applied n times per supervision step)
- `models/action_module.py` - Updates action state y based on refined z
- `models/output_heads.py` - Decodes y into action_type (direct_answer/tool_call), tool_selection, and halt probability

**Data flow:**
1. Encoder processes input tokens → hidden states x
2. For T supervision steps:
   - Recursively refine z: `z = f(x, y, z)` repeated n times
   - Update action state: `y = g(y, z)`
   - Generate outputs from y
   - Detach y, z (no BPTT across steps)

**Adaptive Computation (ACT):** Model learns when to stop refinement early via halt head.

## Configuration Constraints

`hidden_dim` must be divisible by `num_heads`. Default: 768 dim, 12 heads.
When using smaller hidden_dim (e.g., 256), adjust num_heads accordingly (e.g., 8 or 4).

## Data Format

JSONL with tool-calling conversations:
```json
{
  "tools": "[{\"name\": \"calculator\", \"description\": \"...\", \"parameters\": {...}}]",
  "messages": [
    {"role": "user", "content": "What is 25 * 47?"},
    {"role": "tool_call", "content": "{\"name\": \"calculator\", \"arguments\": {...}}"},
    {"role": "tool_response", "content": "{\"result\": 1175}"},
    {"role": "assistant", "content": "The result is 1175."}
  ]
}
```

Roles: `user`, `tool_call`, `tool_response`, `assistant`

## Key Files

- `trm_llm/utils/config.py` - TRMLLMConfig dataclass with all hyperparameters
- `trm_llm/training/loss.py` - Multi-component loss (action, tool, halt)
- `trm_llm/data/tokenizer.py` - GPT-2 based tokenizer with special tokens
- `trm_llm/data/dataset.py` - Dataset loading and tool-to-ID mapping
- `trm_llm/inference/generator.py` - TRMInference class for generation
