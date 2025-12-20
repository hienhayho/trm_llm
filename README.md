# TRM-LLM: Tiny Recursive Model for LLM Tool Calling

A parameter-efficient approach to training LLMs for tool-calling tasks using recursive reasoning and deep supervision from the TRM (Tiny Recursive Models) paper.

## Overview

TRM-LLM applies the key innovations from [Tiny Recursive Models](https://arxiv.org/abs/2510.04871) to LLM tool-calling:

- **Recursive Reasoning**: Small network applied multiple times to refine reasoning state
- **Deep Supervision**: Train on multiple refinement steps, not just final output
- **Adaptive Computation**: Learned early stopping (fewer steps for easy problems)
- **Parameter Efficiency**: ~150M params achieving strong performance on tool-calling tasks

## Key Features

- **Recursive refinement** - Iteratively improve decisions about which action to take
- **Deep supervision** - Multi-step training with supervision at each iteration
- **Adaptive computation time (ACT)** - Dynamic number of refinement steps based on difficulty
- **Parameter efficient** - Achieve competitive performance with 100M-500M params
- **Tool calling focus** - Specialized for deciding when and how to use tools
- **Multi-GPU training** - DDP and DeepSpeed ZeRO support
- **Staged training pipeline** - Train backbone and generation head separately
- **Muon optimizer** - Fast convergence for large matrix parameters
- **EMA (Exponential Moving Average)** - Stable training for recursive models
- **Focal Loss** - Handle imbalanced datasets (tool_call vs direct_answer)
- **SentencePiece tokenizer** - Train custom tokenizer from your data

## Installation

Using `uv` (recommended):

```bash
# Clone the repository
git clone https://github.com/yourusername/trm_llm.git
cd trm_llm

# Install dependencies with uv
uv sync
```

## Quick Start

### 1. Prepare Your Data

Create a JSONL file where each line is a conversation with tools:

```json
{
  "tools": "[{\"name\": \"calculator\", \"description\": \"Perform calculations\", \"parameters\": {\"expression\": {\"type\": \"string\"}}}]",
  "messages": [
    {"role": "user", "content": "What is 25 * 47?"},
    {"role": "tool_call", "content": "{\"name\": \"calculator\", \"arguments\": {\"expression\": \"25 * 47\"}}"},
    {"role": "tool_response", "content": "{\"result\": 1175}"},
    {"role": "assistant", "content": "The result is 1175."}
  ]
}
```

### 2. Train the Model

Basic training (single GPU):

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --batch_size 8 \
    --max_epochs 50 \
    --save_dir checkpoints
```

### 3. Run Inference

```bash
# Interactive mode
uv run scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --interactive

# Single query with analysis
uv run scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --query "Calculate 123 * 456" \
    --analyze
```

## Architecture

TRM-LLM consists of:

```
Input (user query + tools)
  |
  v
Encoder (Transformer, configurable layers/dim)
  |
  v
Deep Supervision Loop (2-8 steps):
  |-- Recursive Reasoning Module
  |     \-- Refine reasoning state z (n times)
  |-- Action State Module
  |     \-- Update action state y based on z
  \-- Output Heads
        |-- Action: direct_answer vs tool_call
        |-- Num Calls: how many parallel tool calls
        |-- Halt: should we stop refining?
        \-- Generation Head: generate tool call JSON or direct answer
```

### Model Configurations

| Config | Params | Hidden | Layers | Heads | Use Case |
|--------|--------|--------|--------|-------|----------|
| Small | ~50M | 512 | 8 | 4 | Fast prototyping |
| Base | ~150M | 768 | 12 | 12 | Recommended |
| Medium | ~300M | 1024 | 16 | 16 | Better accuracy |
| Large | ~500M | 1024 | 24 | 16 | Maximum performance |

## Training

### Single GPU Training

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --eval_path data/val.jsonl \
    --batch_size 8 \
    --max_epochs 50 \
    --hidden_dim 768 \
    --num_layers 12 \
    --num_heads 12 \
    --save_dir checkpoints
```

### Multi-GPU Training with DDP

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py \
    --data_path data/train.jsonl \
    --batch_size 8 \
    --max_epochs 50 \
    --ddp \
    --save_dir checkpoints
```

### Multi-GPU Training with DeepSpeed

```bash
deepspeed --num_gpus=4 scripts/train.py \
    --data_path data/train.jsonl \
    --batch_size 4 \
    --max_epochs 50 \
    --deepspeed \
    --zero_stage 2 \
    --use_bf16 \
    --save_dir checkpoints
```

### Staged Training Pipeline

TRM-LLM supports a 3-stage training pipeline for better convergence:

| Stage | Description | Trains | Freezes |
|-------|-------------|--------|---------|
| 0 | Backbone | encoder, reasoning, action, output_heads | generation_head |
| 1 | Generation | generation_head | all others |
| 2 | Fine-tune | all parameters | none |

```bash
# Stage 0: Train backbone (big dataset)
deepspeed --num_gpus=4 scripts/train.py \
    --data_path data/train.jsonl \
    --training_stage 0 \
    --deepspeed --zero_stage 2 --use_bf16 \
    --max_epochs 30 \
    --save_dir checkpoints/stage0

# Stage 1: Train generation head (big dataset)
deepspeed --num_gpus=4 scripts/train.py \
    --data_path data/train.jsonl \
    --training_stage 1 \
    --stage_checkpoint checkpoints/stage0/best_model.pt \
    --deepspeed --zero_stage 2 --use_bf16 \
    --max_epochs 20 \
    --save_dir checkpoints/stage1

# Stage 2: Fine-tune all (small curated dataset)
deepspeed --num_gpus=4 scripts/train.py \
    --data_path data/finetune.jsonl \
    --training_stage 2 \
    --stage_checkpoint checkpoints/stage1/best_model.pt \
    --deepspeed --zero_stage 2 --use_bf16 \
    --learning_rate 1e-5 \
    --max_epochs 10 \
    --save_dir checkpoints/stage2
```

### Handling Imbalanced Datasets

For datasets with imbalanced action types (e.g., 70% direct_answer, 30% tool_call):

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --use_focal_loss \
    --focal_gamma 2.0 \
    --action_class_weights 0.3 0.7 \
    --action_loss_weight 2.0 \
    --save_dir checkpoints
```

Options:
- `--use_focal_loss`: Enable Focal Loss for action classification
- `--focal_gamma`: Focus parameter (higher = more focus on hard examples, default: 2.0)
- `--action_class_weights DIRECT TOOL`: Manual class weights (e.g., 0.3 0.7 gives 70% weight to tool_call)
- `--action_loss_weight`: Weight for action loss vs other losses (default: 2.0)

### Using Muon Optimizer

Muon optimizer provides faster convergence for transformer training:

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --optimizer muon \
    --muon_lr 0.02 \
    --learning_rate 1e-4 \
    --save_dir checkpoints
```

Note: Muon requires DeepSpeed ZeRO-2 for distributed training.

### Using EMA for Stable Training

EMA (Exponential Moving Average) helps stabilize recursive model training:

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --use_ema \
    --ema_decay 0.9999 \
    --save_dir checkpoints
```

### Custom SentencePiece Tokenizer

Train a custom tokenizer from your data:

```bash
# Train new tokenizer (default)
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --vocab_size 12000 \
    --save_dir checkpoints

# Use pre-trained tokenizer
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --sp_model checkpoints/sp_tokenizer.model \
    --save_dir checkpoints
```

### Resume Training

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --resume checkpoints/checkpoint_epoch_30.pt
```

## Training Arguments Reference

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden_dim` | 768 | Hidden dimension |
| `--num_layers` | 12 | Number of encoder layers |
| `--num_heads` | 12 | Number of attention heads |
| `--reasoning_dim` | 512 | Reasoning state dimension |
| `--action_dim` | 256 | Action state dimension |
| `--num_recursions` | 3 | Recursive refinements per step |
| `--max_seq_len` | 2048 | Maximum input sequence length |
| `--max_generation_len` | 512 | Maximum generation length |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 8 | Batch size per GPU |
| `--max_epochs` | 50 | Maximum training epochs |
| `--learning_rate` | 1e-4 | Learning rate for AdamW |
| `--optimizer` | adamw | Optimizer: adamw or muon |
| `--muon_lr` | 0.02 | Muon learning rate for hidden weights |
| `--max_supervision_steps` | 8 | Maximum deep supervision steps |

### Loss Weights

| Argument | Default | Description |
|----------|---------|-------------|
| `--action_loss_weight` | 2.0 | Weight for action classification loss |
| `--tool_call_gen_weight` | 2.0 | Weight for tool call generation loss |
| `--direct_answer_gen_weight` | 1.0 | Weight for direct answer generation loss |
| `--special_token_weight` | 5.0 | Extra weight for special tokens |
| `--label_smoothing` | 0.1 | Label smoothing for generation loss |
| `--num_calls_loss_weight` | 1.0 | Weight for num_calls loss (0 to disable) |

### Focal Loss (Imbalanced Data)

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_focal_loss` | True | Use Focal Loss for action classification |
| `--no_focal_loss` | False | Disable Focal Loss |
| `--focal_gamma` | 2.0 | Focal Loss gamma (higher = focus on hard examples) |
| `--action_class_weights` | None | Manual class weights: DIRECT TOOL (e.g., 0.3 0.7) |

### Staged Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--training_stage` | -1 | Stage: -1=standard, 0=backbone, 1=generation, 2=finetune |
| `--stage_checkpoint` | None | Checkpoint from previous stage |
| `--finetune_data_path` | None | Dataset for stage 2 fine-tuning |

### Distributed Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--ddp` | False | Enable DDP multi-GPU training |
| `--deepspeed` | False | Enable DeepSpeed |
| `--zero_stage` | 2 | DeepSpeed ZeRO stage (0, 1, 2, 3) |
| `--use_bf16` | True | Use BF16 mixed precision |
| `--use_fp16` | False | Use FP16 mixed precision |

### EMA

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_ema` | False | Enable EMA |
| `--ema_decay` | 0.9999 | EMA decay rate |

## Data Format

### Input Format

JSONL file with tool-calling conversations:

```json
{
  "tools": "[{\"name\": \"get_weather\", \"description\": \"Get weather info\", \"parameters\": {\"city\": {\"type\": \"string\"}}}]",
  "messages": [
    {"role": "user", "content": "What is the weather in Beijing?"},
    {"role": "tool_call", "content": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}"},
    {"role": "tool_response", "content": "{\"temp\": 20, \"condition\": \"sunny\"}"},
    {"role": "assistant", "content": "The weather in Beijing is sunny with 20 degrees."}
  ]
}
```

### Supported Roles

- `user`: User's query
- `tool_call`: Model decides to call a tool (JSON with name and arguments)
- `tool_response`: Result from tool execution
- `assistant`: Final text response (direct answer without tool call)

### Special Tokens

Create a special tokens file (one per line):

```
<tool_call>
</tool_call>
<tool_response>
</tool_response>
```

Use with `--special_tokens data/special_tokens.txt`.

## Metrics

### Training Metrics

| Metric | Description |
|--------|-------------|
| `loss` | Total training loss |
| `F1` | Macro F1 score (average of both classes) |
| `tc_F1` | F1 score for tool_call class |
| `da_F1` | F1 score for direct_answer class |
| `tc_gen` | Tool call generation token accuracy |
| `da_gen` | Direct answer generation token accuracy |
| `tc_fmt` | Valid tool call format accuracy (JSON structure) |

### Per-Class Metrics

For imbalanced datasets, per-class metrics are more informative than overall accuracy:

| Metric | Description |
|--------|-------------|
| `direct_answer_acc` | Accuracy on direct_answer samples |
| `direct_answer_f1` | F1 score for direct_answer |
| `tool_call_acc` | Accuracy on tool_call samples |
| `tool_call_f1` | F1 score for tool_call |
| `macro_f1` | Average F1 across classes |

Best model is selected based on `macro_f1` (validation).

## Project Structure

```
trm_llm/
├── trm_llm/
│   ├── models/
│   │   ├── trm_llm.py           # Main model
│   │   ├── reasoning_module.py  # Recursive reasoning
│   │   ├── action_module.py     # Action state updates
│   │   ├── output_heads.py      # Output heads
│   │   └── transformer_blocks.py
│   ├── data/
│   │   ├── sp_tokenizer.py      # SentencePiece tokenizer
│   │   ├── dataset.py           # Dataset loading
│   │   └── collator.py          # Batch collation
│   ├── training/
│   │   ├── trainer.py           # Training loop with DDP/DeepSpeed
│   │   └── loss.py              # Loss functions (Focal Loss, etc.)
│   ├── inference/
│   │   └── generator.py         # Inference engine
│   └── utils/
│       ├── config.py            # Configuration
│       └── logger.py            # Structured logging
├── scripts/
│   ├── train.py                 # Training script
│   └── inference.py             # Inference script
├── configs/
│   └── ds_config/
│       └── zero2.json           # DeepSpeed ZeRO-2 config
└── pyproject.toml
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 2

# Use DeepSpeed ZeRO-2 or ZeRO-3
--deepspeed --zero_stage 2

# Use gradient accumulation (via DeepSpeed config)
```

### Low tool_call F1 (Imbalanced Data)

```bash
# Use Focal Loss with class weights
--use_focal_loss --focal_gamma 3.0 --action_class_weights 0.3 0.7
```

### FP16 Loss Scale Issues

```bash
# Use BF16 instead (more stable)
--use_bf16

# Or lower focal_gamma for FP16
--use_fp16 --focal_gamma 1.0
```

### Slow Training

```bash
# Use DeepSpeed with multiple GPUs
deepspeed --num_gpus=4 scripts/train.py --deepspeed --zero_stage 2

# Reduce max_supervision_steps
--max_supervision_steps 4

# Skip dataset statistics computation
--skip_stats
```

### Model Collapse (Always Predicts One Class)

```bash
# Lower learning rate
--learning_rate 5e-5

# Use stronger class weights
--action_class_weights 0.2 0.8

# Increase focal_gamma
--focal_gamma 3.0
```

## License

MIT License

## Citation

If you use this code, please cite the TRM paper:

```bibtex
@article{trm2024,
  title={Tiny Recursive Models},
  author={...},
  journal={arXiv preprint arXiv:2510.04871},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
