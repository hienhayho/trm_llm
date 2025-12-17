# TRM-LLM: Tiny Recursive Model for LLM Tool Calling

A parameter-efficient approach to training LLMs for tool-calling tasks using recursive reasoning and deep supervision from the TRM (Tiny Recursive Models) paper.

## Overview

TRM-LLM applies the key innovations from [Tiny Recursive Models](https://arxiv.org/abs/2510.04871) to LLM tool-calling:

- **Recursive Reasoning**: Small network applied multiple times to refine reasoning state
- **Deep Supervision**: Train on multiple refinement steps, not just final output
- **Adaptive Computation**: Learned early stopping (fewer steps for easy problems)
- **Parameter Efficiency**: ~150M params achieving strong performance on tool-calling tasks

## Key Features

- 🔄 **Recursive refinement** - Iteratively improve decisions about which action to take
- 📊 **Deep supervision** - Multi-step training with supervision at each iteration
- ⚡ **Adaptive computation time** - Dynamic number of refinement steps based on difficulty
- 💾 **Parameter efficient** - Achieve competitive performance with 100M-500M params
- 🛠️ **Tool calling focus** - Specialized for deciding when and how to use tools

## Installation

Using `uv` (recommended):

```bash
# Clone the repository
git clone https://github.com/yourusername/trm_llm.git
cd trm_llm

# Install dependencies with uv
uv sync

# Or install in editable mode
uv pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Create a JSONL file where each line is a conversation with tools:

```jsonl
{"tools": "[{\"name\": \"calculator\", \"description\": \"...\", \"parameters\": {...}}]", "messages": [{"role": "user", "content": "What is 25 * 47?"}, {"role": "tool_call", "content": "{\"name\": \"calculator\", \"arguments\": {\"a\": 25, \"b\": 47, \"op\": \"multiply\"}}"}, {"role": "tool_response", "content": "{\"result\": 1175}"}, {"role": "assistant", "content": "The result is 1175."}]}
```

### 2. Train the Model

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

# Single query
uv run scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --query "What is the weather in Beijing?" \
    --tools tools.json

# Analyze refinement process
uv run scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --query "Calculate 123 * 456" \
    --analyze
```

## Architecture

TRM-LLM consists of:

```
Input (user query + tools)
  ↓
Encoder (12-layer Transformer, 768-dim)
  ↓
Deep Supervision Loop (2-8 steps):
  ├─ Recursive Reasoning Module
  │   └─ Refine reasoning state z (n=3 times)
  ├─ Action State Module
  │   └─ Update action state y based on z
  └─ Output Heads
      ├─ Action: direct_answer vs tool_call
      ├─ Tool Selection: which tool to use
      └─ Halt: should we stop refining?
```

### Model Configurations

| Config | Params | Hidden | Layers | Reasoning | Action | Use Case |
|--------|--------|--------|--------|-----------|--------|----------|
| Tiny | ~100M | 640 | 10 | 384 | 192 | Fast prototyping |
| **Base** | **~150M** | **768** | **12** | **512** | **256** | **Recommended** |
| Medium | ~300M | 1024 | 20 | 768 | 384 | Better accuracy |
| Large | ~500M | 1024 | 24 | 768 | 512 | Maximum performance |

## Training

### Basic Training

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --val_split 0.1 \
    --batch_size 8 \
    --max_epochs 50
```

### Advanced Options

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --hidden_dim 768 \
    --num_layers 12 \
    --reasoning_dim 512 \
    --action_dim 256 \
    --num_recursions 3 \
    --max_supervision_steps 8 \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --max_epochs 100 \
    --save_dir checkpoints
```

### Resume from Checkpoint

```bash
uv run scripts/train.py \
    --data_path data/train.jsonl \
    --resume checkpoints/checkpoint_epoch_30.pt
```

## Data Format

### Input Format

JSONL file with tool-calling conversations:

```json
{
  "tools": "[{\"name\": \"realtime_aqi\", \"description\": \"Get air quality\", \"parameters\": {\"city\": {\"type\": \"string\"}}}]",
  "messages": [
    {"role": "user", "content": "What is the weather like in Beijing?"},
    {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"Beijing\"}}"},
    {"role": "tool_response", "content": "{\"aqi\": \"10\", \"unit\": \"celsius\"}"},
    {"role": "assistant", "content": "The air quality in Beijing is good with AQI of 10."}
  ]
}
```

### Supported Roles

- `user`: User's query
- `tool_call`: Model decides to call a tool (JSON with name and arguments)
- `tool_response`: Result from tool execution
- `assistant`: Final text response

## Inference

### Interactive Mode

```bash
uv run scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --interactive
```

Example session:
```
User: What is the weather in Beijing?
Assistant:
  Action: tool_call
  Tool: realtime_aqi
  Confidence: 0.923
  Steps used: 3

User: What is 2+2?
Assistant:
  Action: tool_call
  Tool: calculator
  Confidence: 0.987
  Steps used: 2
```

### Analyzing Refinement

See how the model progressively refines its decision:

```bash
uv run scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --query "Calculate 25 * 47" \
    --analyze
```

Output:
```
--- Refinement Analysis ---
Step 1: tool_call (conf: 0.654, halt: 0.234)
  → Tool: calculator (conf: 0.721)
Step 2: tool_call (conf: 0.891, halt: 0.456)
  → Tool: calculator (conf: 0.934)
Step 3: tool_call (conf: 0.976, halt: 0.823)
  → Tool: calculator (conf: 0.989)

--- Result ---
Action: tool_call
Tool: calculator
Confidence: 0.965
Steps used: 3
```

## Training Details

### Deep Supervision

The model is trained with supervision at each refinement step:

```python
for step in range(max_supervision_steps):
    # Refine reasoning and action states
    z = recursive_reasoning(x, y, z)
    y = action_state(y, z)

    # Compute loss at this step
    loss = compute_loss(y, target)

    # Detach for next iteration (no BPTT)
    y, z = y.detach(), z.detach()
```

### Curriculum Learning

Training gradually increases supervision steps:
- Epochs 1-5: 2 steps
- Epochs 6-10: 3 steps
- Epochs 11-15: 4 steps
- ...
- Epochs 31+: 8 steps (maximum)

### Loss Components

1. **Action Loss**: CrossEntropy for direct_answer vs tool_call
2. **Tool Loss**: CrossEntropy for which tool to select
3. **Halt Loss**: BCE for when to stop refining

Total loss = Action Loss + Tool Loss + 0.5 × Halt Loss

## Expected Performance

On tool-calling datasets (10K-100K examples):

| Metric | Expected | Notes |
|--------|----------|-------|
| Action Accuracy | >90% | Correct decision to call tool or answer directly |
| Tool Selection | >85% | Correct tool chosen when action is tool_call |
| Overall Accuracy | >75% | Both action and tool correct |
| Avg Steps (train) | 3-5 | With ACT early stopping |
| Avg Steps (inference) | 2-4 | Fewer for easy queries |

## Project Structure

```
trm_llm/
├── trm_llm/
│   ├── models/          # Model components
│   │   ├── trm_llm.py           # Main model
│   │   ├── reasoning_module.py  # Recursive reasoning
│   │   ├── action_module.py     # Action state updates
│   │   ├── output_heads.py      # Output heads
│   │   └── transformer_blocks.py
│   ├── data/            # Data processing
│   │   ├── tokenizer.py
│   │   ├── dataset.py
│   │   └── collator.py
│   ├── training/        # Training utilities
│   │   ├── trainer.py
│   │   └── loss.py
│   ├── inference/       # Inference engine
│   │   └── generator.py
│   └── utils/
│       └── config.py
├── scripts/
│   ├── train.py         # Training script
│   └── inference.py     # Inference script
├── pyproject.toml       # Dependencies (uv)
└── README.md
```

## Configuration

All hyperparameters can be configured via the `TRMLLMConfig` class:

```python
from trm_llm.utils.config import TRMLLMConfig

config = TRMLLMConfig(
    # Architecture
    vocab_size=50257,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    ff_dim=3072,

    # TRM-specific
    reasoning_dim=512,
    action_dim=256,
    num_recursions=3,
    max_supervision_steps=8,

    # Training
    learning_rate=1e-4,
    batch_size=8,
    max_epochs=50,

    # ACT
    halt_threshold=0.5,
    halt_loss_weight=0.5,
)
```

## Advanced Usage

### Custom Tool Definitions

```python
tools = [
    {
        "name": "search_web",
        "description": "Search the internet",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }
    }
]
```

### Programmatic Usage

```python
from trm_llm.models.trm_llm import TRMLLM
from trm_llm.data.tokenizer import ToolCallTokenizer
from trm_llm.inference.generator import TRMInference
from trm_llm.utils.config import TRMLLMConfig

# Load model
config = TRMLLMConfig()
model = TRMLLM(config)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Initialize inference
tokenizer = ToolCallTokenizer()
inference = TRMInference(model, tokenizer, config)

# Set tool mapping
inference.set_tool_mapping(tool_name_to_id)

# Generate prediction
result = inference.generate(
    user_query="What's the weather in Paris?",
    tools_json='[{"name": "get_weather", ...}]'
)

print(f"Action: {result['action']}")
print(f"Tool: {result['tool_name']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Out of Memory

Reduce batch size or use gradient accumulation:
```bash
uv run scripts/train.py --batch_size 4
```

### Low Accuracy

- Check data quality and format
- Increase model size or training epochs
- Verify tool mappings are correct
- Use more supervision steps

### Slow Training

- Reduce `max_supervision_steps` during training
- Use fewer recursions (`num_recursions=2`)
- Enable ACT for early stopping

## What's Not Included (Future Work)

This is a minimal working prototype. Future extensions:

- [ ] Full parameter generation (autoregressive decoder for tool arguments)
- [ ] Full response generation (text generation for assistant responses)
- [ ] Data augmentation strategies
- [ ] EMA (Exponential Moving Average) for training stability
- [ ] Multi-GPU distributed training
- [ ] Advanced evaluation metrics
- [ ] Integration with real tool execution
- [ ] Multi-turn conversation support

## Contact

For questions or issues, please open an issue on GitHub.
