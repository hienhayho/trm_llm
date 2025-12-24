# Quick Start Guide for TRM-LLM

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
# Make sure you have uv installed
# Then sync dependencies
uv sync
```

### 2. Test with Example Data

We've included example data to get you started:

```bash
# Train a small model on example data (will overfit, but good for testing)
uv run scripts/train.py \
    --data_path data/example_train.jsonl \
    --batch_size 2 \
    --max_epochs 10 \
    --hidden_dim 256 \
    --num_layers 4 \
    --save_dir checkpoints_test
```

This will train a tiny model (~10M params) on 5 examples in a few minutes.

### 3. Run Inference

```bash
# Interactive mode
uv run scripts/inference.py \
    --checkpoint checkpoints_test/best_model.pt \
    --interactive
```

Try queries like:
- "What is the weather in Beijing?"
- "Calculate 25 * 47"
- "How is the air quality in Tokyo?"

### 4. Analyze How It Works

See how the model refines its decision:

```bash
uv run scripts/inference.py \
    --checkpoint checkpoints_test/best_model.pt \
    --query "What is 25 times 47?" \
    --analyze
```

You'll see output like:
```
--- Refinement Analysis ---
Step 1: tool_call (conf: 0.654, q: 0.234)
  → Tool: calculator (conf: 0.721)
Step 2: tool_call (conf: 0.891, q: 0.456)
  → Tool: calculator (conf: 0.934)
Step 3: tool_call (conf: 0.976, q: 0.823)
  → Tool: calculator (conf: 0.989)
```
Note: `q` is the Q-head prediction (probability that current answer is correct, from TRM paper)

## 📊 Training on Your Data

### Prepare Your JSONL Data

Format:
```json
{
  "tools": "[{\"name\": \"your_tool\", \"description\": \"...\", \"parameters\": {...}}]",
  "messages": [
    {"role": "user", "content": "user query"},
    {"role": "tool_call", "content": "{\"name\": \"tool\", \"arguments\": {...}}"},
    {"role": "tool_response", "content": "{\"result\": ...}"},
    {"role": "assistant", "content": "final response"}
  ]
}
```

### Train Full Model

For production (10K-100K examples):

```bash
uv run scripts/train.py \
    --data_path your_data.jsonl \
    --batch_size 8 \
    --max_epochs 50 \
    --save_dir checkpoints
```

Monitor training:
- Action accuracy should reach >90%
- Tool accuracy should reach >85%
- Overall accuracy should reach >75%

## 🎯 Key Model Behaviors

### Recursive Refinement

The model progressively improves its decision:
- **Step 1**: Quick guess (may be wrong)
- **Steps 2-3**: Refinement based on reasoning
- **Steps 4+**: Fine-tuning (if needed)

### Adaptive Computation

Easy queries stop early:
- "What is 2+2?" → 1-2 steps
- "Calculate complex formula" → 4-6 steps

### Confidence Scores

- >0.9: Very confident (likely correct)
- 0.7-0.9: Confident
- <0.7: Uncertain (may need more steps or data)

## 🔧 Tuning Hyperparameters

### For Better Accuracy
```bash
--hidden_dim 1024 \
--num_layers 20 \
--reasoning_dim 768 \
--max_supervision_steps 16
```

### For Faster Training
```bash
--hidden_dim 512 \
--num_layers 8 \
--num_recursions 2 \
--max_supervision_steps 4
```

### For Parameter Efficiency
```bash
--hidden_dim 768 \
--num_layers 12 \
--reasoning_dim 256 \
--action_dim 128
```

## 📈 Expected Results

With 10K-100K training examples:

| Model Size | Training Time | Action Acc | Tool Acc | Overall Acc |
|------------|---------------|------------|----------|-------------|
| ~100M | 2-4 hours | ~88% | ~82% | ~72% |
| ~150M | 4-6 hours | ~92% | ~87% | ~78% |
| ~300M | 8-12 hours | ~95% | ~91% | ~84% |

*On single GPU (A100/V100)*

## 🐛 Common Issues

### "Out of memory"
```bash
# Reduce batch size
--batch_size 4

# Or reduce model size
--hidden_dim 512 --num_layers 8
```

### "Low accuracy after training"
- Check data format is correct
- Verify tool names are consistent
- Increase training epochs
- Use more training data
- Try larger model

### "Training is slow"
- Reduce `max_supervision_steps` to 4-6
- Use fewer recursions (`--num_recursions 2`)
- Enable ACT (automatically reduces steps)

## 🎓 Understanding the Output

### Training Output
```
Epoch 10/50
Max supervision steps: 3
Learning rate: 0.000095

Training Results:
  Loss: 0.2341
  Action Accuracy: 0.912
  Tool Accuracy: 0.874
  Overall Accuracy: 0.798
```

### Inference Output
```
Action: tool_call
Tool: calculator
Tool ID: 1
Confidence: 0.965
Steps used: 3
```

## 📚 Next Steps

1. **Read full README** - More details on architecture and training
2. **Prepare your data** - Convert your tool-calling logs to JSONL
3. **Experiment with configs** - Find optimal hyperparameters
4. **Analyze results** - Use `--analyze` flag to understand model behavior
5. **Iterate** - Improve data quality and model config

## 🤝 Need Help?

- Check [README.md](README.md) for comprehensive documentation
- Review example data in `data/example_train.jsonl`
- Open an issue on GitHub
- Check the code comments for implementation details

---

**Ready to train?** Just run:
```bash
uv run scripts/train.py --data_path your_data.jsonl
```
