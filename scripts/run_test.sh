#!/bin/bash
# Evaluation script for TRM-LLM tool call generation
#
# Usage:
#   ./scripts/run_test.sh <checkpoint_path> <eval_data_path> [options]
#
# Examples:
#   # Basic usage
#   ./scripts/run_test.sh checkpoints/best_model.pt data/eval.json
#
#   # With external tools and system prompt
#   ./scripts/run_test.sh checkpoints/best_model.pt data/eval.json \
#       --tools data/sample_tools.json \
#       --system data/sample_system_prompt.txt
#
#   # With batch size
#   ./scripts/run_test.sh checkpoints/best_model.pt data/eval.json --batch_size 8
#
#   # With max samples limit
#   ./scripts/run_test.sh checkpoints/best_model.pt data/eval.json --max_samples 100
#
#   # Save results to JSON
#   ./scripts/run_test.sh checkpoints/best_model.pt data/eval.json --output results.json
#
# DDP (Multi-GPU) Usage:
#   torchrun --nproc_per_node=4 scripts/test.py \
#       --checkpoint checkpoints/best_model.pt \
#       --eval_data data/eval.json \
#       --tools data/sample_tools.json \
#       --ddp

set -e

# Default values
CHECKPOINT="${1:-checkpoints/best_model.pt}"
EVAL_DATA="${2:-data/eval.json}"
shift 2 2>/dev/null || true

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check if eval data exists
if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: Eval data not found: $EVAL_DATA"
    exit 1
fi

echo "========================================"
echo "TRM-LLM Tool Call Evaluation"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Eval Data:  $EVAL_DATA"
echo "Extra Args: $@"
echo "========================================"

# Run evaluation
uv run scripts/test.py \
    --checkpoint "$CHECKPOINT" \
    --eval_data "$EVAL_DATA" \
    "$@"
