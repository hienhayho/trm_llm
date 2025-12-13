#!/usr/bin/env python3
"""Inference script for TRM-LLM

Usage:
    uv run scripts/inference.py --checkpoint checkpoints/best_model.pt --query "What is 25 times 47?"
"""

import argparse
import torch
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trm_llm.models.trm_llm import TRMLLM
from trm_llm.data.tokenizer import ToolCallTokenizer
from trm_llm.inference.generator import TRMInference
from trm_llm.utils.config import TRMLLMConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run TRM-LLM inference")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--query", type=str, default=None, help="User query to process")
    parser.add_argument(
        "--tools", type=str, default=None, help="Path to JSON file with tool definitions"
    )
    parser.add_argument(
        "--tool_mapping",
        type=str,
        default=None,
        help="Path to tool mapping JSON (default: same dir as checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON (default: same dir as checkpoint or from checkpoint)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device"
    )
    parser.add_argument("--analyze", action="store_true", help="Analyze refinement across steps")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    return parser.parse_args()


def load_model(checkpoint_path, device, config_path=None):
    """Load model from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on
        config_path: Optional path to config JSON file. If not provided,
                     loads config from checkpoint or tries checkpoint dir.

    Returns:
        model: Loaded TRMLLM model
        config: TRMLLMConfig
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config from JSON file if provided, otherwise from checkpoint
    config = None

    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = TRMLLMConfig(**config_dict)
    else:
        # Try to find config.json in same dir as checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
        default_config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(default_config_path):
            print(f"Loading config from {default_config_path}...")
            with open(default_config_path, "r") as f:
                config_dict = json.load(f)
            config = TRMLLMConfig(**config_dict)
        elif "config" in checkpoint:
            print("Loading config from checkpoint...")
            config = checkpoint["config"]
        else:
            raise ValueError("No config found. Provide --config or ensure config.json exists.")

    print(f"\nModel configuration:")
    print(f"  Parameters: ~{config.estimate_parameters()['total_M']:.1f}M")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Recursions: {config.num_recursions}")
    print(f"  Max supervision steps: {config.max_supervision_steps}")

    model = TRMLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(model)

    metrics = checkpoint.get("metrics", {})
    if metrics:
        print(f"\nCheckpoint metrics:")
        for key, value in metrics.items():
            if "accuracy" in key or "loss" in key:
                print(f"  {key}: {value:.4f}")

    return model, config


def default_tools():
    """Default tool definitions for demo"""
    return [
        {
            "name": "realtime_aqi",
            "description": "Weather forecast. Get real-time air quality, including current air quality, PM2.5, and PM10 information.",
            "parameters": {"city": {"type": "string", "description": "City name, e.g., Shanghai"}},
        },
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"},
                "op": {
                    "type": "string",
                    "description": "Operation: add, subtract, multiply, divide",
                },
            },
        },
    ]


def main():
    args = parse_args()

    # Load model
    model, config = load_model(args.checkpoint, args.device, args.config)

    # Initialize tokenizer
    tokenizer = ToolCallTokenizer()
    config.vocab_size = tokenizer.vocab_size

    # Load tool mapping
    if args.tool_mapping:
        tool_mapping_path = args.tool_mapping
    else:
        # Try to find in same directory as checkpoint
        checkpoint_dir = os.path.dirname(args.checkpoint)
        tool_mapping_path = os.path.join(checkpoint_dir, "tool_mapping.json")

    if os.path.exists(tool_mapping_path):
        print(f"\nLoading tool mapping from {tool_mapping_path}...")
        with open(tool_mapping_path, "r") as f:
            tool_name_to_id = json.load(f)
        print(f"Loaded {len(tool_name_to_id)} tools")
    else:
        print(f"\nWarning: Tool mapping not found at {tool_mapping_path}")
        print("Using empty mapping - tool names may not be resolved correctly")
        tool_name_to_id = {}

    # Initialize inference
    inference = TRMInference(model, tokenizer, config, device=args.device)
    inference.set_tool_mapping(tool_name_to_id)

    # Load tools
    if args.tools:
        with open(args.tools, "r") as f:
            tools = json.load(f)
        tools_json = json.dumps(tools)
    else:
        tools = default_tools()
        tools_json = json.dumps(tools)
        print(f"\nUsing default tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 80)
        print("TRM-LLM Interactive Mode")
        print("=" * 80)
        print("Enter your queries (type 'quit' to exit, 'tools' to show tools)")
        print("=" * 80)

        while True:
            query = input("\nUser: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if query.lower() == "tools":
                print("\nAvailable tools:")
                for tool in tools:
                    print(f"  - {tool['name']}: {tool['description']}")
                continue

            if not query:
                continue

            # Generate
            if args.analyze:
                analysis = inference.analyze_refinement(query, tools_json)
                print("\n--- Refinement Analysis ---")
                for step in analysis["steps"]:
                    print(
                        f"Step {step['step']}: {step['action']} "
                        f"(conf: {step['action_confidence']:.3f}, halt: {step['halt_prob']:.3f})"
                    )
                    if step["action"] == "tool_call":
                        print(
                            f"  → Tool: {step['tool_name']} (conf: {step['tool_confidence']:.3f})"
                        )
                print()

            result = inference.generate(query, tools_json)

            print(f"\nAssistant:")
            print(f"  Action: {result['action']}")
            if result["action"] == "tool_call":
                print(f"  Tool: {result['tool_name']}")
                print(f"  Parallel calls: {result.get('num_parallel_calls', 1)}")
                if result.get("tool_call"):
                    print(f"  Tool Call: {json.dumps(result['tool_call'], indent=4)}")
            elif result["action"] == "direct_answer":
                if result.get("response"):
                    print(f"  Response: {result['response']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Steps used: {result['num_steps']}")

    # Single query mode
    elif args.query:
        print(f"\nQuery: {args.query}")

        if args.analyze:
            print("\n--- Refinement Analysis ---")
            analysis = inference.analyze_refinement(args.query, tools_json)
            for step in analysis["steps"]:
                print(
                    f"Step {step['step']}: {step['action']} "
                    f"(conf: {step['action_confidence']:.3f}, halt: {step['halt_prob']:.3f})"
                )
                if step["action"] == "tool_call":
                    print(f"  → Tool: {step['tool_name']} (conf: {step['tool_confidence']:.3f})")
            print()

        result = inference.generate(args.query, tools_json)

        print(f"\n--- Result ---")
        print(f"Action: {result['action']}")
        if result["action"] == "tool_call":
            print(f"Tool: {result['tool_name']}")
            print(f"Tool ID: {result['tool_id']}")
            print(f"Parallel calls: {result.get('num_parallel_calls', 1)}")
            if result.get("tool_call"):
                print(f"Tool Call: {json.dumps(result['tool_call'], indent=2)}")
        elif result["action"] == "direct_answer":
            if result.get("response"):
                print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Steps used: {result['num_steps']}")

    else:
        print("\nError: Either --query or --interactive must be specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
