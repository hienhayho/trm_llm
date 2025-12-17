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
from trm_llm.utils.logger import log, log_warning, log_error


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
        training_args: Training arguments dict (may contain pretrained_model info)
    """
    log("Loading checkpoint", path=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Load config from JSON file if provided, otherwise from checkpoint
    config = None

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = TRMLLMConfig(**config_dict)
        config_source = config_path
    else:
        # Try to find config.json in same dir as checkpoint
        default_config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(default_config_path):
            with open(default_config_path, "r") as f:
                config_dict = json.load(f)
            config = TRMLLMConfig(**config_dict)
            config_source = default_config_path
        elif "config" in checkpoint:
            config = checkpoint["config"]
            config_source = "checkpoint"
        else:
            raise ValueError("No config found. Provide --config or ensure config.json exists.")

    # Load training_args.json to check for pretrained model
    training_args = {}
    training_args_path = os.path.join(checkpoint_dir, "training_args.json")
    if os.path.exists(training_args_path):
        with open(training_args_path, "r") as f:
            training_args = json.load(f)
        log("Training args loaded", path=training_args_path)

    log("Model configuration",
        config_source=config_source,
        parameters=f"~{config.estimate_parameters()['total_M']:.1f}M",
        hidden_dim=config.hidden_dim,
        recursions=config.num_recursions,
        max_supervision_steps=config.max_supervision_steps)

    model = TRMLLM(config)

    # If trained with pretrained embeddings, we need to set up the same structure
    # before loading state dict
    pretrained_model = training_args.get("pretrained_model")
    if pretrained_model:
        log("Setting up pretrained embedding structure", model=pretrained_model)
        # Initialize tokenizer to get vocab size
        tokenizer_base = training_args.get("tokenizer_base_model", pretrained_model)
        tokenizer = ToolCallTokenizer(base_model=tokenizer_base)
        # Load pretrained embeddings to create the right model structure
        # freeze=False since we'll load the trained weights anyway
        model.load_pretrained_embeddings(
            pretrained_model,
            freeze=False,
            device=device,
            tokenizer_vocab_size=tokenizer.vocab_size,
        )

    # Now load the trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    log("Model weights loaded")

    metrics = checkpoint.get("metrics", {})
    if metrics:
        metrics_info = {k: f"{v:.4f}" for k, v in metrics.items() if "accuracy" in k or "loss" in k}
        if metrics_info:
            log("Checkpoint metrics", **metrics_info)

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

    # Load training args to get tokenizer info
    checkpoint_dir = os.path.dirname(args.checkpoint)
    training_args_path = os.path.join(checkpoint_dir, "training_args.json")
    training_args = {}
    if os.path.exists(training_args_path):
        with open(training_args_path, "r") as f:
            training_args = json.load(f)

    # Initialize tokenizer with same base model as training
    tokenizer_base = training_args.get("tokenizer_base_model", "gpt2")
    tokenizer = ToolCallTokenizer(base_model=tokenizer_base)
    log("Tokenizer initialized", base_model=tokenizer_base, vocab_size=tokenizer.vocab_size)

    # Load model (will use training_args to set up pretrained structure)
    model, config = load_model(args.checkpoint, args.device, args.config)
    config.vocab_size = tokenizer.vocab_size

    # Load tool mapping
    if args.tool_mapping:
        tool_mapping_path = args.tool_mapping
    else:
        # Try to find in same directory as checkpoint
        checkpoint_dir = os.path.dirname(args.checkpoint)
        tool_mapping_path = os.path.join(checkpoint_dir, "tool_mapping.json")

    if os.path.exists(tool_mapping_path):
        with open(tool_mapping_path, "r") as f:
            tool_name_to_id = json.load(f)
        log("Tool mapping loaded", path=tool_mapping_path, num_tools=len(tool_name_to_id))
    else:
        log_warning("Tool mapping not found", path=tool_mapping_path, note="Using empty mapping")
        tool_name_to_id = {}

    # Initialize inference
    inference = TRMInference(model, tokenizer, config, device=args.device)
    inference.set_tool_mapping(tool_name_to_id)

    # Load tools
    if args.tools:
        with open(args.tools, "r") as f:
            tools = json.load(f)
        tools_json = json.dumps(tools)
        log("Tools loaded from file", path=args.tools, num_tools=len(tools))
    else:
        tools = default_tools()
        tools_json = json.dumps(tools)
        tool_names = [t['name'] for t in tools]
        log("Using default tools", tools=tool_names)

    # Interactive mode
    if args.interactive:
        log("TRM-LLM Interactive Mode", note="Enter queries (type 'quit' to exit, 'tools' to show tools)")

        while True:
            query = input("\nUser: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                log("Goodbye!")
                break

            if query.lower() == "tools":
                log("\nAvailable tools:")
                for tool in tools:
                    log(f"  - {tool['name']}: {tool['description']}")
                continue

            if not query:
                continue

            # Generate
            if args.analyze:
                analysis = inference.analyze_refinement(query, tools_json)
                log("\n--- Refinement Analysis ---")
                for step in analysis["steps"]:
                    log(
                        f"Step {step['step']}: {step['action']} "
                        f"(conf: {step['action_confidence']:.3f}, halt: {step['halt_prob']:.3f})"
                    )
                    if step["action"] == "tool_call":
                        log(
                            f"  → Tool: {step['tool_name']} (conf: {step['tool_confidence']:.3f})"
                        )
                log()

            result = inference.generate(query, tools_json)

            log(f"\nAssistant:")
            log(f"  Action: {result['action']}")
            if result["action"] == "tool_call":
                log(f"  Tool: {result['tool_name']}")
                log(f"  Parallel calls: {result.get('num_parallel_calls', 1)}")
                if result.get("tool_call"):
                    log(f"  Tool Call: {json.dumps(result['tool_call'], indent=4)}")
            elif result["action"] == "direct_answer":
                if result.get("response"):
                    log(f"  Response: {result['response']}")
            log(f"  Confidence: {result['confidence']:.3f}")
            log(f"  Steps used: {result['num_steps']}")

    # Single query mode
    elif args.query:
        log("Processing query", query=args.query)

        if args.analyze:
            analysis = inference.analyze_refinement(args.query, tools_json)
            for step in analysis["steps"]:
                step_info = {
                    "step": step['step'],
                    "action": step['action'],
                    "action_confidence": f"{step['action_confidence']:.3f}",
                    "halt_prob": f"{step['halt_prob']:.3f}",
                }
                if step["action"] == "tool_call":
                    step_info["tool_name"] = step['tool_name']
                    step_info["tool_confidence"] = f"{step['tool_confidence']:.3f}"
                log("Refinement step", **step_info)

        result = inference.generate(args.query, tools_json)

        result_info = {
            "action": result['action'],
            "confidence": f"{result['confidence']:.3f}",
            "steps_used": result['num_steps'],
        }
        if result["action"] == "tool_call":
            result_info["tool_name"] = result['tool_name']
            result_info["tool_id"] = result['tool_id']
            result_info["parallel_calls"] = result.get('num_parallel_calls', 1)
            if result.get("tool_call"):
                result_info["tool_call"] = json.dumps(result['tool_call'])
        elif result["action"] == "direct_answer":
            if result.get("response"):
                result_info["response"] = result['response']

        log("Inference result", **result_info)

    else:
        log_error("Either --query or --interactive must be specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
