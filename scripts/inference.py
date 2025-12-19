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
from trm_llm.data.sp_tokenizer import SentencePieceTokenizer
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
        "--system", type=str, default=None, help="Path to text file with system prompt"
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
        "--sp_model",
        type=str,
        default=None,
        help="Path to SentencePiece model (default: same dir as checkpoint)",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        default=None,
        help="Path to special tokens file (default: from training_args.json)",
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

    log("Model configuration",
        config_source=config_source,
        parameters=f"~{config.estimate_parameters()['total_M']:.1f}M",
        hidden_dim=config.hidden_dim,
        recursions=config.num_recursions,
        max_supervision_steps=config.max_supervision_steps)

    model = TRMLLM(config)

    # Load the trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    log("Model weights loaded")

    metrics = checkpoint.get("metrics", {})
    if metrics:
        metrics_info = {k: f"{v:.4f}" for k, v in metrics.items() if "accuracy" in k or "loss" in k}
        if metrics_info:
            log("Checkpoint metrics", **metrics_info)

    return model, config


def parse_model_response(response: str) -> dict:
    """Parse model response and detect if it contains tool_call tags

    The model might generate <tool_call> tags even when action head predicts direct_answer.
    This function detects and extracts the appropriate content.

    Args:
        response: Raw model response

    Returns:
        dict with:
            - type: 'tool_call' or 'direct_answer'
            - content: Extracted content (tool call JSON or answer text)
            - tool_call: Parsed tool call dict if type is 'tool_call'
    """
    import re

    if not response:
        return {'type': 'direct_answer', 'content': '', 'tool_call': None}

    response = response.strip()

    # Method 1: Check if response contains <tool_call> tags with regex
    tool_call_pattern = r'<tool_call>\s*([\s\S]*?)\s*</tool_call>'
    match = re.search(tool_call_pattern, response)

    if match:
        tool_call_content = match.group(1).strip()
        # Try to find and parse JSON within the content
        try:
            # Find JSON object boundaries
            start = tool_call_content.find('{')
            end = tool_call_content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = tool_call_content[start:end]
                tool_call = json.loads(json_str)
                return {
                    'type': 'tool_call',
                    'content': json_str,
                    'tool_call': tool_call
                }
        except json.JSONDecodeError:
            pass

    # Method 2: Check if response starts with <tool_call> (incomplete tag)
    if '<tool_call>' in response:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                tool_call = json.loads(json_str)
                if 'name' in tool_call:
                    return {
                        'type': 'tool_call',
                        'content': json_str,
                        'tool_call': tool_call
                    }
        except json.JSONDecodeError:
            pass

    # Method 3: Check for raw JSON with "name" field (no tags)
    if '{"name"' in response or '{"name":' in response:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                tool_call = json.loads(json_str)
                if 'name' in tool_call:
                    return {
                        'type': 'tool_call',
                        'content': json_str,
                        'tool_call': tool_call
                    }
        except json.JSONDecodeError:
            pass

    # Clean up response for direct answer
    clean_response = response
    for tag in ['<tool_call>', '</tool_call>', '<|im_start|>', '<|im_end|>',
                '<bos>', '<eos>', '<pad>', 'assistant\n', 'assistant']:
        clean_response = clean_response.replace(tag, '')
    clean_response = clean_response.strip()

    return {'type': 'direct_answer', 'content': clean_response, 'tool_call': None}


def simulate_tool_response(tool_name: str, arguments: dict) -> str:
    """Simulate tool execution and return mock response

    In production, this would call actual APIs/functions.
    For demo purposes, returns simulated responses.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments dict

    Returns:
        JSON string with simulated response
    """
    # Default simulated responses based on tool name
    simulated_responses = {
        "get_product_price": {
            "price": "220000 VND/tháng (đã bao gồm VAT)",
            "note": "Giá áp dụng cho khu vực đã chọn"
        },
        "describe_product": {
            "info": "Sản phẩm sử dụng công nghệ tiên tiến, đảm bảo chất lượng cao và ổn định."
        },
        "request_agent": {
            "status": "success",
            "message": "Đã ghi nhận yêu cầu. Nhân viên sẽ liên hệ trong 15 phút."
        },
        "calculator": {
            "result": 1175,
            "expression": "25 * 47"
        },
        "realtime_aqi": {
            "city": arguments.get("city", "Unknown"),
            "aqi": 85,
            "pm25": 35,
            "pm10": 55,
            "status": "Moderate"
        },
    }

    if tool_name in simulated_responses:
        return json.dumps(simulated_responses[tool_name], ensure_ascii=False)
    else:
        # Generic response for unknown tools
        return json.dumps({
            "status": "success",
            "tool": tool_name,
            "arguments_received": arguments,
            "result": "Simulated response for " + tool_name
        }, ensure_ascii=False)


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

    checkpoint_dir = os.path.dirname(args.checkpoint)

    # Load training args to get tokenizer info
    training_args_path = os.path.join(checkpoint_dir, "training_args.json")
    training_args = {}
    if os.path.exists(training_args_path):
        with open(training_args_path, "r") as f:
            training_args = json.load(f)

    # Initialize SentencePiece tokenizer
    sp_model_path = args.sp_model
    if not sp_model_path:
        # Try to find in training_args or checkpoint dir
        sp_model_path = training_args.get("sp_model")
        if not sp_model_path or not os.path.exists(sp_model_path):
            sp_model_path = os.path.join(checkpoint_dir, "sp_tokenizer.model")

    if not os.path.exists(sp_model_path):
        log_error("SentencePiece model not found", path=sp_model_path)
        log_error("Please provide --sp_model or ensure sp_tokenizer.model exists in checkpoint dir")
        sys.exit(1)

    # Load special tokens file path
    special_tokens_file = args.special_tokens
    if not special_tokens_file:
        # Try to find in training_args
        special_tokens_file = training_args.get("special_tokens")

    tokenizer = SentencePieceTokenizer(
        model_path=sp_model_path,
        special_tokens_file=special_tokens_file,
    )
    log("Tokenizer initialized",
        model=sp_model_path,
        vocab_size=tokenizer.vocab_size,
        special_tokens=special_tokens_file or "default")

    # Load model
    model, config = load_model(args.checkpoint, args.device, args.config)
    config.vocab_size = tokenizer.vocab_size

    # Load tool mapping
    if args.tool_mapping:
        tool_mapping_path = args.tool_mapping
    else:
        # Try to find in same directory as checkpoint
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

    # Load system prompt
    system_prompt = None
    if args.system:
        with open(args.system, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        log("System prompt loaded", path=args.system, length=len(system_prompt))

    # Interactive mode with conversation history
    if args.interactive:
        log("TRM-LLM Interactive Mode (Multi-turn Conversation)")
        log("Commands: 'quit' to exit, 'tools' to show tools, 'clear' to reset, 'history' to show conversation")

        # Initialize conversation history
        conversation = []
        if system_prompt:
            conversation.append({'role': 'system', 'content': system_prompt})

        while True:
            query = input("\nUser: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                log("Goodbye!")
                break

            if query.lower() == "tools":
                log("\nAvailable tools:")
                for tool in tools:
                    if "function" in tool:
                        name = tool["function"]["name"]
                        desc = tool["function"].get("description", "")
                    else:
                        name = tool["name"]
                        desc = tool.get("description", "")
                    log(f"  - {name}: {desc}")
                continue

            if query.lower() == "clear":
                conversation = []
                if system_prompt:
                    conversation.append({'role': 'system', 'content': system_prompt})
                log("Conversation cleared.")
                continue

            if query.lower() == "history":
                log("\n--- Conversation History ---")
                for i, msg in enumerate(conversation):
                    role = msg['role']
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    log(f"  [{i}] {role}: {content}")
                log("---")
                continue

            if not query:
                continue

            # Add user message to conversation
            conversation.append({'role': 'user', 'content': query})

            # Generate with full conversation history
            result = inference.generate_with_history(conversation, tools_json)

            log(f"\nAssistant:")
            log(f"  Action: {result['action']}")
            log(f"  Confidence: {result['confidence']:.3f}")
            log(f"  Steps used: {result['num_steps']}")

            # Parse the response to detect actual type (generation may differ from action head)
            raw_response = result.get("response") or ""
            if result["action"] == "tool_call":
                raw_response = result.get("tool_call", {}).get("raw_output", "")

            parsed = parse_model_response(raw_response)

            # Use parsed type if generation contains tool_call tags
            actual_type = parsed['type'] if parsed['type'] == 'tool_call' else result['action']

            if actual_type == "tool_call":
                # Get tool call from parsed response or from result
                if parsed['tool_call']:
                    tool_call = parsed['tool_call']
                else:
                    tool_call = result.get("tool_call", {})

                tool_name = tool_call.get("name", result.get("tool_name", "unknown"))
                arguments = tool_call.get("arguments", {})

                log(f"  Tool: {tool_name}")
                log(f"  Arguments: {json.dumps(arguments, ensure_ascii=False)}")

                # Add tool_call to conversation
                tool_call_content = json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)
                conversation.append({'role': 'tool_call', 'content': tool_call_content})

                # Simulate tool response
                log("\n  [Simulating tool execution...]")
                tool_response_content = simulate_tool_response(tool_name, arguments)
                log(f"  Tool Response: {tool_response_content}")

                # Add tool_response to conversation
                conversation.append({'role': 'tool_response', 'content': tool_response_content})

                # Generate follow-up response after tool execution
                log("\n  [Generating follow-up response...]")
                followup = inference.generate_with_history(conversation, tools_json)

                # Get raw output based on action type
                # For tool_call action: generated text is in tool_call dict
                # For direct_answer action: generated text is in response
                if followup["action"] == "tool_call":
                    tool_call = followup.get("tool_call", {})
                    # If tool_call has valid name (not "unknown"), use it directly
                    if tool_call and tool_call.get("name") and tool_call.get("name") != "unknown":
                        followup_parsed = {
                            'type': 'tool_call',
                            'content': json.dumps(tool_call, ensure_ascii=False),
                            'tool_call': tool_call
                        }
                    else:
                        # Parse from raw output
                        followup_raw = tool_call.get("raw_output", "") if tool_call else ""
                        followup_parsed = parse_model_response(followup_raw)
                else:
                    # direct_answer - check if response contains tool_call tags
                    followup_raw = followup.get("response") or ""
                    followup_parsed = parse_model_response(followup_raw)

                if followup_parsed['type'] == 'tool_call' and followup_parsed['tool_call']:
                    # Model wants another tool call
                    next_tool = followup_parsed['tool_call']
                    next_tool_name = next_tool.get('name', 'unknown')
                    next_tool_args = next_tool.get('arguments', {})
                    log(f"  (Model wants another tool call: {next_tool_name})")
                    log(f"  Tool Arguments: {json.dumps(next_tool_args, ensure_ascii=False)}")

                    # Add tool_call to conversation
                    tool_call_content = json.dumps({"name": next_tool_name, "arguments": next_tool_args}, ensure_ascii=False)
                    conversation.append({'role': 'tool_call', 'content': tool_call_content})

                    # Simulate this tool response too
                    log("\n  [Simulating tool execution...]")
                    next_tool_response = simulate_tool_response(next_tool_name, next_tool_args)
                    log(f"  Tool Response: {next_tool_response}")
                    conversation.append({'role': 'tool_response', 'content': next_tool_response})

                    # Generate another follow-up
                    log("\n  [Generating response after second tool call...]")
                    final_followup = inference.generate_with_history(conversation, tools_json)
                    final_raw = final_followup.get("response") or ""
                    if final_followup["action"] == "tool_call":
                        final_tc = final_followup.get("tool_call", {})
                        final_raw = final_tc.get("raw_output", "") if final_tc else ""
                    final_parsed = parse_model_response(final_raw)

                    if final_parsed['content']:
                        log(f"  Final Response: {final_parsed['content']}")
                        conversation.append({'role': 'assistant', 'content': final_parsed['content']})
                    else:
                        log("  (Max tool calls reached - end of turn)")
                elif followup_parsed['content']:
                    log(f"  Final Response: {followup_parsed['content']}")
                    conversation.append({'role': 'assistant', 'content': followup_parsed['content']})
                else:
                    log("  (No follow-up response generated)")

            else:  # direct_answer
                response = parsed['content'] if parsed['content'] else result.get("response", "")
                if response:
                    log(f"  Response: {response}")
                    conversation.append({'role': 'assistant', 'content': response})
                else:
                    log("  (No response generated)")

    # Single query mode
    elif args.query:
        log("Processing query", query=args.query)

        if args.analyze:
            analysis = inference.analyze_refinement(args.query, tools_json, system_prompt=system_prompt)
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

        result = inference.generate(args.query, tools_json, system_prompt=system_prompt)

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
