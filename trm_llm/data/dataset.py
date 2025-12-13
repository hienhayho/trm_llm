"""Dataset for tool-calling conversations

Loads JSONL file and creates supervision targets for TRM training.
Splits each conversation into multiple decision-point samples.
"""

import json
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from .tokenizer import ToolCallTokenizer


class ToolCallDataset(Dataset):
    """Dataset for tool-calling conversations with multi-sample splitting

    Each conversation is split into decision-point samples:
    1. User message → tool_call decision (how many parallel calls, which tool, params)
    2. After tool responses → direct_answer decision

    Each example in JSONL has format:
    {
        "tools": "[{\"name\": \"func\", \"params\": {...}}]",
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "tool_call", "content": "{\"name\": \"...\", \"arguments\": {...}}"},
            {"role": "tool_response", "content": "{\"result\": ...}"},
            {"role": "assistant", "content": "..."}
        ]
    }
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: ToolCallTokenizer,
        max_length: int = 2048,
        max_param_len: int = 128,
        max_response_len: int = 256,
        compute_stats: bool = True
    ):
        """
        Args:
            jsonl_path: Path to JSONL file
            tokenizer: ToolCallTokenizer instance
            max_length: Maximum input sequence length
            max_param_len: Maximum parameter sequence length
            max_response_len: Maximum response sequence length
            compute_stats: Whether to compute dataset statistics (can be slow)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_param_len = max_param_len
        self.max_response_len = max_response_len

        # Load raw data
        self.raw_data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    self.raw_data.append(json.loads(line))

        # Build tool name to ID mapping (across all examples)
        self.tool_name_to_id = self._build_tool_mapping()

        # Split into decision-point samples
        self.samples = self._split_into_samples()

        # Statistics (can be slow for large datasets)
        self.stats = self._compute_stats() if compute_stats else None

    def _extract_tool_name(self, tool: dict) -> Optional[str]:
        """Extract tool name from tool dict, supporting multiple formats

        Supports:
        - Simple format: {"name": "calculator", ...}
        - OpenAI format: {"type": "function", "function": {"name": "calculator", ...}}
        """
        if not isinstance(tool, dict):
            return None

        if "name" in tool:
            return tool["name"]

        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            return tool["function"].get("name")

        return None

    def _build_tool_mapping(self) -> Dict[str, int]:
        """Build mapping from tool names to IDs"""
        tool_names = set()

        for example in self.raw_data:
            tools = self.tokenizer.get_tools_list(example["tools"])
            for tool in tools:
                if isinstance(tool, str):
                    tool = json.loads(tool)

                tool_name = self._extract_tool_name(tool)
                assert tool_name is not None, (
                    f"Could not extract tool name from: {tool}. "
                    f"Expected format: {{'name': '...'}} or "
                    f"{{'type': 'function', 'function': {{'name': '...'}}}}"
                )
                tool_names.add(tool_name)

        tool_names = sorted(tool_names)
        return {name: idx for idx, name in enumerate(tool_names)}

    def get_tool_id(self, tool_name: str) -> int:
        """Get tool ID from name"""
        return self.tool_name_to_id.get(tool_name, -1)

    def _split_into_samples(self) -> List[Dict]:
        """Split each conversation into decision-point samples

        For each conversation:
        - Sample 1: Input up to user message → predict tool_call action + num_calls + tool + params
        - Sample 2: Input including tool calls and responses → predict direct_answer

        Returns:
            samples: List of sample dicts
        """
        samples = []

        for raw_idx, example in enumerate(self.raw_data):
            tools_json = example["tools"]
            messages = example["messages"]

            # Find decision points in the conversation
            sample_list = self._extract_decision_points(tools_json, messages, raw_idx)
            samples.extend(sample_list)

        return samples

    def _extract_decision_points(
        self, tools_json: str, messages: List[Dict], raw_idx: int
    ) -> List[Dict]:
        """Extract decision point samples from a conversation

        Args:
            tools_json: Tool definitions JSON string
            messages: List of message dicts
            raw_idx: Index in raw data (for debugging)

        Returns:
            samples: List of sample dicts for this conversation
        """
        samples = []
        current_context = []  # Messages up to decision point

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "system":
                # Add system message to context (system prompts)
                current_context.append(msg)
                i += 1

            elif msg["role"] == "user":
                # Add user message to context
                current_context.append(msg)
                i += 1

                # Look ahead to see what comes next
                if i < len(messages):
                    next_msg = messages[i]

                    if next_msg["role"] == "tool_call":
                        # Count consecutive tool_calls (parallel calls)
                        tool_calls = []
                        while i < len(messages) and messages[i]["role"] == "tool_call":
                            tool_calls.append(messages[i])
                            i += 1

                        # Create tool_call sample
                        # Input: context up to (not including) tool_calls
                        # Target: tool_call action, first tool, num parallel calls
                        first_tool_call = tool_calls[0]
                        tool_name = self.tokenizer.get_tool_name_from_call(first_tool_call["content"])

                        sample = {
                            "tools_json": tools_json,
                            "context_messages": list(current_context),
                            "target_action": 1,  # tool_call
                            "target_tool_id": self.get_tool_id(tool_name) if tool_name else -1,
                            "target_num_calls": len(tool_calls),  # Number of parallel calls
                            "target_param_content": first_tool_call["content"],
                            "target_response_content": None,  # No response for tool_call
                            "raw_idx": raw_idx,
                        }
                        samples.append(sample)

                        # Add tool_calls and tool_responses to context
                        current_context.extend(tool_calls)

                        # Consume tool_responses
                        while i < len(messages) and messages[i]["role"] == "tool_response":
                            current_context.append(messages[i])
                            i += 1

                    elif next_msg["role"] == "assistant":
                        # Direct answer after user message (no tool call needed)
                        sample = {
                            "tools_json": tools_json,
                            "context_messages": list(current_context),
                            "target_action": 0,  # direct_answer
                            "target_tool_id": -1,
                            "target_num_calls": 0,
                            "target_param_content": None,
                            "target_response_content": next_msg["content"],  # Target response text
                            "raw_idx": raw_idx,
                        }
                        samples.append(sample)

                        # Add assistant message and move on
                        current_context.append(next_msg)
                        i += 1

            elif msg["role"] == "tool_response":
                # After tool responses, look for what comes next
                current_context.append(msg)
                i += 1

            elif msg["role"] == "assistant":
                # After tool responses, we have assistant response
                # This is a direct_answer decision point
                # Input: context including tool responses
                # Target: direct_answer

                # Only create sample if we have tool_responses in context
                has_tool_response = any(m["role"] == "tool_response" for m in current_context)
                if has_tool_response:
                    sample = {
                        "tools_json": tools_json,
                        "context_messages": list(current_context),
                        "target_action": 0,  # direct_answer
                        "target_tool_id": -1,
                        "target_num_calls": 0,
                        "target_param_content": None,
                        "target_response_content": msg["content"],  # Target response text
                        "raw_idx": raw_idx,
                    }
                    samples.append(sample)

                current_context.append(msg)
                i += 1

            else:
                # Unknown role, skip
                current_context.append(msg)
                i += 1

        return samples

    def _compute_stats(self) -> Dict:
        """Compute dataset statistics including token lengths"""
        from tqdm import tqdm

        tool_call_samples = sum(1 for s in self.samples if s["target_action"] == 1)
        direct_answer_samples = sum(1 for s in self.samples if s["target_action"] == 0)

        # Count by num_calls
        num_calls_dist = {}
        for s in self.samples:
            if s["target_action"] == 1:
                n = s["target_num_calls"]
                num_calls_dist[n] = num_calls_dist.get(n, 0) + 1

        # Compute character-level length stats (fast)
        context_char_lengths = []
        param_char_lengths = []
        response_char_lengths = []

        for s in self.samples:
            # Context length (sum of all message contents)
            ctx_len = sum(len(m.get("content", "")) for m in s["context_messages"])
            ctx_len += len(s["tools_json"])  # Add tools
            context_char_lengths.append(ctx_len)

            # Param length (for tool_call)
            if s.get("target_param_content"):
                param_char_lengths.append(len(s["target_param_content"]))

            # Response length (for direct_answer)
            if s.get("target_response_content"):
                response_char_lengths.append(len(s["target_response_content"]))

        # Compute token-level length stats (tokenize all samples)
        # We compute both original (untruncated) and truncated lengths
        input_token_lengths = []
        input_token_lengths_orig = []
        input_truncated_count = 0

        param_token_lengths = []
        param_token_lengths_orig = []
        param_truncated_count = 0

        response_token_lengths = []
        response_token_lengths_orig = []
        response_truncated_count = 0

        print("  Computing token statistics...")
        for idx in tqdm(range(len(self.samples)), desc="  Tokenizing samples", leave=False):
            sample = self.samples[idx]

            # Input tokens (with truncation)
            input_ids = self.tokenizer.encode_conversation(
                sample["tools_json"],
                sample["context_messages"],
                truncation=True,
                max_length=self.max_length
            )
            input_token_lengths.append(len(input_ids))

            # Input tokens (without truncation) - to detect truncation
            input_ids_orig = self.tokenizer.encode_conversation(
                sample["tools_json"],
                sample["context_messages"],
                truncation=False
            )
            input_token_lengths_orig.append(len(input_ids_orig))
            if len(input_ids_orig) > self.max_length:
                input_truncated_count += 1

            # Param tokens
            if sample.get("target_param_content"):
                param_ids = self.tokenizer.encode_text(
                    sample["target_param_content"],
                    truncation=True,
                    max_length=self.max_param_len
                )
                param_token_lengths.append(len(param_ids))

                param_ids_orig = self.tokenizer.encode_text(
                    sample["target_param_content"],
                    truncation=False
                )
                param_token_lengths_orig.append(len(param_ids_orig))
                if len(param_ids_orig) > self.max_param_len:
                    param_truncated_count += 1

            # Response tokens
            if sample.get("target_response_content"):
                response_ids = self.tokenizer.encode_text(
                    sample["target_response_content"],
                    truncation=True,
                    max_length=self.max_response_len
                )
                response_token_lengths.append(len(response_ids))

                response_ids_orig = self.tokenizer.encode_text(
                    sample["target_response_content"],
                    truncation=False
                )
                response_token_lengths_orig.append(len(response_ids_orig))
                if len(response_ids_orig) > self.max_response_len:
                    response_truncated_count += 1

        # Unique tools
        num_unique_tools = len(self.tool_name_to_id)

        return {
            "raw_conversations": len(self.raw_data),
            "total_samples": len(self.samples),
            "tool_call_samples": tool_call_samples,
            "direct_answer_samples": direct_answer_samples,
            "num_calls_distribution": num_calls_dist,
            "num_unique_tools": num_unique_tools,
            # Character-level stats
            "context_char_lengths": {
                "min": min(context_char_lengths) if context_char_lengths else 0,
                "max": max(context_char_lengths) if context_char_lengths else 0,
                "avg": sum(context_char_lengths) / len(context_char_lengths) if context_char_lengths else 0,
            },
            "param_char_lengths": {
                "min": min(param_char_lengths) if param_char_lengths else 0,
                "max": max(param_char_lengths) if param_char_lengths else 0,
                "avg": sum(param_char_lengths) / len(param_char_lengths) if param_char_lengths else 0,
                "count": len(param_char_lengths),
            },
            "response_char_lengths": {
                "min": min(response_char_lengths) if response_char_lengths else 0,
                "max": max(response_char_lengths) if response_char_lengths else 0,
                "avg": sum(response_char_lengths) / len(response_char_lengths) if response_char_lengths else 0,
                "count": len(response_char_lengths),
            },
            # Token-level stats (truncated - what's used in training)
            "input_token_lengths": {
                "min": min(input_token_lengths) if input_token_lengths else 0,
                "max": max(input_token_lengths) if input_token_lengths else 0,
                "avg": sum(input_token_lengths) / len(input_token_lengths) if input_token_lengths else 0,
                "total": len(input_token_lengths),
                "truncated_count": input_truncated_count,
                "max_length": self.max_length,
            },
            "param_token_lengths": {
                "min": min(param_token_lengths) if param_token_lengths else 0,
                "max": max(param_token_lengths) if param_token_lengths else 0,
                "avg": sum(param_token_lengths) / len(param_token_lengths) if param_token_lengths else 0,
                "count": len(param_token_lengths),
                "truncated_count": param_truncated_count,
                "max_length": self.max_param_len,
            },
            "response_token_lengths": {
                "min": min(response_token_lengths) if response_token_lengths else 0,
                "max": max(response_token_lengths) if response_token_lengths else 0,
                "avg": sum(response_token_lengths) / len(response_token_lengths) if response_token_lengths else 0,
                "count": len(response_token_lengths),
                "truncated_count": response_truncated_count,
                "max_length": self.max_response_len,
            },
            # Token-level stats (original - before truncation)
            "input_token_lengths_orig": {
                "min": min(input_token_lengths_orig) if input_token_lengths_orig else 0,
                "max": max(input_token_lengths_orig) if input_token_lengths_orig else 0,
                "avg": sum(input_token_lengths_orig) / len(input_token_lengths_orig) if input_token_lengths_orig else 0,
            },
            "param_token_lengths_orig": {
                "min": min(param_token_lengths_orig) if param_token_lengths_orig else 0,
                "max": max(param_token_lengths_orig) if param_token_lengths_orig else 0,
                "avg": sum(param_token_lengths_orig) / len(param_token_lengths_orig) if param_token_lengths_orig else 0,
            },
            "response_token_lengths_orig": {
                "min": min(response_token_lengths_orig) if response_token_lengths_orig else 0,
                "max": max(response_token_lengths_orig) if response_token_lengths_orig else 0,
                "avg": sum(response_token_lengths_orig) / len(response_token_lengths_orig) if response_token_lengths_orig else 0,
            },
        }

    def compute_token_stats(self, sample_size: int = 100) -> Dict:
        """Compute token-level statistics by sampling

        This is more expensive than character stats, so we sample.

        Args:
            sample_size: Number of samples to use for stats

        Returns:
            stats: Dict with token length statistics
        """
        import random

        # Sample indices
        indices = list(range(len(self.samples)))
        if len(indices) > sample_size:
            indices = random.sample(indices, sample_size)

        input_lengths = []
        param_lengths = []
        response_lengths = []

        for idx in indices:
            item = self[idx]
            input_lengths.append(len(item["input_ids"]))
            if item.get("target_param_ids"):
                param_lengths.append(len(item["target_param_ids"]))
            if item.get("target_response_ids"):
                response_lengths.append(len(item["target_response_ids"]))

        return {
            "sample_size": len(indices),
            "input_token_lengths": {
                "min": min(input_lengths) if input_lengths else 0,
                "max": max(input_lengths) if input_lengths else 0,
                "avg": sum(input_lengths) / len(input_lengths) if input_lengths else 0,
            },
            "param_token_lengths": {
                "min": min(param_lengths) if param_lengths else 0,
                "max": max(param_lengths) if param_lengths else 0,
                "avg": sum(param_lengths) / len(param_lengths) if param_lengths else 0,
                "count": len(param_lengths),
            },
            "response_token_lengths": {
                "min": min(response_lengths) if response_lengths else 0,
                "max": max(response_lengths) if response_lengths else 0,
                "avg": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
                "count": len(response_lengths),
            },
        }

    def __len__(self) -> int:
        """Get dataset size (number of decision-point samples)"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample

        Args:
            idx: Index

        Returns:
            sample: Dict with input_ids, target_action, target_tool_id, target_num_calls,
                    target_param_ids, target_response_ids
        """
        sample = self.samples[idx]

        # Tokenize context (tools + messages up to decision point)
        input_ids = self.tokenizer.encode_conversation(
            sample["tools_json"],
            sample["context_messages"],
            truncation=True,
            max_length=self.max_length
        )

        # Tokenize tool call parameters if present
        target_param_ids = []
        if sample.get("target_param_content"):
            target_param_ids = self.tokenizer.encode_text(
                sample["target_param_content"],
                truncation=True,
                max_length=self.max_param_len
            )

        # Tokenize response content if present (for direct_answer samples)
        target_response_ids = []
        if sample.get("target_response_content"):
            target_response_ids = self.tokenizer.encode_text(
                sample["target_response_content"],
                truncation=True,
                max_length=self.max_response_len
            )

        return {
            "input_ids": input_ids,
            "target_action": sample["target_action"],
            "target_tool_id": sample["target_tool_id"],
            "target_num_calls": sample["target_num_calls"],
            "target_param_ids": target_param_ids,
            "target_response_ids": target_response_ids,
        }


class ToolCallDatasetWithAugmentation(ToolCallDataset):
    """Dataset with data augmentation

    Future extension: add data augmentation strategies
    - Paraphrase user queries
    - Shuffle tool order
    - Add distractor tools
    """

    def __init__(self, jsonl_path: str, tokenizer: ToolCallTokenizer, max_length: int = 2048):
        super().__init__(jsonl_path, tokenizer, max_length)
        # TODO: Initialize augmentation strategies

    def augment_example(self, example: Dict) -> List[Dict]:
        """Augment a single example

        Args:
            example: Original example

        Returns:
            augmented_examples: List of augmented examples
        """
        # TODO: Implement augmentation
        # For now, just return original
        return [example]
