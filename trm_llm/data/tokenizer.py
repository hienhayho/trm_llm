"""Tokenizer for tool-calling conversations

Wraps GPT-2 tokenizer and adds special tokens for tool-calling format
"""

import json
from typing import Dict, List, Optional
from transformers import GPT2Tokenizer


class ToolCallTokenizer:
    """Tokenizer wrapper for tool-calling conversations

    Adds special tokens:
    - <|system|>: System prompt
    - <|tools|>: Marker for tool definitions
    - <|user|>: User message
    - <|assistant|>: Assistant response
    - <|tool_call|>: Tool call
    - <|tool_response|>: Tool execution result
    - <|pad|>: Padding token
    """

    def __init__(self, base_model: str = 'gpt2'):
        """
        Args:
            base_model: HuggingFace model name for tokenizer
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)

        # Add special tokens
        special_tokens = {
            'additional_special_tokens': [
                '<|system|>',
                '<|tools|>',
                '<|user|>',
                '<|assistant|>',
                '<|tool_call|>',
                '<|tool_response|>',
            ],
            'pad_token': '<|pad|>'
        }

        self.tokenizer.add_special_tokens(special_tokens)

        # Store special token IDs for easy access
        self.system_token = '<|system|>'
        self.tools_token = '<|tools|>'
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'
        self.tool_call_token = '<|tool_call|>'
        self.tool_response_token = '<|tool_response|>'
        self.pad_token = '<|pad|>'

        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode_conversation(
        self,
        tools_json: str,
        messages: List[Dict[str, str]],
        truncation: bool = True,
        max_length: int = 2048
    ) -> List[int]:
        """Encode a tool-calling conversation

        Format:
        <|system|>SYSTEM_PROMPT<|tools|>TOOLS_JSON<|user|>USER_QUERY<|tool_call|>TOOL_CALL<|tool_response|>RESULT<|assistant|>RESPONSE

        Args:
            tools_json: JSON string with tool definitions
            messages: List of message dicts with 'role' and 'content'
            truncation: whether to truncate to max_length
            max_length: maximum sequence length

        Returns:
            token_ids: List of token IDs
        """
        # Start with tools
        text = f"{self.tools_token}{tools_json}"

        # Add messages in order
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                # System message goes at the beginning (before tools ideally, but after for simplicity)
                text = f"{self.system_token}{content}" + text
            elif role == 'user':
                text += f"{self.user_token}{content}"
            elif role == 'tool_call':
                text += f"{self.tool_call_token}{content}"
            elif role == 'tool_response':
                text += f"{self.tool_response_token}{content}"
            elif role == 'assistant':
                text += f"{self.assistant_token}{content}"

        # Tokenize
        token_ids = self.tokenizer.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False  # We already added our special tokens
        )

        return token_ids

    def encode_text(self, text: str, truncation: bool = True, max_length: int = 2048) -> List[int]:
        """Encode plain text

        Args:
            text: Text to encode
            truncation: whether to truncate
            max_length: maximum length

        Returns:
            token_ids: List of token IDs
        """
        return self.tokenizer.encode(text, truncation=truncation, max_length=max_length)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special_tokens: whether to skip special tokens

        Returns:
            text: Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, token_ids_batch: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        """Decode batch of token IDs

        Args:
            token_ids_batch: List of lists of token IDs
            skip_special_tokens: whether to skip special tokens

        Returns:
            texts: List of decoded texts
        """
        return self.tokenizer.batch_decode(token_ids_batch, skip_special_tokens=skip_special_tokens)

    def get_tool_name_from_call(self, tool_call_content: str) -> Optional[str]:
        """Extract tool name from tool call content

        Args:
            tool_call_content: JSON string like '{"name": "calculator", "arguments": {...}}'

        Returns:
            tool_name: Name of the tool, or None if parsing fails
        """
        try:
            tool_call = json.loads(tool_call_content)
            return tool_call.get('name')
        except json.JSONDecodeError:
            return None

    def get_tools_list(self, tools_json: str) -> List[Dict]:
        """Parse tools JSON string to list

        Args:
            tools_json: JSON string like '[{"name": "calc", ...}, ...]'

        Returns:
            tools: List of tool dictionaries
        """
        try:
            return json.loads(tools_json)
        except json.JSONDecodeError:
            return []

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size

    def __len__(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size

    def __call__(self, *args, **kwargs):
        """Make tokenizer callable (delegates to tokenizer.encode)"""
        return self.tokenizer(*args, **kwargs)
