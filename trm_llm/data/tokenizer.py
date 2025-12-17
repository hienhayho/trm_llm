"""Tokenizer for tool-calling conversations

Uses Hermes chat template format for tool-calling, which is compatible with
models like Qwen, Mistral, and other models trained with this format.

Hermes format:
<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "func", "description": "...", "parameters": {...}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "func", "arguments": {...}}
</tool_call>
<|im_end|>
<|im_start|>user
User message<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "func", "arguments": {...}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"result": "..."}
</tool_response><|im_end|>
<|im_start|>assistant
Final answer<|im_end|>
"""

import json
from typing import Dict, List, Optional
from transformers import AutoTokenizer


class ToolCallTokenizer:
    """Tokenizer wrapper using Hermes chat template for tool-calling

    Uses Hermes-style special tokens:
    - <|im_start|>: Start of message
    - <|im_end|>: End of message
    - <tools>: Start of tools definition
    - </tools>: End of tools definition
    - <tool_call>: Start of tool call
    - </tool_call>: End of tool call
    - <tool_response>: Start of tool response
    - </tool_response>: End of tool response
    """

    # Hermes special tokens
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    TOOLS_START = "<tools>"
    TOOLS_END = "</tools>"
    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TOOL_RESPONSE_START = "<tool_response>"
    TOOL_RESPONSE_END = "</tool_response>"

    # System prompt template for tool-calling (Hermes format)
    SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>"""

    def __init__(self, base_model: str = 'gpt2', trust_remote_code: bool = True):
        """
        Args:
            base_model: HuggingFace model name for tokenizer (e.g., 'gpt2', 'Qwen/Qwen2.5-0.5B')
            trust_remote_code: Whether to trust remote code (needed for models like Qwen)
        """
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code
        )

        # Collect tokens that need to be added
        tokens_to_add = []
        for token in [
            self.IM_START,
            self.IM_END,
            self.TOOLS_START,
            self.TOOLS_END,
            self.TOOL_CALL_START,
            self.TOOL_CALL_END,
            self.TOOL_RESPONSE_START,
            self.TOOL_RESPONSE_END,
        ]:
            if token not in self.tokenizer.get_vocab():
                tokens_to_add.append(token)

        # Add special tokens if not present
        if tokens_to_add:
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': tokens_to_add
            })

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            # Use eos_token as pad_token if available, otherwise add one
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id

        # Store token IDs for easy access
        self.im_start_id = self.tokenizer.convert_tokens_to_ids(self.IM_START)
        self.im_end_id = self.tokenizer.convert_tokens_to_ids(self.IM_END)

    def _format_tools(self, tools_json: str) -> str:
        """Format tools JSON for system prompt in Hermes format

        Converts tools to Hermes format:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

        Args:
            tools_json: JSON string of tools (can be simple format or OpenAI format)

        Returns:
            Formatted tools string with one tool per line
        """
        try:
            tools = json.loads(tools_json)
            formatted_tools = []

            for tool in tools:
                if isinstance(tool, str):
                    tool = json.loads(tool)

                # Check if already in OpenAI/Hermes format
                if tool.get("type") == "function" and "function" in tool:
                    # Already in correct format
                    formatted_tools.append(json.dumps(tool))
                else:
                    # Convert simple format to Hermes format
                    hermes_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {})
                        }
                    }
                    formatted_tools.append(json.dumps(hermes_tool))

            # One tool per line
            return "\n".join(formatted_tools)
        except json.JSONDecodeError:
            return tools_json

    def _format_message(self, role: str, content: str) -> str:
        """Format a single message in Hermes style

        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content

        Returns:
            Formatted message string
        """
        return f"{self.IM_START}{role}\n{content}{self.IM_END}\n"

    def _format_tool_call(self, tool_call_content: str) -> str:
        """Format tool call content with tags

        Args:
            tool_call_content: Tool call JSON string

        Returns:
            Formatted tool call string
        """
        return f"{self.TOOL_CALL_START}\n{tool_call_content}\n{self.TOOL_CALL_END}"

    def _format_tool_response(self, response_content: str) -> str:
        """Format tool response content with tags

        Args:
            response_content: Tool response string

        Returns:
            Formatted tool response string
        """
        return f"{self.TOOL_RESPONSE_START}\n{response_content}\n{self.TOOL_RESPONSE_END}"

    def encode_conversation(
        self,
        tools_json: str,
        messages: List[Dict[str, str]],
        truncation: bool = True,
        max_length: int = 2048
    ) -> List[int]:
        """Encode a tool-calling conversation in Hermes format

        Args:
            tools_json: JSON string with tool definitions
            messages: List of message dicts with 'role' and 'content'
            truncation: whether to truncate to max_length
            max_length: maximum sequence length

        Returns:
            token_ids: List of token IDs
        """
        text_parts = []

        # Check if there's a system message in the conversation
        has_system = any(msg['role'] == 'system' for msg in messages)

        # Add system message with tools in Hermes format
        if has_system:
            # Find the system message and append tools section
            for msg in messages:
                if msg['role'] == 'system':
                    # Append tools section to existing system message
                    tools_section = f"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{self._format_tools(tools_json)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>"""
                    system_content = msg['content'] + tools_section
                    text_parts.append(self._format_message('system', system_content))
                    break
        else:
            # Create system message with tools using template
            system_content = self.SYSTEM_PROMPT_TEMPLATE.format(tools=self._format_tools(tools_json))
            text_parts.append(self._format_message('system', system_content))

        # Add other messages
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                # Already handled above
                continue
            elif role == 'user':
                text_parts.append(self._format_message('user', content))
            elif role == 'tool_call':
                # Tool call is an assistant message with tool_call tags
                formatted_call = self._format_tool_call(content)
                text_parts.append(self._format_message('assistant', formatted_call))
            elif role == 'tool_response':
                # Tool response is a user message with tool_response tags (Hermes format)
                formatted_response = self._format_tool_response(content)
                text_parts.append(self._format_message('user', formatted_response))
            elif role == 'assistant':
                text_parts.append(self._format_message('assistant', content))

        # Join all parts
        text = ''.join(text_parts)

        # Tokenize
        token_ids = self.tokenizer.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False  # We already added our special tokens
        )

        return token_ids

    def encode_tool_call(self, tool_call_json: str, truncation: bool = True, max_length: int = 512) -> List[int]:
        """Encode tool call JSON with Hermes format tags

        Args:
            tool_call_json: Tool call JSON string (single object or array)
            truncation: whether to truncate
            max_length: maximum length

        Returns:
            token_ids: List of token IDs
        """
        # Wrap in tool_call tags
        text = self._format_tool_call(tool_call_json)
        return self.tokenizer.encode(text, truncation=truncation, max_length=max_length)

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

    # Legacy token names for compatibility (map to Hermes equivalents)
    @property
    def system_token(self):
        return f"{self.IM_START}system\n"

    @property
    def tools_token(self):
        return self.TOOLS_START  # <tools> tag

    @property
    def user_token(self):
        return f"{self.IM_START}user\n"

    @property
    def assistant_token(self):
        return f"{self.IM_START}assistant\n"

    @property
    def tool_call_token(self):
        return self.TOOL_CALL_START

    @property
    def tool_response_token(self):
        return self.TOOL_RESPONSE_START

    @property
    def pad_token(self):
        return self.tokenizer.pad_token
