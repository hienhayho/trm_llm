"""SentencePiece Tokenizer for tool-calling conversations

Trains a SentencePiece model from the dataset or loads a pre-trained model.
Uses Hermes chat template format for tool-calling.
"""

import json
import os
import tempfile
from typing import Dict, List, Optional

import sentencepiece as spm


class SentencePieceTokenizer:
    """SentencePiece-based tokenizer for tool-calling conversations

    Can either:
    1. Train a new SentencePiece model from JSONL data
    2. Load a pre-trained SentencePiece model

    Uses Hermes-style special tokens:
    - <|im_start|>: Start of message
    - <|im_end|>: End of message
    - <tools>: Start of tools definition
    - </tools>: End of tools definition
    - <tool_call>: Start of tool call
    - </tool_call>: End of tool call
    - <tool_response>: Start of tool response
    - </tool_response>: End of tool response
    - <pad>: Padding token
    - <unk>: Unknown token
    - <bos>: Beginning of sequence
    - <eos>: End of sequence
    - user: Role token for user messages
    - assistant: Role token for assistant messages
    - system: Role token for system messages
    """

    # Special tokens (must be defined before training)
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    TOOLS_START = "<tools>"
    TOOLS_END = "</tools>"
    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TOOL_RESPONSE_START = "<tool_response>"
    TOOL_RESPONSE_END = "</tool_response>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    # Role tokens
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"

    # All user-defined special tokens (order matters for IDs)
    USER_DEFINED_SYMBOLS = [
        "<pad>",           # ID 0
        "<unk>",           # ID 1
        "<bos>",           # ID 2
        "<eos>",           # ID 3
        "<|im_start|>",    # ID 4
        "<|im_end|>",      # ID 5
        "<tools>",         # ID 6
        "</tools>",        # ID 7
        "<tool_call>",     # ID 8
        "</tool_call>",    # ID 9
        "<tool_response>", # ID 10
        "</tool_response>",# ID 11
        "user",            # ID 12 - role token
        "assistant",       # ID 13 - role token
        "system",          # ID 14 - role token
    ]

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

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 8000,
    ):
        """
        Args:
            model_path: Path to pre-trained SentencePiece model (.model file).
                       If None, must call train() before using.
            vocab_size: Vocabulary size for training (ignored if loading model)
        """
        self.model_path = model_path
        self.target_vocab_size = vocab_size
        self.sp_model: Optional[spm.SentencePieceProcessor] = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(
        self,
        data_paths: List[str],
        output_dir: str,
        model_prefix: str = "sp_tokenizer",
        vocab_size: Optional[int] = None,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
    ) -> str:
        """Train SentencePiece model from JSONL data files

        Args:
            data_paths: List of paths to JSONL files
            output_dir: Directory to save the model
            model_prefix: Prefix for output model files
            vocab_size: Vocabulary size (default: use self.target_vocab_size)
            character_coverage: Character coverage for training
            model_type: Model type: "bpe", "unigram", "char", or "word"

        Returns:
            model_path: Path to the trained model file
        """
        from trm_llm.utils.logger import log

        vocab_size = vocab_size or self.target_vocab_size
        os.makedirs(output_dir, exist_ok=True)

        # Extract all text from JSONL files
        log("Extracting text from dataset for SentencePiece training...")
        texts = []
        for data_path in data_paths:
            texts.extend(self._extract_texts_from_jsonl(data_path))

        log(f"Extracted {len(texts)} text segments from {len(data_paths)} file(s)")

        # Write texts to temporary file for SentencePiece training
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name
            for text in texts:
                # Write each text on a line (SentencePiece expects one sentence per line)
                f.write(text.replace("\n", " ") + "\n")

        try:
            # Train SentencePiece model
            model_prefix_path = os.path.join(output_dir, model_prefix)

            # Build user_defined_symbols string
            user_defined_symbols = ",".join(self.USER_DEFINED_SYMBOLS)

            log(f"Training SentencePiece model...",
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage)

            spm.SentencePieceTrainer.train(
                input=temp_path,
                model_prefix=model_prefix_path,
                vocab_size=vocab_size,
                character_coverage=character_coverage,
                model_type=model_type,
                # Special tokens configuration
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece=self.PAD_TOKEN,
                unk_piece=self.UNK_TOKEN,
                bos_piece=self.BOS_TOKEN,
                eos_piece=self.EOS_TOKEN,
                # User-defined symbols (will be assigned IDs starting from 4)
                user_defined_symbols=user_defined_symbols[len("<pad>,<unk>,<bos>,<eos>,"):],  # Skip first 4
                # Training options
                normalization_rule_name="identity",  # Don't normalize (preserve case, etc.)
                remove_extra_whitespaces=False,
                split_by_whitespace=True,
                byte_fallback=True,  # Handle unknown characters via byte encoding
            )

            model_path = f"{model_prefix_path}.model"
            log(f"SentencePiece model trained", path=model_path)

            # Load the trained model
            self.load(model_path)

            return model_path

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    def _extract_texts_from_jsonl(self, jsonl_path: str) -> List[str]:
        """Extract all text content from a JSONL file for training

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            texts: List of text strings
        """
        texts = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                example = json.loads(line)

                # Extract tools JSON
                if "tools" in example:
                    texts.append(example["tools"])

                # Extract all message contents
                if "messages" in example:
                    for msg in example["messages"]:
                        content = msg.get("content", "")
                        if content:
                            texts.append(content)

        return texts

    def load(self, model_path: str):
        """Load a pre-trained SentencePiece model

        Args:
            model_path: Path to .model file
        """
        from trm_llm.utils.logger import log

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.model_path = model_path

        log(f"SentencePiece model loaded",
            path=model_path,
            vocab_size=self.vocab_size)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.sp_model is None:
            return self.target_vocab_size
        return self.sp_model.get_piece_size()

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID"""
        if self.sp_model is None:
            return 0
        return self.sp_model.pad_id()

    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID"""
        if self.sp_model is None:
            return 1
        return self.sp_model.unk_id()

    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID"""
        if self.sp_model is None:
            return 2
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID"""
        if self.sp_model is None:
            return 3
        return self.sp_model.eos_id()

    @property
    def pad_token(self) -> str:
        """Get padding token"""
        return self.PAD_TOKEN

    def _format_tools(self, tools_json: str) -> str:
        """Format tools JSON for system prompt in Hermes format"""
        try:
            tools = json.loads(tools_json)
            formatted_tools = []

            for tool in tools:
                if isinstance(tool, str):
                    tool = json.loads(tool)

                if tool.get("type") == "function" and "function" in tool:
                    formatted_tools.append(json.dumps(tool))
                else:
                    hermes_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {})
                        }
                    }
                    formatted_tools.append(json.dumps(hermes_tool))

            return "\n".join(formatted_tools)
        except json.JSONDecodeError:
            return tools_json

    def _format_message(self, role: str, content: str) -> str:
        """Format a single message in Hermes style"""
        return f"{self.IM_START}{role}\n{content}{self.IM_END}\n"

    def _format_tool_call(self, tool_call_content: str) -> str:
        """Format tool call content with tags"""
        return f"{self.TOOL_CALL_START}\n{tool_call_content}\n{self.TOOL_CALL_END}"

    def _format_tool_response(self, response_content: str) -> str:
        """Format tool response content with tags"""
        return f"{self.TOOL_RESPONSE_START}\n{response_content}\n{self.TOOL_RESPONSE_END}"

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs

        Args:
            text: Text to encode
            add_bos: Add beginning of sequence token
            add_eos: Add end of sequence token

        Returns:
            token_ids: List of token IDs
        """
        if self.sp_model is None:
            raise RuntimeError("SentencePiece model not loaded. Call train() or load() first.")

        ids = self.sp_model.encode(text)

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            text: Decoded text
        """
        if self.sp_model is None:
            raise RuntimeError("SentencePiece model not loaded. Call train() or load() first.")

        if skip_special_tokens:
            # Filter out special token IDs (0-11 are special tokens)
            token_ids = [tid for tid in token_ids if tid >= len(self.USER_DEFINED_SYMBOLS)]

        return self.sp_model.decode(token_ids)

    def decode_batch(
        self,
        token_ids_batch: List[List[int]],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        """Decode batch of token IDs

        Args:
            token_ids_batch: List of lists of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            texts: List of decoded texts
        """
        return [self.decode(ids, skip_special_tokens) for ids in token_ids_batch]

    def encode_conversation(
        self,
        tools_json: str,
        messages: List[Dict[str, str]],
        truncation: bool = True,
        max_length: int = 2048,
    ) -> List[int]:
        """Encode a tool-calling conversation in Hermes format

        Args:
            tools_json: JSON string with tool definitions
            messages: List of message dicts with 'role' and 'content'
            truncation: Whether to truncate to max_length
            max_length: Maximum sequence length

        Returns:
            token_ids: List of token IDs
        """
        text_parts = []

        # Check if there's a system message in the conversation
        has_system = any(msg['role'] == 'system' for msg in messages)

        if has_system:
            for msg in messages:
                if msg['role'] == 'system':
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
            system_content = self.SYSTEM_PROMPT_TEMPLATE.format(tools=self._format_tools(tools_json))
            text_parts.append(self._format_message('system', system_content))

        # Add other messages
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                continue
            elif role == 'user':
                text_parts.append(self._format_message('user', content))
            elif role == 'tool_call':
                formatted_call = self._format_tool_call(content)
                text_parts.append(self._format_message('assistant', formatted_call))
            elif role == 'tool_response':
                formatted_response = self._format_tool_response(content)
                text_parts.append(self._format_message('user', formatted_response))
            elif role == 'assistant':
                text_parts.append(self._format_message('assistant', content))

        text = ''.join(text_parts)
        token_ids = self.encode(text, add_bos=True)  # Add BOS for input sequences

        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def encode_tool_call(
        self,
        tool_call_json: str,
        truncation: bool = True,
        max_length: int = 512,
    ) -> List[int]:
        """Encode tool call JSON with Hermes format tags

        Args:
            tool_call_json: Tool call JSON string
            truncation: Whether to truncate
            max_length: Maximum length

        Returns:
            token_ids: List of token IDs
        """
        text = self._format_tool_call(tool_call_json)
        token_ids = self.encode(text, add_eos=True)  # Add EOS for generation targets

        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def encode_text(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 2048,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode plain text

        Args:
            text: Text to encode
            truncation: Whether to truncate
            max_length: Maximum length
            add_eos: Add EOS token at end (default True for generation targets)

        Returns:
            token_ids: List of token IDs
        """
        token_ids = self.encode(text, add_eos=add_eos)

        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def get_tool_name_from_call(self, tool_call_content: str) -> Optional[str]:
        """Extract tool name from tool call content"""
        try:
            tool_call = json.loads(tool_call_content)
            return tool_call.get('name')
        except json.JSONDecodeError:
            return None

    def get_tools_list(self, tools_json: str) -> List[Dict]:
        """Parse tools JSON string to list"""
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

    def convert_tokens_to_ids(self, token: str) -> int:
        """Convert a token string to its ID

        Args:
            token: Token string

        Returns:
            token_id: Token ID
        """
        if self.sp_model is None:
            raise RuntimeError("SentencePiece model not loaded.")

        return self.sp_model.piece_to_id(token)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        """Convert a token ID to its string

        Args:
            token_id: Token ID

        Returns:
            token: Token string
        """
        if self.sp_model is None:
            raise RuntimeError("SentencePiece model not loaded.")

        return self.sp_model.id_to_piece(token_id)

    # Legacy token names for compatibility
    @property
    def system_token(self):
        return f"{self.IM_START}system\n"

    @property
    def tools_token(self):
        return self.TOOLS_START

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

    # For compatibility with ToolCallTokenizer interface
    @property
    def tokenizer(self):
        """Return self for compatibility with code expecting .tokenizer attribute"""
        return self

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as dict mapping tokens to IDs"""
        if self.sp_model is None:
            return {}

        vocab = {}
        for i in range(self.vocab_size):
            vocab[self.sp_model.id_to_piece(i)] = i
        return vocab
