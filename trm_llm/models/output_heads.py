"""Output Heads for TRM-LLM

Multiple specialized heads that decode the action state y into concrete decisions:
1. Action Head: Should I answer directly or call a tool?
2. Tool Selection Head: Which tool should I call?
3. Halt Head: Should I stop refining? (ACT - Adaptive Computation Time)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputHeads(nn.Module):
    """Multi-head output module for tool calling decisions

    Takes action state y and produces:
    - Action logits: [P(direct_answer), P(tool_call)]
    - Tool selection logits: [P(tool_1), P(tool_2), ..., P(tool_n)]
    - Num parallel calls logits: [P(1), P(2), ..., P(max_parallel)]
    - Halt logit: P(should_stop_refining)
    """

    def __init__(self, action_dim: int, num_action_types: int = 2, max_tools: int = 50,
                 max_parallel_calls: int = 5):
        """
        Args:
            action_dim: Dimension of action state y
            num_action_types: Number of action types (2: direct_answer, tool_call)
            max_tools: Maximum number of tools that can be in context
            max_parallel_calls: Maximum number of parallel tool calls (1-5)
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_action_types = num_action_types
        self.max_tools = max_tools
        self.max_parallel_calls = max_parallel_calls

        # 1. Action decision head: should I answer directly or call a tool?
        self.action_head = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim // 2, num_action_types)
        )

        # 2. Tool selection head: which tool to call?
        self.tool_head = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim // 2, max_tools)
        )

        # 3. Num parallel calls head: how many tools to call in parallel?
        # Predicts 1, 2, 3, ... max_parallel_calls
        self.num_calls_head = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim // 2, max_parallel_calls)
        )

        # 4. Halt decision head: should I stop refining? (for ACT)
        self.halt_head = nn.Linear(action_dim, 1)

    def forward(self, y_state):
        """Generate all outputs from action state

        Args:
            y_state: (batch_size, action_dim) - current action state

        Returns:
            dict with:
                - action_logits: (batch_size, num_action_types)
                - tool_logits: (batch_size, max_tools)
                - num_calls_logits: (batch_size, max_parallel_calls)
                - halt_logit: (batch_size, 1)
                - y_state: (batch_size, action_dim) - for downstream use
        """
        # Action decision (direct answer vs tool call)
        action_logits = self.action_head(y_state)  # (batch_size, 2)

        # Tool selection (which tool)
        tool_logits = self.tool_head(y_state)  # (batch_size, max_tools)

        # Number of parallel calls (1, 2, 3, ...)
        num_calls_logits = self.num_calls_head(y_state)  # (batch_size, max_parallel_calls)

        # Halting decision (should stop refining)
        halt_logit = self.halt_head(y_state)  # (batch_size, 1)

        return {
            'action_logits': action_logits,
            'tool_logits': tool_logits,
            'num_calls_logits': num_calls_logits,
            'halt_logit': halt_logit,
            'y_state': y_state  # Keep for potential parameter/response generation
        }

    def get_action_probs(self, y_state):
        """Get action probabilities

        Args:
            y_state: (batch_size, action_dim)

        Returns:
            action_probs: (batch_size, num_action_types)
        """
        logits = self.action_head(y_state)
        return F.softmax(logits, dim=-1)

    def get_tool_probs(self, y_state):
        """Get tool selection probabilities

        Args:
            y_state: (batch_size, action_dim)

        Returns:
            tool_probs: (batch_size, max_tools)
        """
        logits = self.tool_head(y_state)
        return F.softmax(logits, dim=-1)

    def get_halt_prob(self, y_state):
        """Get halting probability

        Args:
            y_state: (batch_size, action_dim)

        Returns:
            halt_prob: (batch_size, 1)
        """
        logit = self.halt_head(y_state)
        return torch.sigmoid(logit)

    def get_num_calls_probs(self, y_state):
        """Get num parallel calls probabilities

        Args:
            y_state: (batch_size, action_dim)

        Returns:
            num_calls_probs: (batch_size, max_parallel_calls)
        """
        logits = self.num_calls_head(y_state)
        return F.softmax(logits, dim=-1)

    def decode_action(self, y_state):
        """Decode action state into discrete decision

        Args:
            y_state: (batch_size, action_dim)

        Returns:
            dict with:
                - action: (batch_size,) - 0 for direct_answer, 1 for tool_call
                - tool_id: (batch_size,) - selected tool ID (-1 if direct_answer)
                - num_calls: (batch_size,) - number of parallel calls (0 if direct_answer)
                - confidence: (batch_size,) - confidence score
        """
        # Get probabilities
        action_probs = self.get_action_probs(y_state)  # (batch_size, 2)
        tool_probs = self.get_tool_probs(y_state)  # (batch_size, max_tools)
        num_calls_probs = self.get_num_calls_probs(y_state)  # (batch_size, max_parallel_calls)

        # Decode action
        action = action_probs.argmax(dim=-1)  # (batch_size,)
        action_confidence = action_probs.max(dim=-1)[0]  # (batch_size,)

        # Decode tool (only relevant if action == 1)
        tool_id = tool_probs.argmax(dim=-1)  # (batch_size,)
        tool_confidence = tool_probs.max(dim=-1)[0]  # (batch_size,)

        # Decode num calls (index 0 = 1 call, index 1 = 2 calls, etc.)
        num_calls = num_calls_probs.argmax(dim=-1) + 1  # (batch_size,) - add 1 for 1-indexed

        # Set tool_id to -1 and num_calls to 0 for direct_answer
        tool_id = torch.where(action == 0, torch.tensor(-1, device=action.device), tool_id)
        num_calls = torch.where(action == 0, torch.tensor(0, device=action.device), num_calls)

        # Overall confidence
        confidence = torch.where(
            action == 0,
            action_confidence,
            action_confidence * tool_confidence
        )

        return {
            'action': action,
            'tool_id': tool_id,
            'num_calls': num_calls,
            'confidence': confidence
        }


class ParameterGenerationHead(nn.Module):
    """Generate tool parameters autoregressively

    Uses the action state y as context to generate parameter tokens.
    Architecture: Cross-attention decoder that attends to action state.
    """

    def __init__(self, action_dim: int, vocab_size: int, hidden_dim: int = 256,
                 num_layers: int = 2, num_heads: int = 4, max_param_len: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_param_len = max_param_len

        # Token embedding for decoder input
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_param_len, hidden_dim)

        # Project action state to decoder hidden dim
        self.context_proj = nn.Linear(action_dim, hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to vocab
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Causal mask cache
        self.register_buffer("causal_mask", None)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, y_state: torch.Tensor, target_ids: torch.Tensor = None,
                max_length: int = None) -> torch.Tensor:
        """Forward pass for training (teacher forcing) or inference

        Args:
            y_state: (batch_size, action_dim) - action state from TRM
            target_ids: (batch_size, seq_len) - target token IDs for teacher forcing
            max_length: max generation length for inference

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size = y_state.size(0)
        device = y_state.device

        # Project action state as memory for cross-attention
        memory = self.context_proj(y_state).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        if target_ids is not None:
            # Training mode: teacher forcing
            seq_len = target_ids.size(1)

            # Embed target tokens
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embedding(target_ids) + self.position_embedding(positions)

            # Causal mask
            causal_mask = self._get_causal_mask(seq_len, device)

            # Decode
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output)  # (batch_size, seq_len, vocab_size)

            return logits
        else:
            # Inference mode: autoregressive generation
            max_length = max_length or self.max_param_len

            # Start with BOS token (use pad token id as start)
            generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            for step in range(max_length - 1):
                seq_len = generated.size(1)
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                tgt = self.token_embedding(generated) + self.position_embedding(positions)

                causal_mask = self._get_causal_mask(seq_len, device)
                output = self.decoder(tgt, memory, tgt_mask=causal_mask)

                # Get next token
                next_logits = self.output_proj(output[:, -1, :])  # (batch_size, vocab_size)
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (batch_size, 1)

                generated = torch.cat([generated, next_token], dim=1)

                # Stop if all sequences have generated end token (simplified: check for closing brace)
                # In practice, should check for EOS token

            # Return logits for the full sequence
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embedding(generated) + self.position_embedding(positions)
            causal_mask = self._get_causal_mask(seq_len, device)
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output)

            return logits

    def generate(self, y_state: torch.Tensor, max_length: int = None,
                 temperature: float = 1.0, eos_token_id: int = None) -> torch.Tensor:
        """Generate parameter tokens autoregressively

        Args:
            y_state: (batch_size, action_dim)
            max_length: maximum generation length
            temperature: sampling temperature (1.0 = greedy)
            eos_token_id: token ID to stop generation

        Returns:
            generated_ids: (batch_size, seq_len)
        """
        batch_size = y_state.size(0)
        device = y_state.device
        max_length = max_length or self.max_param_len

        # Project action state
        memory = self.context_proj(y_state).unsqueeze(1)

        # Start token
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embedding(generated) + self.position_embedding(positions)

            causal_mask = self._get_causal_mask(seq_len, device)
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)

            # Get next token logits
            next_logits = self.output_proj(output[:, -1, :]) / temperature

            if temperature == 1.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


class ResponseGenerationHead(nn.Module):
    """Generate direct answer text responses autoregressively

    Uses the action state y as context to generate response tokens.
    Similar architecture to ParameterGenerationHead but with larger capacity.
    """

    def __init__(self, action_dim: int, vocab_size: int, hidden_dim: int = 384,
                 num_layers: int = 3, num_heads: int = 6, max_response_len: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_response_len = max_response_len

        # Token embedding for decoder input
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_response_len, hidden_dim)

        # Project action state to decoder hidden dim
        self.context_proj = nn.Linear(action_dim, hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to vocab
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Causal mask cache
        self.register_buffer("causal_mask", None)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, y_state: torch.Tensor, target_ids: torch.Tensor = None,
                max_length: int = None) -> torch.Tensor:
        """Forward pass for training (teacher forcing) or inference

        Args:
            y_state: (batch_size, action_dim) - action state from TRM
            target_ids: (batch_size, seq_len) - target token IDs for teacher forcing
            max_length: max generation length for inference

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size = y_state.size(0)
        device = y_state.device

        # Project action state as memory for cross-attention
        memory = self.context_proj(y_state).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        if target_ids is not None:
            # Training mode: teacher forcing
            seq_len = target_ids.size(1)

            # Embed target tokens
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embedding(target_ids) + self.position_embedding(positions)

            # Causal mask
            causal_mask = self._get_causal_mask(seq_len, device)

            # Decode
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output)  # (batch_size, seq_len, vocab_size)

            return logits
        else:
            # Inference mode: autoregressive generation
            max_length = max_length or self.max_response_len

            # Start with BOS token (use pad token id as start)
            generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            for step in range(max_length - 1):
                seq_len = generated.size(1)
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                tgt = self.token_embedding(generated) + self.position_embedding(positions)

                causal_mask = self._get_causal_mask(seq_len, device)
                output = self.decoder(tgt, memory, tgt_mask=causal_mask)

                # Get next token
                next_logits = self.output_proj(output[:, -1, :])
                next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

            # Return logits for the full sequence
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embedding(generated) + self.position_embedding(positions)
            causal_mask = self._get_causal_mask(seq_len, device)
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output)

            return logits

    def generate(self, y_state: torch.Tensor, max_length: int = None,
                 temperature: float = 1.0, eos_token_id: int = None) -> torch.Tensor:
        """Generate response tokens autoregressively

        Args:
            y_state: (batch_size, action_dim)
            max_length: maximum generation length
            temperature: sampling temperature (1.0 = greedy)
            eos_token_id: token ID to stop generation

        Returns:
            generated_ids: (batch_size, seq_len)
        """
        batch_size = y_state.size(0)
        device = y_state.device
        max_length = max_length or self.max_response_len

        # Project action state
        memory = self.context_proj(y_state).unsqueeze(1)

        # Start token
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embedding(generated) + self.position_embedding(positions)

            causal_mask = self._get_causal_mask(seq_len, device)
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)

            # Get next token logits
            next_logits = self.output_proj(output[:, -1, :]) / temperature

            if temperature == 1.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
