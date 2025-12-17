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
    - Num parallel calls logits: [P(1), P(2), ..., P(max_parallel)]
    - Halt logit: P(should_stop_refining)

    Note: Tool selection is done via generation (model generates tool name in JSON)
    """

    def __init__(self, action_dim: int, num_action_types: int = 2,
                 max_parallel_calls: int = 5, **kwargs):
        """
        Args:
            action_dim: Dimension of action state y
            num_action_types: Number of action types (2: direct_answer, tool_call)
            max_parallel_calls: Maximum number of parallel tool calls (1-5)
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_action_types = num_action_types
        self.max_parallel_calls = max_parallel_calls

        # 1. Action decision head: should I answer directly or call a tool?
        self.action_head = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim // 2, num_action_types)
        )

        # 2. Num parallel calls head: how many tools to call in parallel?
        # Predicts 1, 2, 3, ... max_parallel_calls
        self.num_calls_head = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim // 2, max_parallel_calls)
        )

        # 3. Halt decision head: should I stop refining? (for ACT)
        self.halt_head = nn.Linear(action_dim, 1)

    def forward(self, y_state):
        """Generate all outputs from action state

        Args:
            y_state: (batch_size, action_dim) - current action state

        Returns:
            dict with:
                - action_logits: (batch_size, num_action_types)
                - num_calls_logits: (batch_size, max_parallel_calls)
                - halt_logit: (batch_size, 1)
                - y_state: (batch_size, action_dim) - for downstream use
        """
        # Action decision (direct answer vs tool call)
        action_logits = self.action_head(y_state)  # (batch_size, 2)

        # Number of parallel calls (1, 2, 3, ...)
        num_calls_logits = self.num_calls_head(y_state)  # (batch_size, max_parallel_calls)

        # Halting decision (should stop refining)
        halt_logit = self.halt_head(y_state)  # (batch_size, 1)

        return {
            'action_logits': action_logits,
            'num_calls_logits': num_calls_logits,
            'halt_logit': halt_logit,
            'y_state': y_state  # Keep for generation
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
                - num_calls: (batch_size,) - number of parallel calls (0 if direct_answer)
                - confidence: (batch_size,) - confidence score
        """
        # Get probabilities
        action_probs = self.get_action_probs(y_state)  # (batch_size, 2)
        num_calls_probs = self.get_num_calls_probs(y_state)  # (batch_size, max_parallel_calls)

        # Decode action
        action = action_probs.argmax(dim=-1)  # (batch_size,)
        action_confidence = action_probs.max(dim=-1)[0]  # (batch_size,)

        # Decode num calls (index 0 = 1 call, index 1 = 2 calls, etc.)
        num_calls = num_calls_probs.argmax(dim=-1) + 1  # (batch_size,) - add 1 for 1-indexed

        # Set num_calls to 0 for direct_answer
        num_calls = torch.where(action == 0, torch.tensor(0, device=action.device), num_calls)

        return {
            'action': action,
            'num_calls': num_calls,
            'confidence': action_confidence
        }


class UnifiedGenerationHead(nn.Module):
    """Unified generation head for both tool calls and direct answers

    Generates:
    - Tool call JSON: {"name": "tool", "arguments": {...}} or [{"name": "tool1", ...}, ...]
    - Direct answer text

    Cross-attends to full encoder output (not just y_state) for better generation.
    Optionally uses y_state as additional context via a learned prefix token.
    """

    def __init__(self, action_dim: int, vocab_size: int, hidden_dim: int = 384,
                 num_layers: int = 3, num_heads: int = 6, max_gen_len: int = 512,
                 encoder_hidden_dim: int = None):
        """
        Args:
            action_dim: Dimension of action state y
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension for decoder
            num_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            max_gen_len: Maximum generation length
            encoder_hidden_dim: Hidden dimension of encoder output (for projection)
        """
        super().__init__()
        self.action_dim = action_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_gen_len = max_gen_len
        self.encoder_hidden_dim = encoder_hidden_dim or hidden_dim

        # Token embedding for decoder input
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_gen_len, hidden_dim)

        # Project action state to decoder hidden dim (used as prefix token)
        self.context_proj = nn.Linear(action_dim, hidden_dim)

        # Project encoder output to decoder hidden dim (if dimensions differ)
        if self.encoder_hidden_dim != hidden_dim:
            self.encoder_proj = nn.Linear(self.encoder_hidden_dim, hidden_dim)
        else:
            self.encoder_proj = None

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

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _prepare_memory(self, y_state: torch.Tensor, encoder_output: torch.Tensor = None) -> torch.Tensor:
        """Prepare memory for cross-attention

        Combines encoder output with y_state context token.

        Args:
            y_state: (batch_size, action_dim) - action state from TRM
            encoder_output: (batch_size, seq_len, encoder_hidden_dim) - encoder output

        Returns:
            memory: (batch_size, memory_len, hidden_dim)
        """
        device = y_state.device

        # Project y_state as a context token
        y_context = self.context_proj(y_state).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        if encoder_output is not None:
            # Project encoder output if needed
            if self.encoder_proj is not None:
                encoder_output = self.encoder_proj(encoder_output)

            # Prepend y_context to encoder output
            # This gives the decoder: [y_state_context, encoder_token_1, encoder_token_2, ...]
            memory = torch.cat([y_context, encoder_output], dim=1)
        else:
            # Fallback: use only y_state (like before)
            memory = y_context

        return memory

    def forward(self, y_state: torch.Tensor, target_ids: torch.Tensor = None,
                max_length: int = None, encoder_output: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for training (teacher forcing) or inference

        Args:
            y_state: (batch_size, action_dim) - action state from TRM
            target_ids: (batch_size, seq_len) - target token IDs for teacher forcing
            max_length: max generation length for inference
            encoder_output: (batch_size, seq_len, encoder_hidden_dim) - full encoder output
                           for cross-attention (enables attending to all input tokens)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size = y_state.size(0)
        device = y_state.device

        # Prepare memory for cross-attention (encoder output + y_state context)
        memory = self._prepare_memory(y_state, encoder_output)

        if target_ids is not None:
            # Training mode: teacher forcing
            seq_len = target_ids.size(1)

            # Embed target tokens
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embedding(target_ids) + self.position_embedding(positions)

            # Causal mask
            causal_mask = self._get_causal_mask(seq_len, device)

            # Decode with cross-attention to full encoder output
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output)  # (batch_size, seq_len, vocab_size)

            return logits
        else:
            # Inference mode: autoregressive generation
            max_length = max_length or self.max_gen_len

            # Start with BOS token (use 0 as start token)
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
                 temperature: float = 1.0, eos_token_id: int = None,
                 encoder_output: torch.Tensor = None,
                 repetition_penalty: float = 1.2,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 no_repeat_ngram_size: int = 3) -> torch.Tensor:
        """Generate tokens autoregressively with repetition prevention

        Args:
            y_state: (batch_size, action_dim)
            max_length: maximum generation length
            temperature: sampling temperature (lower = more deterministic)
            eos_token_id: token ID to stop generation
            encoder_output: (batch_size, seq_len, encoder_hidden_dim) - encoder output for cross-attention
            repetition_penalty: penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize)
            top_k: keep only top k tokens for sampling (0 = disabled)
            top_p: nucleus sampling - keep tokens with cumulative prob < top_p (1.0 = disabled)
            no_repeat_ngram_size: prevent repeating n-grams of this size (0 = disabled)

        Returns:
            generated_ids: (batch_size, seq_len)
        """
        batch_size = y_state.size(0)
        device = y_state.device
        max_length = max_length or self.max_gen_len

        # Prepare memory for cross-attention
        memory = self._prepare_memory(y_state, encoder_output)

        # Start token
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embedding(generated) + self.position_embedding(positions)

            causal_mask = self._get_causal_mask(seq_len, device)
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)

            # Get next token logits
            next_logits = self.output_proj(output[:, -1, :])  # (batch, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        if token_id < next_logits.size(-1):
                            # Penalize tokens that have appeared before
                            if next_logits[i, token_id] > 0:
                                next_logits[i, token_id] /= repetition_penalty
                            else:
                                next_logits[i, token_id] *= repetition_penalty

            # Apply no_repeat_ngram blocking
            if no_repeat_ngram_size > 0 and seq_len >= no_repeat_ngram_size:
                for i in range(batch_size):
                    # Get the last (n-1) tokens as the prefix
                    ngram_prefix = generated[i, -(no_repeat_ngram_size - 1):].tolist()
                    # Find all positions where this prefix appeared before
                    for j in range(seq_len - no_repeat_ngram_size + 1):
                        prev_ngram = generated[i, j:j + no_repeat_ngram_size - 1].tolist()
                        if prev_ngram == ngram_prefix:
                            # Block the token that followed this prefix
                            blocked_token = generated[i, j + no_repeat_ngram_size - 1].item()
                            if blocked_token < next_logits.size(-1):
                                next_logits[i, blocked_token] = float('-inf')

            # Apply temperature
            next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < min_top_k,
                    torch.full_like(next_logits, float('-inf')),
                    next_logits
                )

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep first token above threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                # Scatter back to original indices
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_logits[i, indices_to_remove] = float('-inf')

            # Sample or greedy
            probs = F.softmax(next_logits, dim=-1)
            if temperature < 0.1:  # Essentially greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
