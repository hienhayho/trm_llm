"""Main TRM-LLM Model

Integrates all components:
- Input embedding + encoder
- Recursive reasoning module (iteratively refines z)
- Action state module (updates y based on z)
- Output heads (decode y into decisions)
- Deep supervision loop (progressively improves answer)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from ..utils.config import TRMLLMConfig
from .transformer_blocks import TransformerEncoder, PositionalEncoding
from .reasoning_module import RecursiveReasoningModule
from .action_module import ActionStateModule
from .output_heads import OutputHeads, ParameterGenerationHead, ResponseGenerationHead


class TRMLLM(nn.Module):
    """Tiny Recursive Model for LLM Tool Calling

    Key innovations from TRM:
    1. Recursive reasoning: Small network applied n times to refine z
    2. Deep supervision: Multiple supervision steps to progressively improve answer
    3. State detaching: Gradients don't flow across supervision steps
    4. Adaptive computation: Learned early stopping

    Architecture:
        Input → Encoder → [Recursive Reasoning → Action Update] × T → Outputs
    """

    def __init__(self, config: TRMLLMConfig):
        super().__init__()
        self.config = config

        # ===== Input Embeddings =====
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Alternative: Sinusoidal positional encoding
        # self.position_encoding = PositionalEncoding(config.hidden_dim, config.max_seq_len)

        self.embedding_dropout = nn.Dropout(config.dropout)

        # ===== Encoder (Standard Transformer) =====
        self.encoder = TransformerEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout
        )

        # ===== TRM Components =====
        # Recursive reasoning module (core innovation)
        self.reasoning_module = RecursiveReasoningModule(
            hidden_dim=config.hidden_dim,
            reasoning_dim=config.reasoning_dim,
            action_dim=config.action_dim,
            num_recursions=config.num_recursions
        )

        # Action state module
        self.action_module = ActionStateModule(
            reasoning_dim=config.reasoning_dim,
            action_dim=config.action_dim
        )

        # Output heads
        self.output_heads = OutputHeads(
            action_dim=config.action_dim,
            num_action_types=config.num_action_types,
            max_tools=config.max_tools,
            max_parallel_calls=config.max_parallel_calls
        )

        # Parameter generation head (for tool arguments)
        self.param_head = ParameterGenerationHead(
            action_dim=config.action_dim,
            vocab_size=config.vocab_size,
            hidden_dim=config.action_dim,  # Use same dim for simplicity
            num_layers=2,
            num_heads=4,
            max_param_len=config.max_param_len
        )

        # Response generation head (for direct answer text)
        self.response_head = ResponseGenerationHead(
            action_dim=config.action_dim,
            vocab_size=config.vocab_size,
            hidden_dim=getattr(config, 'response_hidden_dim', 384),
            num_layers=getattr(config, 'response_num_layers', 3),
            num_heads=getattr(config, 'response_num_heads', 6),
            max_response_len=config.max_response_len
        )

        # ===== Learnable Initial State =====
        # Initial action state y (learned)
        self.init_y = nn.Parameter(torch.randn(config.action_dim) * 0.02)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following best practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_input(self, input_ids):
        """Embed and encode input tokens

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            x_encoded: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # (batch_size, seq_len, hidden_dim)

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)  # (1, seq_len, hidden_dim)

        # Combine
        x = token_embeds + pos_embeds
        x = self.embedding_dropout(x)

        # Encode through transformer
        x_encoded = self.encoder(x)

        return x_encoded

    def forward(
        self,
        input_ids: torch.Tensor,
        max_supervision_steps: Optional[int] = None,
        training: bool = True,
        target_param_ids: Optional[torch.Tensor] = None,
        target_response_ids: Optional[torch.Tensor] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Forward pass with deep supervision

        This is the core TRM algorithm:
        1. Encode input
        2. For each supervision step:
            a. Recursively refine reasoning state z
            b. Update action state y
            c. Generate outputs
            d. Detach states for next iteration (no BPTT)

        Args:
            input_ids: (batch_size, seq_len) - input token IDs
            max_supervision_steps: number of supervision iterations (default: config value)
            training: whether in training mode (affects ACT behavior)
            target_param_ids: (batch_size, param_seq_len) - target parameter token IDs for training
            target_response_ids: (batch_size, response_seq_len) - target response token IDs for training

        Returns:
            outputs_per_step: List of dicts, one per supervision step
                Each dict contains:
                    - action_logits: (batch_size, num_action_types)
                    - tool_logits: (batch_size, max_tools)
                    - halt_logit: (batch_size, 1)
                    - y_state: (batch_size, action_dim)
                    - param_logits: (batch_size, param_seq_len, vocab_size) if target_param_ids provided
                    - response_logits: (batch_size, response_seq_len, vocab_size) if target_response_ids provided
        """
        batch_size = input_ids.size(0)
        max_steps = max_supervision_steps or self.config.max_supervision_steps

        # ===== Step 1: Encode Input =====
        x_encoded = self.embed_input(input_ids)  # (batch_size, seq_len, hidden_dim)

        # ===== Step 2: Initialize States =====
        # Action state y (initialized to learned parameter)
        y = self.init_y.unsqueeze(0).expand(batch_size, -1)  # (batch_size, action_dim)

        # Reasoning state z (initialized to None, will be created in reasoning module)
        z = None

        # ===== Step 3: Deep Supervision Loop =====
        outputs_per_step = []

        for step in range(max_steps):
            # 3a. Recursive reasoning: refine z based on (x, y, z)
            z = self.reasoning_module(
                x_encoded=x_encoded,
                y_current=y,
                z_current=z,
                n_recursions=self.config.num_recursions
            )  # (batch_size, reasoning_dim)

            # 3b. Update action state: y = g(y, z)
            y = self.action_module(y_current=y, z_reasoning=z)  # (batch_size, action_dim)

            # 3c. Generate outputs from current y
            outputs = self.output_heads(y)  # Dict with action_logits, tool_logits, halt_logit

            # 3d. Generate parameter/response logits if target provided (only on last step for efficiency)
            if step == max_steps - 1:
                if target_param_ids is not None:
                    param_logits = self.param_head(y, target_ids=target_param_ids)
                    outputs['param_logits'] = param_logits
                if target_response_ids is not None:
                    response_logits = self.response_head(y, target_ids=target_response_ids)
                    outputs['response_logits'] = response_logits

            outputs_per_step.append(outputs)

            # 3e. Check for early stopping (ACT) - only in inference
            if not training:
                halt_prob = torch.sigmoid(outputs['halt_logit']).mean().item()
                if halt_prob > self.config.halt_threshold:
                    # Generate params/response on early stop too
                    if target_param_ids is not None and 'param_logits' not in outputs:
                        param_logits = self.param_head(y, target_ids=target_param_ids)
                        outputs['param_logits'] = param_logits
                    if target_response_ids is not None and 'response_logits' not in outputs:
                        response_logits = self.response_head(y, target_ids=target_response_ids)
                        outputs['response_logits'] = response_logits
                    break

            # 3f. Detach states for next iteration
            # Key TRM technique: don't backprop through all supervision steps
            # Only the last step will have gradients
            if step < max_steps - 1:
                y = y.detach()
                z = z.detach()

        return outputs_per_step

    def generate_params(self, y_state: torch.Tensor, max_length: int = 64) -> torch.Tensor:
        """Generate parameter tokens from action state

        Args:
            y_state: (batch_size, action_dim) - final action state
            max_length: maximum parameter sequence length

        Returns:
            param_ids: (batch_size, seq_len) - generated parameter token IDs
        """
        return self.param_head.generate(y_state, max_length=max_length)

    def generate_response(self, y_state: torch.Tensor, max_length: int = 128,
                          temperature: float = 1.0, eos_token_id: int = None) -> torch.Tensor:
        """Generate response tokens from action state

        Args:
            y_state: (batch_size, action_dim) - final action state
            max_length: maximum response sequence length
            temperature: sampling temperature (1.0 = greedy)
            eos_token_id: token ID to stop generation

        Returns:
            response_ids: (batch_size, seq_len) - generated response token IDs
        """
        return self.response_head.generate(y_state, max_length=max_length,
                                           temperature=temperature, eos_token_id=eos_token_id)

    def get_num_params(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TRMLLMWithCache(TRMLLM):
    """TRM-LLM with KV-cache for efficient inference

    Extension for production use - caches encoder output to avoid recomputation
    """

    def __init__(self, config: TRMLLMConfig):
        super().__init__(config)
        self.encoder_cache = None

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = True,
        max_supervision_steps: Optional[int] = None
    ):
        """Forward with encoder caching

        Args:
            input_ids: (batch_size, seq_len)
            use_cache: whether to use cached encoder output
            max_supervision_steps: max supervision iterations

        Returns:
            outputs_per_step: same as regular forward
        """
        # Check if can use cache
        if use_cache and self.encoder_cache is not None:
            x_encoded = self.encoder_cache
        else:
            x_encoded = self.embed_input(input_ids)
            if use_cache:
                self.encoder_cache = x_encoded

        # Rest is same as regular forward
        batch_size = x_encoded.size(0)
        max_steps = max_supervision_steps or self.config.max_supervision_steps

        y = self.init_y.unsqueeze(0).expand(batch_size, -1)
        z = None

        outputs_per_step = []

        for step in range(max_steps):
            z = self.reasoning_module(x_encoded, y, z, self.config.num_recursions)
            y = self.action_module(y, z)
            outputs = self.output_heads(y)
            outputs_per_step.append(outputs)

            # ACT early stopping
            halt_prob = torch.sigmoid(outputs['halt_logit']).mean().item()
            if halt_prob > self.config.halt_threshold:
                break

            if step < max_steps - 1:
                y = y.detach()
                z = z.detach()

        return outputs_per_step

    def clear_cache(self):
        """Clear encoder cache"""
        self.encoder_cache = None
