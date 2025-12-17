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

from trm_llm.utils.config import TRMLLMConfig
from trm_llm.models.transformer_blocks import TransformerEncoder, PositionalEncoding
from trm_llm.models.reasoning_module import RecursiveReasoningModule
from trm_llm.models.action_module import ActionStateModule
from trm_llm.models.output_heads import OutputHeads, UnifiedGenerationHead


class TRMLLM(nn.Module):
    """Tiny Recursive Model for LLM Tool Calling

    Key innovations from TRM:
    1. Recursive reasoning: Small network applied n times to refine z
    2. Deep supervision: Multiple supervision steps to progressively improve answer
    3. State detaching: Gradients don't flow across supervision steps
    4. Adaptive computation: Learned early stopping

    Architecture:
        Input → Encoder → [Recursive Reasoning → Action Update] × T → Outputs

    Generation:
        - direct_answer: generates answer text
        - tool_call: generates JSON like {"name": "tool", "arguments": {...}}
                     or [{"name": "tool1", ...}, {"name": "tool2", ...}] for parallel calls
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
            dropout=config.dropout,
            use_sdp_attention=getattr(config, 'use_flash_attention', True),
        )

        # ===== TRM Components =====
        # Recursive reasoning module (core innovation)
        self.reasoning_module = RecursiveReasoningModule(
            hidden_dim=config.hidden_dim,
            reasoning_dim=config.reasoning_dim,
            action_dim=config.action_dim,
            num_recursions=config.num_recursions,
        )

        # Action state module
        self.action_module = ActionStateModule(
            reasoning_dim=config.reasoning_dim, action_dim=config.action_dim
        )

        # Output heads (action type + num parallel calls + halt)
        self.output_heads = OutputHeads(
            action_dim=config.action_dim,
            num_action_types=config.num_action_types,
            max_parallel_calls=config.max_parallel_calls,
        )

        # Unified generation head (for both tool calls and direct answers)
        # Now cross-attends to full encoder output for better generation
        self.generation_head = UnifiedGenerationHead(
            action_dim=config.action_dim,
            vocab_size=config.vocab_size,
            hidden_dim=getattr(config, "generation_hidden_dim", 384),
            num_layers=getattr(config, "generation_num_layers", 3),
            num_heads=getattr(config, "generation_num_heads", 6),
            max_gen_len=config.max_generation_len,
            encoder_hidden_dim=config.hidden_dim,  # Pass encoder hidden dim for projection
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
        # Use causal attention if configured (for pure LLM training)
        is_causal = getattr(self.config, 'use_causal_encoder', False)
        x_encoded = self.encoder(x, is_causal=is_causal)

        return x_encoded

    def forward(
        self,
        input_ids: torch.Tensor,
        max_supervision_steps: Optional[int] = None,
        training: bool = True,
        target_generation_ids: Optional[torch.Tensor] = None,
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
            target_generation_ids: (batch_size, gen_seq_len) - target token IDs for generation
                For tool_call: JSON like {"name": "tool", "arguments": {...}}
                For direct_answer: answer text

        Returns:
            outputs_per_step: List of dicts, one per supervision step
                Each dict contains:
                    - action_logits: (batch_size, num_action_types)
                    - num_calls_logits: (batch_size, max_parallel_calls)
                    - halt_logit: (batch_size, 1)
                    - y_state: (batch_size, action_dim)
                    - generation_logits: (batch_size, gen_seq_len, vocab_size) if target provided
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
                n_recursions=self.config.num_recursions,
            )  # (batch_size, reasoning_dim)

            # 3b. Update action state: y = g(y, z)
            y = self.action_module(y_current=y, z_reasoning=z)  # (batch_size, action_dim)

            # 3c. Generate outputs from current y
            outputs = self.output_heads(y)  # Dict with action_logits, num_calls_logits, halt_logit

            # 3d. Generate logits if target provided (only on last step for efficiency)
            # Pass encoder output and action_logits for action-conditioned generation
            if step == max_steps - 1 and target_generation_ids is not None:
                generation_logits = self.generation_head(
                    y, target_ids=target_generation_ids, encoder_output=x_encoded,
                    action_logits=outputs["action_logits"]
                )
                outputs["generation_logits"] = generation_logits

            # Store encoder output for inference generation
            outputs["encoder_output"] = x_encoded

            outputs_per_step.append(outputs)

            # 3e. Check for early stopping (ACT) - only in inference
            if not training:
                halt_prob = torch.sigmoid(outputs["halt_logit"]).mean().item()
                if halt_prob > self.config.halt_threshold:
                    # Generate on early stop too (with action conditioning)
                    if target_generation_ids is not None and "generation_logits" not in outputs:
                        generation_logits = self.generation_head(
                            y, target_ids=target_generation_ids, encoder_output=x_encoded,
                            action_logits=outputs["action_logits"]
                        )
                        outputs["generation_logits"] = generation_logits
                    break

            # 3f. Optionally detach states for next iteration
            # Key TRM technique: don't backprop through all supervision steps
            # Only the last step will have gradients (when detaching is enabled)
            # Set detach_between_steps=False to allow gradients through all steps
            should_detach = getattr(self.config, 'detach_between_steps', True)
            if step < max_steps - 1 and should_detach:
                y = y.detach()
                z = z.detach()

        return outputs_per_step

    def generate(
        self,
        y_state: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.7,
        eos_token_id: int = None,
        encoder_output: torch.Tensor = None,
        repetition_penalty: float = 1.2,
        top_k: int = 50,
        top_p: float = 0.9,
        no_repeat_ngram_size: int = 3,
        action_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate tokens from action state

        Args:
            y_state: (batch_size, action_dim) - final action state
            max_length: maximum generation length
            temperature: sampling temperature (lower = more deterministic, default 0.7)
            eos_token_id: token ID to stop generation
            encoder_output: (batch_size, seq_len, hidden_dim) - encoder output for cross-attention
            repetition_penalty: penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize)
            top_k: keep only top k tokens for sampling (0 = disabled)
            top_p: nucleus sampling - keep tokens with cumulative prob < top_p
            no_repeat_ngram_size: prevent repeating n-grams of this size (0 = disabled)
            action_logits: (batch_size, num_action_types) - action prediction logits
                          for conditioning generation on predicted action

        Returns:
            generated_ids: (batch_size, seq_len) - generated token IDs
        """
        return self.generation_head.generate(
            y_state, max_length=max_length, temperature=temperature,
            eos_token_id=eos_token_id, encoder_output=encoder_output,
            repetition_penalty=repetition_penalty, top_k=top_k,
            top_p=top_p, no_repeat_ngram_size=no_repeat_ngram_size,
            action_logits=action_logits
        )

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
        max_supervision_steps: Optional[int] = None,
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
            halt_prob = torch.sigmoid(outputs["halt_logit"]).mean().item()
            if halt_prob > self.config.halt_threshold:
                break

            if step < max_steps - 1:
                y = y.detach()
                z = z.detach()

        return outputs_per_step

    def clear_cache(self):
        """Clear encoder cache"""
        self.encoder_cache = None
