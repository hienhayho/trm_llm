"""Recursive Reasoning Module - Core TRM Innovation

This module implements recursive refinement of reasoning state z:
    z_{i+1} = f(x, y, z_i) for i in range(n_recursions)

The key insight from TRM paper:
- Recursively improve reasoning BEFORE making a decision
- Use small 2-layer network (parameter efficient)
- Recurse multiple times to get deep reasoning (recursive depth > model depth)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from trm_llm.models.transformer_blocks import TinyTransformer


class AttentionPooling(nn.Module):
    """Attention-based pooling with learnable query

    Instead of mean pooling (which loses positional information),
    this uses cross-attention with a learnable query to selectively
    aggregate information from the sequence.

    This allows the model to learn WHAT to attend to when pooling,
    preserving important positional and content information.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, num_queries: int = 1):
        """
        Args:
            hidden_dim: Dimension of input features
            num_heads: Number of attention heads
            num_queries: Number of learnable query vectors (default 1 for single pooled output)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Learnable query vector(s)
        self.query = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Cross-attention: query attends to sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Pool sequence using attention

        Args:
            x: (batch_size, seq_len, hidden_dim) - input sequence
            key_padding_mask: (batch_size, seq_len) - True for padding positions

        Returns:
            pooled: (batch_size, hidden_dim) if num_queries=1
                    (batch_size, num_queries, hidden_dim) otherwise
        """
        batch_size = x.size(0)

        # Expand query for batch
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_queries, dim)

        # Cross-attention: query attends to input sequence
        pooled, _ = self.attention(
            query, x, x,
            key_padding_mask=key_padding_mask
        )  # (batch, num_queries, dim)

        # Apply layer norm
        pooled = self.norm(pooled)

        # Squeeze if single query
        if self.num_queries == 1:
            pooled = pooled.squeeze(1)  # (batch, dim)

        return pooled


class RecursiveReasoningModule(nn.Module):
    """Recursive reasoning module that iteratively refines latent reasoning state

    This is the heart of TRM - instead of using a huge network,
    we use a small network recursively to achieve deep reasoning.
    """

    def __init__(self, hidden_dim: int, reasoning_dim: int, action_dim: int, num_recursions: int = 3):
        """
        Args:
            hidden_dim: Dimension of encoded input (from encoder)
            reasoning_dim: Dimension of reasoning state z
            action_dim: Dimension of action state y
            num_recursions: Number of recursive refinement iterations (n)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reasoning_dim = reasoning_dim
        self.action_dim = action_dim
        self.num_recursions = num_recursions

        # Attention pooling instead of mean pooling
        # This preserves positional information by learning what to attend to
        self.attention_pool = AttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_queries=1
        )

        # Project pooled input to reasoning space
        self.project_input = nn.Linear(hidden_dim, reasoning_dim)

        # Project action state to reasoning space
        self.project_action = nn.Linear(action_dim, reasoning_dim)

        # Small 2-layer transformer for recursive refinement (TRM principle)
        # This network is applied n times recursively
        self.reasoning_network = TinyTransformer(
            dim=reasoning_dim,
            num_layers=2,
            num_heads=8,
            dropout=0.1
        )

        # Combine x_proj, y_proj, z
        self.combiner = nn.Linear(reasoning_dim * 3, reasoning_dim)

    def forward(self, x_encoded, y_current, z_current=None, n_recursions=None):
        """Recursively refine reasoning state

        Args:
            x_encoded: (batch_size, seq_len, hidden_dim) - encoded input from encoder
            y_current: (batch_size, action_dim) - current action state
            z_current: (batch_size, reasoning_dim) - current reasoning state (None to initialize)
            n_recursions: Number of recursion steps (uses self.num_recursions if None)

        Returns:
            z_refined: (batch_size, reasoning_dim) - refined reasoning state
        """
        batch_size = x_encoded.size(0)
        n = n_recursions if n_recursions is not None else self.num_recursions

        # Pool sequence using attention (learns what to attend to)
        # This replaces mean pooling which loses positional information
        x_pooled = self.attention_pool(x_encoded)  # (batch_size, hidden_dim)
        x_proj = self.project_input(x_pooled)  # (batch_size, reasoning_dim)

        # Project action state to reasoning space
        y_proj = self.project_action(y_current)  # (batch_size, reasoning_dim)

        # Initialize z if not provided
        if z_current is None:
            # Use same dtype as input to support FP16/DeepSpeed
            z = torch.zeros(batch_size, self.reasoning_dim, device=x_encoded.device, dtype=x_encoded.dtype)
        else:
            z = z_current

        # Recursive refinement loop
        # Key TRM innovation: recursively improve z by considering (x, y, z)
        for i in range(n):
            # Combine input, action, and current reasoning
            combined = torch.cat([x_proj, y_proj, z], dim=-1)  # (batch_size, reasoning_dim * 3)
            combined = self.combiner(combined)  # (batch_size, reasoning_dim)

            # Refine through small transformer
            z_refined = self.reasoning_network(combined)  # (batch_size, reasoning_dim)

            # Update z for next iteration
            # Residual connection helps training
            z = z + z_refined

        return z


class RecursiveReasoningWithSequence(nn.Module):
    """Alternative reasoning module that processes full sequence

    This version processes the full sequence instead of pooling,
    which may be useful for tasks requiring fine-grained reasoning.
    """

    def __init__(self, hidden_dim: int, reasoning_dim: int, action_dim: int, num_recursions: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reasoning_dim = reasoning_dim
        self.num_recursions = num_recursions

        # Project to reasoning space
        self.project_input = nn.Linear(hidden_dim, reasoning_dim)
        self.project_action = nn.Linear(action_dim, reasoning_dim)

        # Reasoning network (processes sequences)
        self.reasoning_network = TinyTransformer(
            dim=reasoning_dim,
            num_layers=2,
            num_heads=8,
            dropout=0.1
        )

    def forward(self, x_encoded, y_current, z_current=None, n_recursions=None):
        """
        Args:
            x_encoded: (batch_size, seq_len, hidden_dim)
            y_current: (batch_size, action_dim)
            z_current: (batch_size, seq_len, reasoning_dim)

        Returns:
            z_refined: (batch_size, seq_len, reasoning_dim)
        """
        batch_size, seq_len, _ = x_encoded.shape
        n = n_recursions if n_recursions is not None else self.num_recursions

        # Project input
        x_proj = self.project_input(x_encoded)  # (batch_size, seq_len, reasoning_dim)

        # Project and broadcast action
        y_proj = self.project_action(y_current)  # (batch_size, reasoning_dim)
        y_proj = y_proj.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, reasoning_dim)

        # Initialize z
        if z_current is None:
            # Use same dtype as input to support FP16/DeepSpeed
            z = torch.zeros(batch_size, seq_len, self.reasoning_dim, device=x_encoded.device, dtype=x_encoded.dtype)
        else:
            z = z_current

        # Recursive refinement
        for i in range(n):
            # Combine and refine
            combined = x_proj + y_proj + z
            z_refined = self.reasoning_network(combined)
            z = z + z_refined

        return z
