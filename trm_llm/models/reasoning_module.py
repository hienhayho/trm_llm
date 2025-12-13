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
from .transformer_blocks import TinyTransformer


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

        # Project encoded input to reasoning space
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

        # Project input to reasoning space (pool sequence)
        # Average pooling over sequence length
        x_pooled = x_encoded.mean(dim=1)  # (batch_size, hidden_dim)
        x_proj = self.project_input(x_pooled)  # (batch_size, reasoning_dim)

        # Project action state to reasoning space
        y_proj = self.project_action(y_current)  # (batch_size, reasoning_dim)

        # Initialize z if not provided
        if z_current is None:
            z = torch.zeros(batch_size, self.reasoning_dim, device=x_encoded.device)
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
            z = torch.zeros(batch_size, seq_len, self.reasoning_dim, device=x_encoded.device)
        else:
            z = z_current

        # Recursive refinement
        for i in range(n):
            # Combine and refine
            combined = x_proj + y_proj + z
            z_refined = self.reasoning_network(combined)
            z = z + z_refined

        return z
