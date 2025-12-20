"""Action State Module - Updates current action based on reasoning

This module implements:
    y_{t+1} = g(y_t, z_t)

where:
- y: action state (what to do: direct answer or tool call)
- z: refined reasoning state (from RecursiveReasoningModule)
"""

import torch
import torch.nn as nn
from trm_llm.models.transformer_blocks import TinyTransformer


class ActionStateModule(nn.Module):
    """Updates action state based on refined reasoning

    After recursive reasoning refines z, this module uses z to update
    the action state y, which represents the current plan/decision.
    """

    def __init__(self, reasoning_dim: int, action_dim: int):
        """
        Args:
            reasoning_dim: Dimension of reasoning state z
            action_dim: Dimension of action state y
        """
        super().__init__()
        self.reasoning_dim = reasoning_dim
        self.action_dim = action_dim

        # Project reasoning to action space
        self.project_reasoning = nn.Linear(reasoning_dim, action_dim)

        # Small 2-layer transformer to refine action state (TRM principle)
        self.action_network = TinyTransformer(
            dim=action_dim,
            num_layers=2,
            num_heads=4,  # Fewer heads for smaller dimension
            dropout=0.1
        )

        # Combiner for y and z_proj
        self.combiner = nn.Linear(action_dim * 2, action_dim)

    def forward(self, y_current, z_reasoning):
        """Update action state using refined reasoning

        Args:
            y_current: (batch_size, action_dim) - current action state
            z_reasoning: (batch_size, reasoning_dim) - refined reasoning state

        Returns:
            y_updated: (batch_size, action_dim) - updated action state
        """
        # Project reasoning to action space
        z_proj = self.project_reasoning(z_reasoning)  # (batch_size, action_dim)

        # Combine current action with reasoning
        combined = torch.cat([y_current, z_proj], dim=-1)  # (batch_size, action_dim * 2)
        combined = self.combiner(combined)  # (batch_size, action_dim)

        # Refine through action network
        y_refined = self.action_network(combined)  # (batch_size, action_dim)

        # Residual connection
        y_updated = y_current + y_refined

        return y_updated


class ActionStateWithGating(nn.Module):
    """Action state module with gating mechanism

    Uses gating to control how much of the new reasoning to incorporate
    """

    def __init__(self, reasoning_dim: int, action_dim: int):
        super().__init__()
        self.reasoning_dim = reasoning_dim
        self.action_dim = action_dim

        # Project reasoning to action space
        self.project_reasoning = nn.Linear(reasoning_dim, action_dim)

        # Action refinement network
        self.action_network = TinyTransformer(
            dim=action_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )

        # Gating mechanism (how much to update)
        self.gate = nn.Sequential(
            nn.Linear(action_dim + reasoning_dim, action_dim),
            nn.Sigmoid()
        )

    def forward(self, y_current, z_reasoning):
        """
        Args:
            y_current: (batch_size, action_dim)
            z_reasoning: (batch_size, reasoning_dim)

        Returns:
            y_updated: (batch_size, action_dim)
        """
        # Project reasoning
        z_proj = self.project_reasoning(z_reasoning)

        # Compute update
        combined = y_current + z_proj
        y_update = self.action_network(combined)

        # Gate the update
        gate_input = torch.cat([y_current, z_reasoning], dim=-1)
        gate_value = self.gate(gate_input)  # (batch_size, action_dim)

        # Gated residual update
        y_updated = y_current + gate_value * y_update

        return y_updated
