"""
Dueling DQN architecture (Wang et al., 2016):

    Q(s,a) = V(s) + ( A(s,a) - mean_a'[A(s,a')] )

A shared two-layer FC backbone, then two streams:
    - Value stream:     FC -> 1 value          V(s)
    - Advantage stream: FC -> n_actions values A(s,a)

SUBTRACTING THE MEAN advantage is required by the original formula. Without it,
V(s) and A(s,a) could shift by any constant while Q(s,a) stays the same (the
"unidentifiable" problem), which would make the two streams meaningless.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from src.envs.state_encoder import STATE_DIM
from src.envs.snake_env import ACTION_SPACE

class DuelingQNetwork(nn.Module):
    """Dueling DQN network: input state (STATE_DIM,) -> output Q-values (ACTION_SPACE,)."""
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = ACTION_SPACE, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s), one value
        self.value_head = nn.Linear(hidden_dim, 1)
        # Advantage stream: A(s,a), n_actions values
        self.advantage_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: shape (batch_size, STATE_DIM)

        Returns:
            q_values: shape (batch_size, ACTION_SPACE)
        """
        assert state.dim() == 2 and state.shape[1] == self.state_dim, (
            f"wrong state shape: {tuple(state.shape)}, expected (batch_size, {self.state_dim})"
        )

        features = self.backbone(state) # (batch_size, hidden_dim)
        value = self.value_head(features) # (batch_size, 1)          -- V(s)
        advantage = self.advantage_head(features) # (batch_size, n_actions)  -- A(s,a)

        """
        Q(s,a) = V(s) + (A(s,a) - mean_a'[A(s,a')])
        keepdim=True keeps the mean at shape (batch_size, 1) so it broadcasts against advantage (batch_size, n_actions)
        """
        advantage_mean = advantage.mean(dim=1, keepdim=True) # (batch_size, 1)
        q_values = value + (advantage - advantage_mean)  # (batch_size, n_actions)

        assert q_values.shape == (state.shape[0], self.n_actions), (
            f"wrong q_values shape: {tuple(q_values.shape)}, expected ({state.shape[0]}, {self.n_actions})"
        )

        return q_values