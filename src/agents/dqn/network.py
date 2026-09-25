"""
Kiến trúc Dueling DQN (Wang et al., 2016):

    Q(s,a) = V(s) + ( A(s,a) - mean_a'[A(s,a')] )

Backbone chung 2 lớp FC, sau đó tách 2 nhánh:
    - Value stream:     FC -> 1 số      V(s)
    - Advantage stream: FC -> n_actions số   A(s,a)

Việc TRỪ MEAN của advantage là bắt buộc theo công thức gốc — nếu không trừ,
V(s) và A(s,a) có thể cộng/trừ cho nhau 1 hằng số bất kỳ mà Q(s,a) không đổi
(bài toán "unidentifiable"), khiến việc tách 2 nhánh trở nên vô nghĩa.
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

        # Backbone chung
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s), 1 số
        self.value_head = nn.Linear(hidden_dim, 1)
        # Advantage stream: A(s,a), n_actions số
        self.advantage_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: shape (batch_size, STATE_DIM)

        Returns:
            q_values: shape (batch_size, ACTION_SPACE)
        """
        assert state.dim() == 2 and state.shape[1] == self.state_dim, (
            f"state shape sai: {tuple(state.shape)}, kỳ vọng (batch_size, {self.state_dim})"
        )

        features = self.backbone(state) # (batch_size, hidden_dim)
        value = self.value_head(features) # (batch_size, 1)          -- V(s)
        advantage = self.advantage_head(features) # (batch_size, n_actions)  -- A(s,a)

        """
        Q(s,a) = V(s) + (A(s,a) - mean_a'[A(s,a')])
        keepdim=True để mean có shape (batch_size, 1), broadcast đúng với advantage (batch_size, n_actions)
        """
        advantage_mean = advantage.mean(dim=1, keepdim=True) # (batch_size, 1)
        q_values = value + (advantage - advantage_mean)  # (batch_size, n_actions)

        assert q_values.shape == (state.shape[0], self.n_actions), (
            f"q_values shape sai: {tuple(q_values.shape)}, kỳ vọng ({state.shape[0]}, {self.n_actions})"
        )

        return q_values