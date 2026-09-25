"""
Actor-Critic network: 1 backbone chung, tách 2 head:
    - Actor:  logits cho phân phối Categorical trên 4 action (softmax ngầm định
              qua torch.distributions.Categorical)
    - Critic: V(s), 1 số — ước lượng giá trị trạng thái, dùng để tính advantage (GAE)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.envs.state_encoder import STATE_DIM
from src.envs.snake_env import ACTION_SPACE

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = ACTION_SPACE, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, n_actions)   # -> logits, chưa softmax
        self.critic_head = nn.Linear(hidden_dim, 1)           # -> V(s)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: shape (batch_size, STATE_DIM)

        Returns:
            action_logits: shape (batch_size, ACTION_SPACE)
            value:         shape (batch_size,)   -- đã squeeze chiều cuối
        """
        assert state.dim() == 2 and state.shape[1] == self.state_dim, (
            f"state shape sai: {tuple(state.shape)}, kỳ vọng (batch_size, {self.state_dim})"
        )
        features = self.backbone(state)                 # (batch_size, hidden_dim)
        action_logits = self.actor_head(features)         # (batch_size, ACTION_SPACE)
        value = self.critic_head(features).squeeze(-1)     # (batch_size,)

        assert action_logits.shape == (state.shape[0], self.n_actions)
        assert value.shape == (state.shape[0],)
        return action_logits, value

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample 1 action từ policy hiện tại — dùng lúc thu thập rollout (không backprop qua bước này).

        Args:
            state: shape (batch_size, STATE_DIM) — thường batch_size=1 khi chạy trong env

        Returns:
            action:   (batch_size,)  int64, action được SAMPLE (không phải argmax) -- giữ tính stochastic của policy
            log_prob: (batch_size,)  log pi_theta(a|s) tại action đã sample -- dùng làm log_prob "cũ" cho PPO ratio
            value:    (batch_size,)  V(s)
        """
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()               # (batch_size,)
        log_prob = dist.log_prob(action)      # (batch_size,)

        assert action.shape == (state.shape[0],)
        assert log_prob.shape == (state.shape[0],)
        return action, log_prob, value

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tính lại log_prob VÀ entropy của các action đã lấy trong rollout, dưới policy HIỆN TẠI
        (policy có thể đã thay đổi qua vài lần update kể từ lúc thu thập dữ liệu).
        Dùng trong hàm loss PPO — chính là pi_theta(a_t|s_t) ở tử số của probability ratio r_t(theta).

        Args:
            states:  (batch_size, STATE_DIM)
            actions: (batch_size,) — action ĐÃ được sample lúc thu thập rollout (giữ nguyên, không sample lại)

        Returns:
            log_probs: (batch_size,)
            entropy:   (batch_size,)  -- dùng cho entropy bonus S[pi_theta] trong loss
            values:    (batch_size,)
        """
        action_logits, values = self.forward(states)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        assert log_probs.shape == (states.shape[0],)
        assert entropy.shape == (states.shape[0],)
        return log_probs, entropy, values