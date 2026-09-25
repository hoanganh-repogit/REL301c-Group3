"""
Double DQN agent (van Hasselt et al., 2016), dùng kiến trúc Dueling DQN
(network.py) làm hàm xấp xỉ Q.

Công thức target của Double DQN (khác DQN thuần ở chỗ TÁCH việc chọn action
và đánh giá action ra 2 network riêng, để giảm overestimation bias):

    a* = argmax_a  Q_online(s', a)              # ONLINE network CHỌN action tốt nhất
    y  = r + gamma * (1 - done) * Q_target(s', a*)   # TARGET network ĐÁNH GIÁ action đó

So với DQN thuần:  y = r + gamma * (1-done) * max_a Q_target(s', a)
                    (1 network vừa chọn vừa đánh giá -> dễ overestimate)
"""
from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as  F
from src.envs.snake_env import ACTION_SPACE
from src.envs.state_encoder import STATE_DIM
from src.agents.dqn.network import DuelingQNetwork
from src.agents.dqn.replay_buffer import ReplayBuffer

@dataclass
class DQNConfig:
    lr: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 10000
    min_buffer_size_before_train: int = 500
    target_update_freq: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 20000
    hidden_dim: int = 128

class DoubleDQNAgent:
    def __init__(self, config: DQNConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = DuelingQNetwork(STATE_DIM, ACTION_SPACE, config.hidden_dim).to(self.device)
        self.target_net = DuelingQNetwork(STATE_DIM, ACTION_SPACE, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() # target network không train trực tiếp (không cần dropout/batchnorm behavior, nhưng để rõ ý
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size)

        self.total_env_steps = 0 # dùng cho epsilon decay VÀ target update freq

    # ------------------------------------------------------------------ #
    # Epsilon schedule
    # ------------------------------------------------------------------ #
    def epsilon(self) -> float:
        """Epsilon giảm TUYẾN TÍNH từ epsilon_start -> epsilon_end trong epsilon_decay_steps bước."""
        frac = min(1.0, self.total_env_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def reset_epsilon_schedule(self) -> None:
        """
        Reset lại bộ đếm bước -> epsilon quay về epsilon_start.

        Dùng ở Giai đoạn 3 (curriculum): mỗi khi agent lên level mới, tăng lại
        exploration để agent khám phá môi trường mới thay vì exploit ngay
        theo thói quen học được ở level cũ.
        """
        self.total_env_steps = 0

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Chọn action theo epsilon-greedy (hoặc thuần greedy nếu greedy=True, dùng lúc evaluation).

        Args:
            state: shape (STATE_DIM,)
            greedy: True -> luôn chọn argmax Q, bỏ qua epsilon (dùng khi evaluate, không train).

        Returns:
            action: int trong [0, ACTION_SPACE)
        """
        assert state.shape == (STATE_DIM,), f"act() nhận state sai shape: {state.shape}"

        if not greedy:
            self.total_env_steps += 1
            if random.random() < self.epsilon():
                return random.randrange(ACTION_SPACE)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, STATE_DIM)
            q_values = self.online_net(state_t)  # (1, ACTION_SPACE)
            action = int(q_values.argmax(dim=1).item())
        return action

    # ------------------------------------------------------------------ #
    # Store transition
    # ------------------------------------------------------------------ #
    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #

    def update(self) -> None:
        """
        1 bước gradient update trên 1 minibatch sample từ buffer.

        Returns:
            loss (float) nếu đã update, None nếu buffer chưa đủ transition.
        """
        min_needed = max(self.config.batch_size, self.config.min_buffer_size_before_train)
        if len(self.buffer) < min_needed:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size, self.device)
        # states:      (B, STATE_DIM)
        # actions:     (B,)
        # rewards:     (B,)
        # next_states: (B, STATE_DIM)
        # dones:       (B,)  0.0 hoặc 1.0

        with torch.no_grad():
            # a* = argmax_a Q_online(s', a)   -- ONLINE network chọn action
            next_q_online = self.online_net(next_states)  # (B, ACTION_SPACE)
            best_next_actions = next_q_online.argmax(dim=1)  # (B,)

            # Q_target(s', a*)                -- TARGET network đánh giá action đó
            next_q_target = self.target_net(next_states)  # (B, ACTION_SPACE)
            next_q_selected = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)  # (B,)

            # y = r + gamma * (1-done) * Q_target(s', a*)
            td_target = rewards + self.config.gamma * (1.0 - dones) * next_q_selected  # (B,)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        assert current_q.shape == td_target.shape == (self.config.batch_size,)

        loss = F.mse_loss(current_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_env_steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_env_steps": self.total_env_steps,
                "config": self.config.__dict__,
            },
            path,
        )
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_env_steps = checkpoint["total_env_steps"]


