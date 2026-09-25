"""
Replay buffer dạng ring-buffer đơn giản (dùng collections.deque với maxlen)
để lưu transition (s, a, r, s', done) và sample ngẫu nhiên thành minibatch.

Off-policy nên buffer có thể chứa transition thu thập từ NHIỀU episode/level
trước đó — đây chính là lý do cần cẩn thận khi chuyển level trong curriculum
(Giai đoạn 3): buffer cũ có thể lẫn kinh nghiệm từ level trước, gây nhiễu
khi vừa mới lên level mới.
"""
from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from src.envs.state_encoder import STATE_DIM

class Transition(NamedTuple):
    state: np.ndarray  # shape (STATE_DIM,)
    action: int
    reward: float
    next_state: np.ndarray  # shape (STATE_DIM,)
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push (self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        assert state.shape == (STATE_DIM,), f"state shape sai khi push vào buffer: {state.shape}"
        assert next_state.shape == (STATE_DIM,), f"next_state shape sai khi push vào buffer: {next_state.shape}"
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """
        Xoá toàn bộ buffer — dùng khi chuyển level trong curriculum để tránh
        kinh nghiệm level cũ áp đảo lúc vừa mới lên level mới (xem plan Giai đoạn 3).
        """
        self.buffer.clear()

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """
        Sample ngẫu nhiên 1 minibatch, trả về tensor sẵn sàng đưa vào network.

        Returns:
            states:      (batch_size, STATE_DIM) float32
            actions:     (batch_size,)            int64
            rewards:     (batch_size,)             float32
            next_states: (batch_size, STATE_DIM)  float32
            dones:       (batch_size,)             float32 (0.0 hoặc 1.0)
        """
        assert len(self.buffer) >= batch_size, (
            f"Buffer chỉ có {len(self.buffer)} transition, không đủ để sample batch_size={batch_size}"
        )

        batch = random.sample(self.buffer, batch_size)

        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=device)

        assert states.shape == (batch_size, STATE_DIM)
        assert next_states.shape == (batch_size, STATE_DIM)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert dones.shape == (batch_size,)

        return states, actions, rewards, next_states, dones





















