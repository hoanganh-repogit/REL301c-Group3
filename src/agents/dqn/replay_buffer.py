"""
A simple ring-buffer replay memory (collections.deque with maxlen) that stores
(s, a, r, s', done) transitions and samples random minibatches.

DQN is off-policy, so the buffer holds transitions from MANY past episodes. The
`level` field only records which level a transition came from; it does not
affect sampling.
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
    level: str

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        level: str = "default",
    ) -> None:
        assert state.shape == (STATE_DIM,), f"wrong state shape pushed to buffer: {state.shape}"
        assert next_state.shape == (STATE_DIM,), f"wrong next_state shape pushed to buffer: {next_state.shape}"
        self.buffer.append(Transition(state, action, reward, next_state, done, level))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Remove all transitions; used before restoring from a checkpoint."""
        self.buffer.clear()

    def state_dict(self) -> dict:
        """Serialize transitions compactly for a resumable training checkpoint."""
        transitions = list(self.buffer)
        if not transitions:
            return {"capacity": self.capacity, "size": 0}
        return {
            "capacity": self.capacity,
            "size": len(transitions),
            "states": torch.from_numpy(np.stack([t.state for t in transitions])).float(),
            "actions": torch.tensor([t.action for t in transitions], dtype=torch.int64),
            "rewards": torch.tensor([t.reward for t in transitions], dtype=torch.float32),
            "next_states": torch.from_numpy(
                np.stack([t.next_state for t in transitions])
            ).float(),
            "dones": torch.tensor([t.done for t in transitions], dtype=torch.bool),
            "levels": [t.level for t in transitions],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the newest transitions that fit this buffer's capacity."""
        self.clear()
        size = int(state.get("size", 0))
        if size == 0:
            return

        states = state["states"].cpu().numpy()
        actions = state["actions"].cpu().numpy()
        rewards = state["rewards"].cpu().numpy()
        next_states = state["next_states"].cpu().numpy()
        dones = state["dones"].cpu().numpy()
        levels = state.get("levels", ["default"] * size)
        start = max(0, size - self.capacity)
        for index in range(start, size):
            self.push(
                states[index].copy(),
                int(actions[index]),
                float(rewards[index]),
                next_states[index].copy(),
                bool(dones[index]),
                str(levels[index]),
            )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """
        Sample a random minibatch as tensors ready for the network.

        Returns:
            states:      (batch_size, STATE_DIM) float32
            actions:     (batch_size,)            int64
            rewards:     (batch_size,)             float32
            next_states: (batch_size, STATE_DIM)  float32
            dones:       (batch_size,)             float32 (0.0 or 1.0)
        """
        assert len(self.buffer) >= batch_size, (
            f"Buffer holds only {len(self.buffer)} transitions, not enough for batch_size={batch_size}"
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





















