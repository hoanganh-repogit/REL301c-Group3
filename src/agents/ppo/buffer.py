"""
Rollout buffer: lưu 1 đoạn trajectory (on-policy, thu thập bằng chính policy
hiện tại) rồi tính GAE (Generalized Advantage Estimation, Schulman et al. 2016):

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}     (đệ quy ngược từ cuối trajectory)
    return_t = A_t + V(s_t)          -- dùng làm target cho critic loss (L_VF)

Vì PPO là on-policy, buffer này PHẢI được reset() sau mỗi lần update() —
không giữ lại dữ liệu cũ như replay buffer của DQN.
"""
from __future__ import annotations
import numpy as np
import torch
from src.envs.state_encoder import STATE_DIM

class RolloutBuffer:
    def __init__(self, gamma: float, gae_lambda: float) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        assert state.shape == (STATE_DIM,), f"state shape sai khi add vào rollout buffer: {state.shape}"
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.states)

    def compute_returns_and_advantages(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """Tính GAE advantage và return cho toàn bộ trajectory đang lưu.

        Args:
            last_value: V(s_T) — giá trị bootstrap của state NGAY SAU bước cuối cùng
                trong buffer (dùng khi rollout bị cắt giữa chừng episode, chưa done).
                Nếu bước cuối done=True thì giá trị này không ảnh hưởng (bị nhân với (1-done)=0).

        Returns:
            advantages: np.ndarray shape (T,)
            returns:    np.ndarray shape (T,)   = advantages + values  (target cho critic)
        """
        T = len(self.rewards)
        assert T > 0, "Rollout buffer rỗng, không thể tính GAE"

        values_extended = self.values + [last_value]  # length T+1, values_extended[t+1] = V(s_{t+1})
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            not_done = 1.0 - float(self.dones[t])
            # delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)
            delta = self.rewards[t] + self.gamma * values_extended[t + 1] * not_done - values_extended[t]
            # A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        assert advantages.shape == (T,)
        assert returns.shape == (T,)
        return advantages, returns

    def get_tensors(self, device: torch.device, last_value: float) -> dict[str, torch.Tensor]:
        """Chuyển toàn bộ buffer thành tensor, kèm advantages/returns đã tính sẵn.

        Args:
            last_value: V(s_T) bootstrap — TRUYỀN TƯỜNG MINH từ training loop, bằng
                0.0 nếu bước cuối buffer done=True, hoặc bằng critic(state_hiện_tại)
                nếu rollout bị cắt giữa chừng episode (xem cách gọi trong agent.py).

        Advantage được CHUẨN HOÁ (trừ mean, chia std) — bước này giúp training PPO
        ổn định hơn đáng kể, là thực hành chuẩn trong hầu hết implementation PPO.
        """
        T = len(self)
        advantages, returns = self.compute_returns_and_advantages(last_value=last_value)

        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        tensors = {
            "states": torch.tensor(np.stack(self.states), dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.int64, device=device),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "advantages": torch.tensor(advantages_norm, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
        }

        assert tensors["states"].shape == (T, STATE_DIM)
        assert tensors["actions"].shape == (T,)
        assert tensors["old_log_probs"].shape == (T,)
        assert tensors["advantages"].shape == (T,)
        assert tensors["returns"].shape == (T,)
        return tensors