"""
PPO agent (Schulman et al., 2017), dùng ActorCriticNetwork (network.py) +
RolloutBuffer (buffer.py).

Loss tổng (mỗi minibatch):
    L(theta) = L_CLIP(theta) - c1 * L_VF(theta) + c2 * S[pi_theta]

    r_t(theta)   = exp( log pi_theta(a_t|s_t) - log pi_theta_old(a_t|s_t) )   -- probability ratio
    L_CLIP       = E_t[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ]
    L_VF         = MSE( V_theta(s_t), return_t )
    S[pi_theta]  = entropy trung bình của policy hiện tại (khuyến khích exploration)

Vì tối ưu bằng gradient DESCENT (Adam), code dùng loss = -L_CLIP + c1*L_VF - c2*S
(đổi dấu L_CLIP và S vì ta muốn MAXIMIZE chúng, nhưng optimizer chỉ MINIMIZE).
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.ppo.buffer import RolloutBuffer
from src.agents.ppo.network import ActorCriticNetwork
from src.envs.snake_env import ACTION_SPACE
from src.envs.state_encoder import STATE_DIM


@dataclass
class PPOConfig:
    lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    k_epochs: int = 4
    rollout_steps: int = 2048
    minibatch_size: int = 64
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    hidden_dim: int = 128


class PPOAgent:
    def __init__(self, config: PPOConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCriticNetwork(STATE_DIM, ACTION_SPACE, config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer(config.gamma, config.gae_lambda)

    # ------------------------------------------------------------------ #
    # Thu thập rollout
    # ------------------------------------------------------------------ #
    def act(self, state: np.ndarray) -> tuple[int, float, float]:
        """Sample 1 action từ policy hiện tại, dùng khi đang thu thập rollout.

        Returns:
            action:   int
            log_prob: float — log pi_theta_old(a|s), sẽ dùng làm "old_log_prob" trong ratio
            value:    float — V(s), dùng để tính GAE sau này
        """
        assert state.shape == (STATE_DIM,), f"act() nhận state sai shape: {state.shape}"
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, STATE_DIM)
        with torch.no_grad():
            action_t, log_prob_t, value_t = self.net.act(state_t)
        return int(action_t.item()), float(log_prob_t.item()), float(value_t.item())

    def act_greedy(self, state: np.ndarray) -> int:
        """Chọn action argmax theo policy — dùng lúc evaluation (không sample ngẫu nhiên)."""
        assert state.shape == (STATE_DIM,), f"act_greedy() nhận state sai shape: {state.shape}"
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = self.net(state_t)
            action = int(action_logits.argmax(dim=1).item())
        return action

    def store(self, state: np.ndarray, action: int, reward: float, done: bool, log_prob: float, value: float) -> None:
        self.buffer.add(state, action, reward, done, log_prob, value)

    def bootstrap_value(self, next_state: np.ndarray, done: bool) -> float:
        """V(s_T) dùng để bootstrap GAE khi rollout bị cắt giữa chừng episode.
        Nếu done=True thì trả về 0.0 (không có giá trị tương lai sau khi chết)."""
        if done:
            return 0.0
        state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value_t = self.net(state_t)
        return float(value_t.item())

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def update(self, last_value: float) -> dict[str, float]:
        """K epoch update trên toàn bộ rollout hiện có trong buffer, theo minibatch.

        Args:
            last_value: bootstrap value, xem RolloutBuffer.compute_returns_and_advantages().

        Returns:
            dict thống kê loss trung bình, để logging.
        """
        tensors = self.buffer.get_tensors(self.device, last_value)
        states = tensors["states"]              # (T, STATE_DIM)
        actions = tensors["actions"]              # (T,)
        old_log_probs = tensors["old_log_probs"]  # (T,)
        advantages = tensors["advantages"]        # (T,)  đã chuẩn hoá
        returns = tensors["returns"]              # (T,)

        T = states.shape[0]
        indices = np.arange(T)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

        for _ in range(self.config.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.config.minibatch_size):
                batch_idx = indices[start : start + self.config.minibatch_size]
                if len(batch_idx) < 2:
                    continue  # bỏ minibatch quá nhỏ (ví dụ 1 sample) để std() không lỗi ở nơi khác dùng advantage

                mb_states = states[batch_idx]
                mb_actions = actions[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_returns = returns[batch_idx]

                log_probs, entropy, values = self.net.evaluate_actions(mb_states, mb_actions)
                assert log_probs.shape == mb_old_log_probs.shape

                # r_t(theta) = exp(log pi_theta(a|s) - log pi_theta_old(a|s))
                ratio = torch.exp(log_probs - mb_old_log_probs)  # (batch,)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * mb_advantages
                # L_CLIP = E[min(surr1, surr2)] -- ta MUỐN maximize, nên trong loss lấy dấu âm
                policy_loss = -torch.min(surr1, surr2).mean()

                # L_VF = MSE(V_theta(s), return) -- muốn minimize trực tiếp, không cần đổi dấu
                value_loss = F.mse_loss(values, mb_returns)

                entropy_bonus = entropy.mean()  # muốn maximize entropy -> trừ nó khỏi loss

                loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_bonus.item())
                stats["n_updates"] += 1

        n = max(1, stats["n_updates"])
        stats["policy_loss"] /= n
        stats["value_loss"] /= n
        stats["entropy"] /= n

        self.buffer.reset()  # BẮT BUỘC: PPO on-policy, không giữ lại dữ liệu rollout cũ
        return stats

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
