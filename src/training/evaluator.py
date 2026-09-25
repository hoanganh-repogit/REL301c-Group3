"""
Chạy N episode GREEDY (không update gradient, không epsilon-explore) trên
1 level, trả về thống kê score. Dùng chung cho:
  - Retention evaluation trong lúc train (Giai đoạn 3, bước 19)
  - Evaluation cuối cùng train/test trên cả 5 level (Giai đoạn 4)

Nhận `act_fn` (callable) thay vì nhận thẳng agent, để tách rời khỏi việc
agent là DQN (agent.act(state, greedy=True)) hay PPO (agent.act_greedy(state)) —
caller tự truyền lambda phù hợp.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.envs.snake_env import SnakeEnv
from src.utils.config_loader import EnvConfig


def evaluate_agent(
    act_fn: Callable[[np.ndarray], int],
    env_config: EnvConfig,
    n_episodes: int = 10,
    seed: int = 1234,
) -> dict:
    """
    Args:
        act_fn: hàm nhận state (STATE_DIM,) trả về action int, KHÔNG update gradient.
        env_config: level cần đánh giá.
        n_episodes: số episode chạy để lấy trung bình.
        seed: seed cố định -- đảm bảo evaluation TÁI LẬP ĐƯỢC giữa các lần gọi
            (khác với training dùng seed=None để có tính ngẫu nhiên).

    Returns:
        {"mean_score": float, "std_score": float, "scores": list[int]}
    """
    env = SnakeEnv(env_config, seed=seed)
    scores: list[int] = []

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)  # seed khác nhau mỗi episode nhưng vẫn tái lập được
        done = False
        info = {"score": 0}
        while not done:
            action = act_fn(state)
            state, reward, done, info = env.step(action)
        scores.append(info["score"])

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "scores": scores,
    }


def evaluate_on_levels(
    act_fn: Callable[[np.ndarray], int],
    levels: list[EnvConfig],
    n_episodes: int = 10,
    seed: int = 1234,
) -> list[dict]:
    """Đánh giá trên NHIỀU level, trả về list kết quả (mỗi phần tử kèm level.name)."""
    results = []
    for level in levels:
        result = evaluate_agent(act_fn, level, n_episodes=n_episodes, seed=seed)
        results.append({"level": level.name, **result})
    return results
