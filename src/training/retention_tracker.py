"""
Cứ mỗi `eval_every_episodes` episode, tạm dừng train, chạy vài episode
đánh giá (greedy, không update) trên TỪNG level mà strategy đã "biết" tới
thời điểm hiện tại (strategy.known_levels_for_retention()), log lại score
trung bình -> dùng vẽ "retention curve" ở Giai đoạn 5, phát hiện agent có
quên level cũ khi đang tập trung học level mới hay không.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.training.evaluator import evaluate_agent
from src.utils.config_loader import EnvConfig


class RetentionTracker:
    def __init__(self, eval_every_episodes: int = 200, n_eval_episodes: int = 10, eval_seed: int = 999) -> None:
        self.eval_every_episodes = eval_every_episodes
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed

    def should_evaluate(self, episode_idx: int) -> bool:
        return episode_idx % self.eval_every_episodes == 0

    def maybe_evaluate(
        self,
        episode_idx: int,
        levels: list[EnvConfig],
        act_fn: Callable[[np.ndarray], int],
    ) -> list[dict] | None:
        """Trả về list dict {"episode","level","mean_score","std_score"} nếu tới mốc đánh giá, else None."""
        if not self.should_evaluate(episode_idx):
            return None

        rows = []
        for level in levels:
            result = evaluate_agent(act_fn, level, n_episodes=self.n_eval_episodes, seed=self.eval_seed)
            rows.append(
                {
                    "episode": episode_idx,
                    "level": level.name,
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                }
            )
        return rows
