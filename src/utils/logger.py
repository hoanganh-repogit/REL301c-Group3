"""
Ghi log dạng CSV, dùng chung cho cả DQN và PPO training loop:
  - EpisodeLogger:   1 dòng / episode (level, score, steps, death_cause, leveled_up, mastered_level)
  - RetentionLogger: 1 dòng / (episode checkpoint, level được eval) -- dùng đo catastrophic forgetting
"""

from __future__ import annotations
import os
import csv


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

class EpisodeLogger:
    COLUMNS = ["episode", "level", "score", "steps", "death_cause", "leveled_up", "mastered_level"]

    def __init__(self, path: str) -> None:
        self.path = path
        self.rows: list[dict] = []

    def log(
            self,
            episode: int,
            level: str,
            score: int,
            steps: int,
            death_cause: str | None,
            leveled_up: bool,
            mastered_level: str | None,
    ) -> None:
        self.rows.append(
            {
                "episode": episode,
                "level": level,
                "score": score,
                "steps": steps,
                "death_cause": death_cause if death_cause is not None else "",
                "leveled_up": leveled_up,
                "mastered_level": mastered_level if mastered_level is not None else "",
            }
        )

    def save(self) -> None:
        _ensure_parent(self.path)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)


class RetentionLogger:
    COLUMNS = ["episode", "level", "mean_score", "std_score"]

    def __init__(self, path: str) -> None:
        self.path = path
        self.rows: list[dict] = []

    def log(self, episode: int, level: str, mean_score: float, std_score: float) -> None:
        self.rows.append({"episode": episode, "level": level, "mean_score": mean_score, "std_score": std_score})

    def extend(self, rows: list[dict]) -> None:
        """Nhận list dict từ RetentionTracker.evaluate() (đã đúng key: episode/level/mean_score/std_score)."""
        self.rows.extend(rows)

    def save(self) -> None:
        _ensure_parent(self.path)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

