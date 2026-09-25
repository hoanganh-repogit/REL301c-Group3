"""
strategies.py
--------------
3 chiến lược training, CÙNG 1 interface (BaseStrategy) để train_loop.py
dùng chung được, không cần biết chi tiết logic từng chiến lược:

    level_config = strategy.start_episode()      # trước mỗi episode: train trên level nào?
    ... chạy episode trên level_config ...
    result = strategy.end_episode(episode_score)  # sau mỗi episode: có leveled_up không?

Chỉ CurriculumStrategy có thể trả leveled_up=True — Fixed và RandomDR luôn
False, vì bản chất 2 chiến lược đó không có khái niệm "tiến trình" cần
reset epsilon/buffer giữa chừng.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from src.curriculum.level_manager import LevelManager
from src.utils.config_loader import EnvConfig


class BaseStrategy(ABC):
    name: str

    @abstractmethod
    def start_episode(self) -> EnvConfig:
        """Trả về EnvConfig của level sẽ dùng cho episode SẮP chạy."""

    @abstractmethod
    def end_episode(self, episode_score: int) -> dict:
        """Cập nhật trạng thái nội bộ sau khi 1 episode kết thúc.

        Returns dict tối thiểu gồm: {"leveled_up": bool, "mastered_level": str|None, "current_level": str}
        """

    @abstractmethod
    def known_levels_for_retention(self) -> list[EnvConfig]:
        """Danh sách level cần đưa vào retention evaluation định kỳ (Giai đoạn 3, bước 19)."""


class FixedStrategy(BaseStrategy):
    """Luôn train trên level KHÓ NHẤT trong tập train (level cuối cùng)."""

    name = "fixed"

    def __init__(self, train_levels: list[EnvConfig]) -> None:
        assert len(train_levels) >= 1
        self.level = train_levels[-1]

    def start_episode(self) -> EnvConfig:
        return self.level

    def end_episode(self, episode_score: int) -> dict:
        return {"leveled_up": False, "mastered_level": None, "current_level": self.level.name}

    def known_levels_for_retention(self) -> list[EnvConfig]:
        return [self.level]


class RandomDRStrategy(BaseStrategy):
    """Mỗi episode sample ngẫu nhiên (đều) 1 level trong tập train."""

    name = "random_dr"

    def __init__(self, train_levels: list[EnvConfig], seed: int | None = None) -> None:
        assert len(train_levels) >= 1
        self.train_levels = train_levels
        self.rng = random.Random(seed)
        self._current_level: EnvConfig | None = None

    def start_episode(self) -> EnvConfig:
        self._current_level = self.rng.choice(self.train_levels)
        return self._current_level

    def end_episode(self, episode_score: int) -> dict:
        assert self._current_level is not None, "end_episode() gọi trước start_episode()"
        return {"leveled_up": False, "mastered_level": None, "current_level": self._current_level.name}

    def known_levels_for_retention(self) -> list[EnvConfig]:
        # Random DR tiếp xúc đều cả 4 level train ngay từ đầu -> retention check trên toàn bộ
        return self.train_levels


class CurriculumStrategy(BaseStrategy):
    """Level-up tuần tự: đạt level_up_score trong 1 episode -> lên level tiếp theo.
    Không đạt -> tự động replay (episode tiếp theo vẫn ở level hiện tại)."""

    name = "curriculum"

    def __init__(self, train_levels: list[EnvConfig], level_up_score: int = 100) -> None:
        self.level_manager = LevelManager(train_levels, level_up_score)
        self._episode_counter = 0

    def start_episode(self) -> EnvConfig:
        return self.level_manager.current_level

    def end_episode(self, episode_score: int) -> dict:
        self._episode_counter += 1
        return self.level_manager.report_episode_result(episode_score, self._episode_counter)

    def known_levels_for_retention(self) -> list[EnvConfig]:
        # Chỉ những level ĐÃ đi qua tính tới thời điểm hiện tại -> retention check
        # đúng nghĩa "có quên level cũ không" khi đang học level mới.
        return self.level_manager.all_levels_reached_so_far()


_STRATEGY_REGISTRY = {
    "fixed": FixedStrategy,
    "random_dr": RandomDRStrategy,
    "curriculum": CurriculumStrategy,
}


def make_strategy(name: str, train_levels: list[EnvConfig], **kwargs) -> BaseStrategy:
    """Factory: tạo strategy theo tên, dùng trong run_experiment.py (Phase 4)."""
    assert name in _STRATEGY_REGISTRY, f"Strategy không hợp lệ: {name}. Chọn 1 trong {list(_STRATEGY_REGISTRY)}"
    return _STRATEGY_REGISTRY[name](train_levels, **kwargs)
