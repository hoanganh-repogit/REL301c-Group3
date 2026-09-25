"""
Logic lõi của cơ chế "Curriculum Level-up": agent bắt đầu từ level đầu tiên,
đạt >= level_up_score trong 1 episode DUY NHẤT thì lên level tiếp theo.
Điểm/độ thạo level cũ được giữ nguyên (không cần xử lý gì đặc biệt, vì
LevelManager chỉ tăng current_idx chứ không "quên" level cũ).

Đây là class thuần logic, KHÔNG phụ thuộc SnakeEnv hay bất kỳ agent nào —
để dễ unit test độc lập.
"""

from __future__ import annotations

from src.utils.config_loader import EnvConfig


class LevelManager:
    def __init__(self, train_levels: list[EnvConfig], level_up_score: int = 100) -> None:
        assert len(train_levels) >= 1, "Cần ít nhất 1 level để train"
        assert all(not lvl.is_holdout for lvl in train_levels), (
            "LevelManager chỉ được nhận TRAIN levels — phát hiện 1 level có is_holdout=True. "
            "Đây là lỗi cấu hình nghiêm trọng (leak held-out set vào training)."
        )
        self.train_levels = train_levels
        self.level_up_score = level_up_score
        self.current_idx = 0
        # episode_idx (1-indexed) tại đó mỗi level ĐẦU TIÊN đạt ngưỡng level-up
        self.mastered_at_episode: dict[str, int] = {}

    @property
    def current_level(self) -> EnvConfig:
        return self.train_levels[self.current_idx]

    @property
    def is_at_max_level(self) -> bool:
        return self.current_idx == len(self.train_levels) - 1

    def all_levels_reached_so_far(self) -> list[EnvConfig]:
        """Toàn bộ level từ đầu tới level hiện tại — dùng cho retention evaluation
        (kiểm tra agent có quên level cũ khi đang học level mới không)."""
        return self.train_levels[: self.current_idx + 1]

    def report_episode_result(self, episode_score: int, episode_idx: int) -> dict:
        """Gọi sau khi 1 episode kết thúc. Trả về dict thông tin để logging.

        Args:
            episode_score: số mồi ăn được trong episode vừa kết thúc.
            episode_idx: chỉ số episode (1-indexed, dùng để log mốc thời gian mastered).

        Returns:
            {
                "leveled_up": bool,
                "mastered_level": str | None,   # tên level VỪA được master (None nếu không lên level)
                "current_level": str,           # tên level SAU khi xử lý (đã tăng nếu leveled_up=True)
            }
        """
        leveled_up = False
        mastered_level_name = None

        if episode_score >= self.level_up_score and not self.is_at_max_level:
            mastered_level_name = self.current_level.name
            self.mastered_at_episode[mastered_level_name] = episode_idx
            self.current_idx += 1
            leveled_up = True

        return {
            "leveled_up": leveled_up,
            "mastered_level": mastered_level_name,
            "current_level": self.current_level.name,
        }
