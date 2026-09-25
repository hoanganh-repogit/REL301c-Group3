"""
Test THUẦN LOGIC (không cần SnakeEnv) cho LevelManager + 3 Strategy —
đây là phần quan trọng nhất của Phase 3, cần chắc chắn tuyệt đối:
  - Level-up đúng ngưỡng, đúng thứ tự
  - Level 5 (holdout) KHÔNG BAO GIỜ lọt vào bất kỳ strategy nào
  - leveled_up chỉ True ở CurriculumStrategy, không bao giờ ở Fixed/RandomDR
"""

import pytest

from src.curriculum.level_manager import LevelManager
from src.curriculum.strategies import (
    CurriculumStrategy,
    FixedStrategy,
    RandomDRStrategy,
    make_strategy,
)
from src.utils.config_loader import EnvConfig

# 4 level train giả lập, không cần load từ YAML thật cho unit test thuần logic
LEVEL1 = EnvConfig(name="L1", height=10, width=10, obstacles=[], is_holdout=False)
LEVEL2 = EnvConfig(name="L2", height=15, width=15, obstacles=[], is_holdout=False)
LEVEL3 = EnvConfig(name="L3", height=15, width=15, obstacles=[(4, 4)], is_holdout=False)
LEVEL4 = EnvConfig(name="L4", height=20, width=20, obstacles=[(5, 5)], is_holdout=False)
LEVEL5_HOLDOUT = EnvConfig(name="L5", height=20, width=20, obstacles=[(3, 10)], is_holdout=True)

TRAIN_LEVELS = [LEVEL1, LEVEL2, LEVEL3, LEVEL4]


# ---------------------------------------------------------------------- #
# LevelManager
# ---------------------------------------------------------------------- #
def test_level_manager_rejects_holdout_level():
    """LevelManager PHẢI raise lỗi nếu lỡ truyền nhầm level 5 vào — đây là lớp bảo vệ
    cuối cùng chống leak held-out set vào training."""
    with pytest.raises(AssertionError):
        LevelManager([LEVEL1, LEVEL5_HOLDOUT], level_up_score=100)


def test_level_manager_starts_at_level1():
    lm = LevelManager(TRAIN_LEVELS, level_up_score=100)
    assert lm.current_level.name == "L1"
    assert lm.is_at_max_level is False


def test_level_manager_levels_up_on_threshold():
    lm = LevelManager(TRAIN_LEVELS, level_up_score=100)
    result = lm.report_episode_result(episode_score=100, episode_idx=5)
    assert result["leveled_up"] is True
    assert result["mastered_level"] == "L1"
    assert result["current_level"] == "L2"
    assert lm.current_level.name == "L2"


def test_level_manager_does_not_level_up_below_threshold():
    lm = LevelManager(TRAIN_LEVELS, level_up_score=100)
    result = lm.report_episode_result(episode_score=99, episode_idx=5)
    assert result["leveled_up"] is False
    assert result["mastered_level"] is None
    assert result["current_level"] == "L1"
    assert lm.current_level.name == "L1"  # vẫn ở nguyên level cũ -> episode tiếp theo tự động replay


def test_level_manager_stops_at_max_level():
    """Đạt ngưỡng ở level CUỐI CÙNG (L4) thì không được lên nữa (không có level 5 nào để lên
    trong tập train — level 5 là held-out, không thuộc train_levels)."""
    lm = LevelManager(TRAIN_LEVELS, level_up_score=100)
    for _ in range(3):  # lên hết L1->L2->L3->L4
        lm.report_episode_result(episode_score=100, episode_idx=1)

    assert lm.current_level.name == "L4"
    assert lm.is_at_max_level is True

    result = lm.report_episode_result(episode_score=100, episode_idx=99)
    assert result["leveled_up"] is False  # đã max level, không lên nữa
    assert lm.current_level.name == "L4"


def test_level_manager_tracks_mastered_episode():
    lm = LevelManager(TRAIN_LEVELS, level_up_score=100)
    lm.report_episode_result(episode_score=100, episode_idx=42)
    assert lm.mastered_at_episode["L1"] == 42


def test_level_manager_all_levels_reached_so_far():
    lm = LevelManager(TRAIN_LEVELS, level_up_score=100)
    assert [lvl.name for lvl in lm.all_levels_reached_so_far()] == ["L1"]

    lm.report_episode_result(episode_score=100, episode_idx=1)  # -> L2
    assert [lvl.name for lvl in lm.all_levels_reached_so_far()] == ["L1", "L2"]


# ---------------------------------------------------------------------- #
# FixedStrategy
# ---------------------------------------------------------------------- #
def test_fixed_strategy_always_hardest_level():
    strategy = FixedStrategy(TRAIN_LEVELS)
    for _ in range(10):
        level = strategy.start_episode()
        assert level.name == "L4"
        result = strategy.end_episode(episode_score=100)  # dù đạt 100 vẫn không leveled_up
        assert result["leveled_up"] is False


def test_fixed_strategy_retention_levels_is_single_level():
    strategy = FixedStrategy(TRAIN_LEVELS)
    assert [lvl.name for lvl in strategy.known_levels_for_retention()] == ["L4"]


# ---------------------------------------------------------------------- #
# RandomDRStrategy
# ---------------------------------------------------------------------- #
def test_random_dr_strategy_samples_from_train_levels_only():
    strategy = RandomDRStrategy(TRAIN_LEVELS, seed=0)
    seen_names = set()
    for _ in range(200):
        level = strategy.start_episode()
        seen_names.add(level.name)
        result = strategy.end_episode(episode_score=100)
        assert result["leveled_up"] is False  # RandomDR không bao giờ leveled_up

    assert seen_names == {"L1", "L2", "L3", "L4"}  # đủ 200 lần thì kỳ vọng thấy cả 4 level
    assert "L5" not in seen_names


def test_random_dr_strategy_reproducible_with_same_seed():
    s1 = RandomDRStrategy(TRAIN_LEVELS, seed=42)
    s2 = RandomDRStrategy(TRAIN_LEVELS, seed=42)
    sequence1 = [s1.start_episode().name for _ in range(20)]
    sequence2 = [s2.start_episode().name for _ in range(20)]
    assert sequence1 == sequence2


def test_random_dr_strategy_retention_levels_is_all_train_levels():
    strategy = RandomDRStrategy(TRAIN_LEVELS, seed=0)
    assert {lvl.name for lvl in strategy.known_levels_for_retention()} == {"L1", "L2", "L3", "L4"}


# ---------------------------------------------------------------------- #
# CurriculumStrategy
# ---------------------------------------------------------------------- #
def test_curriculum_strategy_progresses_through_levels():
    strategy = CurriculumStrategy(TRAIN_LEVELS, level_up_score=100)

    assert strategy.start_episode().name == "L1"
    result = strategy.end_episode(episode_score=50)  # chưa đủ ngưỡng
    assert result["leveled_up"] is False
    assert strategy.start_episode().name == "L1"  # vẫn L1 -> tự động replay

    result = strategy.end_episode(episode_score=100)  # đủ ngưỡng
    assert result["leveled_up"] is True
    assert result["mastered_level"] == "L1"
    assert strategy.start_episode().name == "L2"  # đã lên L2


def test_curriculum_strategy_retention_only_includes_reached_levels():
    strategy = CurriculumStrategy(TRAIN_LEVELS, level_up_score=100)
    strategy.start_episode()
    assert [lvl.name for lvl in strategy.known_levels_for_retention()] == ["L1"]

    strategy.end_episode(episode_score=100)  # -> L2
    assert [lvl.name for lvl in strategy.known_levels_for_retention()] == ["L1", "L2"]
    # L3, L4 CHƯA từng đi qua -> KHÔNG được đưa vào retention check


def test_curriculum_strategy_never_touches_holdout_level_even_after_full_progression():
    strategy = CurriculumStrategy(TRAIN_LEVELS, level_up_score=100)
    for _ in range(10):  # thừa số lần để chắc chắn đã lên hết mức
        strategy.start_episode()
        strategy.end_episode(episode_score=100)

    all_seen = {lvl.name for lvl in strategy.known_levels_for_retention()}
    assert all_seen == {"L1", "L2", "L3", "L4"}
    assert "L5" not in all_seen


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #
def test_make_strategy_factory():
    assert isinstance(make_strategy("fixed", TRAIN_LEVELS), FixedStrategy)
    assert isinstance(make_strategy("random_dr", TRAIN_LEVELS, seed=0), RandomDRStrategy)
    assert isinstance(make_strategy("curriculum", TRAIN_LEVELS, level_up_score=100), CurriculumStrategy)

    with pytest.raises(AssertionError):
        make_strategy("khong_ton_tai", TRAIN_LEVELS)
