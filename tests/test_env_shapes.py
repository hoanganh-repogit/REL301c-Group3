"""
Kiểm tra: state luôn có shape cố định (STATE_DIM,) trên CẢ 6 level,
bất kể kích thước bàn khác nhau. Đây là điều kiện tiên quyết để 1 network
dùng chung được cho mọi level.
"""

import glob
import os
from collections import deque

import numpy as np
import pytest

from src.envs.snake_env import ACTION_SPACE, SnakeEnv
from src.envs.state_encoder import STATE_DIM
from src.utils.config_loader import load_all_levels, load_env_config

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "envs")
ALL_LEVEL_FILES = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))


@pytest.mark.parametrize("config_path", ALL_LEVEL_FILES)
def test_reset_state_shape(config_path):
    config = load_env_config(config_path)
    env = SnakeEnv(config, seed=0)
    state = env.reset()
    assert state.shape == (STATE_DIM,), f"{config.name}: shape sai sau reset()"
    assert state.dtype == np.float32


@pytest.mark.parametrize("config_path", ALL_LEVEL_FILES)
def test_step_state_shape(config_path):
    config = load_env_config(config_path)
    env = SnakeEnv(config, seed=0)
    env.reset()
    for action in range(ACTION_SPACE):
        state, reward, done, info = env.step(action)
        assert state.shape == (STATE_DIM,), f"{config.name}: shape sai sau step(action={action})"
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "score" in info and "death_cause" in info
        if done:
            env.reset()


def test_all_six_levels_present():
    """Đảm bảo đúng 6 file level tồn tại (không bị thiếu/thừa khi tổ chức lại config)."""
    names = {load_env_config(p).name for p in ALL_LEVEL_FILES}
    expected = {"level1", "level2", "level3", "level4", "level5_holdout", "level6"}
    assert names == expected, f"Thiếu/thừa level: {expected.symmetric_difference(names)}"


def test_level5_is_marked_holdout():
    """Level 5 PHẢI được đánh dấu is_holdout=True — sai cấu hình này sẽ làm hỏng
    toàn bộ thí nghiệm generalization (leak dữ liệu test vào train)."""
    level5_path = os.path.join(CONFIG_DIR, "level5_holdout.yaml")
    config = load_env_config(level5_path)
    assert config.is_holdout is True

    for path in ALL_LEVEL_FILES:
        if path != level5_path:
            config = load_env_config(path)
            assert config.is_holdout is False, f"{config.name} không phải level 5 nhưng lại is_holdout=True"


def test_load_all_levels_returns_all_configs():
    levels = load_all_levels(CONFIG_DIR)
    assert set(levels) == {
        "level1", "level2", "level3", "level4", "level5_holdout", "level6"
    }


@pytest.mark.parametrize("config_path", ALL_LEVEL_FILES)
def test_all_free_cells_are_connected(config_path):
    config = load_env_config(config_path)
    blocked = set(config.obstacles)
    start = next(
        (row, col)
        for row in range(config.height)
        for col in range(config.width)
        if (row, col) not in blocked
    )
    seen = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            point = (row + delta_row, col + delta_col)
            if (
                0 <= point[0] < config.height
                and 0 <= point[1] < config.width
                and point not in blocked
                and point not in seen
            ):
                seen.add(point)
                queue.append(point)

    free_cells = config.height * config.width - len(blocked)
    assert len(seen) == free_cells, (
        f"{config.name}: chỉ kết nối {len(seen)}/{free_cells} ô trống"
    )


def test_timeout_counter_resets_after_eating_food():
    config = load_env_config(os.path.join(CONFIG_DIR, "level1.yaml"))
    env = SnakeEnv(config, seed=0)
    env.reset()
    env.snake_body = [(5, 5), (5, 4), (5, 3)]
    env.direction = "RIGHT"
    env.food_pos = (5, 6)
    env.steps_since_food = 10

    _, reward, done, info = env.step(0)  # STRAIGHT while facing RIGHT

    assert reward > 0
    assert done is False
    assert info["score"] == 1
    assert env.steps_since_food == 0
