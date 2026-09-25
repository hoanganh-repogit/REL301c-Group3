"""
Kiểm tra: state luôn có shape cố định (STATE_DIM,) trên CẢ 5 level,
bất kể kích thước bàn khác nhau. Đây là điều kiện tiên quyết để 1 network
dùng chung được cho mọi level.
"""

import glob
import os

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


def test_all_five_levels_present():
    """Đảm bảo đúng 5 file level tồn tại (không bị thiếu/thừa khi tổ chức lại config)."""
    names = {load_env_config(p).name for p in ALL_LEVEL_FILES}
    expected = {"level1", "level2", "level3", "level4", "level5_holdout"}
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
    assert set(levels) == {"level1", "level2", "level3", "level4", "level5_holdout"}
