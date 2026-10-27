"""

Check that reward signs match the design:
  - Eating food      -> reward > 0
  - Fatal collision  -> reward < 0
  - Normal step      -> reward <= 0 (small penalty that rewards efficiency)

Each test forces the situation instead of waiting for a random policy to reach
it, so the tests are fast and deterministic (independent of RNG luck).
"""

import os

from src.envs.snake_env import (
    REWARD_DEATH,
    REWARD_FARTHER,
    REWARD_FOOD,
    REWARD_STEP,
    SnakeEnv,
)
from src.utils.config_loader import EnvConfig, load_env_config

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "envs")


def _make_env(seed=0):
    config = load_env_config(os.path.join(CONFIG_DIR, "level1.yaml"))
    env = SnakeEnv(config, seed=seed)
    env.reset()
    return env


def test_reward_constants_sign():
    """Sanity check on the reward constants themselves, independent of step()."""
    assert REWARD_FOOD > 0, "The food reward must be positive"
    assert REWARD_DEATH < 0, "The death reward must be negative"
    assert REWARD_STEP <= 0, "The normal step reward must not be positive"


def test_eating_food_gives_positive_reward():
    env = _make_env(seed=1)
    env.snake_body = [(5, 5), (5, 4), (5, 3)]
    env.direction = "RIGHT"
    env.food_pos = (5, 6)

    # action=0 is STRAIGHT; the snake heads RIGHT, so it moves straight onto the food
    _, reward, done, info = env.step(0)

    assert reward == REWARD_FOOD
    assert done is False
    assert info["score"] == 1


def test_hitting_wall_gives_negative_reward():
    env = _make_env(seed=2)
    # Put the head at the right edge heading RIGHT -> the next step must hit the wall
    env.snake_body = [(5, env.width - 1), (5, env.width - 2), (5, env.width - 3)]
    env.direction = "RIGHT"
    env.food_pos = (0, 0)  # far away so it cannot be eaten by accident

    _, reward, done, info = env.step(0)  # STRAIGHT -> off the board

    assert reward == REWARD_DEATH
    assert done is True
    assert info["death_cause"] == "wall"


def test_hitting_self_gives_negative_reward():
    env = _make_env(seed=4)
    # Head at (5,5) heading UP; (4,5) is a MIDDLE body cell (not the tail).
    # Deliberately avoid the tail cell: entering the tail without eating is LEGAL
    # (the tail moves away in the same step; see the comment in step()).
    # Body order: head(5,5) - (5,4) - (4,4) - (4,5)[middle] - (4,6) - tail(5,6)
    env.snake_body = [(5, 5), (5, 4), (4, 4), (4, 5), (4, 6), (5, 6)]
    env.direction = "UP"
    env.food_pos = (0, 0)

    _, reward, done, info = env.step(0)  # action UP: head -> (4,5), a middle body cell

    assert reward == REWARD_DEATH
    assert done is True
    assert info["death_cause"] == "self"


def test_moving_into_own_tail_is_legal_when_not_eating():
    """Counterpart of the test above: entering the TAIL cell (not a middle cell)
    WITHOUT eating must be legal, because the tail moves away in the same step."""
    env = _make_env(seed=5)
    env.snake_body = [(5, 5), (5, 4), (4, 4), (4, 5)]  # current tail = (4,5)
    env.direction = "UP"
    env.food_pos = (0, 0)  # far away, so no food is eaten

    _, reward, done, info = env.step(0)  # action UP: head -> (4,5) = the old tail cell

    assert done is False
    assert reward != REWARD_DEATH
    assert info["death_cause"] is None


def test_normal_step_moves_one_cell_and_does_not_eat():
    """Regression test for the mistaken ``new_head = food_pos`` assignment in step()."""
    env = _make_env(seed=6)
    env.snake_body = [(5, 5), (5, 4), (5, 3)]
    env.direction = "RIGHT"
    env.food_pos = (0, 0)

    _, reward, done, info = env.step(0)

    assert env.snake_body == [(5, 6), (5, 5), (5, 4)]
    assert env.food_pos == (0, 0)
    assert reward == REWARD_STEP + REWARD_FARTHER
    assert done is False
    assert info["score"] == 0


def test_food_respawn_never_overlaps_updated_snake():
    env = _make_env(seed=7)
    env.snake_body = [(5, 5), (5, 4), (5, 3)]
    env.direction = "RIGHT"
    env.food_pos = (5, 6)

    env.step(0)

    assert env.food_pos not in env.snake_body
    assert env.food_pos not in env.obstacles


def test_food_spawn_works_when_only_one_cell_is_free():
    env = SnakeEnv(EnvConfig("nearly_full", 5, 5), seed=0)
    free_cell = (4, 4)
    env.snake_body = [
        (row, col)
        for row in range(5)
        for col in range(5)
        if (row, col) != free_cell
    ]
    assert env._sample_food_position() == free_cell


def test_reaching_level_target_finishes_episode_successfully():
    env = _make_env(seed=8)
    env.snake_body = [(5, 5), (5, 4), (5, 3)]
    env.direction = "RIGHT"
    env.food_pos = (5, 6)
    env.score = env.config.target_score - 1

    _, reward, done, info = env.step(0)

    assert reward == REWARD_FOOD
    assert done is True
    assert info["score"] == env.config.target_score
    assert info["death_cause"] == "target_reached"


def test_level4_has_no_score_limit():
    config = load_env_config(os.path.join(CONFIG_DIR, "level4.yaml"))
    assert config.target_score is None
    env = SnakeEnv(config, seed=9)
    env.reset()
    env.snake_body = [(10, 10), (10, 9), (10, 8)]
    env.direction = "RIGHT"
    env.food_pos = (10, 11)
    env.score = 999

    _, reward, done, info = env.step(0)

    assert reward == REWARD_FOOD
    assert done is False
    assert info["score"] == 1000


def test_no_food_timeout_is_truncation_not_death():
    env = _make_env(seed=10)
    env.snake_body = [(5, 5), (5, 4), (5, 3)]
    env.direction = "RIGHT"
    env.food_pos = (12, 12)
    env.steps_since_food = env.max_steps_without_food - 1

    _, _, done, info = env.step(0)

    assert done is True
    assert info["death_cause"] == "timeout"
    assert info["truncated"] is True


def test_collision_is_terminal_not_truncation():
    env = _make_env(seed=11)
    env.snake_body = [(5, 14), (5, 13), (5, 12)]
    env.direction = "RIGHT"
    env.food_pos = (12, 12)

    _, _, done, info = env.step(0)

    assert done is True
    assert info["death_cause"] == "wall"
    assert info["truncated"] is False
