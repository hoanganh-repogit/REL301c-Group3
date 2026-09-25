"""

Kiểm tra dấu reward đúng theo thiết kế:
  - Ăn mồi        -> reward > 0
  - Va chạm chết  -> reward < 0
  - Bước bình thường -> reward <= 0 (phạt nhẹ, khuyến khích hiệu quả)

Dùng cách "ép" tình huống thay vì chờ random policy tự nhiên gặp phải,
để test chạy nhanh và ổn định (không phụ thuộc may rủi của rng).
"""

import os

from src.envs.snake_env import REWARD_DEATH, REWARD_FOOD, REWARD_STEP, SnakeEnv
from src.utils.config_loader import load_env_config

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "envs")


def _make_env(seed=0):
    config = load_env_config(os.path.join(CONFIG_DIR, "level1.yaml"))
    env = SnakeEnv(config, seed=seed)
    env.reset()
    return env


def test_reward_constants_sign():
    """Sanity check trên chính các hằng số reward, độc lập với logic step()."""
    assert REWARD_FOOD > 0, "Reward khi ăn mồi phải dương"
    assert REWARD_DEATH < 0, "Reward khi chết phải âm"
    assert REWARD_STEP <= 0, "Reward mỗi bước bình thường không được dương"


def test_eating_food_gives_positive_reward():
    env = _make_env(seed=1)
    # Ép vị trí mồi ngay trước đầu rắn (hướng RIGHT) để bước tiếp theo chắc chắn ăn được
    head_r, head_c = env.snake_body[0]
    env.food_pos = (head_r, head_c + 1)

    # action=3 tương ứng RIGHT (đúng hướng đang đi) -> đi thẳng vào mồi
    _, reward, done, info = env.step(3)

    assert reward == REWARD_FOOD
    assert done is False
    assert info["score"] == 1


def test_hitting_wall_gives_negative_reward():
    env = _make_env(seed=2)
    # Đưa đầu rắn ra sát biên phải, hướng RIGHT -> bước tiếp theo chắc chắn đâm tường
    env.snake_body = [(5, env.width - 1), (5, env.width - 2), (5, env.width - 3)]
    env.direction = "RIGHT"
    env.food_pos = (0, 0)  # đặt xa để chắc chắn không vô tình ăn trúng

    _, reward, done, info = env.step(3)  # RIGHT -> ra ngoài bàn

    assert reward == REWARD_DEATH
    assert done is True
    assert info["death_cause"] == "wall"


def test_hitting_self_gives_negative_reward():
    env = _make_env(seed=4)
    # Đầu rắn tại (5,5), hướng UP; (4,5) là 1 đoạn thân GIỮA (không phải đuôi) —
    # cố ý tránh nhắm vào ô đuôi, vì va vào đuôi khi không ăn mồi là HỢP LỆ
    # (đuôi sẽ dịch chuyển đi trong cùng bước đó, xem comment trong step()).
    # Thứ tự thân: đầu(5,5) - (5,4) - (4,4) - (4,5)[giữa] - (4,6) - đuôi(5,6)
    env.snake_body = [(5, 5), (5, 4), (4, 4), (4, 5), (4, 6), (5, 6)]
    env.direction = "UP"
    env.food_pos = (0, 0)

    _, reward, done, info = env.step(0)  # action UP: đầu -> (4,5), trùng đoạn thân giữa

    assert reward == REWARD_DEATH
    assert done is True
    assert info["death_cause"] == "self"


def test_moving_into_own_tail_is_legal_when_not_eating():
    """Đối chứng cho test trên: va vào Ô ĐUÔI (không phải đoạn giữa) khi KHÔNG ăn mồi
    phải là hợp lệ, vì đuôi dịch chuyển đi trong cùng bước đó."""
    env = _make_env(seed=5)
    env.snake_body = [(5, 5), (5, 4), (4, 4), (4, 5)]  # đuôi hiện tại = (4,5)
    env.direction = "UP"
    env.food_pos = (0, 0)  # đặt xa, chắc chắn không ăn mồi

    _, reward, done, info = env.step(0)  # action UP: đầu -> (4,5) = đúng ô đuôi cũ

    assert done is False
    assert reward == REWARD_STEP
    assert info["death_cause"] is None


def test_normal_step_moves_one_cell_and_does_not_eat():
    """Hồi quy cho lỗi gán nhầm ``new_head = food_pos`` trong step()."""
    env = _make_env(seed=6)
    env.snake_body = [(5, 5), (5, 4), (5, 3)]
    env.direction = "RIGHT"
    env.food_pos = (0, 0)

    _, reward, done, info = env.step(3)

    assert env.snake_body == [(5, 6), (5, 5), (5, 4)]
    assert env.food_pos == (0, 0)
    assert reward == REWARD_STEP
    assert done is False
    assert info["score"] == 0


def test_food_respawn_never_overlaps_updated_snake():
    env = _make_env(seed=7)
    env.snake_body = [(5, 5), (5, 4), (5, 3)]
    env.direction = "RIGHT"
    env.food_pos = (5, 6)

    env.step(3)

    assert env.food_pos not in env.snake_body
    assert env.food_pos not in env.obstacles
