"""
Chuyển trạng thái "thô" của game (vị trí rắn, hướng đi, vị trí mồi, vật cản)
thành 1 feature vector CỐ ĐỊNH 11 chiều — không phụ thuộc kích thước bàn.

Đây là điểm mấu chốt để cùng 1 network có thể dùng chung cho cả 5 level
(10x10 tới 20x20), vì input luôn có shape (11,) bất kể bàn to hay nhỏ.

Bố cục vector (thứ tự CỐ ĐỊNH, không được đổi khi đã bắt đầu train):
    index 0-2  : danger [thẳng, phải, trái] tương đối theo hướng đang đi
    index 3-6  : hướng đang di chuyển, one-hot [UP, DOWN, LEFT, RIGHT]
    index 7-10 : vị trí mồi tương đối so với đầu rắn [trái, phải, trên, dưới]
"""
from __future__ import annotations
import numpy as np

STATE_DIM = 11 # hằng số dùng chung cho input_dim của mọi network (DQN/PPO)

# Vector di chuyển tương ứng mỗi hướng: (delta_row, delta_col)
DIRECTION_VECTOR: dict[str, tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT":(0, 1),
}

"""
Thứ tự xoay theo chiều kim đồng hồ — dùng để tính "trái/phải tương đối"
so với hướng đang đi hiện tại.
"""
_CLOCKWISE_ODER = ["UP", "RIGHT", "DOWN", "LEFT"]

def _turn_relative(direction: str, turn: str) -> str:
    """
    Trả về hướng tuyệt đối sau khi rẽ 'turn' (straight/right/left) từ 'direction'
    Ví dụ: direction="UP", turn="right" -> "RIGHT"
    """
    idx = _CLOCKWISE_ODER.index(direction)
    if turn == "straight":
        return _CLOCKWISE_ODER[idx]
    elif turn == "right":
        return _CLOCKWISE_ODER[(idx + 1) % 4]
    elif turn == "left":
        return _CLOCKWISE_ODER[(idx - 1) % 4]
    raise ValueError(f"Unknown direction: {direction}")

def _is_collision(point: tuple[int, int], height: int, width: int, obstacles: set[tuple[int, int]], snake_body: list[tuple[int, int]]) -> bool:
    """
    True nếu điểm `point` là va chạm: ra ngoài bàn, trúng vật cản, hoặc trúng thân rắn.

    Lưu ý (simplification có chủ đích): coi TOÀN BỘ thân rắn (kể cả ô đuôi)
    là nguy hiểm, kể cả khi ô đuôi sẽ di chuyển đi ở bước tiếp theo.
    Đây là cách đơn giản hoá phổ biến cho state 1-step-lookahead, có thể khiến
    agent hơi "thận trọng quá mức" ở 1 số tình huống hiếm — chấp nhận được
    cho phạm vi đồ án.
    """

    row, col = point
    if row < 0 or row >= height or col < 0 or col >= width:
        return True
    if point in obstacles:
        return True
    if point in snake_body:
        return True
    return False

def encode_state(snake_body: list[tuple[int, int]], direction: str, food_pos: tuple[int, int], height: int, width: int, obstacles: set[tuple[int, int]]) -> np.ndarray:
    """
    Encode trạng thái game thô thành vector 11 chiều.

    Args:
        snake_body: list toạ độ (row, col), index 0 LÀ ĐẦU RẮN.
        direction: hướng đang di chuyển hiện tại, một trong DIRECTION_VECTORS.
        food_pos: toạ độ (row, col) của mồi.
        height, width: kích thước bàn hiện tại (đổi theo từng level).
        obstacles: tập toạ độ vật cản tĩnh của level hiện tại.

    Returns:
        np.ndarray shape (STATE_DIM,) = (11,), dtype float32.
    """
    assert direction in DIRECTION_VECTOR, f"Unknown direction: {direction}"
    assert len(snake_body) >= 1, f"snake_boy phải có ít nhất 1 ô (đầu rắn)"

    head = snake_body[0]

    # index 0-2: danger [thẳng, phải, trái]
    dir_straight = _turn_relative(direction, "straight")
    dir_right = _turn_relative(direction, "right")
    dir_left = _turn_relative(direction, "left")

    def _point_after(dir_name: str) -> tuple[int, int]:
        dr, dc = DIRECTION_VECTOR[dir_name]
        return (head[0] + dr, head[1] + dc)

    danger_straight = _is_collision(_point_after(dir_straight), height, width, obstacles, snake_body)
    danger_right = _is_collision(_point_after(dir_right), height, width, obstacles, snake_body)
    danger_left = _is_collision(_point_after(dir_left), height, width, obstacles, snake_body)

    # index 3-6: hướng đang đi, one-hot
    dir_up = direction == "UP"
    dir_down = direction == "DOWN"
    dir_left_flag = direction == "LEFT"
    dir_right_flag = direction == "RIGHT"

    # index 7-10: vị trí mồi tương đối so với đầu rắn
    food_row, food_col = food_pos
    head_row, head_col = head
    food_left = food_col < head_col
    food_right = food_col > head_col
    food_up = food_row < head_row
    food_down = food_row > head_row

    state = np.array(
        [danger_straight,
         danger_right,
         danger_left,
         dir_up,
         dir_down,
         dir_left_flag,
         dir_right_flag,
         food_left,
         food_right,
         food_up,
         food_down,], dtype=np.float32,
    )

    assert state.shape == (STATE_DIM,) , f"State shape sai: {state.shape}, ky vong ({STATE_DIM},)"
    return state

















