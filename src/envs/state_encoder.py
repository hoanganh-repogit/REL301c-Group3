"""
Encode the raw game state (snake position, heading, food, obstacles) into a
FIXED 27-dimensional feature vector that does not depend on board size.

This is what lets one network serve every level (15x15 to 36x36): the input
always has shape (27,), whatever the board size.

Vector layout (FIXED order; never change it once training has started):
    index 0-2  : danger [straight, right, left] relative to the heading
    index 3-6  : current heading, one-hot [UP, DOWN, LEFT, RIGHT]
    index 7-10 : food relative to the head [left, right, up, down]
    index 11-13: normalized ray clearance [straight, right, left]
    index 14-15: normalized food offset [row, col]
    index 16-17: normalized head position [row, col]
    index 18   : snake length over the number of usable cells
    index 19-21: safe-area ratio after [straight, right, left]
    index 22-25: tail relative to the head [left, right, up, down]
    index 26   : normalized Manhattan distance from head to tail
"""
from __future__ import annotations
import numpy as np

STATE_DIM = 27

# Movement vector of each heading: (delta_row, delta_col)
DIRECTION_VECTOR: dict[str, tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT":(0, 1),
}

"""
Clockwise order of headings, used to compute "relative left/right" from the
current heading.
"""
_CLOCKWISE_ODER = ["UP", "RIGHT", "DOWN", "LEFT"]

def _turn_relative(direction: str, turn: str) -> str:
    """
    Return the absolute heading after turning 'turn' (straight/right/left) from
    'direction'. Example: direction="UP", turn="right" -> "RIGHT"
    """
    idx = _CLOCKWISE_ODER.index(direction)
    if turn == "straight":
        return _CLOCKWISE_ODER[idx]
    elif turn == "right":
        return _CLOCKWISE_ODER[(idx + 1) % 4]
    elif turn == "left":
        return _CLOCKWISE_ODER[(idx - 1) % 4]
    raise ValueError(f"Unknown direction: {direction}")

def _is_collision(
    point: tuple[int, int],
    height: int,
    width: int,
    obstacles: set[tuple[int, int]],
    snake_body: list[tuple[int, int]],
    food_pos: tuple[int, int],
) -> bool:
    """
    True if moving to `point` is a collision: outside the board, on an obstacle,
    or on the snake's body.

    The current tail cell counts as free unless `point` is the food: without
    eating, the tail moves away in the same step, which matches SnakeEnv.step().
    """

    row, col = point
    if row < 0 or row >= height or col < 0 or col >= width:
        return True
    if point in obstacles:
        return True
    # Without eating, the tail leaves in the same step, so its cell is free.
    body_to_check = snake_body if point == food_pos else snake_body[:-1]
    if point in body_to_check:
        return True
    return False

def encode_state(snake_body: list[tuple[int, int]], direction: str, food_pos: tuple[int, int], height: int, width: int, obstacles: set[tuple[int, int]]) -> np.ndarray:
    """
    Encode the raw game state as a 27-dimensional vector.

    Args:
        snake_body: (row, col) coordinates; index 0 IS THE HEAD.
        direction: current heading, one of the keys of DIRECTION_VECTOR.
        food_pos: (row, col) of the food.
        height, width: current board size (differs per level).
        obstacles: static obstacle coordinates of the current level.

    Returns:
        np.ndarray shape (STATE_DIM,) = (27,), dtype float32.
    """
    assert direction in DIRECTION_VECTOR, f"Unknown direction: {direction}"
    assert len(snake_body) >= 1, "snake_body must contain at least the head"

    head = snake_body[0]

    # index 0-2: danger [straight, right, left]
    dir_straight = _turn_relative(direction, "straight")
    dir_right = _turn_relative(direction, "right")
    dir_left = _turn_relative(direction, "left")

    def _point_after(dir_name: str) -> tuple[int, int]:
        dr, dc = DIRECTION_VECTOR[dir_name]
        return (head[0] + dr, head[1] + dc)

    danger_straight = _is_collision(
        _point_after(dir_straight), height, width, obstacles, snake_body, food_pos
    )
    danger_right = _is_collision(
        _point_after(dir_right), height, width, obstacles, snake_body, food_pos
    )
    danger_left = _is_collision(
        _point_after(dir_left), height, width, obstacles, snake_body, food_pos
    )

    # index 3-6: current heading, one-hot
    dir_up = direction == "UP"
    dir_down = direction == "DOWN"
    dir_left_flag = direction == "LEFT"
    dir_right_flag = direction == "RIGHT"

    # index 7-10: food position relative to the head
    food_row, food_col = food_pos
    head_row, head_col = head
    food_left = food_col < head_col
    food_right = food_col > head_col
    food_up = food_row < head_row
    food_down = food_row > head_row

    # Free cells along a ray before the first collision. This separates "safe for
    # the next step" from "long corridor or dead end" on large boards.
    def _ray_clearance(dir_name: str) -> float:
        dr, dc = DIRECTION_VECTOR[dir_name]
        row, col = head
        free = 0
        occupied_body = set(snake_body[:-1])
        while True:
            row += dr
            col += dc
            point = (row, col)
            if not (0 <= row < height and 0 <= col < width):
                break
            if point in obstacles or point in occupied_body:
                break
            free += 1
        return free / float(max(height, width))

    clearance_straight = _ray_clearance(dir_straight)
    clearance_right = _ray_clearance(dir_right)
    clearance_left = _ray_clearance(dir_left)
    food_delta_row = (food_row - head_row) / float(max(1, height - 1))
    food_delta_col = (food_col - head_col) / float(max(1, width - 1))
    head_row_norm = head_row / float(max(1, height - 1))
    head_col_norm = head_col / float(max(1, width - 1))
    available_cells = max(1, height * width - len(obstacles))
    snake_length_ratio = len(snake_body) / float(available_cells)

    # Reachable area after each action. The BFS stops once the area is large
    # relative to the snake length, so dead ends are found without scanning the
    # whole board every step.
    def _safe_area_ratio(dir_name: str) -> float:
        candidate = _point_after(dir_name)
        if _is_collision(candidate, height, width, obstacles, snake_body, food_pos):
            return 0.0

        will_grow = candidate == food_pos
        next_body = [candidate, *snake_body]
        if not will_grow:
            next_body.pop()
        blocked = obstacles | set(next_body[1:])
        cap = min(available_cells, max(32, len(next_body) * 2))
        seen = {candidate}
        queue = [candidate]
        cursor = 0
        while cursor < len(queue) and len(seen) < cap:
            row, col = queue[cursor]
            cursor += 1
            for dr, dc in DIRECTION_VECTOR.values():
                point = (row + dr, col + dc)
                if (
                    0 <= point[0] < height
                    and 0 <= point[1] < width
                    and point not in blocked
                    and point not in seen
                ):
                    seen.add(point)
                    queue.append(point)
                    if len(seen) >= cap:
                        break
        return len(seen) / float(cap)

    safe_area_straight = _safe_area_ratio(dir_straight)
    safe_area_right = _safe_area_ratio(dir_right)
    safe_area_left = _safe_area_ratio(dir_left)

    tail_row, tail_col = snake_body[-1]
    tail_left = tail_col < head_col
    tail_right = tail_col > head_col
    tail_up = tail_row < head_row
    tail_down = tail_row > head_row
    tail_distance = (
        abs(tail_row - head_row) + abs(tail_col - head_col)
    ) / float(max(1, height + width - 2))

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
         food_down,
         clearance_straight,
         clearance_right,
         clearance_left,
         food_delta_row,
         food_delta_col,
         head_row_norm,
         head_col_norm,
         snake_length_ratio,
         safe_area_straight,
         safe_area_right,
         safe_area_left,
         tail_left,
         tail_right,
         tail_up,
         tail_down,
         tail_distance], dtype=np.float32,
    )

    assert state.shape == (STATE_DIM,) , f"Wrong state shape: {state.shape}, expected ({STATE_DIM},)"
    return state

















