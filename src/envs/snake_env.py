"""
Môi trường Snake tham số hoá theo level (EnvConfig): kích thước bàn và
vật cản đều đọc từ config, không hard-code — để cùng 1 class dùng được
cho cả 5 level (Giai đoạn 1 của kế hoạch).

Interface theo phong cách Gym cổ điển: reset() -> state, step(action) -> (state, reward, done, info)
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from src.envs.state_encoder import (DIRECTION_VECTOR, STATE_DIM, encode_state)
from src.utils.config_loader import EnvConfig


# Action space: 4 hướng tuyệt đối
ACTION_SPACE = 4
ACTION_TO_DIRECTION = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}

"""
Hướng đối lập — dùng để chặn agent tự đảo chiều 180 độ tức thời
(hành vi chuẩn trong hầu hết game Snake: bấm hướng ngược lại bị bỏ qua,
không phải là chết ngay).
"""
_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# Hệ số reward — CỐ ĐỊNH, dùng chung cho mọi level để so sánh công bằng
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.01 # phạt nhẹ mỗi bước, khuyến khích agent đi hiệu quả

"""
Giới hạn số bước tối đa 1 episode (an toàn, tránh vòng lặp vô hạn khi
agent chưa học được gì, ví dụ với random policy lúc sanity-check).
"""
_MAX_STEPS_PER_CELL = 100 # số bước tối đa = số ô bàn cờ * hệ số này

class SnakeEnv:
    """Môi trường Snake, tham số hoá theo EnvConfig (1 level)."""

    def __init__(self, config: EnvConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self.height = config.height
        self.width = config.width
        self.obstacles: set[tuple[int, int]] = set(config.obstacles)
        self.max_steps = self.height * self.width * _MAX_STEPS_PER_CELL // 100 + 200

        self.rng = np.random.RandomState(seed=seed)

        # Các thuộc tính này được gán thật trong reset(), khai báo trước để rõ kiểu dữ liệu
        self.snake_body: list[tuple[int, int]] = []
        self.direction: str = "RIGHT"
        self.food_ops: tuple[int, int] = (0, 0)
        self.score: int = 0
        self.steps: int = 0

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Bắt đầu episode mới. Trả về state ban đầu, shape (STATE_DIM,)."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.snake_body = self._sample_valid_start()
        self.direction = "RIGHT"
        self.score = 0
        self.steps = 0
        self.food_pos = self._sample_food_position()

        state = encode_state(
            snake_body=self.snake_body,
            direction=self.direction,
            food_pos=self.food_pos,
            height=self.height,
            width=self.width,
            obstacles=self.obstacles,
        )
        assert state.shape == (STATE_DIM,), f"reset() trả state sai shape: {state.shape}"
        return state

    def _sample_valid_start(self) -> list[tuple[int, int]]:
        """
        Tìm vị trí khởi tạo rắn dài 3 ô hợp lệ (không trúng vật cản/ngoài bàn).

        Rắn khởi tạo hướng RIGHT nên thân nằm bên TRÁI đầu:
        body = [head, head-1_col, head-2_col]
        """
        for _ in range(1000):
            r = self.rng.randint(1, self.height - 1)
            c = self.rng.randint(3, self.width) # cần c-2 >= 0 => c >= 2, lấy dư ra 1 chút cho an toàn
            candidate = [(r, c), (r, c - 1), (r, c - 2)]
            if all(0 <= cc < self.width for (_, cc) in candidate) and not any(p in self.obstacles for p in candidate):
                return candidate
        raise RuntimeError(
            f"[{self.config.name}] Không tìm được vị trí khởi tạo hợp lệ cho rắn "
            "sau 1000 lần thử — kiểm tra lại obstacles có chiếm quá nhiều bàn cờ không."
        )

    def _sample_food_position(self) -> tuple[int, int]:
        """Sample vị trí mồi ngẫu nhiên, không trùng thân rắn / vật cản."""
        occupied = self.obstacles | set(self.snake_body)
        for _ in range(1000):
            r = self.rng.randint(0, self.height)
            c = self.rng.randint(0, self.width)
            if (r, c) not in occupied:
                return (r, c)
        raise RuntimeError(
            f"[{self.config.name}] Không tìm được vị trí mồi hợp lệ — bàn cờ có thể đã đầy."
        )

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Thực hiện 1 action.

        Args:
            action: int trong {0,1,2,3}, tra bằng ACTION_TO_DIRECTION.

        Returns:
            (state, reward, done, info) — state shape (STATE_DIM,).
            info gồm: score (số mồi đã ăn), death_cause (None nếu chưa chết).
        """
        assert action in ACTION_TO_DIRECTION, f"action không hợp lệ: {action}"
        requested_dir = ACTION_TO_DIRECTION[action]
        # Chặn đảo chiều 180 độ tức thời khi rắn dài hơn 1 ô
        if requested_dir == _OPPOSITE[self.direction] and len(self.snake_body):
            actual_dir = self.direction
        else:
            actual_dir = requested_dir

        dr, dc = DIRECTION_VECTOR[actual_dir]
        head_r, head_c = self.snake_body[0]
        new_head = (head_r + dr, head_c + dc)

        will_grow = new_head == self.food_pos

        """
        Nếu rắn KHÔNG ăn mồi, đuôi sẽ dịch đi -> va vào ô đuôi hiện tại là hợp lệ.
        Nếu CÓ ăn mồi, đuôi giữ nguyên -> va vào ô đuôi vẫn tính là chết.
        """
        body_to_check_collision = self.snake_body if will_grow else self.snake_body[:-1]

        death_cause: Optional[str] = None
        if not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
            death_cause = "wall"
        elif new_head in self.obstacles:
            death_cause = "obstacle"
        elif new_head in body_to_check_collision:
            death_cause = "self"

        self.steps += 1

        if death_cause is not None:
            reward = REWARD_DEATH
            done = True
            state = encode_state(
                snake_body=self.snake_body,  # giữ nguyên state trước khi chết để log/debug
                direction=self.direction,
                food_pos=self.food_pos,
                height=self.height,
                width=self.width,
                obstacles=self.obstacles,
            )
            info = {"score": self.score, "death_cause": death_cause}
            assert state.shape == (STATE_DIM,), f"step() trả state sai shape: {state.shape}"
            return state, reward, done, info

        # Không va chạm: cập nhật trạng thá
        self.direction = actual_dir
        new_body = [new_head] + self.snake_body
        if will_grow:
            self.score += 1
            reward = REWARD_FOOD
            # không pop đuôi -> rắn dài ra
        else:
            new_body.pop()  # bỏ ô đuôi cũ
            reward = REWARD_STEP
        self.snake_body = new_body

        # Chỉ sample sau khi đã cập nhật thân rắn, nếu không mồi mới có thể
        # spawn ngay trên ô đầu vừa ăn mồi.
        if will_grow:
            free_cells = self.height * self.width - len(self.obstacles) - len(self.snake_body)
            if free_cells == 0:
                done = True
                death_cause = "board_filled"
                state = encode_state(
                    snake_body=self.snake_body,
                    direction=self.direction,
                    food_pos=self.food_pos,
                    height=self.height,
                    width=self.width,
                    obstacles=self.obstacles,
                )
                return state, reward, done, {"score": self.score, "death_cause": death_cause}
            self.food_pos = self._sample_food_position()

        done = self.steps >= self.max_steps
        death_cause = "timeout" if done else None

        state = encode_state(
            snake_body=self.snake_body,
            direction=self.direction,
            food_pos=self.food_pos,
            height=self.height,
            width=self.width,
            obstacles=self.obstacles,
        )
        info = {"score": self.score, "death_cause": death_cause}
        assert state.shape == (STATE_DIM,), f"step() trả state sai shape: {state.shape}"
        return state, reward, done, info

    # ------------------------------------------------------------------ #
    # Render (debug / sanity check)
    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """Trả về chuỗi biểu diễn bàn cờ dạng ASCII để in ra kiểm tra thủ công."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for (r, c) in self.obstacles:
            grid[r][c] = "#"

        for (r, c) in self.snake_body[1:]:
            grid[r][c] = "S"

        head_r, head_c = self.snake_body[0]
        grid[head_r][head_c] = "H"

        food_r, food_c = self.food_pos
        grid[food_r][food_c] = "F"

        lines = ["".join(row) for row in grid]
        header = f"{self.config.name} | score={self.score} | steps={self.steps} | dir={self.direction}"
        return header + "\n" + "\n".join(lines)
