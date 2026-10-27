"""
Snake environment parameterized by level (EnvConfig): board size and obstacles
come from the config rather than being hard-coded, so one class serves every
level.

Classic Gym-style interface: reset() -> state, step(action) -> (state, reward, done, info)

info["truncated"] = True when an episode is cut by the no-food timeout. That is
not a true terminal state, so the trainer keeps bootstrapping from next_state
instead of treating it as a death.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from src.envs.state_encoder import (DIRECTION_VECTOR, STATE_DIM, encode_state)
from src.utils.config_loader import EnvConfig


# Three relative actions: straight, turn right, turn left. The chosen action is
# always the move the environment performs; there is no ignored reverse action.
ACTION_SPACE = 3
ACTION_TO_TURN = {0: "STRAIGHT", 1: "RIGHT", 2: "LEFT"}
_CLOCKWISE = ["UP", "RIGHT", "DOWN", "LEFT"]


def _apply_relative_turn(direction: str, turn: str) -> str:
    idx = _CLOCKWISE.index(direction)
    if turn == "STRAIGHT":
        return direction
    if turn == "RIGHT":
        return _CLOCKWISE[(idx + 1) % 4]
    if turn == "LEFT":
        return _CLOCKWISE[(idx - 1) % 4]
    raise ValueError(f"Unknown relative turn: {turn}")

# Reward constants: FIXED and shared by every level for a fair comparison
REWARD_FOOD = 10.0
REWARD_DEATH = -20.0
REWARD_STEP = -0.005 # very small penalty, so the snake is not pushed toward food on a risky path
REWARD_CLOSER = 0.02
REWARD_FARTHER = -0.02

"""
No-food timeout per board cell: an episode times out after
height * width * this many consecutive steps without food. It prevents endless
loops, e.g. of an untrained or random policy.
"""
_MAX_STEPS_WITHOUT_FOOD_PER_CELL = 2

class SnakeEnv:
    """Snake environment for one level, parameterized by EnvConfig."""

    def __init__(self, config: EnvConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self.height = config.height
        self.width = config.width
        self.obstacles: set[tuple[int, int]] = set(config.obstacles)
        # The timeout counts consecutive steps without food, not the total episode
        # length. Eating resets the counter, so high targets remain reachable.
        self.max_steps_without_food = (
            self.height * self.width * _MAX_STEPS_WITHOUT_FOOD_PER_CELL
        )

        self.rng = np.random.RandomState(seed=seed)

        # Assigned for real in reset(); declared here to document their types
        self.snake_body: list[tuple[int, int]] = []
        self.direction: str = "RIGHT"
        self.food_ops: tuple[int, int] = (0, 0)
        self.score: int = 0
        self.steps: int = 0
        self.steps_since_food: int = 0

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Start a new episode and return the initial state, shape (STATE_DIM,)."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.snake_body = self._sample_valid_start()
        self.direction = "RIGHT"
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.food_pos = self._sample_food_position()

        state = encode_state(
            snake_body=self.snake_body,
            direction=self.direction,
            food_pos=self.food_pos,
            height=self.height,
            width=self.width,
            obstacles=self.obstacles,
        )
        assert state.shape == (STATE_DIM,), f"reset() returned a wrong state shape: {state.shape}"
        return state

    def _sample_valid_start(self) -> list[tuple[int, int]]:
        """
        Find a valid start for a 3-cell snake (inside the board, not on obstacles).

        The snake starts heading RIGHT, so its body lies to the LEFT of the head:
        body = [head, head-1_col, head-2_col]
        """
        for _ in range(1000):
            r = self.rng.randint(1, self.height - 1)
            c = self.rng.randint(3, self.width) # c - 2 >= 0 requires c >= 2; start at 3 for a margin
            candidate = [(r, c), (r, c - 1), (r, c - 2)]
            if all(0 <= cc < self.width for (_, cc) in candidate) and not any(p in self.obstacles for p in candidate):
                return candidate
        raise RuntimeError(
            f"[{self.config.name}] No valid snake start found after 1000 attempts; "
            "check whether the obstacles cover too much of the board."
        )

    def _sample_food_position(self) -> tuple[int, int]:
        """Sample a random food position that is not on the snake or an obstacle."""
        occupied = self.obstacles | set(self.snake_body)
        free_cells = [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in occupied
        ]
        if not free_cells:
            raise RuntimeError(f"[{self.config.name}] No free cell left to spawn food.")
        return free_cells[int(self.rng.randint(0, len(free_cells)))]

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply one action.

        Args:
            action: 0=STRAIGHT, 1=TURN_RIGHT, 2=TURN_LEFT.

        Returns:
            (state, reward, done, info) — state shape (STATE_DIM,).
            info contains: score (food eaten), death_cause (None while alive),
            truncated (True when cut by the timeout).
        """
        assert action in ACTION_TO_TURN, f"invalid action: {action}"
        actual_dir = _apply_relative_turn(self.direction, ACTION_TO_TURN[action])

        dr, dc = DIRECTION_VECTOR[actual_dir]
        head_r, head_c = self.snake_body[0]
        new_head = (head_r + dr, head_c + dc)
        old_food_distance = abs(head_r - self.food_pos[0]) + abs(head_c - self.food_pos[1])

        will_grow = new_head == self.food_pos

        """
        Without eating, the tail moves away, so entering the current tail cell is legal.
        When eating, the tail stays, so entering the tail cell is still a collision.
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
        self.steps_since_food += 1

        if death_cause is not None:
            reward = REWARD_DEATH
            done = True
            state = encode_state(
                snake_body=self.snake_body,  # keep the pre-death state for logging/debugging
                direction=self.direction,
                food_pos=self.food_pos,
                height=self.height,
                width=self.width,
                obstacles=self.obstacles,
            )
            info = {"score": self.score, "death_cause": death_cause, "truncated": False}
            assert state.shape == (STATE_DIM,), f"step() returned a wrong state shape: {state.shape}"
            return state, reward, done, info

        # No collision: update the state
        self.direction = actual_dir
        new_body = [new_head] + self.snake_body
        if will_grow:
            self.score += 1
            self.steps_since_food = 0
            reward = REWARD_FOOD
            # keep the tail -> the snake grows
        else:
            new_body.pop()  # drop the old tail cell
            new_food_distance = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
            distance_reward = REWARD_CLOSER if new_food_distance < old_food_distance else REWARD_FARTHER
            reward = REWARD_STEP + distance_reward
        self.snake_body = new_body

        # Sample only after updating the body; otherwise new food could spawn on
        # the cell the head has just entered.
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
                return state, reward, done, {
                    "score": self.score,
                    "death_cause": death_cause,
                    "truncated": False,
                }
            self.food_pos = self._sample_food_position()

            if self.config.target_score is not None and self.score >= self.config.target_score:
                state = encode_state(
                    snake_body=self.snake_body,
                    direction=self.direction,
                    food_pos=self.food_pos,
                    height=self.height,
                    width=self.width,
                    obstacles=self.obstacles,
                )
                return state, reward, True, {
                    "score": self.score,
                    "death_cause": "target_reached",
                    "truncated": False,
                }

        done = self.steps_since_food >= self.max_steps_without_food
        death_cause = "timeout" if done else None

        state = encode_state(
            snake_body=self.snake_body,
            direction=self.direction,
            food_pos=self.food_pos,
            height=self.height,
            width=self.width,
            obstacles=self.obstacles,
        )
        info = {"score": self.score, "death_cause": death_cause, "truncated": done}
        assert state.shape == (STATE_DIM,), f"step() returned a wrong state shape: {state.shape}"
        return state, reward, done, info

    # ------------------------------------------------------------------ #
    # Render (debug / sanity check)
    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """Return an ASCII drawing of the board for manual inspection."""
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
