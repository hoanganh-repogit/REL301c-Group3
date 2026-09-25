"""Pygame renderer tuy chon cho SnakeEnv.

Module nay tach khoi SnakeEnv de training headless khong can khoi tao cua so va
khong bi cham khi khong truyen --render.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.envs.snake_env import SnakeEnv


class SnakeRenderer:
    BG = (15, 18, 24)
    BOARD_BG = (23, 29, 38)
    GRID = (38, 46, 58)
    BODY = (40, 125, 220)
    BODY_INNER = (76, 165, 245)
    HEAD = (45, 220, 115)
    HEAD_INNER = (135, 255, 180)
    FOOD = (235, 70, 70)
    OBSTACLE = (115, 120, 130)
    OBSTACLE_INNER = (75, 80, 90)
    TEXT = (235, 238, 245)
    MUTED = (160, 170, 185)

    def __init__(self, fps: int = 30, window_size: int = 720, hud_height: int = 112) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Pygame is required for --render. Run: python -m pip install pygame"
            ) from exc

        self.pg = pygame
        pygame.init()
        pygame.font.init()
        self.window_size = window_size
        self.hud_height = hud_height
        self.fps = fps
        self.screen = pygame.display.set_mode((window_size, window_size + hud_height))
        pygame.display.set_caption("Snake RL - Training Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 23)
        self.closed = False

    def _events(self) -> bool:
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.close()
                return False
        return not self.closed

    def draw(self, env: "SnakeEnv", metadata: dict | None = None) -> bool:
        """Ve mot frame; tra False neu nguoi dung dong cua so."""
        if self.closed or not self._events():
            return False

        pg = self.pg
        self.screen.fill(self.BG)
        cell = max(1, min(self.window_size // env.width, self.window_size // env.height))
        board_w = cell * env.width
        board_h = cell * env.height
        offset_x = (self.window_size - board_w) // 2
        offset_y = (self.window_size - board_h) // 2
        board_rect = pg.Rect(offset_x, offset_y, board_w, board_h)
        pg.draw.rect(self.screen, self.BOARD_BG, board_rect)

        for row in range(env.height + 1):
            y = offset_y + row * cell
            pg.draw.line(self.screen, self.GRID, (offset_x, y), (offset_x + board_w, y), 1)
        for col in range(env.width + 1):
            x = offset_x + col * cell
            pg.draw.line(self.screen, self.GRID, (x, offset_y), (x, offset_y + board_h), 1)

        def rect_at(point: tuple[int, int], inset: int = 1):
            row, col = point
            return pg.Rect(
                offset_x + col * cell + inset,
                offset_y + row * cell + inset,
                max(1, cell - 2 * inset),
                max(1, cell - 2 * inset),
            )

        for point in env.obstacles:
            pg.draw.rect(self.screen, self.OBSTACLE, rect_at(point))
            inset = max(3, cell // 5)
            pg.draw.rect(self.screen, self.OBSTACLE_INNER, rect_at(point, inset))

        for point in reversed(env.snake_body[1:]):
            pg.draw.rect(self.screen, self.BODY, rect_at(point))
            inset = max(3, cell // 5)
            pg.draw.rect(self.screen, self.BODY_INNER, rect_at(point, inset))

        if env.snake_body:
            head = env.snake_body[0]
            pg.draw.rect(self.screen, self.HEAD, rect_at(head))
            inset = max(3, cell // 5)
            pg.draw.rect(self.screen, self.HEAD_INNER, rect_at(head, inset))

        food_rect = rect_at(env.food_pos, max(2, cell // 6))
        pg.draw.ellipse(self.screen, self.FOOD, food_rect)

        metadata = metadata or {}
        line1 = (
            f"{metadata.get('algorithm', 'agent').upper()} | {metadata.get('strategy', '-')} | "
            f"seed {metadata.get('seed', '-')} | episode {metadata.get('episode', '-')}"
        )
        line2 = (
            f"{env.config.name}  score: {env.score}  steps: {env.steps}  "
            f"direction: {env.direction}"
        )
        extra = metadata.get("extra", "")
        hud_y = self.window_size + 10
        self.screen.blit(self.font.render(line1, True, self.TEXT), (14, hud_y))
        self.screen.blit(self.small_font.render(line2, True, self.TEXT), (14, hud_y + 34))
        hint = f"{extra}    Close window to disable rendering; training will continue."
        self.screen.blit(self.small_font.render(hint, True, self.MUTED), (14, hud_y + 65))

        pg.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.pg.display.quit()

