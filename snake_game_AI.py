from turtledemo import clock

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from sympy.physics.units import action

# With code logic in snake game AI have some change logic to learn and training
# Code logic for game play
# Reset function
# Reward
# play(action) --> Direction
# game_iteration
# is_collision
pygame.init()
font = pygame.font.SysFont("arial.ttf", 25)

# Class Direction (Up, Down, Left, Right)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x, y")

# color rgb
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Size and Speed
BLOCK_SIZE = 20
SPEED = 10

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        # reset help for reset game when AI lost or game over
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y), Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE) //BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE) // BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action) # agent action with environments
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0 # add reward for check point
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10 #if game over
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10 # if eat food success
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over
        return reward, game_over, self.score

    def is_collision(self, p=None):
        if p is None:
            p = self.head
        # hits boundary
        if p.x > self.w - BLOCK_SIZE or p.x < 0 or p.y > self.h - BLOCK_SIZE or p.y < 0:
            return True
        # hits itself
        if p in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for p in self.snake: # Snake
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(p.x+4, p.y+4, 12, 12))

        # Food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score:" + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    # Move function
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if  self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)


# if __name__ == "__main__":
#     game = SnakeGame()
#
#     # game loop
#     while True:
#         game_over, score = game.play_step()
#         if game_over == True:
#             break
#
#     print(f'Final Score: {score}')
#
#     pygame.quit()













