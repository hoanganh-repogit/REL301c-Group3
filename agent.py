"""
agent.py
Agent điều phối: trích xuất state (trạng thái) từ game, chọn action (epsilon-greedy),
lưu transition vào replay memory, và gọi QTrainer để cập nhật model.
"""

import random
from collections import deque

import numpy as np
import torch

from model import Linear_QNet, QTrainer
from plotlib import plot
from snake_game_AI import Direction, Point, SnakeGameAI

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

STATE_SIZE = 11
ACTION_SIZE = 3
HIDDEN_SIZE = 256

# Epsilon-greedy: exponential decay thay vì "80 - n_games" tuyến tính của bản gốc.
# EPSILON_MIN đảm bảo agent vẫn giữ một chút exploration về lâu dài.
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995  # epsilon *= EPSILON_DECAY sau mọi episode

TARGET_UPDATE_FREQ = 100  # dong bo target network sau mọi N train_step


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() khi dầy

        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE)
        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma,
            target_update_freq=TARGET_UPDATE_FREQ,
        )

    def get_state(self, game) -> np.ndarray:
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location - FIX: ban goc so sanh food.y voi chinh no (luon False),
            # dung phai la so sanh voi head.y giong nhu 2 dong food.x ben tren
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        state_array = np.array(state, dtype=int)
        assert state_array.shape == (STATE_SIZE,), (
            f"get_state: expected shape ({STATE_SIZE},), got {state_array.shape}"
        )
        return state_array

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def decay_epsilon(self):
        """Exponential decay, giu toi thieu EPSILON_MIN de khong mat het exploration."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=self.trainer.device)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    episode_reward = 0   # ← thêm biến tích lũy

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        episode_reward += reward   # ← cộng dồn mỗi bước

        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.decay_epsilon()
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game: {agent.n_games} | Score: {score} | '
                  f'Episode Reward: {episode_reward} | Epsilon: {agent.epsilon:.3f} | Record: {record}')

            episode_reward = 0   # ← reset cho episode tiếp theo

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()