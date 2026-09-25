"""
----------------------------
Train Double DQN + Dueling DQN CHỈ trên level 1 (chưa gắn logic curriculum) —
mục đích debug code training loop trước khi thêm độ phức tạp của Giai đoạn 3.

Kỳ vọng: reward trung bình và score trung bình TĂNG DẦN theo thời gian train.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

from src.agents.dqn.agent import DoubleDQNAgent, DQNConfig
from src.envs.snake_env import SnakeEnv
from src.utils.config_loader import load_env_config

ROOT = os.path.join(os.path.dirname(__file__), "..")
N_EPISODES = 800
LOG_EVERY = 50

if __name__ == "__main__":
    with open(os.path.join(ROOT, "configs", "agents", "dqn.yaml")) as f:
        dqn_config_dict = yaml.safe_load(f)
    dqn_config = DQNConfig(**dqn_config_dict)

    env_config = load_env_config(os.path.join(ROOT, "configs", "envs", "level1.yaml"))
    env = SnakeEnv(env_config, seed=0)
    agent = DoubleDQNAgent(dqn_config)

    print(f"Device: {agent.device}")
    print(f"Training DQN trên {env_config.name} ({env_config.height}x{env_config.width}), {N_EPISODES} episode...")
    print("-" * 70)

    episode_rewards = []
    episode_scores = []
    losses = []
    start_time = time.time()

    for ep in range(1, N_EPISODES + 1):
        state = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            state = next_state
            ep_reward += reward

        episode_rewards.append(ep_reward)
        episode_scores.append(info["score"])

        if ep % LOG_EVERY == 0:
            recent_rewards = episode_rewards[-LOG_EVERY:]
            recent_scores = episode_scores[-LOG_EVERY:]
            recent_loss = np.mean(losses[-500:]) if losses else float("nan")
            print(
                f"Episode {ep:4d} | avg_reward={np.mean(recent_rewards):7.2f} | "
                f"avg_score={np.mean(recent_scores):5.2f} | max_score={max(recent_scores):3d} | "
                f"epsilon={agent.epsilon():.3f} | avg_loss={recent_loss:.4f}"
            )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Hoàn tất sau {elapsed:.1f}s")

    first_quarter = episode_scores[: N_EPISODES // 4]
    last_quarter = episode_scores[-N_EPISODES // 4 :]
    print(f"Score trung bình 1/4 đầu:  {np.mean(first_quarter):.3f}")
    print(f"Score trung bình 1/4 cuối: {np.mean(last_quarter):.3f}")

    if np.mean(last_quarter) > np.mean(first_quarter):
        print("=> Agent CÓ học được (score trung bình tăng theo thời gian). Phase 2 (DQN) OK.")
    else:
        print("=> CẢNH BÁO: score trung bình KHÔNG tăng — cần kiểm tra lại code training loop.")

    checkpoint_path = os.path.join(ROOT, "results", "checkpoints", "dqn_sanity_level1.pt")
    agent.save(checkpoint_path)
    print(f"Đã lưu checkpoint tạm tại: {checkpoint_path}")
