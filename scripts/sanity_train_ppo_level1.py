"""
sanity_train_ppo_level1.py
-----------------------------
Train PPO CHỈ trên level 1 (chưa gắn curriculum), thu thập rollout_steps bước
mỗi lần rồi update — đúng vòng lặp chuẩn của PPO on-policy.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

from src.agents.ppo.agent import PPOAgent, PPOConfig
from src.envs.snake_env import SnakeEnv
from src.utils.config_loader import load_env_config

ROOT = os.path.join(os.path.dirname(__file__), "..")
N_UPDATES = 60  # số lần update (mỗi lần thu thập rollout_steps bước) -- ~60*2048 = ~123k bước

if __name__ == "__main__":
    with open(os.path.join(ROOT, "configs", "agents", "ppo.yaml")) as f:
        ppo_config_dict = yaml.safe_load(f)
    ppo_config = PPOConfig(**ppo_config_dict)

    env_config = load_env_config(os.path.join(ROOT, "configs", "envs", "level1.yaml"))
    env = SnakeEnv(env_config, seed=0)
    agent = PPOAgent(ppo_config)

    print(f"Device: {agent.device}")
    print(f"Training PPO trên {env_config.name} ({env_config.height}x{env_config.width}), {N_UPDATES} update...")
    print("-" * 70)

    state = env.reset()
    episode_reward = 0.0
    episode_scores_window: list[int] = []
    episode_rewards_window: list[float] = []
    start_time = time.time()

    for update_idx in range(1, N_UPDATES + 1):
        agent.buffer.reset()
        for _ in range(ppo_config.rollout_steps):
            action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, done, log_prob, value)
            state = next_state
            episode_reward += reward

            if done:
                episode_scores_window.append(info["score"])
                episode_rewards_window.append(episode_reward)
                episode_reward = 0.0
                state = env.reset()

        last_value = agent.bootstrap_value(state, done=False)  # rollout thường cắt giữa episode
        stats = agent.update(last_value)

        recent_scores = episode_scores_window[-50:] if episode_scores_window else [0]
        recent_rewards = episode_rewards_window[-50:] if episode_rewards_window else [0]
        print(
            f"Update {update_idx:3d} | avg_score(50 ep gần nhất)={np.mean(recent_scores):5.2f} | "
            f"avg_reward={np.mean(recent_rewards):7.2f} | policy_loss={stats['policy_loss']:.4f} | "
            f"value_loss={stats['value_loss']:.4f} | entropy={stats['entropy']:.4f}"
        )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Hoàn tất sau {elapsed:.1f}s")

    first_quarter = episode_scores_window[: len(episode_scores_window) // 4] or [0]
    last_quarter = episode_scores_window[-len(episode_scores_window) // 4 :] or [0]
    print(f"Score trung bình 1/4 episode đầu:  {np.mean(first_quarter):.3f}")
    print(f"Score trung bình 1/4 episode cuối: {np.mean(last_quarter):.3f}")

    if np.mean(last_quarter) > np.mean(first_quarter):
        print("=> Agent CÓ học được (score trung bình tăng theo thời gian). Phase 2 (PPO) OK.")
    else:
        print("=> CẢNH BÁO: score trung bình KHÔNG tăng — cần kiểm tra lại code training loop.")

    checkpoint_path = os.path.join(ROOT, "results", "checkpoints", "ppo_sanity_level1.pt")
    agent.save(checkpoint_path)
    print(f"Đã lưu checkpoint tạm tại: {checkpoint_path}")

