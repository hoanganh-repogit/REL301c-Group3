"""
Vòng lặp training tích hợp 1 trong 3 Strategy (fixed / random_dr / curriculum)
với DQN hoặc PPO. Vì 2 thuật toán có nhịp update khác nhau (DQN update MỖI
BƯỚC, PPO update sau khi gom đủ 1 rollout), 2 hàm train_dqn_with_strategy()
và train_ppo_with_strategy() TÁCH RIÊNG, nhưng dùng CHUNG:
    - BaseStrategy (start_episode / end_episode)
    - EpisodeLogger / RetentionLogger
    - RetentionTracker
để đảm bảo tính nhất quán khi so sánh 6 cấu hình ở Giai đoạn 4.

Quy ước quan trọng khi leveled_up=True (chỉ CurriculumStrategy có thể trả True):
    - DQN: agent.reset_epsilon_schedule() + agent.buffer.clear()
    - PPO: agent.buffer.reset()  (xả toàn bộ rollout đang thu thập dở, KHÔNG update
           trên dữ liệu lẫn giữa 2 level)
"""

from __future__ import annotations

from src.agents.dqn.agent import DoubleDQNAgent
from src.agents.ppo.agent import PPOAgent
from src.curriculum.strategies import BaseStrategy
from src.envs.snake_env import SnakeEnv
from src.training.retention_tracker import RetentionTracker
from src.utils.logger import EpisodeLogger, RetentionLogger


def train_dqn_with_strategy(
    agent: DoubleDQNAgent,
    strategy: BaseStrategy,
    n_episodes: int,
    episode_log_path: str,
    retention_log_path: str,
    retention_eval_every: int = 200,
    retention_n_episodes: int = 10,
    checkpoint_path: str | None = None,
    seed: int | None = None,
    save_logs_every: int = 100,
    render: bool = False,
    render_every: int = 1,
    render_fps: int = 30,
) -> None:
    """Train DQN qua n_episodes, dùng strategy để quyết định level mỗi episode."""
    episode_logger = EpisodeLogger(episode_log_path)
    retention_logger = RetentionLogger(retention_log_path)
    retention_tracker = RetentionTracker(retention_eval_every, retention_n_episodes)
    if render_every <= 0:
        raise ValueError("render_every must be greater than zero")
    renderer = None
    if render:
        from src.envs.pygame_renderer import SnakeRenderer

        renderer = SnakeRenderer(fps=render_fps)

    for episode_idx in range(1, n_episodes + 1):
        level_config = strategy.start_episode()
        env_seed = None if seed is None else seed * 1_000_003 + episode_idx
        env = SnakeEnv(level_config, seed=env_seed)

        state = env.reset()
        done = False
        info = {"score": 0, "death_cause": None}

        while not done:
            action = agent.act(state)  # epsilon-greedy
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.update()  # DQN: update mỗi bước (no-op nếu buffer chưa đủ)
            state = next_state
            if renderer is not None and (episode_idx - 1) % render_every == 0:
                visible = renderer.draw(
                    env,
                    {
                        "algorithm": "dqn",
                        "strategy": strategy.name,
                        "seed": seed,
                        "episode": episode_idx,
                        "extra": f"epsilon: {agent.epsilon():.3f}",
                    },
                )
                if not visible:
                    renderer = None

        result = strategy.end_episode(info["score"])

        episode_logger.log(
            episode=episode_idx,
            level=level_config.name,
            score=info["score"],
            steps=env.steps,
            death_cause=info["death_cause"],
            leveled_up=result["leveled_up"],
            mastered_level=result["mastered_level"],
        )

        if result["leveled_up"]:
            agent.reset_epsilon_schedule()  # explore lại khi vừa lên level mới
            agent.buffer.clear()             # tránh kinh nghiệm level cũ áp đảo

        retention_rows = retention_tracker.maybe_evaluate(
            episode_idx,
            strategy.known_levels_for_retention(),
            act_fn=lambda s: agent.act(s, greedy=True),
        )
        if retention_rows:
            retention_logger.extend(retention_rows)

        if save_logs_every > 0 and episode_idx % save_logs_every == 0:
            episode_logger.save()
            retention_logger.save()

    episode_logger.save()
    retention_logger.save()
    if checkpoint_path:
        agent.save(checkpoint_path)
    if renderer is not None:
        renderer.close()


def train_ppo_with_strategy(
    agent: PPOAgent,
    strategy: BaseStrategy,
    n_episodes: int,
    episode_log_path: str,
    retention_log_path: str,
    min_steps_before_update: int | None = None,
    retention_eval_every: int = 200,
    retention_n_episodes: int = 10,
    checkpoint_path: str | None = None,
    seed: int | None = None,
    save_logs_every: int = 100,
    render: bool = False,
    render_every: int = 1,
    render_fps: int = 30,
) -> None:
    """Train PPO qua n_episodes.

    Thiết kế: PPO chỉ update() ở RANH GIỚI episode (không bao giờ cắt giữa episode),
    nên bootstrap value LUÔN LÀ 0.0 (episode luôn kết thúc bằng done=True) -- đơn giản
    hoá đáng kể so với PPO tổng quát (thường phải bootstrap khi rollout bị cắt ngang).
    Đánh đổi: rollout có thể dài hơn min_steps_before_update một chút (chờ episode
    hiện tại kết thúc trước khi update), chấp nhận được cho quy mô đồ án.
    """
    if min_steps_before_update is None:
        min_steps_before_update = agent.config.rollout_steps

    episode_logger = EpisodeLogger(episode_log_path)
    retention_logger = RetentionLogger(retention_log_path)
    retention_tracker = RetentionTracker(retention_eval_every, retention_n_episodes)
    if render_every <= 0:
        raise ValueError("render_every must be greater than zero")
    renderer = None
    if render:
        from src.envs.pygame_renderer import SnakeRenderer

        renderer = SnakeRenderer(fps=render_fps)

    for episode_idx in range(1, n_episodes + 1):
        level_config = strategy.start_episode()
        env_seed = None if seed is None else seed * 1_000_003 + episode_idx
        env = SnakeEnv(level_config, seed=env_seed)

        state = env.reset()
        done = False
        info = {"score": 0, "death_cause": None}

        while not done:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, done, log_prob, value)
            state = next_state
            if renderer is not None and (episode_idx - 1) % render_every == 0:
                visible = renderer.draw(
                    env,
                    {
                        "algorithm": "ppo",
                        "strategy": strategy.name,
                        "seed": seed,
                        "episode": episode_idx,
                        "extra": f"rollout steps: {len(agent.buffer)}",
                    },
                )
                if not visible:
                    renderer = None

        result = strategy.end_episode(info["score"])

        episode_logger.log(
            episode=episode_idx,
            level=level_config.name,
            score=info["score"],
            steps=env.steps,
            death_cause=info["death_cause"],
            leveled_up=result["leveled_up"],
            mastered_level=result["mastered_level"],
        )

        if result["leveled_up"]:
            # Xả bỏ rollout đang thu thập dở -- có thể lẫn transition từ level cũ,
            # vi phạm giả định on-policy nếu dùng lẫn để update.
            agent.buffer.reset()
        elif len(agent.buffer) >= min_steps_before_update:
            # done=True luôn đúng ở đây (vừa kết thúc episode) -> bootstrap = 0.0 an toàn
            agent.update(last_value=0.0)

        retention_rows = retention_tracker.maybe_evaluate(
            episode_idx,
            strategy.known_levels_for_retention(),
            act_fn=lambda s: agent.act_greedy(s),
        )
        if retention_rows:
            retention_logger.extend(retention_rows)

        if save_logs_every > 0 and episode_idx % save_logs_every == 0:
            episode_logger.save()
            retention_logger.save()

    # Update nốt phần rollout còn dư ở cuối (nếu có), tránh lãng phí dữ liệu đã thu thập
    if len(agent.buffer) > 0:
        agent.update(last_value=0.0)

    episode_logger.save()
    retention_logger.save()
    if checkpoint_path:
        agent.save(checkpoint_path)
    if renderer is not None:
        renderer.close()
