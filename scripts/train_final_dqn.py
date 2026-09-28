"""Train một Dueling Double DQN để master các level được chọn trong config."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.agents.dqn.agent import DQNConfig, DoubleDQNAgent
from src.envs.snake_env import SnakeEnv
from src.utils.config_loader import EnvConfig, load_env_config


DEFAULT_LEVEL_FILES = ["level4"]
EPISODE_FIELDS = [
    "episode", "global_env_step", "level", "target_score", "score",
    "episode_reward", "steps", "steps_per_food", "death_cause",
    "target_reached", "epsilon", "replay_buffer_size", "total_updates",
    "mean_loss", "mean_abs_td_error", "mean_current_q", "mean_target_q",
    "max_abs_q", "mean_gradient_norm",
]
EVAL_FIELDS = [
    "evaluation_index", "global_env_step", "level", "target_score",
    "episodes", "mean_score", "median_score", "std_score", "min_score",
    "max_score", "p25_score", "p75_score", "p90_score", "success_rate",
    "mean_steps", "mean_steps_per_food", "wall_rate", "obstacle_rate",
    "self_rate", "timeout_rate", "target_reached_rate",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_levels(level_files: list[str]) -> list[EnvConfig]:
    return [
        load_env_config(str(ROOT / "configs" / "envs" / f"{name}.yaml"))
        for name in level_files
    ]


def choose_level(
    unlocked: list[EnvConfig],
    rng: random.Random,
    current_probability: float,
    all_levels_unlocked: bool,
) -> EnvConfig:
    if all_levels_unlocked:
        return rng.choice(unlocked)
    current = unlocked[-1]
    if len(unlocked) == 1 or rng.random() < current_probability:
        return current
    return rng.choice(unlocked[:-1])


def evaluate_level(
    agent: DoubleDQNAgent,
    level: EnvConfig,
    n_episodes: int,
    seed: int,
) -> dict:
    scores, steps, steps_per_food = [], [], []
    causes: Counter[str] = Counter()

    for episode in range(n_episodes):
        env = SnakeEnv(level, seed=seed + episode)
        state = env.reset()
        done = False
        info = {"score": 0, "death_cause": None}
        while not done:
            state, _, done, info = env.step(agent.act(state, greedy=True))
        score = int(info["score"])
        scores.append(score)
        steps.append(env.steps)
        steps_per_food.append(env.steps / max(score, 1))
        causes[str(info.get("death_cause") or "unknown")] += 1

    values = np.asarray(scores, dtype=np.float64)
    target = int(level.target_score or 0)
    denominator = float(n_episodes)
    return {
        "level": level.name,
        "target_score": target,
        "episodes": n_episodes,
        "mean_score": float(values.mean()),
        "median_score": float(np.median(values)),
        "std_score": float(values.std()),
        "min_score": int(values.min()),
        "max_score": int(values.max()),
        "p25_score": float(np.percentile(values, 25)),
        "p75_score": float(np.percentile(values, 75)),
        "p90_score": float(np.percentile(values, 90)),
        "success_rate": float(np.mean(values >= target)),
        "mean_steps": float(np.mean(steps)),
        "mean_steps_per_food": float(np.mean(steps_per_food)),
        "wall_rate": causes["wall"] / denominator,
        "obstacle_rate": causes["obstacle"] / denominator,
        "self_rate": causes["self"] / denominator,
        "timeout_rate": causes["timeout"] / denominator,
        "target_reached_rate": causes["target_reached"] / denominator,
        "scores": scores,
    }


def mastery_score(results: list[dict]) -> float:
    normalized = [min(row["mean_score"] / row["target_score"], 1.0) for row in results]
    success = [row["success_rate"] for row in results]
    return 0.5 * float(np.mean(normalized)) + 0.3 * float(np.mean(success)) + 0.2 * min(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final multi-level DQN")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "dqn_final_mastery.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "final_dqn")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-every", type=int, default=50)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument(
        "--render-frame-skip",
        type=int,
        default=1,
        help="Draw one frame every N environment steps (higher is faster)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["seed"] if args.seed is None else args.seed)
    max_env_steps = int(
        config.get("max_env_steps", 0)
        if args.max_env_steps is None
        else args.max_env_steps
    )
    set_seed(seed)
    rng = random.Random(seed)
    level_files = list(config.get("training_levels", DEFAULT_LEVEL_FILES))
    if not level_files:
        raise ValueError("training_levels must contain at least one level")
    if len(level_files) != len(set(level_files)):
        raise ValueError("training_levels must not contain duplicates")
    levels = load_levels(level_files)
    total_levels = len(levels)

    if args.render_frame_skip <= 0:
        raise ValueError("--render-frame-skip must be greater than zero")

    agent_config = DQNConfig(**config["agent"])
    agent = DoubleDQNAgent(agent_config)
    output_dir = args.output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    evaluations_path = output_dir / "evaluations.csv"
    mastery_path = output_dir / "mastery_events.csv"
    status_path = output_dir / "run_status.json"

    # Không ghi nối nhầm vào run cũ.
    for path in (episodes_path, evaluations_path, mastery_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}. Use a new --output-dir.")

    renderer = None
    if args.render:
        from src.envs.pygame_renderer import SnakeRenderer
        renderer = SnakeRenderer(fps=args.render_fps)

    unlocked_count = 1
    mastery_streak = 0
    evaluation_index = 0
    episode_index = 0
    global_env_step = 0
    next_evaluation_step = int(config["evaluation_every_env_steps"])
    best_score = -float("inf")
    started_at = utc_now()
    completed = False

    write_json(status_path, {
        "status": "running",
        "seed": seed,
        "started_at_utc": started_at,
        "device": str(agent.device),
        "unlocked_levels": unlocked_count,
    })

    try:
        while max_env_steps <= 0 or global_env_step < max_env_steps:
            episode_index += 1
            unlocked = levels[:unlocked_count]
            level = choose_level(
                unlocked,
                rng,
                float(config["current_level_probability"]),
                unlocked_count == len(levels),
            )
            env_seed = seed * 1_000_003 + episode_index
            env = SnakeEnv(level, seed=env_seed)
            state = env.reset()
            done = False
            episode_reward = 0.0
            update_metrics: list[dict[str, float]] = []
            info = {"score": 0, "death_cause": None}

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.store(state, action, reward, next_state, done, level=level.name)
                loss = agent.update()
                if loss is not None and agent.last_update_metrics is not None:
                    update_metrics.append(agent.last_update_metrics.copy())
                state = next_state
                episode_reward += reward
                global_env_step += 1

                should_render_episode = (episode_index - 1) % args.render_every == 0
                should_render_frame = env.steps % args.render_frame_skip == 0 or done
                if renderer is not None and should_render_episode and should_render_frame:
                    visible = renderer.draw(env, {
                        "algorithm": "final dqn",
                            "strategy": str(config.get("strategy", "fixed level4")),
                        "seed": seed,
                        "episode": episode_index,
                        "extra": (
                            f"epsilon: {agent.epsilon():.3f} | unlocked: {unlocked_count}/{total_levels} | "
                            f"target: {level.target_score}"
                        ),
                    })
                    if not visible:
                        renderer = None

            metric_names = [
                "loss", "mean_abs_td_error", "mean_current_q", "mean_target_q",
                "max_abs_q", "gradient_norm",
            ]
            means = {
                name: float(np.mean([row[name] for row in update_metrics]))
                if update_metrics else float("nan")
                for name in metric_names
            }
            append_csv(episodes_path, {
                "episode": episode_index,
                "global_env_step": global_env_step,
                "level": level.name,
                "target_score": level.target_score,
                "score": info["score"],
                "episode_reward": episode_reward,
                "steps": env.steps,
                "steps_per_food": env.steps / max(int(info["score"]), 1),
                "death_cause": info.get("death_cause") or "unknown",
                "target_reached": info.get("death_cause") == "target_reached",
                "epsilon": agent.epsilon(),
                "replay_buffer_size": len(agent.buffer),
                "total_updates": agent.total_updates,
                "mean_loss": means["loss"],
                "mean_abs_td_error": means["mean_abs_td_error"],
                "mean_current_q": means["mean_current_q"],
                "mean_target_q": means["mean_target_q"],
                "max_abs_q": means["max_abs_q"],
                "mean_gradient_norm": means["gradient_norm"],
            }, EPISODE_FIELDS)

            if episode_index % int(config["save_every_episodes"]) == 0:
                agent.save(str(output_dir / "checkpoint_latest.pt"))
                write_json(status_path, {
                    "status": "running",
                    "seed": seed,
                    "started_at_utc": started_at,
                    "device": str(agent.device),
                    "episode": episode_index,
                    "global_env_step": global_env_step,
                    "unlocked_levels": unlocked_count,
                    "mastery_streak": mastery_streak,
                })
                print(
                    f"[TRAIN] episode={episode_index:,} steps={global_env_step:,} "
                    f"level={level.name} score={info['score']} "
                    f"epsilon={agent.epsilon():.3f} loss={means['loss']:.5f}"
                )

            if global_env_step < next_evaluation_step:
                continue

            evaluation_index += 1
            eval_seed = 10_000_000 + seed * 100_000 + evaluation_index * 1_000
            results = []
            for level_to_evaluate in levels[:unlocked_count]:
                result = evaluate_level(
                    agent,
                    level_to_evaluate,
                    int(config["evaluation_episodes_per_level"]),
                    eval_seed,
                )
                results.append(result)
                append_csv(evaluations_path, {
                    "evaluation_index": evaluation_index,
                    "global_env_step": global_env_step,
                    **{key: value for key, value in result.items() if key != "scores"},
                }, EVAL_FIELDS)

            threshold = float(config["mastery_success_rate"])
            all_pass = all(row["success_rate"] >= threshold for row in results)
            mastery_streak = mastery_streak + 1 if all_pass else 0
            score = mastery_score(results)

            if unlocked_count == len(levels) and score > best_score:
                best_score = score
                agent.save(str(output_dir / "checkpoint_best.pt"))
                write_json(output_dir / "best_evaluation.json", {
                    "evaluation_index": evaluation_index,
                    "global_env_step": global_env_step,
                    "mastery_score": score,
                    "mastery_streak": mastery_streak,
                    "levels": results,
                })

            required = int(config["required_consecutive_mastery_checks"])
            if mastery_streak >= required:
                if unlocked_count < len(levels):
                    mastered = levels[unlocked_count - 1]
                    unlocked_count += 1
                    append_csv(mastery_path, {
                        "global_env_step": global_env_step,
                        "episode": episode_index,
                        "mastered_level": mastered.name,
                        "unlocked_level": levels[unlocked_count - 1].name,
                        "required_success_rate": threshold,
                    }, [
                        "global_env_step", "episode", "mastered_level",
                        "unlocked_level", "required_success_rate",
                    ])
                    mastery_streak = 0
                    agent.reset_epsilon_schedule(
                        float(config.get("epsilon_on_level_unlock", agent.config.epsilon_start))
                    )
                else:
                    completed = True
                    break

            next_evaluation_step = global_env_step + int(config["evaluation_every_env_steps"])
            print(
                f"[EVAL {evaluation_index}] steps={global_env_step:,} "
                f"unlocked={unlocked_count}/{total_levels} streak={mastery_streak}/{required} "
                + " | ".join(
                    f"{row['level']}: score={row['mean_score']:.2f}, "
                    f"success={row['success_rate']:.1%}"
                    for row in results
                )
            )

    except KeyboardInterrupt:
        print("Training interrupted; saving latest checkpoint.")
    finally:
        agent.save(str(output_dir / "checkpoint_latest.pt"))
        if completed:
            agent.save(str(output_dir / "final_model.pt"))
        if renderer is not None:
            renderer.close()
        write_json(status_path, {
            "status": "mastered" if completed else "stopped",
            "seed": seed,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "device": str(agent.device),
            "episode": episode_index,
            "global_env_step": global_env_step,
            "unlocked_levels": unlocked_count,
            "mastery_streak": mastery_streak,
            "best_mastery_score": best_score if best_score > -float("inf") else None,
        })


if __name__ == "__main__":
    main()
