"""Train Dueling Double DQN liên tục trên môi trường cố định."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import deque
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
    "rolling_mean_score", "best_episode_score",
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


def read_last_csv_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", newline="", encoding="utf-8") as handle:
        for last in csv.DictReader(handle):
            pass
    return last


def read_recent_scores(path: Path, window: int) -> deque[float]:
    scores: deque[float] = deque(maxlen=window)
    if not path.exists():
        return scores
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores.append(float(row["score"]))
    return scores


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train continuous Level 4 DQN")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "dqn_final_mastery.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "level4_continuous_dqn",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the existing seed run from checkpoint_latest.pt",
    )
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
    best_score_window = int(config.get("best_score_window", 100))
    best_score_warmup = int(config.get("best_score_warmup_episodes", 500))
    if best_score_window <= 0:
        raise ValueError("best_score_window must be greater than zero")
    seed = int(config["seed"] if args.seed is None else args.seed)
    max_env_steps = int(
        config.get("max_env_steps", 0)
        if args.max_env_steps is None
        else args.max_env_steps
    )
    set_seed(seed)
    level_files = list(config.get("training_levels", DEFAULT_LEVEL_FILES))
    if len(level_files) != 1:
        raise ValueError("Continuous final trainer requires exactly one training level")
    levels = load_levels(level_files)
    level = levels[0]

    if args.render_frame_skip <= 0:
        raise ValueError("--render-frame-skip must be greater than zero")

    agent_config = DQNConfig(**config["agent"])
    agent = DoubleDQNAgent(agent_config)
    output_dir = args.output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    status_path = output_dir / "run_status.json"

    checkpoint_path = output_dir / "checkpoint_latest.pt"
    if args.resume:
        if not checkpoint_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(
                "Resume requires checkpoint_latest.pt and episodes.csv in "
                f"{output_dir}"
            )
    else:
        # Không ghi nối nhầm vào run cũ.
        for path in (episodes_path,):
            if path.exists():
                raise FileExistsError(
                    f"Output already exists: {path}. Use --resume or a new --output-dir."
                )

    renderer = None
    if args.render:
        from src.envs.pygame_renderer import SnakeRenderer
        renderer = SnakeRenderer(fps=args.render_fps)

    episode_index = 0
    global_env_step = 0
    best_rolling_score = -float("inf")
    best_episode_score = 0
    recent_scores: deque[float] = deque(maxlen=best_score_window)
    started_at = utc_now()

    if args.resume:
        replay_restored = agent.load(str(checkpoint_path))
        last_episode = read_last_csv_row(episodes_path)
        if last_episode is None:
            raise RuntimeError("episodes.csv has no data rows; cannot resume safely")
        episode_index = int(last_episode["episode"])
        csv_env_step = int(last_episode["global_env_step"])
        checkpoint_env_step = int(agent.total_env_steps)
        if checkpoint_env_step < csv_env_step:
            raise RuntimeError(
                "Checkpoint is older than episodes.csv: "
                f"checkpoint steps={checkpoint_env_step}, CSV steps={csv_env_step}. "
                "Use a matching checkpoint or start a new output directory."
            )
        # Ctrl+C có thể đến giữa episode. Checkpoint giữ network/optimizer/replay
        # tới action cuối, còn CSV chỉ chứa episode đã hoàn tất. Bỏ trạng thái env
        # dở và bắt đầu lại episode kế tiếp, nhưng không vứt các transition đã học.
        partial_episode_steps = checkpoint_env_step - csv_env_step
        global_env_step = checkpoint_env_step
        recent_scores = read_recent_scores(episodes_path, best_score_window)
        best_episode_score = max(
            int(float(row["score"]))
            for row in csv.DictReader(episodes_path.open("r", encoding="utf-8"))
        )
        previous_status = load_yaml(status_path) if status_path.exists() else {}
        started_at = str(previous_status.get("started_at_utc", started_at))
        best_training_path = output_dir / "best_training.json"
        if best_training_path.exists():
            best_training = load_yaml(best_training_path)
            best_rolling_score = float(best_training.get("rolling_mean_score", -float("inf")))

        print(
            f"[RESUME] episode={episode_index:,} steps={global_env_step:,} "
            f"epsilon={agent.epsilon():.3f} replay={len(agent.buffer):,} "
            f"({'restored' if replay_restored else 'not present in old checkpoint'}) "
            f"partial_episode_steps={partial_episode_steps:,}"
        )

    write_json(status_path, {
        "status": "running",
        "seed": seed,
        "started_at_utc": started_at,
        "device": str(agent.device),
        "episode": episode_index,
        "global_env_step": global_env_step,
        "level": level.name,
        "evaluation_enabled": False,
        "resumed": args.resume,
    })

    try:
        while max_env_steps <= 0 or global_env_step < max_env_steps:
            episode_index += 1
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
                            f"epsilon: {agent.epsilon():.3f} | level: {level.name} | "
                            "target: unlimited"
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
            episode_score = int(info["score"])
            recent_scores.append(float(episode_score))
            rolling_mean_score = float(np.mean(recent_scores))
            best_episode_score = max(best_episode_score, episode_score)
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
                "rolling_mean_score": rolling_mean_score,
                "best_episode_score": best_episode_score,
            }, EPISODE_FIELDS)

            if episode_index % int(config["save_every_episodes"]) == 0:
                agent.save(str(output_dir / "checkpoint_latest.pt"))
                if (
                    episode_index >= best_score_warmup
                    and len(recent_scores) == best_score_window
                    and rolling_mean_score > best_rolling_score
                ):
                    best_rolling_score = rolling_mean_score
                    agent.save(str(output_dir / "checkpoint_best.pt"))
                    write_json(output_dir / "best_training.json", {
                        "episode": episode_index,
                        "global_env_step": global_env_step,
                        "window": best_score_window,
                        "rolling_mean_score": best_rolling_score,
                        "best_episode_score": best_episode_score,
                    })
                write_json(status_path, {
                    "status": "running",
                    "seed": seed,
                    "started_at_utc": started_at,
                    "device": str(agent.device),
                    "episode": episode_index,
                    "global_env_step": global_env_step,
                    "level": level.name,
                    "evaluation_enabled": False,
                    "rolling_mean_score": rolling_mean_score,
                    "best_rolling_mean_score": (
                        best_rolling_score if best_rolling_score > -float("inf") else None
                    ),
                    "best_episode_score": best_episode_score,
                })
                print(
                    f"[TRAIN] episode={episode_index:,} steps={global_env_step:,} "
                    f"level={level.name} score={info['score']} "
                    f"rolling={rolling_mean_score:.2f} best={best_episode_score} "
                    f"epsilon={agent.epsilon():.3f} loss={means['loss']:.5f}"
                )

    except KeyboardInterrupt:
        print("Training interrupted; saving resumable checkpoint (this may take a moment).")
    finally:
        agent.save(str(checkpoint_path), include_replay_buffer=True)
        # Compact checkpoint để play; checkpoint_latest chứa thêm replay buffer
        # và được dùng riêng cho --resume.
        agent.save(str(output_dir / "final_model.pt"))
        if renderer is not None:
            renderer.close()
        write_json(status_path, {
            "status": "stopped",
            "seed": seed,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "device": str(agent.device),
            "episode": episode_index,
            "global_env_step": global_env_step,
            "level": level.name,
            "evaluation_enabled": False,
            "best_rolling_mean_score": (
                best_rolling_score if best_rolling_score > -float("inf") else None
            ),
            "best_episode_score": best_episode_score,
        })


if __name__ == "__main__":
    main()
