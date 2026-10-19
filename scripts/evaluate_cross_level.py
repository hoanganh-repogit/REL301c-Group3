"""Evaluate frozen Level-4 DQN checkpoints across all six Snake levels.

The script writes per-episode observations, aggregate summaries, and run
metadata. Evaluation is greedy and never updates the networks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.dqn.agent import DQNConfig, DoubleDQNAgent
from src.envs.snake_env import ACTION_SPACE, SnakeEnv
from src.utils.config_loader import load_env_config


LEVEL_FILES = [
    ("level1", "level1.yaml"),
    ("level2", "level2.yaml"),
    ("level3", "level3.yaml"),
    ("level4", "level4.yaml"),
    ("level5", "level5_holdout.yaml"),
    ("level6", "level6.yaml"),
]
CHECKPOINTS = [
    ("best_dqn", ROOT / "results/level4_continuous_dqn/seed_0/checkpoint_best.pt"),
    ("final_dqn", ROOT / "results/level4_continuous_dqn/seed_0/final_model.pt"),
]
RAW_COLUMNS = [
    "policy", "level", "episode", "seed", "score", "steps",
    "steps_per_food", "outcome", "target_score", "success",
    "normalized_score",
]
SUMMARY_COLUMNS = [
    "policy", "level", "episodes", "target_score", "mean_score", "std_score",
    "median_score", "min_score", "max_score", "p25_score", "p75_score",
    "p90_score", "success_rate", "mean_normalized_score", "mean_steps",
    "mean_steps_per_food", "wall_rate", "obstacle_rate", "self_rate",
    "timeout_rate", "board_filled_rate", "target_reached_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/cross_level_evaluation",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_agent(path: Path) -> DoubleDQNAgent:
    config_path = ROOT / "configs/experiments/dqn_final_mastery.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    agent = DoubleDQNAgent(DQNConfig(**config["agent"]), device=torch.device("cpu"))
    agent.load(str(path))
    agent.online_net.eval()
    return agent


@torch.no_grad()
def greedy_batch(agent: DoubleDQNAgent, states: list[np.ndarray]) -> list[int]:
    state_tensor = torch.as_tensor(
        np.asarray(states), dtype=torch.float32, device=agent.device
    )
    return agent.online_net(state_tensor).argmax(dim=1).tolist()


def evaluate_policy(
    policy_name: str,
    act_batch_factory: Callable[[int], Callable[[list[np.ndarray]], list[int]]],
    episodes: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for level_index, (level_name, filename) in enumerate(LEVEL_FILES):
        level = load_env_config(str(ROOT / "configs/envs" / filename))
        act_batch = act_batch_factory(level_index)
        envs = [SnakeEnv(level, seed=seed + episode) for episode in range(episodes)]
        states = [env.reset(seed=seed + episode) for episode, env in enumerate(envs)]
        dones = [False] * episodes
        infos = [{"score": 0, "death_cause": None} for _ in range(episodes)]
        while not all(dones):
            active = [index for index, done in enumerate(dones) if not done]
            actions = act_batch([states[index] for index in active])
            for index, action in zip(active, actions):
                states[index], _, dones[index], infos[index] = envs[index].step(action)

        for episode, (env, info) in enumerate(zip(envs, infos)):
            episode_seed = seed + episode
            score = int(info["score"])
            target = level.target_score
            rows.append(
                {
                    "policy": policy_name,
                    "level": level_name,
                    "episode": episode + 1,
                    "seed": episode_seed,
                    "score": score,
                    "steps": env.steps,
                    "steps_per_food": env.steps / max(1, score),
                    "outcome": info.get("death_cause") or "unknown",
                    "target_score": "" if target is None else target,
                    "success": "" if target is None else int(score >= target),
                    "normalized_score": (
                        "" if target is None else min(score / target, 1.0)
                    ),
                }
            )
        level_rows = rows[-episodes:]
        print(
            f"[{policy_name}] {level_name}: "
            f"mean={np.mean([row['score'] for row in level_rows]):.2f}"
        )
    return rows


def summarize(raw_rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    policies = list(dict.fromkeys(row["policy"] for row in raw_rows))
    levels = [name for name, _ in LEVEL_FILES]
    for policy in policies:
        for level in levels:
            rows = [
                row for row in raw_rows
                if row["policy"] == policy and row["level"] == level
            ]
            scores = np.asarray([row["score"] for row in rows], dtype=float)
            steps = np.asarray([row["steps"] for row in rows], dtype=float)
            steps_per_food = np.asarray(
                [row["steps_per_food"] for row in rows], dtype=float
            )
            outcomes = Counter(row["outcome"] for row in rows)
            target = rows[0]["target_score"]
            successes = [row["success"] for row in rows if row["success"] != ""]
            normalized = [
                row["normalized_score"]
                for row in rows
                if row["normalized_score"] != ""
            ]
            n = len(rows)
            summaries.append(
                {
                    "policy": policy,
                    "level": level,
                    "episodes": n,
                    "target_score": target,
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "median_score": float(np.median(scores)),
                    "min_score": int(np.min(scores)),
                    "max_score": int(np.max(scores)),
                    "p25_score": float(np.percentile(scores, 25)),
                    "p75_score": float(np.percentile(scores, 75)),
                    "p90_score": float(np.percentile(scores, 90)),
                    "success_rate": "" if not successes else float(np.mean(successes)),
                    "mean_normalized_score": (
                        "" if not normalized else float(np.mean(normalized))
                    ),
                    "mean_steps": float(np.mean(steps)),
                    "mean_steps_per_food": float(np.mean(steps_per_food)),
                    "wall_rate": outcomes["wall"] / n,
                    "obstacle_rate": outcomes["obstacle"] / n,
                    "self_rate": outcomes["self"] / n,
                    "timeout_rate": outcomes["timeout"] / n,
                    "board_filled_rate": outcomes["board_filled"] / n,
                    "target_reached_rate": outcomes["target_reached"] / n,
                }
            )
    return summaries


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    checkpoint_metadata = []
    for policy_name, path in CHECKPOINTS:
        if not path.is_file():
            raise FileNotFoundError(path)
        agent = load_agent(path)
        raw_rows.extend(
            evaluate_policy(
                policy_name,
                lambda _level_index, current=agent: (
                    lambda states: greedy_batch(current, states)
                ),
                args.episodes,
                args.seed,
            )
        )
        checkpoint_metadata.append(
            {"policy": policy_name, "path": str(path), "sha256": sha256(path)}
        )

    def random_factory(level_index: int) -> Callable[[list[np.ndarray]], list[int]]:
        rng = random.Random(args.seed + 1_000_000 + level_index)
        return lambda states: [rng.randrange(ACTION_SPACE) for _ in states]

    raw_rows.extend(
        evaluate_policy("random", random_factory, args.episodes, args.seed)
    )
    summary_rows = summarize(raw_rows)
    write_csv(args.output_dir / "episodes.csv", raw_rows, RAW_COLUMNS)
    write_csv(args.output_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes_per_policy_level": args.episodes,
        "evaluation_seed_start": args.seed,
        "environment_seeds": [args.seed + index for index in range(args.episodes)],
        "greedy": True,
        "device": "cpu",
        "git_revision": git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "checkpoints": checkpoint_metadata,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Wrote evaluation results to {args.output_dir}")


if __name__ == "__main__":
    main()
