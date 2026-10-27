"""Measure how a small epsilon changes the frozen policy's scores.

Level 4: the training moving average (~52) is much lower than the greedy
evaluation score (~144); epsilon = 0.01 tests whether residual exploration
explains that gap. Levels 5 and 6: greedy failures are mostly early no-food
timeouts; epsilon = 0.01 tests whether occasional random actions break those
loops. Episode seeds match scripts/evaluate_cross_level.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cross_level import (
    LEVEL_FILES, git_revision, greedy_batch, load_agent, sha256,
)
from src.envs.snake_env import ACTION_SPACE, SnakeEnv
from src.utils.config_loader import load_env_config


CHECKPOINT = ROOT / "results/level4_continuous_dqn/seed_0/checkpoint_best.pt"
LEVEL_CONFIGS = dict(LEVEL_FILES)
SUMMARY_COLUMNS = [
    "level", "epsilon", "episodes", "target_score", "success_rate", "mean_score", "std_score", "median_score",
    "min_score", "max_score", "mean_steps", "wall_rate", "obstacle_rate",
    "self_rate", "timeout_rate", "target_reached_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.01])
    parser.add_argument(
        "--levels", nargs="+", choices=list(LEVEL_CONFIGS),
        default=["level4", "level5", "level6"],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/epsilon_effect"
    )
    return parser.parse_args()


def run_level(agent, level_name: str, epsilon: float, episodes: int, seed: int) -> list[dict]:
    level = load_env_config(str(ROOT / "configs/envs" / LEVEL_CONFIGS[level_name]))
    # A separate RNG for exploration keeps the episode seeds identical to the greedy run.
    rng = random.Random(seed + 2_000_000)
    envs = [SnakeEnv(level, seed=seed + episode) for episode in range(episodes)]
    states = [env.reset(seed=seed + episode) for episode, env in enumerate(envs)]
    dones = [False] * episodes
    infos = [{"score": 0, "death_cause": None} for _ in range(episodes)]
    while not all(dones):
        active = [index for index, done in enumerate(dones) if not done]
        actions = greedy_batch(agent, [states[index] for index in active])
        for index, action in zip(active, actions):
            if rng.random() < epsilon:
                action = rng.randrange(ACTION_SPACE)
            states[index], _, dones[index], infos[index] = envs[index].step(action)
    return [
        {"score": int(info["score"]), "steps": env.steps, "outcome": info["death_cause"]}
        for env, info in zip(envs, infos)
    ]


def summarize(level_name: str, epsilon: float, rows: list[dict]) -> dict:
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    outcomes = Counter(row["outcome"] for row in rows)
    n = len(rows)
    target = load_env_config(
        str(ROOT / "configs/envs" / LEVEL_CONFIGS[level_name])
    ).target_score
    return {
        "level": level_name,
        "epsilon": epsilon,
        "episodes": n,
        "target_score": "" if target is None else target,
        "success_rate": "" if target is None else float(np.mean(scores >= target)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "min_score": int(np.min(scores)),
        "max_score": int(np.max(scores)),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "wall_rate": outcomes["wall"] / n,
        "obstacle_rate": outcomes["obstacle"] / n,
        "self_rate": outcomes["self"] / n,
        "timeout_rate": outcomes["timeout"] / n,
        "target_reached_rate": outcomes["target_reached"] / n,
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    agent = load_agent(CHECKPOINT)
    summaries = []
    for level_name in args.levels:
        for epsilon in args.epsilons:
            rows = run_level(agent, level_name, epsilon, args.episodes, args.seed)
            summary = summarize(level_name, epsilon, rows)
            summaries.append(summary)
            print(
                f"{level_name} epsilon={epsilon}: mean={summary['mean_score']:.2f} "
                f"+/- {summary['std_score']:.2f} median={summary['median_score']:.1f} "
                f"timeout={summary['timeout_rate']:.2f} success={summary['success_rate']}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summaries)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "levels": args.levels,
        "epsilons": args.epsilons,
        "episodes_per_level_epsilon": args.episodes,
        "environment_seeds": [args.seed + index for index in range(args.episodes)],
        "exploration_rng_seed": args.seed + 2_000_000,
        "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)), "sha256": sha256(CHECKPOINT)},
        "git_revision": git_revision(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
