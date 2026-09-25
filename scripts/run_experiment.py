"""Chạy một hoặc toàn bộ 6 thí nghiệm Phase 4.

Ví dụ:
    python scripts/run_experiment.py --config configs/experiments/dqn_fixed.yaml
    python scripts/run_experiment.py --all
    python scripts/run_experiment.py --all --seeds 0 1 2 --episodes 3000

Mỗi run ghi vào results/phase4/<experiment>/seed_<seed>/; Level 5 chỉ
được nạp sau khi train xong để tránh leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.agents.dqn.agent import DQNConfig, DoubleDQNAgent
from src.agents.ppo.agent import PPOAgent, PPOConfig
from src.curriculum.strategies import make_strategy
from src.training.evaluator import evaluate_on_levels
from src.training.train_loop import train_dqn_with_strategy, train_ppo_with_strategy
from src.utils.config_loader import load_env_config


TRAIN_LEVEL_NAMES = ["level1", "level2", "level3", "level4"]
HOLDOUT_NAME = "level5_holdout"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Các network hiện tại chỉ dùng Linear/ReLU; hai cờ này giúp giữ
    # tính tái lập nếu kiến trúc GPU thay đổi về sau.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config không phải mapping YAML: {path}")
    return data


def load_train_levels() -> list:
    levels = [load_env_config(str(ROOT / "configs" / "envs" / f"{name}.yaml")) for name in TRAIN_LEVEL_NAMES]
    if any(level.is_holdout for level in levels):
        raise ValueError("Phát hiện holdout trong TRAIN_LEVEL_NAMES")
    return levels


def build_agent(agent_type: str):
    config_path = ROOT / "configs" / "agents" / f"{agent_type}.yaml"
    raw = load_yaml(config_path)
    if agent_type == "dqn":
        return DoubleDQNAgent(DQNConfig(**raw))
    if agent_type == "ppo":
        return PPOAgent(PPOConfig(**raw))
    raise ValueError(f"agent_type không hợp lệ: {agent_type}")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_and_save(agent, agent_type: str, levels: list, n_episodes: int, seed: int, run_dir: Path) -> dict:
    act_fn = (lambda state: agent.act(state, greedy=True)) if agent_type == "dqn" else agent.act_greedy
    results = evaluate_on_levels(act_fn, levels, n_episodes=n_episodes, seed=100_000 + seed * 10_000)

    raw_rows = []
    level_rows = []
    for result in results:
        for episode, score in enumerate(result["scores"], start=1):
            raw_rows.append({"level": result["level"], "episode": episode, "score": score})
        level_rows.append(
            {
                "level": result["level"],
                "mean_score": result["mean_score"],
                "std_score": result["std_score"],
                "n_episodes": n_episodes,
            }
        )

    train_mean = fmean(row["mean_score"] for row in level_rows if row["level"] in TRAIN_LEVEL_NAMES)
    holdout_mean = next(row["mean_score"] for row in level_rows if row["level"] == HOLDOUT_NAME)
    summary = {
        "seed": seed,
        "train_levels_mean": train_mean,
        "holdout_mean": holdout_mean,
        "generalization_gap": train_mean - holdout_mean,
        "levels": level_rows,
    }
    write_csv(run_dir / "evaluation_episodes.csv", raw_rows, ["level", "episode", "score"])
    write_csv(run_dir / "evaluation_by_level.csv", level_rows, ["level", "mean_score", "std_score", "n_episodes"])
    write_json(run_dir / "evaluation_summary.json", summary)
    return summary


def aggregate_experiment(experiment_dir: Path) -> None:
    summaries = []
    for path in sorted(experiment_dir.glob("seed_*/evaluation_summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    if not summaries:
        return

    rows = []
    for level_name in TRAIN_LEVEL_NAMES + [HOLDOUT_NAME]:
        values = [next(row["mean_score"] for row in item["levels"] if row["level"] == level_name) for item in summaries]
        rows.append(
            {
                "metric": level_name,
                "mean": fmean(values),
                "std_across_seeds": pstdev(values),
                "n_seeds": len(values),
            }
        )
    gaps = [item["generalization_gap"] for item in summaries]
    rows.append(
        {
            "metric": "generalization_gap",
            "mean": fmean(gaps),
            "std_across_seeds": pstdev(gaps),
            "n_seeds": len(gaps),
        }
    )
    write_csv(experiment_dir / "aggregate.csv", rows, ["metric", "mean", "std_across_seeds", "n_seeds"])


def run_one(config_path: Path, args: argparse.Namespace) -> None:
    experiment = load_yaml(config_path)
    agent_type = experiment.get("agent_type")
    strategy_name = experiment.get("strategy")
    if agent_type not in {"dqn", "ppo"}:
        raise ValueError(f"agent_type không hợp lệ trong {config_path}: {agent_type}")
    if strategy_name not in {"fixed", "random_dr", "curriculum"}:
        raise ValueError(f"strategy không hợp lệ trong {config_path}: {strategy_name}")

    seeds = args.seeds if args.seeds is not None else experiment["seeds"]
    n_episodes = args.episodes if args.episodes is not None else int(experiment["n_episodes"])
    eval_episodes = args.eval_episodes
    experiment_dir = args.output_dir / config_path.stem
    experiment_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_dir = experiment_dir / f"seed_{seed}"
        completed_marker = run_dir / "evaluation_summary.json"
        if completed_marker.exists() and not args.force:
            print(f"[SKIP] {config_path.stem} seed={seed}: đã có evaluation_summary.json")
            continue

        # Khi force rerun, bo completion marker CU ngay lap tuc. Neu run moi bi
        # gian doan, lan chay sau se khong nham ket qua cu la run da hoan tat.
        if args.force and completed_marker.exists():
            completed_marker.unlink()

        print(f"[RUN] {config_path.stem} seed={seed}, episodes={n_episodes}, eval={eval_episodes}")
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path = run_dir / "run_status.json"
        started_at = utc_now()
        write_json(
            status_path,
            {
                "status": "running",
                "experiment": config_path.stem,
                "seed": seed,
                "started_at_utc": started_at,
                "completed_at_utc": None,
            },
        )
        set_global_seed(seed)
        train_levels = load_train_levels()
        strategy_kwargs = {}
        if strategy_name == "random_dr":
            strategy_kwargs["seed"] = seed
        elif strategy_name == "curriculum":
            strategy_kwargs["level_up_score"] = int(experiment["level_up_score"])
        strategy = make_strategy(strategy_name, train_levels, **strategy_kwargs)
        agent = build_agent(agent_type)

        common = dict(
            agent=agent,
            strategy=strategy,
            n_episodes=n_episodes,
            episode_log_path=str(run_dir / "episodes.csv"),
            retention_log_path=str(run_dir / "retention.csv"),
            retention_eval_every=int(experiment.get("retention_eval_every", 200)),
            retention_n_episodes=int(experiment.get("retention_n_episodes", 10)),
            checkpoint_path=str(run_dir / "checkpoint.pt"),
            seed=seed,
            render=args.render,
            render_every=args.render_every,
            render_fps=args.render_fps,
        )
        if agent_type == "dqn":
            train_dqn_with_strategy(**common)
        else:
            train_ppo_with_strategy(**common)

        # Holdout chỉ được load sau training.
        holdout = load_env_config(str(ROOT / "configs" / "envs" / f"{HOLDOUT_NAME}.yaml"))
        if not holdout.is_holdout:
            raise ValueError("Level 5 phải có is_holdout=true")
        evaluate_and_save(agent, agent_type, train_levels + [holdout], eval_episodes, seed, run_dir)
        write_json(
            run_dir / "run_config.json",
            {**experiment, "seed": seed, "n_episodes": n_episodes, "eval_episodes": eval_episodes},
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "experiment": config_path.stem,
                "seed": seed,
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
            },
        )
        print(f"[DONE] {run_dir}")

    aggregate_experiment(experiment_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake RL Phase 4 experiment runner")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", type=Path, help="One YAML file in configs/experiments")
    selection.add_argument("--all", action="store_true", help="Run all six experiment configs")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the seed list in YAML")
    parser.add_argument("--episodes", type=int, help="Override n_episodes (useful for smoke tests)")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase4")
    parser.add_argument("--force", action="store_true", help="Overwrite completed runs")
    parser.add_argument("--render", action="store_true", help="Show training in a Pygame window")
    parser.add_argument("--render-every", type=int, default=1, help="Render one episode every N episodes")
    parser.add_argument("--render-fps", type=int, default=30, help="Maximum rendered frames per second")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be greater than zero")
    if args.render_every <= 0:
        parser.error("--render-every must be greater than zero")
    if args.render_fps <= 0:
        parser.error("--render-fps must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    if args.all:
        paths = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
    else:
        paths = [args.config.resolve()]
    for path in paths:
        run_one(path, args)


if __name__ == "__main__":
    main()
