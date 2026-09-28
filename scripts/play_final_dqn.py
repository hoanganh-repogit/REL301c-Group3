"""Chạy model DQN cuối cùng trên một hoặc cả năm level với Pygame."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.agents.dqn.agent import DQNConfig, DoubleDQNAgent
from src.envs.pygame_renderer import SnakeRenderer
from src.envs.snake_env import SnakeEnv
from src.utils.config_loader import load_env_config


LEVEL_FILES = {
    "level1": "level1",
    "level2": "level2",
    "level3": "level3",
    "level4": "level4",
    "level5": "level5_holdout",
    "level5_holdout": "level5_holdout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play final DQN Snake model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "final_dqn" / "seed_0" / "final_model.pt",
    )
    parser.add_argument(
        "--level",
        choices=["all", *LEVEL_FILES],
        default="all",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per level")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config_path = ROOT / "configs" / "experiments" / "dqn_final_mastery.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    agent = DoubleDQNAgent(DQNConfig(**raw["agent"]))
    agent.load(str(args.checkpoint))
    agent.online_net.eval()

    configured_levels = [
        "level5" if name == "level5_holdout" else name
        for name in raw.get("training_levels", ["level4"])
    ]
    selected = configured_levels if args.level == "all" else [args.level]
    renderer = SnakeRenderer(fps=args.fps)

    try:
        for level_index, level_name in enumerate(selected):
            config_name = LEVEL_FILES[level_name]
            level = load_env_config(str(ROOT / "configs" / "envs" / f"{config_name}.yaml"))
            for episode in range(args.episodes):
                env = SnakeEnv(level, seed=args.seed + level_index * 10_000 + episode)
                state = env.reset()
                done = False
                info = {"score": 0, "death_cause": None}
                while not done:
                    action = agent.act(state, greedy=True)
                    state, _, done, info = env.step(action)
                    if not renderer.draw(env, {
                        "algorithm": "final dqn",
                        "strategy": "greedy play",
                        "seed": args.seed,
                        "episode": episode + 1,
                        "extra": f"target: {level.target_score}",
                    }):
                        return
                print(
                    f"{level.name} episode={episode + 1} score={info['score']} "
                    f"steps={env.steps} outcome={info['death_cause']}"
                )
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
