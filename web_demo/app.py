"""Web demo for the frozen Snake DQN checkpoint.

Run locally from the repository root:
    uvicorn web_demo.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.dqn.agent import DQNConfig, DoubleDQNAgent
from src.envs.snake_env import SnakeEnv
from src.utils.config_loader import load_env_config

LEVEL_FILES = {
    "level1": "level1", "level2": "level2", "level3": "level3",
    "level4": "level4", "level5": "level5_holdout", "level6": "level6",
}
DEFAULT_CHECKPOINT = ROOT / "results/level4_continuous_dqn/seed_0/checkpoint_best.pt"
CHECKPOINT = Path(os.getenv("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))

app = FastAPI(title="Snake DQN demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "web_demo/static"), name="static")


@lru_cache(maxsize=1)
def load_agent() -> DoubleDQNAgent:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    with (ROOT / "configs/experiments/dqn_final_mastery.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    agent = DoubleDQNAgent(DQNConfig(**config["agent"]), device=torch.device("cpu"))
    agent.load(str(CHECKPOINT))
    agent.online_net.eval()
    return agent


def snapshot(env: SnakeEnv, action: int | None = None) -> dict:
    return {"snake": [list(cell) for cell in env.snake_body], "food": list(env.food_pos),
            "score": env.score, "steps": env.steps, "action": action}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "web_demo/static/index.html")


@app.get("/api/health")
def health() -> dict:
    checkpoint = str(CHECKPOINT.relative_to(ROOT)) if CHECKPOINT.is_relative_to(ROOT) else str(CHECKPOINT)
    return {"status": "ok" if CHECKPOINT.is_file() else "checkpoint_missing", "checkpoint": checkpoint, "device": "cpu"}


@app.get("/api/episode")
def episode(
    level: str = Query("level4", pattern="^level[1-6]$"),
    seed: int = Query(1234, ge=0, le=2_147_483_647),
    frame_stride: int = Query(2, ge=1, le=20),
) -> dict:
    if level not in LEVEL_FILES:
        raise HTTPException(status_code=400, detail="Unsupported level")
    try:
        agent = load_agent()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    config = load_env_config(str(ROOT / "configs/envs" / f"{LEVEL_FILES[level]}.yaml"))
    env = SnakeEnv(config, seed=seed)
    state, frames, done = env.reset(), [snapshot(env)], False
    info = {"score": 0, "death_cause": None}
    while not done:
        action = int(agent.act(state, greedy=True))
        state, _, done, info = env.step(action)
        if done or env.steps % frame_stride == 0:
            frames.append(snapshot(env, action))
    return {"level": level, "seed": seed, "height": env.height, "width": env.width,
            "obstacles": [list(cell) for cell in sorted(env.obstacles)], "target_score": config.target_score,
            "outcome": info["death_cause"], "score": info["score"], "steps": env.steps,
            "frame_stride": frame_stride, "frames": frames}
