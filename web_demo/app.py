"""Web demo for the frozen Snake DQN checkpoint.

Run locally from the repository root:
    uvicorn web_demo.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
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
BEST_EVALUATION_SEEDS = {
    "level1": 1304,
    "level2": 1316,
    "level3": 1235,
    "level4": 1275,
    "level5": 1248,
    "level6": 1295,
}
DEFAULT_CHECKPOINT = ROOT / "results/level4_continuous_dqn/seed_0/checkpoint_best.pt"
CHECKPOINT = Path(os.getenv("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))
DEMO_FRAME_STRIDE = 3  # must match the frame_stride sent by static/app.js

# The policy runs one tiny forward pass per step. Spreading it over every CPU core
# makes each step slower, so a single thread is about 2.7x faster on a 20-core PC.
torch.set_num_threads(1)


def _warm_episode_cache() -> None:
    # Fixed seeds and a frozen model make every episode deterministic, so compute
    # each level once in the background; "Run episode" then answers immediately.
    for level in ("level4", *(name for name in LEVEL_FILES if name != "level4")):
        try:
            episode_json(level, DEMO_FRAME_STRIDE)
        except FileNotFoundError:
            return


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_episode_cache, name="episode-warmup", daemon=True).start()
    yield


app = FastAPI(title="Snake DQN demo", version="1.0.0", lifespan=lifespan)
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
    return FileResponse(
        ROOT / "web_demo/static/index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/health")
def health() -> dict:
    checkpoint = str(CHECKPOINT.relative_to(ROOT)) if CHECKPOINT.is_relative_to(ROOT) else str(CHECKPOINT)
    return {"status": "ok" if CHECKPOINT.is_file() else "checkpoint_missing", "checkpoint": checkpoint, "device": "cpu"}


@app.get("/api/episode")
def episode(
    level: str = Query("level4", pattern="^level[1-6]$"),
    frame_stride: int = Query(2, ge=1, le=20),
) -> Response:
    if level not in LEVEL_FILES:
        raise HTTPException(status_code=400, detail="Unsupported level")
    try:
        payload = episode_json(level, frame_stride)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(content=payload, media_type="application/json")


_episode_locks: dict[tuple[str, int], threading.Lock] = {}


def episode_json(level: str, frame_stride: int) -> bytes:
    # A request that arrives while the warm-up thread is computing the same level
    # waits for that result instead of computing it a second time.
    with _episode_locks.setdefault((level, frame_stride), threading.Lock()):
        return _cached_episode_json(level, frame_stride)


@lru_cache(maxsize=12)
def _cached_episode_json(level: str, frame_stride: int) -> bytes:
    # Cache the serialized trace: re-encoding a 2 MB Level 4 trace takes ~2 s.
    return json.dumps(run_episode(level, frame_stride), separators=(",", ":")).encode()


def run_episode(level: str, frame_stride: int) -> dict:
    agent = load_agent()
    config = load_env_config(str(ROOT / "configs/envs" / f"{LEVEL_FILES[level]}.yaml"))
    env = SnakeEnv(config, seed=BEST_EVALUATION_SEEDS[level])
    state, frames, done = env.reset(), [snapshot(env)], False
    info = {"score": 0, "death_cause": None}
    while not done:
        action = int(agent.act(state, greedy=True))
        state, _, done, info = env.step(action)
        if done or env.steps % frame_stride == 0:
            frames.append(snapshot(env, action))
    return {"level": level, "height": env.height, "width": env.width,
            "obstacles": [list(cell) for cell in sorted(env.obstacles)], "target_score": config.target_score,
            "outcome": info["death_cause"], "score": info["score"], "steps": env.steps,
            "frame_stride": frame_stride, "frames": frames}
