"""Record the README demo by replaying real evaluation episodes.

Every clip replays one episode of results/cross_level_evaluation with the same
checkpoint, level, and seed, using greedy actions on CPU. The script stops if
the replayed score or outcome differs from episodes.csv, so the video can only
show results that were actually recorded.

Clip selection rule: seed 1234 (the first evaluation episode) on every level,
plus seed 1235 on Level 6 (the first Level 6 failure, an early no-food timeout).

Requires imageio-ffmpeg for the MP4 output:
    python -m pip install imageio-ffmpeg
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # render without opening a window

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cross_level import LEVEL_FILES, load_agent
from src.envs.pygame_renderer import SnakeRenderer
from src.envs.snake_env import SnakeEnv
from src.utils.config_loader import load_env_config


CHECKPOINT = ROOT / "results/level4_continuous_dqn/seed_0/checkpoint_best.pt"
EPISODES_CSV = ROOT / "results/cross_level_evaluation/episodes.csv"
SUMMARY_CSV = ROOT / "results/cross_level_evaluation/summary.csv"
LEVEL_CONFIGS = dict(LEVEL_FILES)
FIRST_EVALUATION_SEED = 1234
# (level, evaluation seed, environment steps per video frame)
VIDEO_CLIPS = [
    ("level1", 1234, 1),
    ("level2", 1234, 2),
    ("level3", 1234, 3),
    ("level4", 1234, 6),
    ("level5", 1234, 6),
    ("level6", 1234, 10),
    ("level6", 1235, 10),
]
GIF_CLIP = ("level3", 1234, 4)
FPS = 30
GIF_FRAME_MS = 50
OUTCOME_TEXT = {
    "target_reached": "target reached",
    "self": "self-collision",
    "wall": "wall collision",
    "obstacle": "obstacle collision",
    "timeout": "no-food timeout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/demo")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def recorded_episode(level: str, seed: int) -> dict:
    with EPISODES_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == "best_dqn" and row["level"] == level and int(row["seed"]) == seed:
                return row
    raise KeyError(f"No best_dqn evaluation episode for {level}, seed {seed}")


def success_summary() -> str:
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["policy"] == "best_dqn"]
    parts = [
        f"L{row['level'][-1]} {100 * float(row['success_rate']):.0f}%"
        for row in rows if row["success_rate"]
    ]
    return "Success over 100 seeds per level: " + ", ".join(parts)


class Recorder:
    def __init__(self, agent) -> None:
        self.agent = agent
        # A very high fps keeps pygame's frame limiter from slowing the recording.
        self.renderer = SnakeRenderer(fps=100_000, show_close_hint=False)
        self.pg = self.renderer.pg
        self.screen = self.renderer.screen
        self.big_font = self.pg.font.Font(None, 46)

    def frame(self) -> np.ndarray:
        return self.pg.surfarray.array3d(self.screen).swapaxes(0, 1).copy()

    def card(self, lines: list[str]) -> np.ndarray:
        self.screen.fill(SnakeRenderer.BG)
        width, height = self.screen.get_size()
        y = height // 2 - 30 * len(lines)
        for index, line in enumerate(lines):
            font = self.big_font if index == 0 else self.renderer.font
            color = SnakeRenderer.TEXT if index == 0 else SnakeRenderer.MUTED
            text = font.render(line, True, color)
            self.screen.blit(text, ((width - text.get_width()) // 2, y))
            y += 60 if index == 0 else 40
        return self.frame()

    def banner(self, text: str) -> np.ndarray:
        width = self.renderer.window_size
        overlay = self.pg.Surface((width, 70), self.pg.SRCALPHA)
        overlay.fill((10, 12, 16, 215))
        self.screen.blit(overlay, (0, width // 2 - 35))
        label = self.big_font.render(text, True, SnakeRenderer.TEXT)
        self.screen.blit(label, ((width - label.get_width()) // 2, width // 2 - label.get_height() // 2))
        return self.frame()

    @torch.no_grad()
    def episode(self, level_name: str, seed: int, stride: int):
        """Yield frames of one greedy episode, then check it against the CSV."""
        config = load_env_config(str(ROOT / "configs/envs" / LEVEL_CONFIGS[level_name]))
        env = SnakeEnv(config, seed=seed)
        state = env.reset(seed=seed)
        target = config.target_score if config.target_score is not None else "none"
        metadata = {
            "algorithm": "dueling double dqn",
            "strategy": "evaluation replay",
            "seed": seed,
            "episode": f"{seed - FIRST_EVALUATION_SEED + 1}/100",
            "extra": f"greedy | target: {target} | speed x{stride} | checkpoint_best.pt",
        }
        self.renderer.draw(env, metadata)
        yield self.frame()
        done, info = False, {"score": 0, "death_cause": None}
        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action = int(self.agent.online_net(state_tensor).argmax(dim=1).item())
            state, _, done, info = env.step(action)
            if done or env.steps % stride == 0:
                self.renderer.draw(env, metadata)
                yield self.frame()

        expected = recorded_episode(level_name, seed)
        replay = (int(info["score"]), info["death_cause"], env.steps)
        recorded = (int(expected["score"]), expected["outcome"], int(expected["steps"]))
        if replay != recorded:
            raise RuntimeError(f"{level_name} seed {seed}: replay {replay} != recorded {recorded}")
        print(f"{level_name} seed {seed}: score {replay[0]}, {replay[1]}, {replay[2]} steps (matches CSV)")
        self.last_result = f"Score {replay[0]}: {OUTCOME_TEXT[replay[1]]}"

    def hold(self, frame: np.ndarray, seconds: float, fps: int):
        for _ in range(int(seconds * fps)):
            yield frame


def level_card(level_name: str, seed: int, stride: int) -> list[str]:
    config = load_env_config(str(ROOT / "configs/envs" / LEVEL_CONFIGS[level_name]))
    role = "training level" if level_name == "level4" else "unseen during training"
    target = config.target_score if config.target_score is not None else "no limit"
    return [
        f"Level {level_name[-1]}: {config.height}x{config.width}, "
        f"{len(config.obstacles)} obstacles, target {target}",
        f"{role} | evaluation seed {seed} | speed x{stride}",
    ]


def video_frames(recorder: Recorder):
    yield from recorder.hold(recorder.card([
        "Snake: Dueling Double DQN",
        "Trained only on Level 4, evaluated zero-shot on six levels",
        "Replay of recorded evaluation episodes (greedy, checkpoint_best.pt)",
        "Seed 1234 on every level, plus the first Level 6 failure (seed 1235)",
    ]), 3.5, FPS)
    for level_name, seed, stride in VIDEO_CLIPS:
        yield from recorder.hold(recorder.card(level_card(level_name, seed, stride)), 2.0, FPS)
        last = None
        for last in recorder.episode(level_name, seed, stride):
            yield last
        yield from recorder.hold(recorder.banner(recorder.last_result), 2.0, FPS)
    yield from recorder.hold(recorder.card([
        "Results, not highlights",
        success_summary(),
        "Most failures on unseen levels are early no-food timeouts",
        "Full analysis: latex/main.tex",
    ]), 4.0, FPS)


def write_video(recorder: Recorder, path: Path) -> None:
    import imageio_ffmpeg

    width, height = recorder.screen.get_size()
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=FPS, codec="libx264", quality=None,
        output_params=["-crf", "28", "-preset", "slow", "-movflags", "+faststart"],
    )
    writer.send(None)
    count = 0
    for frame in video_frames(recorder):
        writer.send(np.ascontiguousarray(frame).tobytes())
        count += 1
    writer.close()
    print(f"Wrote {path} ({count} frames, {count / FPS:.0f} s)")


def write_gif(recorder: Recorder, path: Path) -> None:
    level_name, seed, stride = GIF_CLIP
    frames = [Image.fromarray(frame) for frame in recorder.episode(level_name, seed, stride)]
    hold = int(1500 / GIF_FRAME_MS)
    frames += [Image.fromarray(recorder.banner(recorder.last_result))] * hold
    small = [
        frame.resize((frame.width // 2, frame.height // 2), Image.LANCZOS).quantize(colors=64)
        for frame in frames
    ]
    small[0].save(path, save_all=True, append_images=small[1:], duration=GIF_FRAME_MS,
                  loop=0, optimize=True)
    print(f"Wrote {path} ({len(small)} frames)")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(load_agent(CHECKPOINT))
    if not args.skip_gif:
        write_gif(recorder, args.output_dir / "demo_level3.gif")
    if not args.skip_video:
        write_video(recorder, args.output_dir / "snake_dqn_demo.mp4")


if __name__ == "__main__":
    main()
