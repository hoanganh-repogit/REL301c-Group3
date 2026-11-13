"""Record the Level 6 demo video in the look of the web demo.

The video shows the real web demo page (web_demo/static: same HTML, CSS, and
drawing code) in a headless browser, playing one Level 6 episode with
checkpoint_best.pt under greedy control. The level target of 150 is switched
off, so the snake plays on until it dies.

Up to the target this is the recorded evaluation episode (seed 1234, the first
evaluation seed): the script stops unless the target is reached at the step
listed in results/cross_level_evaluation/episodes.csv. The steps after the
target are not part of the evaluation set.

Outputs docs/demo/level6_until_death.mp4 (1920x1080) and
docs/demo/level6_until_death.gif (README preview). Requirements:
    python -m pip install playwright imageio-ffmpeg pillow
    python -m playwright install chromium   # not needed if Microsoft Edge is installed
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cross_level import load_agent
from src.envs.snake_env import SnakeEnv
from src.utils.config_loader import load_env_config

CHECKPOINT = ROOT / "results/level4_continuous_dqn/seed_0/checkpoint_best.pt"
EPISODES_CSV = ROOT / "results/cross_level_evaluation/episodes.csv"
STATIC = ROOT / "web_demo/static"
LEVEL, LEVEL_FILE, SEED = "level6", "level6.yaml", 1234
VIDEO_STRIDE = 10  # environment steps per video frame
GIF_STRIDE = 30  # environment steps per GIF frame (a multiple of VIDEO_STRIDE)
FPS = 30
GIF_FRAME_MS = 50
VIEWPORT = {"width": 1280, "height": 720}  # the page fits without scrolling
SCALE = 1.5  # 1280x720 CSS pixels -> 1920x1080 video
OUTCOME_TEXT = {"self": "self-collision", "wall": "wall collision",
                "obstacle": "obstacle collision", "timeout": "no-food timeout",
                "board_filled": "board filled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/demo")
    parser.add_argument("--browser-channel", default="msedge",
                        help="installed browser to use (msedge, chrome); 'chromium' uses Playwright's own")
    return parser.parse_args()


def recorded_episode() -> dict:
    with EPISODES_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == "best_dqn" and row["level"] == LEVEL and int(row["seed"]) == SEED:
                return row
    raise KeyError(f"No best_dqn evaluation episode for {LEVEL}, seed {SEED}")


def snapshot(env: SnakeEnv) -> dict:
    # Same frame format as web_demo/app.py, which static/app.js draws.
    return {"snake": [list(cell) for cell in env.snake_body], "food": list(env.food_pos),
            "score": env.score, "steps": env.steps}


@torch.no_grad()
def play_until_death() -> dict:
    expected = recorded_episode()
    if expected["outcome"] != "target_reached":
        raise RuntimeError(f"{LEVEL} seed {SEED} did not reach the target in the evaluation")
    target = int(expected["target_score"])
    agent = load_agent(CHECKPOINT)
    base = load_env_config(str(ROOT / "configs/envs" / LEVEL_FILE))
    env = SnakeEnv(dataclasses.replace(base, target_score=None), seed=SEED)
    state = env.reset(seed=SEED)
    frames, target_step, done, info = [snapshot(env)], None, False, {}
    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        state, _, done, info = env.step(int(agent.online_net(state_tensor).argmax(dim=1).item()))
        reached = target_step is None and env.score >= target
        if reached:
            target_step = env.steps
            if target_step != int(expected["steps"]):
                raise RuntimeError(f"Target reached at step {target_step}, CSV says {expected['steps']}")
        if reached or done or env.steps % VIDEO_STRIDE == 0:
            frames.append(snapshot(env))
    if target_step is None:
        raise RuntimeError("The replay never reached the target")
    print(f"{LEVEL} seed {SEED}: target {target} at step {target_step} (matches CSV), "
          f"then score {info['score']}, {info['death_cause']} at step {env.steps}")
    return {"level": LEVEL, "height": env.height, "width": env.width,
            "obstacles": [list(cell) for cell in sorted(env.obstacles)],
            "target": target, "target_step": target_step, "score": info["score"],
            "outcome": OUTCOME_TEXT[info["death_cause"]], "steps": env.steps, "frames": frames}


def serve_static(route) -> None:
    path = route.request.url.split("demo.local", 1)[1].split("?", 1)[0]
    route.fulfill(path=str(STATIC / ("index.html" if path == "/" else path.removeprefix("/static/"))))


# Runs inside the page and reuses its own globals from static/app.js
# (draw, renderSummary, message, run, pause), so every frame is drawn by the
# web demo's code.
SHOW = """([frame, text, running]) => {
    if (frame) { draw(frame); renderSummary({Level: 'L6', Score: frame.score, Step: frame.steps}); }
    message.textContent = text;
    run.disabled = running; pause.disabled = running || !frame;
}"""


def record(episode: dict, channel: str):
    """Yield (image, seconds_to_hold, in_gif) for every distinct screen."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        launch = {} if channel == "chromium" else {"channel": channel}
        browser = playwright.chromium.launch(**launch)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
        page.route("http://demo.local/**", serve_static)
        page.goto("http://demo.local/")
        page.select_option("#level", LEVEL)
        page.evaluate("""data => {
            episode = {height: data.height, width: data.width, obstacles: data.obstacles};
            canvas.width = canvas.height = 1080;   // sharper board at 1.5x
        }""", episode)

        def shot() -> Image.Image:
            return Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")

        page.evaluate(SHOW, [None, "Choose a level and run the frozen checkpoint.", False])
        yield shot(), 1.2, True
        page.evaluate(SHOW, [None, "Running the policy on the server…", True])
        yield shot(), 0.8, True
        target_note = (f"Target {episode['target']} reached at step {episode['target_step']}, where the "
                       "evaluation episode ends. Playing on until the snake dies.")
        for index, frame in enumerate(episode["frames"]):
            past_target = frame["steps"] >= episode["target_step"]
            message = target_note if past_target else "Playing episode trace."
            page.evaluate(SHOW, [frame, message, False])
            hold = 2.0 if frame["steps"] == episode["target_step"] else 1 / FPS
            yield shot(), hold, frame["steps"] % GIF_STRIDE == 0 or hold > 1 / FPS
        finished = (f"Finished: {episode['outcome']} after {episode['steps']} steps, "
                    f"score {episode['score']}.")
        page.evaluate(SHOW, [episode["frames"][-1], finished, False])
        yield shot(), 4.0, True
        browser.close()


def main() -> None:
    import imageio_ffmpeg

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode = play_until_death()
    video_path = args.output_dir / "level6_until_death.mp4"
    width, height = int(VIEWPORT["width"] * SCALE), int(VIEWPORT["height"] * SCALE)
    writer = imageio_ffmpeg.write_frames(
        str(video_path), (width, height), fps=FPS, codec="libx264", quality=None,
        macro_block_size=1,  # keep exactly 1920x1080 instead of padding to 1088
        output_params=["-crf", "28", "-preset", "slow", "-movflags", "+faststart"],
    )
    writer.send(None)
    gif_frames, gif_durations, count = [], [], 0
    for image, seconds, in_gif in record(episode, args.browser_channel):
        data = np.ascontiguousarray(np.asarray(image)).tobytes()
        for _ in range(max(1, round(seconds * FPS))):
            writer.send(data)
            count += 1
        if in_gif:
            small = image.resize((image.width // 3, image.height // 3), Image.LANCZOS)
            gif_frames.append(small.quantize(colors=32))
            gif_durations.append(max(GIF_FRAME_MS, int(seconds * 1000)))
    writer.close()
    print(f"Wrote {video_path} ({count} frames, {count / FPS:.0f} s)")
    gif_path = args.output_dir / "level6_until_death.gif"
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=gif_durations, loop=0, optimize=True)
    print(f"Wrote {gif_path} ({len(gif_frames)} frames)")


if __name__ == "__main__":
    main()
