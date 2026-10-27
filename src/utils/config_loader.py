"""
Load an environment (level) YAML config into an EnvConfig dataclass, with basic
validation so that config errors fail fast instead of surfacing during
training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
import yaml

@dataclass
class EnvConfig:
    """
    Configuration of one SnakeEnv level.

    Attributes:
        name: level name, used to name log and checkpoint files.
        height: board height (number of rows).
        width: board width (number of columns).
        obstacles: (row, col) coordinates of the static obstacles.
        is_holdout: True for Level 5, which must never be used for training.
    """
    name: str
    height: int
    width: int
    obstacles: list[tuple[int, int]] = field(default_factory=list)
    is_holdout: bool = False
    target_score: int | None = None

    def __post_init__(self) -> None:
        # Basic validation: fail early on an invalid config
        assert self.height >= 5 and self.width >= 5, (
            f"[{self.name}] Board too small ({self.height}x{self.width}); "
            "at least 5x5 is needed for the snake to move."
        )

        for (row, col) in self.obstacles:
            assert 0 <= row < self.height and 0 <= col < self.width, (
                f"[{self.name}] Obstacle {(row, col)} is outside the board "
                f"of size ({self.height}, {self.width})."
            )
        # Obstacles must not share coordinates
        if self.target_score is not None:
            assert self.target_score > 0, f"[{self.name}] target_score must be greater than zero"

        assert len(self.obstacles) == len(set(self.obstacles)), (
            f"[{self.name}] Duplicate obstacle coordinates."
        )


def load_env_config(yaml_path: str) -> EnvConfig:
    """
    Load one level YAML file and return a validated EnvConfig.

    Args:
        yaml_path: path to the YAML file, e.g. configs/envs/level1.yaml

    Returns:
        The corresponding EnvConfig.
    """
    assert os.path.isfile(yaml_path), f"Config file not found: {yaml_path}"
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    height, width = raw["size"]
    obstacles = [tuple(o) for o in raw.get("obstacles", [])]

    return EnvConfig(
        name=raw["name"],
        height=height,
        width=width,
        obstacles=obstacles,
        is_holdout=raw["is_holdout"],
        target_score=raw.get("target_score"),
    )

def load_all_levels(envs_dir: str) -> dict[str, EnvConfig]:
    """
    Load every *.yaml file in configs/envs/ into a dict {name: EnvConfig}.

    Used for sanity checks and whenever all available levels are needed.
    """
    configs: dict[str, EnvConfig] = {}
    for fname in sorted(os.listdir(envs_dir)):
        if fname.endswith(".yaml") or fname.endswith(".yml"):
            cfg = load_env_config(os.path.join(envs_dir, fname))
            configs[cfg.name] = cfg
    return configs
