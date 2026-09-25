"""
Đọc file YAML config của môi trường (level) thành dataclass EnvConfig,
kèm validate cơ bản để phát hiện lỗi config sớm (fail-fast) thay vì
để lỗi lan xuống tận lúc chạy training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
import yaml

@dataclass
class EnvConfig:
    """
    Cấu hình 1 level của SnakeEnv.

    Attributes:
        name: tên level, dùng để đặt tên file log/checkpoint.
        height: chiều cao bàn cờ (số hàng).
        width: chiều rộng bàn cờ (số cột).
        obstacles: danh sách toạ độ (row, col) của vật cản tĩnh.
        is_holdout: True nếu đây là level 5 — không được phép dùng để train.
    """
    name: str
    height: int
    width: int
    obstacles: list[tuple[int, int]] = field(default_factory=list)
    is_holdout: bool = False

    def __post_init__(self) -> None:
        # Validate cơ bản, fail sớm nếu config sai
        assert self.height >= 5 and self.width >= 5, (
            f"[{self.name}] Bàn quá nhỏ ({self.height}x{self.width}), "
            "cần tối thiểu 5x5 để rắn có chỗ di chuyển."
        )

        for (row, col) in self.obstacles:
            assert 0 <= row < self.height and 0 <= col < self.width, (
                f"[{self.name}] Vật cản {(row, col)} nằm ngoài bàn cờ "
                f"kích thước ({self.height}, {self.width})."
            )
        # Không cho phép vật cản trùng toạ độ nhau
        assert len(self.obstacles) == len(set(self.obstacles)), (
            f"[{self.name}] Có toạ độ vật cản bị trùng lặp."
        )


def load_env_config(yaml_path: str) -> EnvConfig:
    """
    Đọc 1 file YAML level và trả về EnvConfig đã validate.

    Args:
        yaml_path: đường dẫn tới file yaml, ví dụ configs/envs/level1.yaml

    Returns:
        EnvConfig tương ứng.
    """
    assert os.path.isfile(yaml_path), f"Không tìm thấy file config: {yaml_path}"
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    height, width = raw["size"]
    obstacles = [tuple(o) for o in raw.get("obstacles", [])]

    return EnvConfig(name=raw["name"], height=height, width=width, obstacles=obstacles, is_holdout=raw["is_holdout"])

def load_all_levels(envs_dir: str) -> dict[str, EnvConfig]:
    """
    Đọc toàn bộ file *.yaml trong thư mục configs/envs/ thành dict {name: EnvConfig}.

    Dùng ở bước sanity check / khi cần liệt kê tất cả level có sẵn.
    """
    configs: dict[str, EnvConfig] = {}
    for fname in sorted(os.listdir(envs_dir)):
        if fname.endswith(".yaml") or fname.endswith(".yml"):
            cfg = load_env_config(os.path.join(envs_dir, fname))
            configs[cfg.name] = cfg
    return configs
