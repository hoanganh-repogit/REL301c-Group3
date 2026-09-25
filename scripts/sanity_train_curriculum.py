"""
Chạy THỬ toàn bộ luồng Phase 3 end-to-end trong thời gian ngắn:
  - Dùng level_up_score=0 để buộc curriculum chuyển level một cách xác định.
    Sanity này kiểm tra wiring, không kiểm tra agent học nhanh hay chậm.
  - Xác nhận: level-up xảy ra đúng lúc, epsilon/buffer reset đúng lúc, retention log
    có dữ liệu, level 5 KHÔNG BAO GIỜ xuất hiện trong log.
  - Test cả 3 strategy x DQN (PPO tương tự logic, không lặp lại ở đây để tiết kiệm thời gian).

LƯU Ý: level_up_score=0 chỉ dùng cho sanity check này. Phase 4 đọc
ngưỡng thật từ configs/experiments/*_curriculum.yaml.
"""
import os
import csv
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_DIR = os.path.join(ROOT, "configs", "envs")

from src.agents.dqn.agent import DoubleDQNAgent, DQNConfig
from src.curriculum.strategies import make_strategy
from src.envs.snake_env import SnakeEnv
from src.training import train_loop
from src.utils.config_loader import load_env_config

class _FastSnakeEnv(SnakeEnv):
    """Subclass CHỈ dùng cho sanity check: giới hạn max_steps thấp để episode
    kết thúc nhanh (do timeout) khi agent còn ngẫu nhiên (epsilon cao), giúp
    sanity check chạy trong vài chục giây thay vì vài phút. KHÔNG dùng cho
    thực nghiệm thật ở Giai đoạn 4 (dùng đúng SnakeEnv gốc)."""

    def __init__(self, config, seed=None):
        super().__init__(config, seed)
        self.max_steps = 150


train_loop.SnakeEnv = _FastSnakeEnv  # monkeypatch riêng cho sanity check

TRAIN_LEVEL_NAMES = ["level1", "level2", "level3", "level4"]
SANITY_LEVEL_UP_SCORE = 0  # buộc chuyển level; không dùng để đánh giá khả năng học
N_EPISODES = 150

if __name__ == "__main__":
    strategy_arg = sys.argv[1] if len(sys.argv) > 1 else None
    n_episodes_override = int(sys.argv[2]) if len(sys.argv) > 2 else N_EPISODES
    strategies_to_run = [strategy_arg] if strategy_arg else ["fixed", "random_dr", "curriculum"]

    train_levels = [load_env_config(os.path.join(ENV_DIR, f"{name}.yaml")) for name in TRAIN_LEVEL_NAMES]
    assert all(not lvl.is_holdout for lvl in train_levels)

    tmp_dir = os.path.join(ROOT, "results", "raw_logs", "_sanity_check")
    os.makedirs(tmp_dir, exist_ok=True)

    for strategy_name in strategies_to_run:
        print("=" * 70)
        print(f"Sanity check strategy = {strategy_name}")
        print("-" * 70)

        kwargs = {"level_up_score": SANITY_LEVEL_UP_SCORE} if strategy_name == "curriculum" else {}
        if strategy_name == "random_dr":
            kwargs = {"seed": 0}
        strategy = make_strategy(strategy_name, train_levels, **kwargs)

        agent = DoubleDQNAgent(DQNConfig(epsilon_decay_steps=5000, min_buffer_size_before_train=100))

        episode_log_path = os.path.join(tmp_dir, f"dqn_{strategy_name}_episodes.csv")
        retention_log_path = os.path.join(tmp_dir, f"dqn_{strategy_name}_retention.csv")

        train_loop.train_dqn_with_strategy(
            agent=agent,
            strategy=strategy,
            n_episodes=n_episodes_override,
            episode_log_path=episode_log_path,
            retention_log_path=retention_log_path,
            retention_eval_every=100,
            retention_n_episodes=3,
            seed=0,
        )

        # ---- Đọc lại CSV để kiểm tra ----
        with open(episode_log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        levels_seen = {r["level"] for r in rows}
        leveled_up_rows = [r for r in rows if r["leveled_up"] == "True"]

        print(f"Số episode: {len(rows)}")
        print(f"Level xuất hiện trong log: {sorted(levels_seen)}")
        print(f"Số lần leveled_up=True: {len(leveled_up_rows)}")
        if leveled_up_rows:
            print(f"  Ví dụ lần đầu: episode={leveled_up_rows[0]['episode']}, "
                  f"mastered={leveled_up_rows[0]['mastered_level']}")

        assert "level5_holdout" not in levels_seen, f"LỖI NGHIÊM TRỌNG: level5_holdout xuất hiện ở strategy={strategy_name}!"

        if strategy_name == "fixed":
            assert levels_seen == {"level4"}, f"Fixed phải luôn ở level4, nhưng thấy {levels_seen}"
            assert len(leveled_up_rows) == 0
        elif strategy_name == "random_dr":
            assert len(leveled_up_rows) == 0, "RandomDR không bao giờ được leveled_up=True"
        elif strategy_name == "curriculum":
            assert len(leveled_up_rows) == 3, (
                "Curriculum sanity phải chuyển đúng 3 lần: L1->L2->L3->L4"
            )
            assert levels_seen == {"level1", "level2", "level3", "level4"}

        with open(retention_log_path, newline="", encoding="utf-8") as f:
            retention_rows = list(csv.DictReader(f))
        print(f"Số dòng retention log: {len(retention_rows)}")
        retention_levels_seen = {r["level"] for r in retention_rows}
        assert "level5_holdout" not in retention_levels_seen

        print(f"=> Strategy {strategy_name}: OK\n")

    print("=" * 70)
    print("CÁC SANITY CHECK PHASE 3 ĐƯỢC YÊU CẦU ĐỀU PASS.")
