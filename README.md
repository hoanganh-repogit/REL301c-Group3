# Snake RL Generalization Study

Dự án so sánh Dueling Double DQN và PPO trên ba chiến lược huấn luyện:
`fixed`, `random_dr` và `curriculum`. Level 1–4 là training environments;
`level5_holdout` chỉ được nạp sau khi training kết thúc.

## Cài đặt và kiểm tra

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

Sanity training riêng cho hai thuật toán:

```powershell
python scripts/sanity_train_dqn_level1.py
python scripts/sanity_train_ppo_level1.py
python scripts/sanity_train_curriculum.py
```

## Phase 4

Chạy một cấu hình theo danh sách seed trong YAML:

```powershell
python scripts/run_experiment.py --config configs/experiments/dqn_curriculum.yaml
```

Chạy toàn bộ 6 cấu hình:

```powershell
python scripts/run_experiment.py --all
```

Xem agent chơi trong lúc training bằng Pygame:

```powershell
python scripts/run_experiment.py --config configs/experiments/dqn_curriculum.yaml --seeds 0 --render --render-every 20 --render-fps 30
```

`--render-every 20` chỉ hiển thị episode 1, 21, 41,... để training không bị
chậm quá nhiều. Dùng `--render-every 1` nếu muốn xem mọi episode. Phần
đầu rắn màu xanh lá, thân xanh dương, mồi đỏ và vật cản xám.
Đóng cửa sổ Pygame chỉ tắt renderer; training và ghi log vẫn tiếp tục.

Smoke test nhanh trước khi dùng nhiều compute:

```powershell
python scripts/run_experiment.py --all --seeds 0 --episodes 2 --eval-episodes 2 --output-dir results/smoke_phase4 --force
```

Mặc định, run đã có `evaluation_summary.json` sẽ được bỏ qua. Dùng
`--force` khi thực sự muốn ghi đè bằng kết quả mới nhất. `run_status.json`
ghi `running/completed` và thời gian UTC; completion marker cũ được gỡ ngay khi
force rerun để run lỗi không bị nhầm là đã hoàn tất. Kết quả chính nằm trong:

```text
results/phase4/<algorithm_strategy>/
├── aggregate.csv
└── seed_<n>/
    ├── episodes.csv
    ├── retention.csv
    ├── checkpoint.pt
    ├── evaluation_episodes.csv
    ├── evaluation_by_level.csv
    ├── evaluation_summary.json
    ├── run_status.json
    └── run_config.json
```

`aggregate.csv` chứa mean, standard deviation qua các seed cho từng level và
`generalization_gap = mean(Level 1–4) - mean(Level 5)`.

Ngưỡng curriculum trong config thực nghiệm là 20. Ngưỡng 100 không thể
đạt trên bàn 10×10 vì rắn ban đầu đã chiếm ba ô.
