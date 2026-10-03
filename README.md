# Snake RL — Generalization Study và Final DQN

Repository gồm hai phần:

1. **Phase 1–5**: so sánh Dueling Double DQN và PPO với `fixed`, `random_dr`, `curriculum`.
2. **Final DQN Level 4 Continuous**: model phục vụ chơi game, được train liên tục trên một Level 4 mở rộng có vật cản và không giới hạn điểm.

Kết quả Phase 4 cũ trong `results/phase4/` và final run 19 chiều cũ được giữ nguyên. Continuous Level 4 training ghi riêng vào `results/level4_continuous_dqn/`.

## Môi trường cuối

| Level | Kích thước | Target kết thúc game |
|---|---:|---:|
| Level 1 | 15×15, không vật cản | 25 |
| Level 2 | 20×20, 8 obstacle cells | 35 |
| Level 3 | 24×24, 16 obstacle cells | 50 |
| Level 4 | 30×30 + 20 obstacle cells | Không giới hạn |
| Level 5 | 32×32, 34 obstacle cells lệch tầng | 100 |
| Level 6 | 36×36, 68 obstacle cells dạng cổng | 150 |

Workflow final hiện chỉ train Level 4. Các level khác và kết quả Phase 4 vẫn được giữ để phân tích, nhưng không được nạp vào final trainer.

Trong workflow final Level 4, episode chỉ kết thúc khi rắn chết, timeout vì quá lâu không ăn mồi hoặc lấp đầy bàn. Các target của level khác chỉ phục vụ thí nghiệm Phase 4 cũ.

## Action và state

Final model dùng ba action tương đối:

```text
0 = STRAIGHT
1 = TURN_RIGHT
2 = TURN_LEFT
```

Action tương đối loại bỏ tình trạng agent chọn hướng ngược nhưng môi trường thực hiện một hướng khác. Thay đổi này làm checkpoint 4-action cũ không tương thích; final model phải train từ đầu.

State là feature vector cố định 27 chiều. Ngoài danger, hướng và vị trí mồi, state có ray clearance, vị trí đầu, tỷ lệ chiều dài, vùng an toàn sau ba action và vị trí tương đối của đuôi. Các feature vùng an toàn giúp agent phát hiện ngõ cụt do thân và vật cản khi rắn dài.

## Cài đặt

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

Với RTX 5070 Ti, cần PyTorch CUDA build hỗ trợ `sm_120` (ví dụ CUDA 13.2), không dùng build CUDA 12.6 cũ.

## Final training strategy

Final trainer sử dụng:

- Dueling Double DQN và Double-DQN target.
- Huber TD loss và gradient clipping.
- Fixed-environment training chỉ trên Level 4.
- Balanced replay sampling theo level.
- Không chạy evaluation riêng và không tự dừng theo mastery.

Mọi episode train đều chạy trên Level 4. Bàn 30×30 có sáu cụm cản ngắn bố trí đối xứng, giữ các hành lang thông nhau và không tạo vùng kín.

### Điều kiện chạy và lưu model

Level 4 không có `target_score`. Episode tiếp tục cho đến khi rắn chết, timeout do quá lâu không ăn, hoặc lấp đầy toàn bộ phần bàn có thể sử dụng. Training chạy cho đến khi người dùng nhấn `Ctrl+C`.

Trainer lưu `checkpoint_best.pt` theo điểm trung bình trượt 100 episode sau giai đoạn warm-up 500 episode. Cách này tránh chọn model chỉ vì một episode may mắn. `checkpoint_latest.pt` chứa replay buffer để resume; `final_model.pt` là checkpoint gọn để chơi và được cập nhật khi dừng an toàn.

## Chạy final training

Không render để đạt tốc độ cao nhất:

```powershell
python scripts/train_final_dqn.py
```

Xem một episode mỗi 50 episode train:

```powershell
python scripts/train_final_dqn.py --render --render-every 50 --render-fps 30
```

Xem nhanh và mượt hơn: chỉ vẽ mỗi 2 environment steps ở 60 FPS, và chỉ xem một episode sau mỗi 100 episode train:

```powershell
python scripts/train_final_dqn.py --render --render-every 100 --render-fps 60 --render-frame-skip 2
```

`--render-frame-skip` chỉ bỏ qua frame hiển thị, không bỏ transition train và không thay đổi kết quả thuật toán.

Chạy seed khác và output riêng:

```powershell
python scripts/train_final_dqn.py --seed 1 --output-dir results/final_dqn_seed1
```

Config chính: `configs/experiments/dqn_final_mastery.yaml`.

`max_env_steps: 0` nghĩa là không giới hạn episode/environment steps; trainer chỉ dừng khi nhấn `Ctrl+C`. Có thể đặt safety budget:

```powershell
python scripts/train_final_dqn.py --max-env-steps 10000000
```

Nhấn `Ctrl+C` sẽ lưu `checkpoint_latest.pt` trước khi dừng.

### Dừng và train tiếp

Dừng an toàn bằng `Ctrl+C` trong terminal. Hãy chờ đến khi xuất hiện thông báo đã lưu checkpoint và terminal trả lại prompt. Checkpoint khi dừng chứa network, optimizer, epsilon, bộ đếm bước và replay buffer.

Tiếp tục đúng run `seed_0` đang dở:

```powershell
python scripts/train_final_dqn.py --resume
```

Tiếp tục và bật cửa sổ quan sát nhanh:

```powershell
python scripts/train_final_dqn.py --resume --render --render-every 100 --render-fps 60 --render-frame-skip 2
```

Resume nối tiếp `episodes.csv`, không tạo lại episode 1. Không dùng `--resume` khi muốn train mới hoàn toàn; khi đó cần dùng output directory mới hoặc xoá run cũ có chủ đích.

## Output final training

```text
results/level4_continuous_dqn/seed_0/
├── episodes.csv
├── checkpoint_latest.pt
├── checkpoint_best.pt
├── best_training.json
├── final_model.pt
└── run_status.json
```

`checkpoint_best.pt` là model có rolling mean 100 episode tốt nhất tại các mốc lưu. `final_model.pt` và checkpoint resume được tạo khi training dừng an toàn.

### Episode metrics

`episodes.csv` ghi:

- Global environment step, level, score, reward, steps và steps-per-food.
- Death cause và target reached.
- Epsilon, replay-buffer size và total updates.
- Huber TD loss và absolute TD error.
- Current/target Q, max absolute Q và gradient norm.
- Rolling mean score và best episode score.

## Chạy final model

Chơi Level 4 đã train:

```powershell
python scripts/play_final_dqn.py --level level4 --episodes 1 --fps 30
```

Chơi lần lượt cả sáu level (mỗi level chạy đúng số episode yêu cầu, không bắt buộc pass để chuyển):

```powershell
python scripts/play_final_dqn.py --level all --episodes 1 --fps 60
```

Chơi theo tiến trình có level-up, tự bỏ Level 4 vì không có target:

```powershell
python scripts/play_final_dqn.py --level progression --episodes 1 --fps 60
```

Progression chạy `Level 1 → 2 → 3 → 5 → 6`. Chỉ khi đạt đúng target của level hiện tại mới chuyển màn; nếu chết trước target, agent tự chơi lại level đó.

Chỉ định Level 4 tường minh:

```powershell
python scripts/play_final_dqn.py --level level4 --episodes 5 --fps 30
```

Chơi riêng Level 6:

```powershell
python scripts/play_final_dqn.py --level level6 --episodes 3 --fps 60
```

Chọn checkpoint khác:

```powershell
python scripts/play_final_dqn.py --checkpoint results/level4_continuous_dqn/seed_0/checkpoint_best.pt --level all
```

## Phase 4 cũ

```powershell
python scripts/run_experiment.py --all
```

Phase 4 cũ được thiết kế với action tuyệt đối bốn chiều. Sau khi action space chuyển sang ba action tương đối, checkpoint Phase 4 cũ chỉ dùng để phân tích CSV; không thể load trực tiếp vào network final mới.
