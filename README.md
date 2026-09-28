# Snake RL — Generalization Study và Final DQN

Repository gồm hai phần:

1. **Phase 1–5**: so sánh Dueling Double DQN và PPO với `fixed`, `random_dr`, `curriculum`.
2. **Final DQN Level 4 Mastery**: model phục vụ chơi game, được train chuyên sâu trên một Level 4 mở rộng có vật cản.

Kết quả Phase 4 cũ trong `results/phase4/` được giữ nguyên. Final training ghi riêng vào `results/final_dqn/`.

## Môi trường cuối

| Level | Kích thước | Target kết thúc game |
|---|---:|---:|
| Level 1 | 15×15 | 25 |
| Level 2 | 20×20 | 35 |
| Level 3 | 20×20 + obstacles | 45 |
| Level 4 | 30×30 + 20 obstacle cells | 55 |
| Level 5 | 30×30 + layout obstacles khác | 100 |

Workflow final hiện chỉ train Level 4. Các level khác và kết quả Phase 4 vẫn được giữ để phân tích, nhưng không được nạp vào final trainer.

Episode kết thúc khi rắn chết, timeout vì quá lâu không ăn mồi, lấp đầy bàn hoặc đạt target. Riêng Level 5 chỉ hoàn thành thành công khi score đạt 100.

## Action và state

Final model dùng ba action tương đối:

```text
0 = STRAIGHT
1 = TURN_RIGHT
2 = TURN_LEFT
```

Action tương đối loại bỏ tình trạng agent chọn hướng ngược nhưng môi trường thực hiện một hướng khác. Thay đổi này làm checkpoint 4-action cũ không tương thích; final model phải train từ đầu.

State là feature vector cố định 19 chiều nên cùng một network dùng được trên mọi kích thước bàn. Ngoài danger/hướng/mồi cũ, state có thêm khoảng trống nhìn theo ba hướng, delta chuẩn hoá tới mồi, vị trí đầu và tỷ lệ chiều dài rắn. Reward shaping nhỏ khuyến khích tiến gần mồi nhưng vẫn giữ food/death reward là tín hiệu chính.

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
- Evaluation greedy định kỳ, không cập nhật gradient.

Mọi episode train đều chạy trên Level 4. Bàn 30×30 có sáu cụm cản ngắn bố trí đối xứng, giữ các hành lang thông nhau và không tạo vùng kín.

### Điều kiện mastery

Cứ 50.000 environment steps, Level 4 được đánh giá bằng 50 episode greedy. Một evaluation pass khi:

```text
success_rate >= 80% trong 2 lần evaluation liên tiếp
```

Một episode thành công khi đạt 55 điểm trước khi chết. Không có level-up; hai evaluation liên tiếp đạt chuẩn chỉ dùng để xác nhận model đã thuần thục và tạo `final_model.pt`.

Đây không phải điều kiện “một episode may mắn”: ít nhất 40/50 episode phải đạt 55 điểm trong mỗi evaluation, lặp lại hai lần liên tiếp.

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

`max_env_steps: 0` nghĩa là không giới hạn episode/environment steps; trainer dừng bằng mastery. Có thể đặt safety budget:

```powershell
python scripts/train_final_dqn.py --max-env-steps 10000000
```

Nhấn `Ctrl+C` sẽ lưu `checkpoint_latest.pt` trước khi dừng.

## Output final training

```text
results/final_dqn/seed_0/
├── episodes.csv
├── evaluations.csv
├── mastery_events.csv
├── checkpoint_latest.pt
├── checkpoint_best.pt
├── best_evaluation.json
├── final_model.pt
└── run_status.json
```

`final_model.pt` chỉ tồn tại khi Level 4 đạt chuẩn hai lần liên tiếp. `checkpoint_best.pt` là checkpoint có kết quả evaluation tốt nhất trong quá trình train.

### Episode metrics

`episodes.csv` ghi:

- Global environment step, level, score, reward, steps và steps-per-food.
- Death cause và target reached.
- Epsilon, replay-buffer size và total updates.
- Huber TD loss và absolute TD error.
- Current/target Q, max absolute Q và gradient norm.

### Evaluation metrics

`evaluations.csv` ghi riêng cho từng level:

- Mean, median, standard deviation, min/max score.
- P25, P75, P90 score và success rate.
- Mean steps và steps-per-food.
- Wall/obstacle/self/timeout/target-reached rates.

DQN không có classification accuracy. `success_rate` là metric tương đương accuracy phù hợp cho bài toán này.

## Chạy final model

Chơi Level 4 đã train:

```powershell
python scripts/play_final_dqn.py --level all --episodes 1 --fps 30
```

Chỉ định Level 4 tường minh:

```powershell
python scripts/play_final_dqn.py --level level4 --episodes 5 --fps 30
```

Chọn checkpoint khác:

```powershell
python scripts/play_final_dqn.py --checkpoint results/final_dqn/seed_0/checkpoint_best.pt --level all
```

## Phase 4 cũ

```powershell
python scripts/run_experiment.py --all
```

Phase 4 cũ được thiết kế với action tuyệt đối bốn chiều. Sau khi action space chuyển sang ba action tương đối, checkpoint Phase 4 cũ chỉ dùng để phân tích CSV; không thể load trực tiếp vào network final mới.
