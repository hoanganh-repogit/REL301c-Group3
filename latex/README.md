# Phase 6 — Kế hoạch viết báo cáo LaTeX

Tài liệu này chốt hướng nghiên cứu và kế hoạch viết báo cáo. Toàn bộ source
LaTeX, bảng, hình và tài liệu tham khảo của báo cáo nằm trong `latex/`.

## 1. Hướng đề tài đã chốt

Đề tài **không còn so sánh DQN với PPO**. Đối tượng nghiên cứu duy nhất là
**Dueling Double DQN**:

- Agent chỉ được huấn luyện trên **Level 4**.
- Sau khi huấn luyện, giữ nguyên network và chạy greedy trên các level khác.
- Không fine-tune, không cập nhật gradient và không thêm dữ liệu từ level test.
- Mục tiêu là đánh giá khả năng **zero-shot cross-level generalization**: một
  policy học từ một môi trường có chuyển được sang bàn có kích thước và bố cục
  vật cản khác hay không.

PPO và bộ thí nghiệm Phase 4 cũ không nằm trong câu hỏi nghiên cứu chính. Có thể
nhắc rất ngắn trong phần lịch sử phát triển dự án hoặc appendix, nhưng không đưa
vào tiêu đề, Methodology chính, bảng kết quả chính hay Conclusion.

## 2. Tên đề tài

### Tên chính thức đề xuất

**Tiếng Việt**

> **Đánh giá khả năng khái quát hóa đa cấp độ của Dueling Double DQN huấn luyện
> trên một môi trường trong trò chơi Snake**

**Tiếng Anh**

> **Evaluating Cross-Level Generalization of a Dueling Double DQN Trained in a
> Single Snake Environment**

Tên này phản ánh đúng thiết kế: một thuật toán, một môi trường train và nhiều
môi trường evaluation. Không dùng các cụm như “đạt mastery” hoặc “hoạt động tốt
trên mọi level” trong tên vì đó là kết luận cần được kiểm chứng bằng evaluation.

### Tên ngắn

> **Cross-Level Generalization of DQN in Snake**

## 3. Phát biểu bài toán

Một Dueling Double DQN được train duy nhất trên Level 4 (bàn 30×30, 20 ô vật
cản, không giới hạn điểm). State 27 chiều và ba action tương đối làm input/output
của network không phụ thuộc kích thước bàn. Nhờ vậy cùng một checkpoint có thể
được nạp và chạy trên Level 1, 2, 3, 5 và 6 mà không đổi kiến trúc.

Tuy nhiên, **chạy được về mặt kỹ thuật không đồng nghĩa với generalize tốt**.
Báo cáo chỉ được kết luận sau khi chạy evaluation lặp lại trên nhiều episode,
với seed cố định và metric định lượng.

### Câu hỏi nghiên cứu

**RQ1.** Dueling Double DQN được train chỉ trên Level 4 đạt hiệu năng như thế
nào trên chính môi trường train khi chạy greedy?

**RQ2.** Không fine-tune, policy đó chuyển sang Level 1, 2, 3, 5 và 6 tốt đến
mức nào?

**RQ3.** Score, success rate, độ ổn định và nguyên nhân kết thúc episode thay
đổi như thế nào giữa các level có kích thước bàn và bố cục vật cản khác nhau?

### Mục tiêu

1. Xây dựng môi trường Snake tham số hóa theo level.
2. Huấn luyện Dueling Double DQN ổn định trên duy nhất Level 4.
3. Đánh giá cùng một policy trên sáu level mà không huấn luyện lại.
4. Phân tích điểm mạnh, failure modes và giới hạn của single-environment
   training đối với cross-level transfer.

## 4. Phạm vi level

| Level | Vai trò trong nghiên cứu mới | Kích thước | Target | Ghi chú |
|---|---|---:|---:|---|
| Level 1 | Evaluation unseen | 15×15 | 25 | Không vật cản |
| Level 2 | Evaluation unseen | 20×20 | 35 | 8 ô vật cản |
| Level 3 | Evaluation unseen | 24×24 | 50 | 16 ô vật cản |
| Level 4 | **Training + in-domain evaluation** | 30×30 | Không giới hạn | 20 ô vật cản |
| Level 5 | Held-out evaluation | 32×32 | 100 | 34 ô vật cản lệch tầng |
| Level 6 | Challenge evaluation | 36×36 | 150 | 68 ô vật cản dạng cổng |

Trong continuous run hiện tại, agent chỉ thấy Level 4. Vì vậy Level 1–3 cũng là
unseen đối với checkpoint này dù config của chúng có `is_holdout: false` từ
thiết kế cũ. Báo cáo cần phân biệt:

- **In-domain:** Level 4.
- **Unseen transfer:** Level 1, 2, 3, 5 và 6.
- **Explicit held-out:** Level 5 theo metadata của repository.

## 5. Trạng thái train hiện có

Continuous DQN seed 0 đã dừng trong episode 19.095:

| Thuộc tính | Giá trị |
|---|---:|
| Training level | Level 4 |
| Episode hoàn chỉnh đã ghi log | 19.094 |
| Episode đang chạy khi dừng | 19.095 |
| Environment steps | 14.293.725 |
| Best checkpoint episode | 17.000 |
| Best rolling mean (100 training episodes) | 52,09 |
| Best single training episode | 157 |
| State dimension | 27 |
| Action space | 3 action tương đối |

Các số 52,09 và 157 là **training statistics**, không phải kết quả evaluation.
Run hiện chỉ có một training seed. Bộ greedy evaluation độc lập đã hoàn thành
trên sáu level với 100 seed cho mỗi tổ hợp policy--level. Vì vậy, độ biến thiên
trong bảng kết quả mô tả các episode của một policy đã học, không phải độ biến
thiên giữa nhiều lần huấn luyện độc lập.

Hai checkpoint cần đánh giá:

- `results/level4_continuous_dqn/seed_0/checkpoint_best.pt`
- `results/level4_continuous_dqn/seed_0/final_model.pt`

Checkpoint chính cho báo cáo nên được chọn theo quy tắc định trước. Đề xuất dùng
`checkpoint_best.pt` làm model chính vì được chọn bằng rolling mean 100 episode,
và dùng `final_model.pt` làm kiểm tra độ nhạy.

## 6. Protocol evaluation đã thực hiện

### 6.1 Thiết lập bắt buộc

- Chạy greedy hoàn toàn (`epsilon = 0`), không update model.
- Chạy cùng danh sách seed evaluation cho mọi level và checkpoint.
- Tối thiểu 100 episode/level; ưu tiên 200 nếu thời gian cho phép.
- Lưu raw result theo từng episode, không chỉ lưu giá trị trung bình.
- Ghi checkpoint hash/path, config level, ngày chạy và phiên bản code.
- Evaluation không render để tránh ảnh hưởng tốc độ; render chỉ dùng minh họa.

### 6.2 Ma trận evaluation

| Checkpoint | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|
| Best DQN | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 |
| Final DQN | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 |
| Random baseline | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 |

Random policy là baseline tối thiểu để chứng minh learned policy tốt hơn hành
vi ngẫu nhiên. Nếu đủ thời gian, bổ sung untrained-network baseline nhưng không
bắt buộc.

### 6.3 Metric cần báo cáo

Cho mọi level:

- Mean, standard deviation, median, min/max score.
- P25, P75 và P90 score.
- Mean episode steps và steps per food.
- Tỷ lệ chết do wall, obstacle, self-collision và timeout.

Cho Level 1, 2, 3, 5 và 6 có target:

- Success rate đạt target.
- Normalized score: `min(score / target_score, 1)`.

Cho Level 4 không có target:

- Không tự đặt success rate tùy ý sau khi nhìn kết quả.
- Dùng score distribution và survival/episode steps làm metric in-domain.

Không dùng một “generalization gap” duy nhất giữa L4 và level khác vì L4 không
có target, board size khác nhau và episode termination không đồng nhất. So sánh
theo từng level, success rate và normalized score sẽ trung thực hơn.

### 6.4 Điều kiện để phát biểu kết luận

- “Model chạy được trên level khác”: checkpoint load và hoàn tất episode.
- “Có khả năng transfer”: tốt hơn random baseline một cách rõ ràng trên metric
  định lượng.
- “Generalize tốt”: cần hiệu năng ổn định trên nhiều unseen level, không chỉ một
  episode đẹp hoặc riêng Level 1.
- “Generalize tới level khó hơn”: cần bằng chứng trên Level 5 và Level 6.

Nếu chưa chạy đủ evaluation, chỉ được viết “thiết kế hỗ trợ chạy đa level”, chưa
được viết “model tổng quát hóa tốt”.

## 7. Cấu trúc báo cáo

### 1. Introduction

Viết sau khi đã có Results:

- Vấn đề: RL agent thường phụ thuộc phân phối môi trường đã train.
- Khoảng trống: một policy Snake train đơn môi trường có chuyển sang board và
  obstacle layout khác không?
- Nêu RQ1–RQ3.
- Đóng góp dự kiến:
  1. Môi trường Snake tham số hóa sáu level.
  2. Dueling Double DQN với state/action độc lập kích thước bàn.
  3. Protocol đánh giá zero-shot từ một train level sang năm unseen level.
  4. Phân tích score, success và failure modes theo độ khó môi trường.

Không đưa PPO hoặc kết quả Phase 4 cũ vào contribution.

### 2. Related Work

Chỉ cần ba nhóm sát đề tài:

1. DQN, Double DQN và Dueling Network Architecture.
2. Generalization và zero-shot transfer trong reinforcement learning.
3. State representation/domain invariance trong game hoặc navigation.

Mọi citation phải kiểm tra tác giả, năm, venue và DOI/URL trước khi đưa vào
`references.bib`.

### 3. Methodology

#### 3.1 Snake environments

- Luật chơi, reward, termination và cách sinh food.
- Bảng sáu level và vai trò train/evaluation.
- Nhấn mạnh chỉ Level 4 xuất hiện trong replay buffer khi training.

#### 3.2 State and action representation

- Giải thích đủ 27 feature theo nhóm, không chỉ liệt kê con số.
- Ba action tương đối: straight, turn right, turn left.
- Lý do thiết kế có thể dùng chung network cho board size khác nhau.

#### 3.3 Dueling Double DQN

- Dueling value/advantage decomposition.
- Double-DQN target.
- Replay buffer, epsilon-greedy, target network, Huber loss và gradient clipping.
- Bảng hyperparameter lấy từ config của continuous run.

#### 3.4 Single-environment training

- Level 4 continuous, không target score.
- Điều kiện kết thúc episode và điều kiện dừng thủ công của toàn run.
- Cách lưu best checkpoint bằng rolling mean 100 episode.
- Nêu rõ chỉ có một training seed như một limitation.

#### 3.5 Cross-level evaluation

- Frozen checkpoint, greedy policy, không fine-tune.
- Danh sách seed, số episode, baseline và metric.
- Quy tắc chọn best checkpoint được quyết định trước khi xem test results.

### 4. Results

Chỉ mô tả điều quan sát được, chưa giải thích nguyên nhân:

1. **Training curve Level 4:** raw score và rolling mean theo environment steps.
2. **Bảng kết quả chính:** Best DQN trên L1–L6.
3. **Best versus final checkpoint:** kiểm tra model cuối có suy giảm không.
4. **DQN versus random baseline:** theo từng level.
5. **Success/normalized score:** cho năm level có target.
6. **Failure modes:** wall, obstacle, self-collision, timeout.
7. **Score distribution:** boxplot/violin plot thay vì chỉ bar chart mean.

Mỗi bảng/hình có 2–3 câu mô tả số liệu cụ thể. Không chọn vài episode render
đẹp làm bằng chứng chính.

### 5. Discussion

- Trả lời lần lượt RQ1–RQ3.
- Phân tích transfer từ L4 sang level nhỏ hơn và level lớn/khó hơn.
- Liên hệ kết quả với state 27 chiều và relative actions, nhưng không khẳng định
  quan hệ nhân quả vì chưa có ablation.
- Phân tích failure mode thay đổi theo board/obstacle layout.
- Threats to validity:
  - Chỉ một training seed.
  - Chỉ một môi trường train.
  - Các level khác đồng thời ở size, obstacle và target nên chưa cô lập biến.
  - Best checkpoint được chọn từ training rolling mean.
  - Không có ablation cho state feature và reward shaping.
  - Target/termination khác nhau khiến raw score không so sánh trực tiếp.

### 6. Conclusion

Ba đoạn ngắn:

1. Agent nào được train, train ở đâu và đánh giá thế nào.
2. Trả lời RQ bằng số thật từ evaluation.
3. Hướng tiếp theo: nhiều training seed, nhiều held-out layout, ablation,
   domain randomization và curriculum như future work.

### Abstract

Viết cuối cùng theo năm ý: bối cảnh → vấn đề → phương pháp → kết quả có số → ý
nghĩa. Không nhắc PPO. Không dùng training best episode 157 làm kết quả chính;
abstract phải dùng greedy evaluation trên unseen levels.

## 8. Thứ tự thực hiện và viết

### P6.1 — Đóng băng thiết kế

- [x] Bỏ PPO khỏi phạm vi đề tài.
- [x] Chốt Level 4 là môi trường train duy nhất.
- [x] Chốt L1, L2, L3, L5, L6 là môi trường unseen evaluation.
- [x] Chốt tên đề tài mới.
- [x] Ghi commit hash, checkpoint hash và snapshot config trong metadata.

### P6.2 — Hoàn thiện evaluation

- [x] Viết evaluation script chạy headless trên L1–L6.
- [x] Đánh giá `checkpoint_best.pt` với 100 seed/level.
- [x] Đánh giá `final_model.pt` với cùng protocol.
- [x] Chạy random-policy baseline.
- [x] Xuất raw CSV, summary CSV và metadata JSON.
- [x] Kiểm tra smoke test và các giá trị aggregate.

### P6.3 — Sinh bảng và hình tự động

- [x] Training curve theo environment steps.
- [x] Bảng kết quả L1–L6.
- [x] Biểu đồ kết quả theo level/checkpoint.
- [x] Boxplot score distribution.
- [x] Failure-mode chart.
- [x] Best-versus-final checkpoint comparison.

Script phân tích đặt trong `latex/scripts/`, output đặt trong
`latex/generated/`; không copy số thủ công vào nhiều file `.tex`.

### P6.4 — Viết phần lõi

- [x] Methodology.
- [x] Experimental Setup.
- [x] Results.
- [x] Discussion và Threats to Validity.

### P6.5 — Hoàn thiện bài

- [x] Xác minh tài liệu tham khảo và viết Related Work.
- [x] Viết Introduction dựa trên kết quả thật.
- [x] Viết Conclusion, sau đó Abstract.
- [x] Kiểm tra mọi số trong prose với CSV.
- [x] Bổ sung Wilson 95% CI cho success rate.
- [x] Sinh các dòng của bảng kết quả chính trực tiếp từ raw CSV.
- [ ] Build LaTeX sạch và sửa citation/cross-reference warning.
- [ ] Điền tên tác giả, đơn vị và email trước khi nộp.

## 9. Cấu trúc thư mục LaTeX

```text
latex/
├── README.md
├── main.tex
├── references.bib
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_results.tex
│   ├── 05_discussion.tex
│   └── 06_conclusion.tex
├── figures/
├── generated/
└── scripts/
```

## 10. Tiêu chí hoàn thành

Phase 6 hoàn tất khi:

- Báo cáo build được từ đầu.
- Cùng một frozen DQN checkpoint được đánh giá trên L1–L6.
- Kết quả dựa trên nhiều episode với seed tái lập và có random baseline.
- Mọi bảng/hình truy ngược được về raw CSV.
- Kết luận phân biệt rõ “chạy được”, “transfer” và “generalize tốt”.
- PPO và kết quả Phase 4 cũ không còn làm lệch câu chuyện nghiên cứu chính.

## 11. Tài liệu tham khảo và build

Các citation được lưu tập trung trong `references.bib`. `main.tex` dùng BibTeX
với style `abbrv`; không viết tay danh mục tài liệu tham khảo trong file paper.

Nhóm nguồn hiện đã có:

- Q-learning và giáo trình nền tảng reinforcement learning.
- DQN, Double DQN và Dueling Network Architecture.
- Generalization benchmarks: CoinRun và Procgen.
- Reproducibility/statistical evaluation trong deep RL.
- Ứng dụng deep reinforcement learning trực tiếp cho Snake.

Build đầy đủ citation theo thứ tự:

```powershell
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

LaTeX Workshop thường tự chạy chuỗi build này. Trước khi nộp, kiểm tra lại mọi
DOI/URL, bảo đảm không có citation hoặc reference warning và chỉ giữ những nguồn
thực sự được nhắc trong nội dung.
