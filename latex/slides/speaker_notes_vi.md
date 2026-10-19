# Presentation script

Slides: English. Speaker Notes: Vietnamese. Main talk: slides 1–19; backup: 20–21.
Suggested duration: 12–15 minutes, excluding questions.

## 01. Cross-Level Generalization of DQN in Snake

Mục tiêu của đề tài là kiểm tra một Dueling Double DQN chỉ học trên Level 4 có thể chơi các level khác đến đâu. Đây là đánh giá zero-shot: giữ nguyên checkpoint và không fine-tune. Bài tập trung DQN. Tên đầy đủ trùng với báo cáo; bổ sung tên tác giả và đơn vị trước buổi trình bày.

## 02. The research question

Mở đầu bằng sự khác nhau giữa chơi tốt ở môi trường train và chuyển được sang môi trường khác. RQ1 đánh giá Level 4; RQ2 đánh giá năm level chưa thấy; RQ3 mô tả khác biệt về score, success và kết thúc episode. Các level thay đổi nhiều yếu tố cùng lúc nên RQ3 không khẳng định nguyên nhân riêng biệt.

## 03. Study at a glance

Tóm tắt thiết kế trước khi đi vào kỹ thuật: train một seed trên một level, dùng state 27 chiều và ba action tương đối. Đánh giá ba policy trên sáu level, mỗi tổ hợp 100 episode, tổng 1.800 episode. Random là baseline tối thiểu; final là kiểm tra độ nhạy checkpoint.

## 04. Six environments, one source

Level 4 là nguồn dữ liệu duy nhất của run được báo cáo. Level 1, 2, 3, 5, 6 đều chưa xuất hiện trong replay buffer của run này. Level 4 không có target, vì vậy không có success rate cho Level 4. Không so sánh raw score giữa các level như cùng một độ khó.

## 05. How the agent sees the game

State gồm 27 feature: danger và ray 6; hướng 4; thông tin food 6; vị trí đầu và chiều dài 3; safe-area 3; thông tin tail 5. Một số feature được chuẩn hóa theo kích thước bàn. Safe-area dùng BFS có giới hạn để ước lượng vùng có thể tới. Thiết kế cho phép một network chạy trên nhiều board; hiệu quả transfer phải được xác nhận bằng evaluation.

## 06. Relative actions and rewards

Ba action tương đối giữ cùng ý nghĩa ở mọi level: đi thẳng, phải, trái. Reward ăn food +10, va chạm -20, bước tiến gần food +0.015, bước khác -0.025. Các giá trị bước đã gồm step penalty. Timeout khi không ăn sau hai lần số ô bàn; bộ đếm reset mỗi lần ăn food.

## 07. Dueling Double DQN

Network dùng MLP hai lớp 256 ReLU, chia value và advantage. Dueling kết hợp V và A để ra Q cho ba action. Double DQN dùng online network chọn action ở next state, target network định giá action đó. Hai kỹ thuật này khác nhau: một là kiến trúc, một là cách tính target.

## 08. Training configuration

Run dùng seed 0, 14.293.725 environment steps; có 19.094 episode hoàn chỉnh và dừng trong episode 19.095. Best checkpoint xét ở mốc lưu 250 episode sau warm-up 500 episode, dựa rolling mean 100 episode. Chọn episode 17.000 với rolling mean 52,09. Các số training này không phải test score.

## 09. Learning on Level 4

Đường cong cho thấy learning tăng chậm giai đoạn đầu rồi tăng rõ sau đó. Rolling mean 52,09 là tiêu chí chọn checkpoint tại episode 17.000. Biến thiên episode còn lớn. Training dùng epsilon-greedy nên không đánh đồng điểm train với điểm greedy evaluation.

## 10. Frozen-policy evaluation

Mỗi policy–level chạy 100 episode với seed 1234–1333. Network giữ nguyên; không gradient và không fine-tune. Best là policy chính, final để xem kết luận nhạy với checkpoint không, random làm baseline. Wilson CI đo bất định qua episode của policy này, không đo biến thiên giữa training seed.

## 11. Zero-shot success across levels

Đây là kết quả chính của best checkpoint: 100%, 97%, 93%, 74%, 54% cho L1, L2, L3, L5, L6. Các thanh thể hiện success rate và đường ngang là Wilson CI 95%. L4 không có target nên không đưa vào biểu đồ success. Đặc biệt 100/100 không bảo đảm xác suất thành công thật là 100%.

## 12. Score distributions reveal instability

Không chỉ xem mean: Level 6 median 150 nhưng P25 chỉ 1, cho thấy nhiều episode hoặc hoàn thành target hoặc gần như không tiến triển. L4 mean 144,49 SD 27,28 và median 154. Các score không trực tiếp cùng thang độ khó vì có target và kích thước khác nhau.

## 13. Checkpoint choice changes transfer

Final checkpoint tăng L4 mean lên 153,83 nhưng L3 success giảm từ 93 xuống 83 và L5 từ 74 xuống 65. L6 mean tăng nhẹ nhưng success giảm 54 xuống 52. Đây là so sánh mô tả với cùng seed, không tuyên bố khác biệt có ý nghĩa thống kê. Random mean chỉ 0,02–0,10 và không hoàn thành target nào.

## 14. Where the policy fails

L5 timeout 25%, L6 timeout 40%, thường biểu hiện thiếu tiến triển hơn là đâm tường. L4 self-collision 79% nhưng không nên diễn giải là kém an toàn hơn một cách đơn giản: L4 không có target nên episode tiếp tục khi snake dài. Không thấy wall collision ngoài L4 trong 100 episode mỗi level của best checkpoint.

## 15. What the evidence supports

Kết quả trả lời RQ1: policy có hiệu năng nguồn cao trong evaluation. RQ2: có transfer nhưng chưa đồng đều, đặc biệt L5 L6. RQ3: phân bố và failure thay đổi giữa các level; chưa tách riêng ảnh hưởng size, obstacle, target. State và relative action là cơ chế hợp lý giúp dùng chung network nhưng chưa có ablation để khẳng định đóng góp nhân quả.

## 16. Limitations and next experiments

Nói rõ phạm vi một seed. 100 episode không thay thế được nhiều training run. Best chọn theo training rolling mean và chưa dùng validation riêng. Thí nghiệm tiếp theo theo ưu tiên: nhiều seed và layout mới, ablation state/reward/action, rồi so sánh curriculum và domain randomization với cùng step budget.

## 17. Conclusion

Chốt ba ý: một Dueling Double DQN học duy nhất Level 4 có thể chơi các level khác không fine-tune; kết quả tốt trên ba level nhỏ hơn nhưng thiếu ổn định ở hai level lớn; mạnh hơn trên nguồn không luôn có nghĩa transfer tốt hơn. Giữ kết luận trong phạm vi policy và suite hiện tại, không gọi là giải quyết mọi môi trường Snake.

## 18. Selected references

Các tài liệu nền tảng đã có trong references.bib. Slide này chỉ chọn nguồn cho DQN, Double DQN, Dueling, generalization và đánh giá RL. Các số liệu trong bài trình bày là từ thực nghiệm dự án, không phải kết quả của những paper này.

## 19. Thank you / Questions

Kết thúc phần chính. Khi được hỏi về seed, ablation hay unseen levels, có thể chuyển đến các backup slide sau.

## 20. Backup / detailed primary results

Bảng chi tiết best checkpoint để trả lời câu hỏi về mean, SD, median, success và normalized score. L4 N/A là do không có target. SD là độ lệch chuẩn mô tả qua episode, không phải khoảng tin cậy.

## 21. Backup / anticipated questions

Các câu trả lời cần giữ nhất quán: unseen là so với run được báo cáo; random baseline chưa đủ chứng minh ưu việt so với heuristic; không có PPO trong phạm vi nghiên cứu chính. Cần nhiều seed để kết luận độ ổn định của thuật toán. Điểm greedy cao hơn training có thể liên quan bỏ exploration, nhưng chưa có thí nghiệm tách riêng để khẳng định toàn bộ chênh lệch do exploration.
