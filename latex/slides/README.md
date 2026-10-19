# Snake DQN research presentation

File chính: `Snake_DQN_Research_Presentation.pptx`.
Bản đọc nhanh: `Snake_DQN_Research_Presentation.pdf`.
Xem toàn bộ bố cục: `contact_sheet.png`.

## Nội dung và cách sử dụng

- Slide tiếng Anh, lời dẫn tiếng Việt nằm trong **Speaker Notes** của từng slide.
- Lời dẫn cũng có trong `speaker_notes_vi.md` để luyện nói riêng.
- Slide 1–19: phần trình bày chính, dự kiến 12–15 phút.
- Slide 20–21: bảng chi tiết và câu hỏi dự phòng cho phần bảo vệ.
- Trong PowerPoint, dùng **Slide Show → Presenter View** để xem Notes khi trình bày.
- Điền tên tác giả, đơn vị và người hướng dẫn trên slide đầu trước khi nộp.

## Phong cách

Được tạo từ `Green and Black Minimalist Pixel Game Night Presentation.pptx`.
Giữ kích thước 16:9, nền xanh xám `#788B73`, khung đen, trang trí pixel ở
slide đầu/cuối và font tiêu đề Pixellance được nhúng trong template.
Nội dung khoa học dùng Aptos để đọc được bảng, công thức và số liệu.
Text, bảng và thanh success rate vẫn chỉnh sửa được trong PowerPoint;
các hình phân tích là ảnh từ pipeline báo cáo.

## Dữ liệu

Nội dung bám theo `../main.tex`, `../../configs/experiments/dqn_final_mastery.yaml`,
`../../results/cross_level_evaluation/summary.csv` và các biểu đồ trong `../generated/`.
Chỉ trình bày Dueling Double DQN train trên Level 4. Có best checkpoint,
final checkpoint và random baseline. Không bổ sung kết quả thí nghiệm mới.

## Tạo lại

Từ thư mục gốc repository, với Python và các thư viện tương ứng:

```powershell
py -m pip install --target .tools/presentation python-pptx matplotlib numpy
py latex/scripts/generate_report_assets.py
py latex/scripts/generate_presentation.py
```

Mở PPTX bằng PowerPoint và Export sang PDF. Nếu xuất slide PNG vào thư mục
`latex/slides/preview`, có thể tạo ảnh tổng quan bằng:

```powershell
py latex/scripts/preview_presentation.py
```

Các file đầu ra sẽ được ghi lại khi chạy script. File template gốc được giữ nguyên.
