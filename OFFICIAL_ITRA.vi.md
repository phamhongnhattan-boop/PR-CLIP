# Bóc tách đường chạy `ItraModel` của PR-CLIP

Tài liệu này nói về đường chạy dựa trên **mã tác giả** trong `model/model.py` và `training/`, không phải bản học tập độc lập `repro/`. PR-CLIP dùng hai bộ mã hóa CLIP ViT-B/32, rồi thêm ba phần huấn luyện: trích xuất vị trí xuyên phương thức, lọc phần dư và tái dựng.

## 1. Đặc trưng đầu vào

- Ảnh 224×224 đi qua ViT-B/32 và tạo 50 token (một CLS và 49 ô ảnh), mỗi token 768 chiều. `model/openclip_hidden.py` lấy đúng chuỗi token trước bước gộp/projection cuối; bài kiểm thử xác nhận đường gộp của nó khớp đầu ra OpenCLIP.
- Câu đi qua CLIP text encoder thành tối đa 77 token, mỗi token 512 chiều. Vị trí sau EOT được xem là padding trong chế độ `paper`.
- Hai phép chiếu `proj_image` và `proj_text` đưa cả hai chuỗi về 512 chiều: ảnh `[B,50,512]`, câu `[B,77,512]`.

## 2. Trích xuất vị trí xuyên phương thức: phần được tách thành mã riêng

`model/position_extraction.py` chứa hàm `extract_positional_residuals`, được `ItraModel.forward` gọi trực tiếp:

1. Ghép ảnh trước, câu sau: `[B,127,512]`.
2. Transformer encoder chung `it_encoder` có **2 lớp**, cho phép mỗi vị trí nhìn các vị trí còn lại ở cả hai phương thức.
3. Cắt 50 vị trí ảnh đầu và 77 vị trí câu sau, giữ nguyên thứ tự.
4. Trừ đặc trưng sau tương tác khỏi đặc trưng gốc tại **cùng vị trí**: `image - attended_image`, `text - attended_text`.

Kết quả vẫn là `[B,50,512]` và `[B,77,512]`. Đây là phần dư mang thông tin còn thiếu sau khi đối chiếu ảnh với câu. Việc tách hàm giữ lại `it_encoder` và tên tham số của mô hình gốc để không đổi cấu trúc checkpoint.

## 3. Lọc

Hai lớp tuyến tính `fc_image` và `fc_text` biến đổi phần dư của từng phương thức. Chúng là bộ lọc học được, không phải phép chọn thủ công những token quan trọng. Đầu ra có cùng kích thước với phần dư.

## 4. Tái dựng

`fusion_model` là BERT decoder **8 lớp** có cross-attention và được dùng chung cho hai chiều:

- Tái dựng token ảnh từ phần dư ảnh, dùng token câu làm nguồn thông tin.
- Tái dựng token câu từ phần dư câu, dùng token ảnh làm nguồn thông tin.

Ở chế độ `paper`, `model/paper_objective.py` đo trung bình bình phương khoảng cách L2 trên token ảnh và token câu hợp lệ, rồi lấy trung bình hai chiều. Mask bỏ vị trí câu sau EOT là lựa chọn triển khai được ghi rõ vì bài báo không nêu chi tiết padding. `training/train.py` kết hợp `0,01 × CMC + 1 × PRC` để cập nhật toàn bộ mô hình. CMC dùng InfoNCE giữa vector ảnh/câu toàn cục trong batch.

Ở chế độ `released`, mã phát hành tính MSE trên vector tái dựng **đã gộp token và chuẩn hóa**, sau đó `training/train.py` dùng `(1,5 × align_loss + rec_loss) / 4` khi không bật loss phụ. Hai mục tiêu này khác nhau; phải ghi rõ chế độ khi báo cáo kết quả.

## 5. Truy hồi sau huấn luyện

Khi đánh giá, `evaluation/retrieval.py` gọi `encode_image` và `encode_text`, so độ tương đồng của vector toàn cục và tính Recall@1/5/10, mR. Các bộ trích xuất, lọc và tái dựng chỉ tạo tín hiệu học trong lúc train; chúng không được gọi cho từng cặp ảnh–câu khi truy hồi.

## Chạy và giới hạn hiện tại

`scripts/run_official_smoke.ps1` kiểm tra toàn tuyến trên vài cặp RET-2 và có thể đo toàn bộ validation với `-FullValidation`. `scripts/train_official_paper.ps1` dùng số epoch, learning rate và batch size nêu trong bài báo. Mã phát hành mặc định yêu cầu file khởi tạo `RET-2.pt`, nhưng ZIP được cung cấp không chứa file này. Cờ `-InitializeFromOpenAI` cho phép huấn luyện từ OpenAI CLIP; kết quả của cách khởi tạo đó cần báo cáo riêng.

Máy dùng trong lần thử có GPU 8 GB nên các phép chạy batch 2 chỉ xác nhận đường chạy và độ đo. Chưa có huấn luyện hội tụ theo batch 100/160 và 7/20 epoch. `rsicd_val.csv` trùng với `rsicd_test.csv`; không thể chọn checkpoint bằng một file rồi trình bày file kia như test độc lập.
