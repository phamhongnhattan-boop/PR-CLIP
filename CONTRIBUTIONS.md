# Đóng góp và nguồn gốc

Kho mã này được tạo từ bản sao của [kho PR-CLIP chính thức của nhóm tác giả](https://github.com/ADMIS-TONGJI/PR-CLIP). Lịch sử commit và thông tin ghi công ban đầu vẫn được giữ nguyên. Công trình nghiên cứu và mô hình gốc thuộc về Jihong Guan, Yulou Shu, Wengen Li, Zihan Song và Yichao Zhang. Kho mã nguồn cho biết PR-CLIP được xây dựng dựa trên ITRA.

Các commit mới trên nhánh `reproduction` dùng tác giả **Codex <codex@local>** để phân biệt phần triển khai và kiểm toán này với công trình của nhóm tác giả bài báo, đồng thời không dùng danh tính của chủ máy. Các đóng góp mới gồm:

1. Bản triển khai PyTorch độc lập cho mục tiêu CMC hai chiều, PRC ở cấp token, module trích xuất xuyên phương thức, lọc và tái dựng được trình bày trong bài báo.
2. Công cụ tải dữ liệu, kiểm toán tập dữ liệu, tính metric truy hồi, huấn luyện và đánh giá.
3. Kết quả đo trên dữ liệu thật, bài kiểm thử, báo cáo tái lập và tài liệu học sâu bằng tiếng Việt.

Các thay đổi hiện chỉ nằm trong kho Git cục bộ này. Chưa tạo pull request hay kho GitHub từ xa. Checkpoint thử nghiệm nhanh dung lượng 770 MB nằm cục bộ trong `results/` và không được đưa vào lịch sử Git; có thể tạo lại bằng script huấn luyện.

## Thực hành của NHAT TAN

- Commit `57e2b52`: triển khai và kiểm thử CMC hai chiều trong `exercises/cmc_loss.py` với 2 cặp giả lập.