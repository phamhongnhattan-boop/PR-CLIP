# Đóng góp và nguồn gốc

Kho mã này được tạo từ bản sao của [kho PR-CLIP chính thức của nhóm tác giả](https://github.com/ADMIS-TONGJI/PR-CLIP). Lịch sử commit và thông tin ghi công ban đầu vẫn được giữ nguyên. Công trình nghiên cứu và mô hình gốc thuộc về Jihong Guan, Yulou Shu, Wengen Li, Zihan Song và Yichao Zhang. Kho mã nguồn cho biết PR-CLIP được xây dựng dựa trên ITRA.

Các commit triển khai và kiểm toán ban đầu dùng tác giả **Codex <codex@local>** để phân biệt với công trình của nhóm tác giả bài báo, đồng thời không dùng danh tính của chủ máy. Các commit thực hành trên nhánh `learn-cmc` ghi tác giả NHAT TAN; xem lịch sử Git để biết tác giả từng thay đổi. Các đóng góp mới gồm:

1. Bản triển khai PyTorch độc lập cho mục tiêu CMC hai chiều, PRC ở cấp token, module trích xuất xuyên phương thức, lọc và tái dựng được trình bày trong bài báo.
2. Công cụ tải dữ liệu, kiểm toán tập dữ liệu, tính metric truy hồi, huấn luyện và đánh giá.
3. Kết quả đo trên dữ liệu thật, bài kiểm thử, báo cáo tái lập và tài liệu học sâu bằng tiếng Việt.

Nhánh `learn-cmc` đã được đẩy lên [fork của NHAT TAN](https://github.com/phamhongnhattan-boop/PR-CLIP/tree/learn-cmc). Chưa tạo pull request lên kho của nhóm tác giả. Checkpoint thử nghiệm nhanh dung lượng khoảng 770 MB nằm cục bộ và không được đưa vào lịch sử Git; có thể tạo lại bằng script huấn luyện.

## Thực hành của NHAT TAN

- Commit `57e2b52`: triển khai và kiểm thử CMC hai chiều trong `exercises/cmc_loss.py` với 2 cặp giả lập.
- Commit `b02a9b1`: triển khai PRC cấp token, bỏ qua token đệm và kiểm thử kết quả 2,5.
- Commit `aaea929`: triển khai loss tái tạo theo mã phát hành trong `exercises/released_rec_loss.py` và kiểm thử ví dụ cho kết quả `1,0`.
- Commit `07b256c`: thêm chế độ loss `released_like` để so sánh riêng với công thức theo bài báo.
- NHAT TAN chạy ba phép huấn luyện đối chứng 100 bước trên 200 cặp RET-2 và đo validation cho hai checkpoint đầu. Codex kiểm tra log, đo validation cho `released_like`, đo test cho checkpoint CMC-only và tổng hợp bảng kết quả trong `ABLATION.md`; các số đo không được trình bày như kết quả PR-CLIP công bố.

## Bóc tách cấu trúc để trình bày

- Codex tách phần trích xuất, lọc và tái dựng đang nằm trong `repro/prclip.py` thành ba lớp trong ba tệp `repro/position_*.py`, giữ nguyên tên trọng số và kết quả tính toán để checkpoint cũ vẫn tương thích.
- Codex bổ sung `MODULE_BREAKDOWN.vi.md` trình bày cả bộ mã hóa CLIP, kích thước tensor, luồng tính loss và đường suy luận; đây là tài liệu giải thích bản triển khai độc lập, không tuyên bố đã tái lập điểm số của bài báo.
