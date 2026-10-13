# Đối chứng loss trên máy cá nhân

## Mục đích và cấu hình

Các phép chạy này so sánh ba mục tiêu huấn luyện trong **cùng một bản triển khai độc lập**, không phải huấn luyện PR-CLIP theo đúng cấu hình công bố. Cả ba bắt đầu từ OpenAI CLIP ViT-B/32, vì checkpoint tiền huấn luyện `RET-2.pt` không có trong dữ liệu được cung cấp. Mỗi phép chạy lấy cùng 200 cặp từ `dataset/train/ret2.csv` với seed 3407, huấn luyện 100 bước, batch 2, một epoch, learning rate 5e-6, warm-up 10 bước, hai lớp extraction và tám lớp reconstruction. Checkpoint khoảng 770 MB mỗi file được giữ cục bộ ngoài Git.

| Tên | Lệnh chọn mục tiêu | Loss thực sự dùng để cập nhật |
| --- | --- | --- |
| Theo công thức bài báo | `--loss-mode paper` | `0.01 × CMC + PRC` cấp token |
| Chỉ CMC | `--loss-mode paper --lambda-prc 0` | `0.01 × CMC` |
| Tái tạo kiểu mã phát hành | `--loss-mode released_like` | Tổng hai MSE trên vector tái tạo đã gộp token và chuẩn hóa |

Chế độ `released_like` chỉ bắt chước **loss tái tạo nội bộ** ở `model/model.py`, chưa bắt chước toàn bộ vòng huấn luyện của tác giả. Trong mã phát hành, `model/model.py` trả `lou_loss = rec_loss`, còn `training/train.py` tính thêm `align_loss` và tối ưu `(1.5 × (w_align × align_loss + ssl_loss) + lou_loss) / 4`. Vì thế không được gọi chế độ `released_like` là bản tái lập chính xác mã gốc. Mã phát hành cũng dùng module fusion khác với `repro/prclip.py`.

## Validation RSITMD

Tập validation có 856 ảnh và 4.280 caption. Tất cả giá trị mR dưới đây được đo trên cùng file `dataset/test/rsitmd_val.csv`. Mốc CLIP chưa huấn luyện được đo bằng `repro.evaluate`; ba checkpoint được đo bằng cùng chương trình.

| Mô hình | mR (%) |
| --- | ---: |
| OpenAI CLIP ban đầu | 16,7679 |
| Theo công thức bài báo, 100 bước | 1,4097 |
| Chỉ CMC, 100 bước | **18,5864** |
| Tái tạo kiểu mã phát hành, 100 bước | 0,7516 |

Trong phép thử nhỏ này, CMC-only cao hơn CLIP ban đầu 1,8185 điểm phần trăm. Hai chế độ có tái tạo làm giảm mạnh chất lượng truy hồi. Loss tái tạo giảm không bảo đảm thứ hạng ảnh–câu được giữ hoặc cải thiện. Kết quả không chứng minh PRC trong bài báo vô ích: trọng số RET-2, batch size, kiến trúc fusion và công thức loss của mã phát hành khác thiết lập ở đây. Quy mô loss cũng không đủ để suy ra quy mô gradient nếu chưa đo trực tiếp.

Các JSON validation và log từng bước được lưu trong `results/learner_ablation/`. Log giữ nguyên cấu hình và đường dẫn tuyệt đối của máy chạy. Checkpoint tương ứng nằm trong `work/tmp/` ngoài Git; có thể tạo lại từ các cấu hình trên.

## Đánh giá test sau khi chọn trên validation

Sau khi so sánh trên validation, checkpoint CMC-only 100 bước được chọn để đo test **một lần**. Đây là phép đối chứng CMC, không phải kết quả PR-CLIP. RSICD không có validation độc lập trong dữ liệu phát hành vì `rsicd_val.csv` trùng `rsicd_test.csv`.

| Tập test | OpenAI CLIP mR (%) | CMC-only 100 bước mR (%) | PR-CLIP trong bài báo mR (%) |
| --- | ---: | ---: | ---: |
| RSITMD | 24,2330 | 25,0959 | 52,41 |
| RSICD | 15,6908 | 17,3132 | 39,23 |

Các file chi tiết gồm `results/rsitmd_cmc_only_100steps.json`, `results/rsicd_cmc_only_100steps.json`, `results/rsitmd_openai_baseline.json` và `results/rsicd_openai_baseline.json`. Một seed và 200 cặp train chưa đủ để kết luận cải thiện ổn định. Không có phép chạy nào ở đây tái lập được kết quả PR-CLIP công bố.

## Để tiếp tục tái lập sát bài báo

1. Lấy đúng checkpoint RET-2 và xác minh định dạng trọng số trước khi nạp.
2. Làm rõ công thức loss và cấu hình thực sự dùng cho các bảng kết quả, vì bài báo và mã phát hành khác nhau.
3. Có tài nguyên phù hợp với batch size 100/160 và nhiều epoch; batch 2 thay đổi số mẫu âm của InfoNCE.
4. Dùng tập validation độc lập để chọn checkpoint, rồi chỉ đánh giá test sau khi cố định cấu hình.
