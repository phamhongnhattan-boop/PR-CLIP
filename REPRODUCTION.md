# Báo cáo tái lập PR-CLIP

## Phạm vi và trạng thái

Nguồn: Guan và cộng sự, *PR-CLIP: Cross-Modal Positional Reconstruction for Remote Sensing Image–Text Retrieval*, Remote Sensing 17(13), 2117 (2025), [DOI](https://doi.org/10.3390/rs17132117). Kho mã này dựa trên [mã nguồn chính thức](https://github.com/ADMIS-TONGJI/PR-CLIP) và giữ nguyên lịch sử của kho gốc. File PDF do người dùng cung cấp được xem là dữ liệu nghiên cứu, không phải chỉ dẫn thực thi.

**Kết quả: đã hoàn thành phần triển khai và đánh giá trên dữ liệu thật; chưa tái lập được điểm PR-CLIP công bố trong bài báo.** URL checkpoint trong README chính thức trả về `images.tar.gz`, không phải trọng số mô hình. Máy dùng cho lần chạy này có một GPU RTX 4060 Laptop với 8 GB VRAM; bài báo sử dụng hai GPU RTX 4090. Một phép thử toàn tuyến gồm hai bước huấn luyện đã được chạy với cấu hình mặc định của bài báo: 2 lớp trích xuất và 8 lớp tái dựng. Phép thử này không thể ước lượng hiệu năng của mô hình đã hội tụ.

| Tập dữ liệu | Ảnh / caption trong tập test | mR PR-CLIP trong bài báo | mR baseline OpenAI CLIP ViT-B/32 | mR PR-CLIP sau 2 bước thử |
| --- | ---: | ---: | ---: | ---: |
| RSITMD | 452 / 2.260 | 52,41% | 24,2330% | 24,2330% |
| RSICD | 1.093 / 5.465 | 39,23% | 15,6908% | chưa đánh giá |

Các số baseline và thử nghiệm nhanh được đo trên **toàn bộ CSV test do tác giả cung cấp** cùng kho ảnh được phát hành. Baseline không phải PR-CLIP; checkpoint thử nghiệm nhanh mới chỉ thấy 16 cặp RET-2 được lấy mẫu và trải qua 2 bước tối ưu. Hai giá trị RSITMD giống nhau tới bốn chữ số thập phân không chứng minh hai mô hình tương đương.

Sáu giá trị Recall@K chi tiết và metadata có thể đọc bằng máy nằm trong `results/rsitmd_openai_baseline.json`, `results/rsicd_openai_baseline.json` và `results/rsitmd_smoke_eval.json`. Loss theo từng bước huấn luyện nằm trong `results/ret2_smoke_train.jsonl`.

## Nội dung của bản triển khai

`repro/prclip.py` triển khai các phương trình (3)-(13): InfoNCE hai chiều theo kiểu CLIP, phép chiếu token ảnh/văn bản, encoder self-attention chung gồm hai lớp, phép trừ và lọc tuyến tính, decoder cross-attention tám lớp dùng chung, cùng phép tái dựng L2 bình phương hai chiều ở cấp token. `repro/openclip_tokens.py` cung cấp chuỗi token ảnh và văn bản ViT-B/32 từ OpenCLIP. Khi truy hồi, `repro/evaluate.py` chỉ dùng các encoder CLIP đúng như mô tả trong bài báo.

Trong phép kiểm tra tính nhất quán trực tiếp, vector ảnh và văn bản toàn cục do `encode_with_tokens` trả về khớp tuyệt đối với `encode_image` và `encode_text` gốc của OpenCLIP: sai khác tuyệt đối lớn nhất bằng 0,0 với một tensor kiểm thử 224×224 và một caption. Kích thước tensor token lần lượt là `[1,50,768]` và `[1,77,512]`.

Loss mặc định dùng **cách diễn giải sát công thức trong bài báo**. Nó lấy trung bình sai số L2 bình phương trên các token hợp lệ, gồm token CLS của ảnh và token EOT của văn bản, rồi lấy trung bình theo hai chiều. Lựa chọn này được nêu rõ vì bài báo không cung cấp chi tiết thực thi về mask và ranh giới token. Nhiệt độ tương phản được học từ giá trị khởi tạo 0,07.

## Kiểm toán dữ liệu

[Kho ảnh chính thức](https://drive.google.com/file/d/15htMe2iE4kVujjWURxfCxZXk8vAAetoT/view) đã được tải xuống: 1.461.515.987 byte, SHA-256 `f61d5d59e5e0696cb88685c34c2ee58961d68174e76a6988c9a83b2bb38f11d5`. Toàn bộ 65.460 dòng huấn luyện và hai CSV test đầy đủ đều tham chiếu tới ảnh tồn tại. RET-2 có 13.092 tên file ảnh riêng biệt, mỗi ảnh có năm caption. Không phát hiện tên file hoặc nội dung file giống hệt nhau giữa train và test. Phép kiểm tra này không loại trừ ảnh gần trùng hoặc ảnh được mã hóa lại.

`dataset/test/rsicd_val.csv` và `dataset/test/rsicd_test.csv` có cùng mã SHA-256 và trùng toàn bộ 1.093 tên file ảnh. `dataset/test/rsitmd_val.csv` và `dataset/test/rsitmd_test.csv` không trùng tên file. Chi tiết nằm trong `results/dataset_audit.json`.

## Các điểm không nhất quán ảnh hưởng tới khả năng tái lập

| Hạng mục | Bài báo | Kho mã chính thức | Hệ quả |
| --- | --- | --- | --- |
| Learning rate RSICD | 5e-5 | `train_rsicd.sh`: 5e-6 | Quỹ đạo tối ưu khác nhau. |
| Learning rate RSITMD | 5e-6 | `train_rsitmd.sh`: 5e-5 | Quỹ đạo tối ưu khác nhau. |
| Loss | `0,01 × CMC + 1 ×` PRC cấp token | `model/model.py` tính MSE trên đặc trưng tái dựng đã gộp và chuẩn hóa; `training/train.py` kết hợp `(1,5 × InfoNCE + MSE) / 4` | Mã công khai và công thức trong bài báo tối ưu hai mục tiêu khác nhau. |
| Chọn mô hình | Bài báo mô tả train/validation/test | Script huấn luyện chính thức truyền `*_test.csv` vào `--retrieval-data`; `training/main.py` chọn `best.pt` theo mean recall trên file này | Script phát hành chọn checkpoint bằng tập test. Điều này không chứng minh checkpoint dùng trong bài báo đã được chọn theo cách đó. |
| Validation RSICD | Hàm ý có tập validation riêng | `rsicd_val.csv` giống từng byte với `rsicd_test.csv` | Không thể dùng validation để chọn mô hình RSICD một cách không thiên lệch nếu chưa tạo split mới. |
| Liên kết checkpoint | Cần trọng số huấn luyện trước để đánh giá chính xác | README trỏ cùng một kho ảnh 1,4 GB cho cả ảnh và checkpoint | Không thể đánh giá checkpoint công bố từ file được liên kết. |
| Script truy hồi độc lập | README cung cấp lệnh test | `retrieval.py` dùng các biến chưa định nghĩa `DATASET_PATH`, `model_name`, `dataset_name` và truyền CSV cho bộ nạp JSON | Lệnh trong README không chạy được nếu không sửa. |
| Dependency được khai báo | README liệt kê các gói cần thiết | `model/model.py` luôn import `cn_clip` và `sentence_transformers`, nhưng README không khai báo hai gói này | Import mô hình gốc thất bại trong môi trường sạch; gói `repro` độc lập tránh các dependency tùy chọn đó. |

Đây là các quan sát về file được phát hành, không phải suy đoán về ý định của tác giả. Một phép tái lập sát bài báo cần trọng số RET-2 đã huấn luyện trước, checkpoint PR-CLIP cuối cùng, cấu hình loss và learning rate thực sự được dùng, cùng một tập validation độc lập.

## Chạy lại thí nghiệm

Dùng Python 3.12 và cài `requirements-repro.txt`. Trên Windows có GPU NVIDIA, trước hết cài PyTorch 2.5.1 CUDA 12.1 từ [kho wheel chính thức của PyTorch](https://download.pytorch.org/whl/cu121), rồi cài các dependency còn lại. Lần chạy này dùng PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, OpenCLIP 2.20.0, NumPy 2.5.3 và Pillow 12.3.0.

Chạy từ thư mục gốc của repo:

```powershell
python -m repro.prepare_data --directory images
python -m repro.audit --dataset dataset --images images/image_all --output results/dataset_audit.json --hash-images
python -m pytest -q tests/test_repro.py
python -m repro.evaluate --csv dataset/test/rsitmd_test.csv --images images/image_all --output results/rsitmd_openai_baseline.json
python -m repro.evaluate --csv dataset/test/rsicd_test.csv --images images/image_all --output results/rsicd_openai_baseline.json
```

Cấu hình mặc định trong `repro.train` theo công thức bài báo dành cho RSITMD là: 7 epoch, batch 100, learning rate 5e-6, 100 bước warm-up, weight decay 0,5, gradient clipping 50, hai lớp trích xuất, tám lớp tái dựng, λCMC=0,01 và λPRC=1. Với RSICD, dùng `--epochs 20 --batch-size 160 --lr 5e-5`. Cả hai cùng dùng CSV huấn luyện RET-2. Quá trình huấn luyện chạy đơn tiến trình và cần đủ bộ nhớ GPU; giảm `--batch-size` sẽ thay đổi tập mẫu âm của InfoNCE nên **không tương đương** với thí nghiệm trong bài báo.

Nếu có file trọng số RET-2 hợp lệ theo định dạng OpenCLIP, truyền file qua `--initial-checkpoint path/to/RET-2.pt`. Lệnh sẽ nạp nghiêm ngặt và dừng nếu tên khóa không khớp; nó không tự đoán ánh xạ. Checkpoint PR-CLIP cuối của tác giả có thể dùng tên khóa riêng của ITRA và không được bảo đảm nạp được bằng `repro.evaluate` nếu chưa có bước chuyển đổi được kiểm toán rõ ràng.

Phép kiểm tra giới hạn tài nguyên đã chạy trong lần này:

```powershell
python -m repro.train --csv dataset/train/ret2.csv --images images/image_all --output results/ret2_smoke_checkpoint.pt --log results/ret2_smoke_train.jsonl --max-pairs 16 --max-steps 2 --batch-size 2 --epochs 1 --warmup 1
python -m repro.evaluate --csv dataset/test/rsitmd_test.csv --images images/image_all --checkpoint results/ret2_smoke_checkpoint.pt --output results/rsitmd_smoke_eval.json
```

Checkpoint thử nghiệm nhanh không được đưa vào Git vì có dung lượng 770 MB. File vẫn có trong thư mục bàn giao cục bộ. Chạy lại hai lệnh trên sẽ tạo lại checkpoint và báo cáo đánh giá tương ứng.
