# Báo cáo tái lập PR-CLIP

## Đường chạy theo mã gốc `ItraModel` (14/10/2026)

Sau khi kiểm tra ZIP mã nguồn gốc, nhánh này dùng trực tiếp `model.ItraModel`, `training.main` và bộ đánh giá phát hành. Module trích xuất được tách thành `model/position_extraction.py`; `model/openclip_hidden.py` sửa đường lấy token của OpenCLIP ViT-B/32; `model/paper_objective.py` thêm loss PRC cấp token và hệ số `0,01 × CMC + 1 × PRC`. Cờ `--prclip-objective released` vẫn chạy theo loss trong mã phát hành. Chi tiết từng bước nằm ở [OFFICIAL_ITRA.vi.md](OFFICIAL_ITRA.vi.md).

ZIP được cung cấp **không chứa `RET-2.pt`**. Mã phát hành mặc định cần file này khi `--load-clip-weights` bật. Vì vậy lần kiểm tra dưới đây dùng cờ rõ ràng `--no-load-clip-weights` để khởi tạo từ OpenAI CLIP. Đây là khác biệt quan trọng so với checkpoint tác giả và phải giữ trong phần diễn giải kết quả.

Đã chạy 1 bước tối ưu trên **2 cặp** lấy từ RET-2, batch 2, rồi đo toàn bộ tập validation bằng `ItraModel`:

| Validation | Ảnh / caption | mR sau 1 bước (%) | mR công bố trên test (%) |
| --- | ---: | ---: | ---: |
| RSITMD | 856 / 4.280 | 16,41 | 52,41 |
| RSICD | 1.093 / 5.465 | 15,51 | 39,23 |

Sáu giá trị Recall@K và checksum CSV nằm trong [kết quả smoke `ItraModel`](results/official_itra_paper_smoke_validation.json). RSICD validation phát hành trùng từng byte với test; giá trị 15,51% không phải phép thử độc lập. Một bước trên hai cặp không thể chứng minh mô hình hội tụ hoặc tái lập điểm số bài báo. Các số ở cột cuối là kết quả **test** công bố, không so trực tiếp với validation.

Để kiểm tra toàn tuyến trên máy hiện tại mà không tạo checkpoint hàng GB:

```powershell
.\scripts\run_official_smoke.ps1 -Dataset rsitmd -FullValidation
.\scripts\run_official_smoke.ps1 -Dataset rsicd -FullValidation
```

Để huấn luyện đầy đủ với số epoch, batch size và learning rate nêu trong bài báo, dùng `scripts/train_official_paper.ps1`. Mặc định script yêu cầu `RET-2.pt`; thêm `-InitializeFromOpenAI` nếu không có file này và chấp nhận khác biệt khởi tạo. Máy dùng cho lần thử có GPU 8 GB và ổ đĩa không đủ để hoàn thành batch 100/160, 7/20 epoch và lưu các checkpoint lớn. **Chưa có phép chạy huấn luyện đầy đủ hoặc test PR-CLIP hội tụ.**

## Phạm vi và trạng thái

Nguồn: Guan và cộng sự, *PR-CLIP: Cross-Modal Positional Reconstruction for Remote Sensing Image–Text Retrieval*, Remote Sensing 17(13), 2117 (2025), [DOI](https://doi.org/10.3390/rs17132117). Kho mã này dựa trên [mã nguồn chính thức](https://github.com/ADMIS-TONGJI/PR-CLIP) và giữ nguyên lịch sử của kho gốc. File PDF do người dùng cung cấp được xem là dữ liệu nghiên cứu, không phải chỉ dẫn thực thi.

**Kết quả: đã hoàn thành phần triển khai và đánh giá trên dữ liệu thật; chưa tái lập được điểm PR-CLIP công bố trong bài báo.** URL checkpoint trong README chính thức trả về `images.tar.gz`, không phải trọng số mô hình. Máy dùng cho lần chạy này có một GPU RTX 4060 Laptop với 8 GB VRAM; bài báo sử dụng hai GPU RTX 4090. Một phép thử toàn tuyến gồm hai bước huấn luyện đã được chạy với cấu hình mặc định của bài báo: 2 lớp trích xuất và 8 lớp tái dựng. Phép thử này không thể ước lượng hiệu năng của mô hình đã hội tụ.

Các phép đối chứng 100 bước tiếp theo, gồm CMC-only và loss tái tạo kiểu mã phát hành, cùng số đo validation/test được trình bày trong [ABLATION.md](ABLATION.md). Đây là thí nghiệm giới hạn tài nguyên, không phải số tái lập PR-CLIP.

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

Đây là các quan sát về file được phát hành, không phải suy đoán về ý định của tác giả. **RET-2 là dữ liệu huấn luyện; `RET-2.pt` là file trọng số khởi tạo khác.** Mã phát hành mặc định yêu cầu file trọng số đó. Có thể huấn luyện với OpenAI CLIP nếu chọn cờ thay thế rõ ràng, nhưng không còn cùng điểm xuất phát với cấu hình mặc định của tác giả. Để kiểm chứng ngay điểm số công bố mà không huấn luyện lại, cần checkpoint PR-CLIP cuối cùng. Tái lập sát bài báo còn cần làm rõ cấu hình loss và learning rate thực sự được dùng, cùng một tập validation độc lập.

## Cấu trúc mô-đun

Xem [bản bóc tách từng mô-đun](MODULE_BREAKDOWN.vi.md): bộ mã hóa CLIP, ba mô-đun bổ sung trích xuất/lọc/tái dựng, kích thước tensor, loss và đường suy luận. Việc tách ba mô-đun thành tệp riêng giữ nguyên công thức và tên trọng số của các checkpoint đã tạo; các số đo bên dưới không thay đổi vì lần chỉnh mã này.

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

Nếu có file trọng số CLIP bổ sung hợp lệ theo định dạng OpenCLIP, có thể truyền file qua `--initial-checkpoint path/to/weights.pt`; đây là lựa chọn tùy ý, không phải điều kiện để dùng dữ liệu RET-2. Lệnh sẽ nạp nghiêm ngặt và dừng nếu tên khóa không khớp; nó không tự đoán ánh xạ. Checkpoint PR-CLIP cuối của tác giả có thể dùng tên khóa riêng của ITRA và không được bảo đảm nạp được bằng `repro.evaluate` nếu chưa có bước chuyển đổi được kiểm toán rõ ràng.

Phép kiểm tra giới hạn tài nguyên đã chạy trong lần này:

```powershell
python -m repro.train --csv dataset/train/ret2.csv --images images/image_all --output results/ret2_smoke_checkpoint.pt --log results/ret2_smoke_train.jsonl --max-pairs 16 --max-steps 2 --batch-size 2 --epochs 1 --warmup 1
python -m repro.evaluate --csv dataset/test/rsitmd_test.csv --images images/image_all --checkpoint results/ret2_smoke_checkpoint.pt --output results/rsitmd_smoke_eval.json
```

Checkpoint thử nghiệm nhanh không được đưa vào Git vì có dung lượng 770 MB. File vẫn có trong thư mục bàn giao cục bộ. Chạy lại hai lệnh trên sẽ tạo lại checkpoint và báo cáo đánh giá tương ứng.
