# PR-CLIP: Mô hình truy hồi viễn thám đa phương thức dựa trên tái dựng thông tin vị trí xuyên phương thức

## Phần tái lập độc lập

Nhánh này giữ nguyên lịch sử kho mã của tác giả và bổ sung một bản triển khai độc lập, kiểm toán dữ liệu, kết quả đo trên dữ liệu thật cùng tài liệu học sâu. Hãy bắt đầu với [bản bóc tách từng mô-đun](MODULE_BREAKDOWN.vi.md), [báo cáo tái lập](REPRODUCTION.md), [hướng dẫn học sâu bằng tiếng Việt](LEARNING_GUIDE.vi.md) và [thông tin đóng góp](CONTRIBUTIONS.md). Điểm số PR-CLIP được công bố trong bài báo **chưa được tái lập**; liên kết checkpoint chính thức hiện trỏ tới một kho ảnh. Báo cáo trình bày kết quả chính xác, lệnh thiết lập và các khác biệt đã phát hiện giữa bài báo với mã nguồn.

Đây là kho mã của PR-CLIP: mô hình truy hồi viễn thám đa phương thức dựa trên tái dựng thông tin vị trí xuyên phương thức.

PR-CLIP được huấn luyện bằng nền tảng mã nguồn [`ITRA`](https://itra.readthedocs.io/).

![framework](framework.png)

### Yêu cầu môi trường

```
torch==2.5.1
torchvision==0.20.1
scikit-learn==1.5.1
transformers==4.45.2
adapter-transformers==4.0.0
open_clip_torch==2.20.0
clip-benchmark==1.6.1
numpy==1.24.3
pandas==2.2.3
pillow==11.0.0
```

### Tải dữ liệu

- Tải ảnh từ https://drive.google.com/file/d/15htMe2iE4kVujjWURxfCxZXk8vAAetoT/view?usp=drive_link và đặt trong thư mục `./images/`.

- Theo README gốc, tải checkpoint từ cùng liên kết trên và đặt trong `./cache/weights/`. Tuy nhiên, liên kết này hiện trả về kho ảnh chứ không phải checkpoint; xem báo cáo tái lập để biết chi tiết.

### RSITR:

Huấn luyện và kiểm thử trên RSICD:

```
sh train_rsicd.sh [gpu number] [port] [log_name]
```

Huấn luyện và kiểm thử trên RSITMD:

```
sh train_rsitmd.sh [gpu number] [port] [log_name] 
```

Chỉ kiểm thử:

```
python retrieval.py
```
