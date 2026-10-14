# PR-CLIP: bóc tách từng mô-đun từ đầu vào đến truy hồi

Tài liệu này mô tả **bản triển khai độc lập trong nhánh này** và đối chiếu với [bài báo PR-CLIP](https://doi.org/10.3390/rs17132117). Bài báo gọi kiến trúc là **4 mô-đun chính**: (0) bộ mã hóa đơn phương thức CLIP, (1) trích xuất thông tin vị trí xuyên phương thức, (2) lọc thông tin vị trí đơn phương thức, (3) tái dựng thông tin vị trí xuyên phương thức. Vì CLIP là nền tảng có sẵn, có thể gọi ba phần sau là **3 mô-đun PR-CLIP bổ sung**. CMC và PRC là **hàm loss**, không phải mô-đun thứ tư/thứ năm.

## 1. Bản đồ mã nguồn

| Khâu | Mã trong bản triển khai này | Vai trò |
| --- | --- | --- |
| 0. Mã hóa ảnh/câu bằng CLIP | [`repro/openclip_tokens.py`](repro/openclip_tokens.py) — `encode_with_tokens` | Sinh token, vector toàn cục và mask token đệm. |
| 1. Trích xuất xuyên phương thức | [`repro/position_extraction.py`](repro/position_extraction.py) — `CrossModalPositionExtractor` | Cho token ảnh và câu tương tác trong cùng Transformer encoder. |
| 2. Lọc từng phương thức | [`repro/position_filtering.py`](repro/position_filtering.py) — `UnimodalPositionFilter` | Lấy token gốc trừ token vừa trích xuất, rồi chiếu tuyến tính. |
| 3. Tái dựng từ phương thức kia | [`repro/position_reconstruction.py`](repro/position_reconstruction.py) — `CrossModalPositionReconstructor` | Transformer decoder dự đoán token ảnh từ câu và token câu từ ảnh. |
| Ghép luồng, tính loss | [`repro/prclip.py`](repro/prclip.py) — `PositionalReconstruction` | Gọi ba mô-đun trên, tính CMC/PRC, trả `loss` để train. |
| Huấn luyện và truy hồi | [`repro/train.py`](repro/train.py), [`repro/evaluate.py`](repro/evaluate.py) | Cập nhật trọng số CLIP lúc train; lúc đánh giá chỉ dùng CLIP. |

Lưu ý: `PositionalReconstruction` là **lớp điều phối** cả quy trình huấn luyện, không có nghĩa mô hình chỉ có một mô-đun tái dựng.

## 2. Ký hiệu và ví dụ kích thước

`B` là số cặp ảnh–câu trong batch; `Ni` là số token ảnh, `Nt` là số vị trí token câu, `D` là chiều đặc trưng chung. Với cấu hình ViT-B/32, ảnh 224×224, `Ni=50` (49 patch + CLS), `Nt=77`, `D=512`. Nếu `B=2`:

| Tên | Kích thước | Nghĩa |
| --- | --- | --- |
| `image_tokens` | `[2,50,768]` | Token ảnh do CLIP tạo ra. |
| `text_tokens` | `[2,77,512]` | Token câu, gồm cả các vị trí đệm. |
| `image_global`, `text_global` | Mỗi tensor `[2,512]` | Một vector cho mỗi ảnh/câu, dùng để so độ giống nhau. |
| `text_padding_mask` | `[2,77]` | `True` ở vị trí sau EOT cần bỏ qua. |
| `image`, `text` sau chiếu | `[2,50,512]`, `[2,77,512]` | Hai chuỗi cùng số chiều, sẵn sàng tương tác. |

## 3. Mô-đun 0 — bộ mã hóa CLIP

**Đầu vào:** ảnh RGB `[B,3,224,224]` và mã token câu `[B,77]`.

**Xử lý:** nhánh ViT chia ảnh thành patch, thêm token CLS và mã hóa; nhánh text mã hóa từ/các vị trí văn bản. `encode_with_tokens` giữ lại **chuỗi token** cho bài tập tái dựng và lấy **vector toàn cục** cho truy hồi. Mask đánh dấu các vị trí sau token EOT; EOT vẫn là token thật.

**Đầu ra:** `image_tokens`, `text_tokens`, `image_global`, `text_global`, `text_padding_mask`. CLIP được cập nhật bằng gradient trong `repro/train.py`, trừ khi chạy tùy chọn đóng băng backbone.

**Tại sao cần hai loại đầu ra?** Ba mô-đun bổ sung thao tác trên từng token để học quan hệ vùng ảnh–từ; phép truy hồi cuối dùng vector toàn cục để chấm điểm nhanh.

## 4. Mô-đun 1 — trích xuất vị trí xuyên phương thức

**Đầu vào:** `image [B,Ni,D]`, `text [B,Nt,D]`, `text_padding_mask [B,Nt]`. Hai phép chiếu tuyến tính trong `repro/prclip.py` đổi 768→512 cho ảnh và 512→512 cho câu.

**Từng bước:**

1. Nối token ảnh trước, token câu sau: `joint = concat(image,text)` → `[B,Ni+Nt,D]`, ví dụ `[2,127,512]`.
2. Nối mask ảnh toàn `False` với mask câu; Transformer bỏ qua token câu đệm khi attention đọc các khóa/giá trị.
3. Chạy Transformer encoder self-attention 2 lớp theo cấu hình mặc định. Token ảnh có thể nhận thông tin từ câu; token câu có thể nhận thông tin từ patch ảnh.
4. Cắt đầu ra tại `Ni`: `image_position = joint_out[:,:Ni]` và `text_position = joint_out[:,Ni:]`, lần lượt `[B,Ni,D]` và `[B,Nt,D]`.

**Ý nghĩa:** tác giả gọi đầu ra này là thông tin vị trí xuyên phương thức. Về mặt mã, đó là đặc trưng được học qua tương tác; không có nhãn tọa độ bắt buộc hay bảo đảm mọi chiều đều biểu diễn vị trí.

## 5. Mô-đun 2 — lọc vị trí đơn phương thức

**Đầu vào:** token đã chiếu `image`, `text` và hai đầu ra trích xuất ở bước 4; mỗi cặp có cùng kích thước.

**Từng bước:**

```text
image_filtered = Linear_image(image - image_position)  # [B,Ni,D]
text_filtered  = Linear_text(text - text_position)    # [B,Nt,D]
```

Hai lớp Linear có trọng số **khác nhau** cho ảnh và câu. Phép trừ diễn đạt ý tưởng làm thiếu thông tin mà mô-đun sau phải bù lại từ phương thức đối diện. Nó không chứng minh toán học rằng mọi thông tin vị trí đã biến mất; điều đó cần được kiểm chứng bằng thực nghiệm.

## 6. Mô-đun 3 — tái dựng xuyên phương thức

**Đầu vào/đầu ra theo từng chiều:**

| Chiều | Target/query của decoder | Memory/key-value từ phương thức kia | Token dự đoán |
| --- | --- | --- | --- |
| Tái dựng ảnh | `image_filtered [B,Ni,D]` | `text [B,Nt,D]`, che padding | `image_reconstructed [B,Ni,D]` |
| Tái dựng câu | `text_filtered [B,Nt,D]`, che padding target | `image [B,Ni,D]` | `text_reconstructed [B,Nt,D]` |

Decoder làm self-attention trên target rồi cross-attention sang memory; phần feed-forward tạo token mới. **Cùng một instance Transformer decoder 8 lớp** được gọi hai lần, nên hai chiều dùng chung trọng số. Số token đầu ra luôn bằng số token target: câu có 77 vị trí không làm đầu ra ảnh thành 77 token.

Ví dụ “đường bên trái sân bay”: khi tái dựng token ảnh đã bị lọc, decoder có thể lấy ngữ cảnh từ các từ “đường”, “bên trái”, “sân bay”. Bài tập này tạo tín hiệu học cho encoder; bản thân decoder không được đem đi so tất cả ảnh với câu khi tìm kiếm.

## 7. Loss và đường gradient

`repro/prclip.py` tính hai loại loss sau khi chạy đủ ba mô-đun:

1. **PRC:** với mỗi token, cộng bình phương sai lệch trên `D` chiều giữa token tái dựng và token gốc. Lấy trung bình trên token ảnh và batch. Với câu, chỉ lấy trung bình trên token thật (`~text_padding_mask`). Sau đó lấy trung bình hai hướng ảnh/câu. PRC ở đây tính trên **đặc trưng token**, không phải khôi phục từng pixel của ảnh.
2. **CMC:** chuẩn hóa hai vector toàn cục, tính `logits = image_global @ text_global.T × scale` kích thước `[B,B]`. Cặp đúng nằm trên đường chéo. Tính cross-entropy ảnh→câu và câu→ảnh, rồi lấy trung bình.

Ở chế độ mặc định `paper`: `loss = 0,01 × CMC + 1 × PRC`. Lệnh `backward()` trong `repro/train.py` đẩy gradient về ba mô-đun và CLIP. Ở chế độ `released_like`, `loss` là bài tập tái dựng vector gộp trong `exercises/released_rec_loss.py`; CMC và PRC vẫn được ghi log để so sánh, nhưng **không** cộng vào `loss` huấn luyện của chế độ này. `released_like` chỉ mô phỏng một phần mã tác giả, không phải tái tạo nguyên xi checkpoint hay kết quả bài báo.

## 8. Khi đánh giá và tìm kiếm

`repro/evaluate.py` chỉ nạp nhánh `clip` từ checkpoint: mã hóa từng ảnh/câu thành vector `[512]`, chuẩn hóa và lấy tích vô hướng để xếp hạng. Ba mô-đun trích xuất/lọc/tái dựng là **đường huấn luyện phụ**. Chúng giúp truy hồi chỉ nếu gradient đã làm trọng số CLIP học tốt hơn. Vì thế PRC giảm không tự động bảo đảm R@K/mR tăng.

## 9. Có thể nói gì về mức tái lập hiện tại?

Các file ba mô-đun giúp **thấy rõ cấu trúc triển khai**; việc tách file không làm thay đổi công thức, checkpoint hay điểm số đã đo. Repo đã có train thử và đánh giá validation/test, nhưng chưa đạt điểm bài báo công bố. Thiếu checkpoint khởi tạo RET-2 được tác giả sử dụng, batch/epoch/tài nguyên chưa tương đương, và mục tiêu loss trong bài báo khác một phần mã phát hành. Xem `REPRODUCTION.md` và `ABLATION.md` trước khi so sánh các con số.
