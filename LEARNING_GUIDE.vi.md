# Học sâu PR-CLIP: từ bài toán đến giới hạn tái lập

Tài liệu này dựa trên Guan và cộng sự, [*PR-CLIP: Cross-Modal Positional Reconstruction for Remote Sensing Image–Text Retrieval*](https://doi.org/10.3390/rs17132117), và kiểm tra mã nguồn [ADMIS-TONGJI/PR-CLIP](https://github.com/ADMIS-TONGJI/PR-CLIP). Hãy đọc `REPRODUCTION.md` trước khi dùng bất kỳ con số đo nào ở đây để so sánh mô hình.

## 1. Bài toán thật sự là gì?

Đầu vào là một ảnh viễn thám và một mô tả ngôn ngữ. Ta cần xếp hạng ảnh phù hợp với câu (text-to-image, T2I) và xếp hạng câu phù hợp với ảnh (image-to-text, I2T). RSICD và RSITMD gán **năm mô tả cho mỗi ảnh**. Khi ảnh là truy vấn, chỉ cần một trong năm mô tả đúng xuất hiện trong top K để tính thành công. Khi câu là truy vấn, ảnh gốc của nó phải nằm trong top K.

CLIP truyền thống tạo một vector toàn cục cho ảnh và một vector toàn cục cho câu, rồi lấy cosine similarity. Điều này hiệu quả để tìm kiếm hàng loạt, nhưng cùng một tập đối tượng có thể tạo ra những cảnh khác nhau: “đường ở bên trái sân bay” khác “đường ở bên phải sân bay”. Một vector toàn cục có thể bỏ qua quan hệ vị trí này.

PR-CLIP đưa một **bài tập phụ khi huấn luyện** vào CLIP: mô hình phải tái dựng đặc trưng đã bị lọc thông tin vị trí của một modality bằng thông tin đầy đủ của modality còn lại. Khi tìm kiếm, chỉ dùng lại hai encoder CLIP. Bởi vậy, chi phí bổ sung tập trung ở huấn luyện, không nằm ở việc so sánh hàng triệu cặp ảnh–câu khi suy luận.

## 2. Sơ đồ và kích thước tensor

Với ViT-B/32 và ảnh 224×224, ảnh có 7×7 patch cộng CLS: `V ∈ R[B,50,768]`. CLIP text dùng tối đa 77 vị trí: `T ∈ R[B,77,512]`. Hai phép chiếu đưa chúng về bề rộng 512. Sau phép nối, encoder tương tác nhận `X ∈ R[B,127,512]`.

```text
Ảnh ── ViT ── V ── chiếu ──┐
                           ├─ nối → self-attention 2 lớp → Vp, Tp
Câu ── CLIP text ── T ─────┘                         │
     │                  │                            │
     ├─ (V - Vp) → lọc tuyến tính → decoder nhìn T → V tái dựng
     └─ (T - Tp) → lọc tuyến tính → decoder nhìn V → T tái dựng

Vector toàn cục của ViT/text ── InfoNCE hai chiều ── retrieval
V/T tái dựng so với V/T gốc ── PRC loss
```

`V` và `T` ở sơ đồ sau phép chiếu. Mã độc lập nằm ở `repro/prclip.py`; cách lấy token từ OpenCLIP nằm ở `repro/openclip_tokens.py`.

## 3. Đi qua từng phương trình

### 3.1. CMC: học cặp nào khớp

Chuẩn hóa vector toàn cục ảnh `v_i` và câu `t_j`, rồi tính `s_ij = v_i · t_j / τ`. Với một batch N cặp, nhãn đúng là đường chéo của ma trận `N×N`. Loss ảnh→câu là cross entropy theo từng hàng; loss câu→ảnh là cross entropy theo từng cột. `L_CMC = (L_I2T + L_T2I)/2`.

Ý nghĩa gradient: một cặp đúng được kéo gần, các cặp khác trong batch bị đẩy ra. Nhiệt độ `τ` nhỏ làm sự khác biệt giữa score lớn và nhỏ rõ hơn. Mã của chúng ta học `log(1/τ)` như một tham số, khởi tạo ở `τ=0,07`.

**Bẫy với bộ dữ liệu này:** năm caption của cùng một ảnh có thể cùng xuất hiện trong một batch. InfoNCE đường chéo khi đó xem các caption đúng khác là âm tính giả. Chỉ tăng batch size hoặc cộng dồn gradient không giải quyết lỗi nhãn này. Muốn loại bỏ cần loss nhiều dương tính hoặc lấy một caption/ảnh trong từng batch, nhưng đó là thay đổi so với mô tả triển khai gốc.

### 3.2. Trích xuất quan hệ vị trí xuyên modality

Mỗi token ảnh và mỗi token câu được chiếu về 512 chiều. Nối hai chuỗi và cho self-attention thấy toàn bộ 127 vị trí. Token ảnh có thể chú ý đến từ “bên trái”, “giữa”, “sát”; token câu có thể chú ý đến các patch chứa đường, sân vận động hoặc sông. Cắt đầu ra thành `V_p` và `T_p` theo đúng độ dài ban đầu.

Đây là **đặc trưng có tương tác**, không phải tọa độ được gắn nhãn. Bài báo gọi thành phần này là “thông tin vị trí” dựa trên giả thuyết học biểu diễn. Attention map có thể gợi ý mô hình nhìn vào đâu, nhưng riêng attention không chứng minh nó đã hiểu quan hệ không gian.

### 3.3. Lọc rồi tái dựng

Phương trình lọc là `V_f = Linear(V - V_p)` và `T_f = Linear(T - T_p)`. Tác giả diễn giải phép trừ là bỏ thông tin vị trí đã trích xuất. Về toán học, phép trừ này không bảo đảm phần còn lại hoàn toàn không còn vị trí: `V_p` có thể mang cả ngữ nghĩa, và `V_f` có thể còn vị trí. Tác dụng thực tế phải được kiểm chứng bằng ablation và probe.

Decoder ảnh lấy `V_f` làm target, `T` đầy đủ làm memory cho cross-attention, rồi cố dự đoán lại `V`. Decoder câu làm chiều ngược lại. Bản triển khai này dùng chung trọng số decoder cho hai chiều, giống lựa chọn trong mã tác giả. Padding sau token EOT của câu được che trong attention và loss.

Theo phương trình (12), `L_PRC` là trung bình bình phương khoảng cách L2 của token tái dựng và token gốc ở hai chiều. Tổng loss theo phương trình (13): `L = 0,01 L_CMC + 1 L_PRC`. Bản triển khai độc lập lấy công thức token-level đúng chữ. Nó cho PRC khoảng 6.743 rồi 6.375 trong hai bước smoke; CMC lần lượt khoảng 0,385 và 0,272. **Không nên diễn giải mức giảm qua hai bước là mô hình đã học retrieval**, vì trọng số nhỏ, dữ liệu rất ít, và hai batch khác nhau.

### 3.4. Điểm số lúc suy luận

Sau huấn luyện, bỏ encoder tương tác và decoder tái dựng. Mỗi ảnh/câu được mã hóa một lần. Score `S = normalize(T_global) · normalize(V_global)^T`. Điều này giúp lập chỉ mục ảnh trước và truy vấn nhanh. Các module tái dựng chỉ có ích nếu gradient huấn luyện của chúng làm encoder CLIP tốt hơn; một module phụ chỉ học trên backbone bị đóng băng sẽ không thay đổi retrieval.

## 4. Đọc đúng Recall@K

Với `N_image` ảnh và `N_text` caption, ma trận score có kích thước `N_text × N_image`. Đối với T2I, mỗi hàng có đúng một ảnh gốc. Đối với I2T, mỗi cột có năm caption gốc. `R@K` là tỷ lệ truy vấn có **ít nhất một** kết quả đúng trong K kết quả cao nhất; không phải trung bình tỷ lệ năm caption đúng được trả về. `mR` trong bài báo là trung bình cộng **sáu** số: T2I và I2T tại K=1,5,10. Nó không phải trung bình theo lớp dù văn bản diễn giải là “across categories”.

Ví dụ: một ảnh có năm caption, top-5 gồm một caption đúng và bốn caption sai. I2T R@5 của ảnh đó bằng 1; “fraction of relevant captions retrieved” sẽ là 1/5. Hai metric khác nhau và có thể dẫn đến kết luận khác nhau.

## 5. Kết quả thực nghiệm và điều chúng ta được phép kết luận

| Dataset | Paper PR-CLIP mR | OpenAI CLIP baseline trên test thật | Hai bước PR-CLIP |
| --- | ---: | ---: | ---: |
| RSITMD | 52,41% | 24,23% | 24,23% |
| RSICD | 39,23% | 15,69% | chưa đo |

Chúng ta đã xác minh code tính Recall@K bằng test nhân tạo và đo toàn bộ test set thật. Hai bước huấn luyện cho thấy forward, backward, optimizer, lưu và nạp checkpoint, rồi đánh giá đều chạy được trên RTX 4060. **Chúng ta chưa tái lập độ chính xác của bài báo.** Không thể dùng baseline OpenAI CLIP hoặc checkpoint hai bước làm bằng chứng PR-CLIP tốt/xấu hơn giá trị công bố.

## 6. Những điểm phải kiểm tra trước khi tin một reproduction

1. **Đúng trọng số khởi tạo?** Mã tác giả mặc định nạp `RET-2.pt`, nhưng URL README cho checkpoint trả về `images.tar.gz`. OpenAI CLIP chỉ là baseline thay thế.
2. **Đúng loss?** Công thức bài báo dùng PRC trên token; mã chính thức dùng MSE giữa các vector gộp đã chuẩn hóa. Hệ số CMC trong mã cũng không trùng `0,01` ghi trong bài. Đây là khác biệt về mục tiêu tối ưu, không phải sai số làm tròn.
3. **Đúng learning rate?** Hai script shell của tác giả dùng learning rate ngược với phần 4.4 của bài báo. Phải biết checkpoint cuối đã dùng cấu hình nào.
4. **Chọn checkpoint trên tập nào?** Script shell truyền test CSV vào đánh giá mỗi epoch, còn `training/main.py` chọn `best.pt` theo mR nhận được. RSICD val CSV còn giống hệt test CSV. Cần tập validation độc lập trước khi chọn epoch/hyperparameter, rồi đánh giá test đúng một lần.
5. **Đủ tài nguyên và batch?** Bài báo dùng hai RTX 4090, batch 100/160, 7/20 epoch. Một RTX 4060 8 GB chỉ chạy được batch 2 ở kiểm tra này. Batch nhỏ tạo tập âm tính InfoNCE khác hẳn; cộng dồn gradient không tái tạo đủ ma trận âm tính của batch lớn.
6. **Đúng định nghĩa dữ liệu?** File RET-2 có 65.460 cặp và 13.092 ảnh; test RSITMD có 452 ảnh, test RSICD có 1.093 ảnh. Kiểm toán đã tìm 0 ảnh thiếu, 0 trùng tên và 0 trùng byte giữa train/test; kiểm tra này không loại trừ ảnh gần trùng.

## 7. Lộ trình học và bài tập kiểm chứng

1. Đọc `repro/metrics.py`; tự tính bằng tay R@1/R@5 trên ví dụ 2 ảnh, mỗi ảnh 5 caption. Sau đó đổi thứ tự caption và xác nhận metric không đổi.
2. Đọc `repro/prclip.py`; ghi kích thước từng tensor ở một batch B=2. Giải thích vì sao cross-attention của decoder ảnh nhận `V_f` làm query và `T` làm key/value.
3. Chạy `pytest`, rồi đặt `lambda_prc=0`. Xem gradient của module tái dựng có mất đi không. Đây là kiểm tra ablation tối thiểu, khác với đo mR sau huấn luyện.
4. So sánh PRC token-level, PRC trên pooled vector và PRC chuẩn hóa theo số chiều. Giữ nguyên seed, batch, checkpoint đầu. Ghi lại tỷ lệ độ lớn gradient vào CLIP thay vì chỉ so giá trị loss.
5. Tạo validation độc lập, chọn epoch bằng validation, khóa cấu hình, rồi đánh giá test một lần. Lặp ít nhất ba seed và báo trung bình/độ lệch chuẩn. Nếu không có checkpoint gốc, kết quả chỉ là **tái triển khai**, không phải xác nhận số liệu tác giả.

## 8. Câu hỏi tự kiểm tra

**Vì sao decoder tái dựng không làm inference chậm?** Vì nó bị bỏ trong đường tính score; tác động kỳ vọng của nó đã được “chuyển” vào trọng số hai encoder qua gradient huấn luyện.

**Nếu PRC giảm nhưng Recall@K không tăng thì sao?** Module phụ có thể học sao chép đặc trưng dễ mà không giúp phân biệt cặp. Cần kiểm tra mR trên validation độc lập và ablation CMC/PRC.

**Vì sao không thể gọi kết quả 24,23% là thất bại tái lập PR-CLIP?** Nó là baseline OpenAI CLIP; checkpoint PR-CLIP công bố không được cung cấp qua liên kết của tác giả, còn checkpoint smoke mới học hai bước trên 16 cặp.
