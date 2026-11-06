# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU (hoặc T4 GPU)

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

- **Record hạng 1:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

```json
{
  "sample_id": "traffic",
  "coco_image_id": 210273,
  "image_width": 640,
  "image_height": 428,
  "task": "image_classification",
  "taxonomy_name": "ImageNet-1K",
  "model_file": "yolo11n-cls.pt",
  "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
  "ultralytics_version": "8.4.145",
  "rank": 1,
  "class_id": 468,
  "class_name": "cab",
  "score": 0.510915
}
```

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

- **Trả lời:** Record này nói về **cả bức ảnh**. Bởi vì bài toán Phân loại ảnh (Image Classification) chỉ gán một nhãn tổng thể đại diện cho chủ thể hoặc ngữ cảnh chính của toàn bộ bức ảnh (Image-level label), mô hình không xác định tọa độ vị trí hay khoanh vùng riêng từng vật thể như bài toán Detection.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

- **Trả lời:** Danh sách tên lớp đi kèm với **taxonomy/checkpoint** có sẵn (ở đây là bộ danh mục ImageNet-1K gồm 1000 lớp cố định), không phải do model tự nghĩ ra.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

- **Trả lời:** Cần ghi đầy đủ cả 3 trường vì:
  - `class_id`: Là mã định danh số duy nhất trong hệ thống, giúp máy tính xử lý và đánh giá thống kê chuẩn xác, tránh nhầm lẫn do cách viết hoặc dịch thuật.
  - `class_name`: Giúp con người (annotator, developer) đọc hiểu trực tiếp nhãn ngữ nghĩa đó là gì.
  - `taxonomy_name`: Xác định nhãn này thuộc hệ quy chiếu / bộ từ điển phân loại nào (ở đây là ImageNet-1K gồm 1000 lớp), phân biệt với các taxonomy khác (như COCO-80), vì cùng một mã hoặc tên ở các bộ nhãn khác nhau có thể mang định nghĩa khác nhau.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

- **Trả lời:** Nếu trong ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần nói rõ quy tắc ưu tiên chọn lớp chính: ví dụ ưu tiên chọn đối tượng chiếm diện tích lớn nhất (dominant subject), vật thể nằm ở vị trí trung tâm, hoặc chủ đề mang lại nhiều giá trị sử dụng nhất cho bài toán.

6. Vì sao không được lấy `score` của model làm ground truth?

- **Trả lời:** Không được lấy `score` làm ground truth vì `score` (confidence) chỉ là độ tự tin toán học do mô hình tự dự đoán ra; mô hình hoàn toàn có thể có score rất cao nhưng vẫn dự đoán sai (nhận nhầm đối tượng). Trong khi đó, `ground truth` là nhãn thực tế khách quan do con người thẩm định và xác nhận theo quy tắc (guideline).

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

* **Thống kê số lượng prediction theo ngưỡng:**
  - Ngưỡng `0.20`: 17 predictions
  - Ngưỡng `0.35`: 11 predictions
  - Ngưỡng `0.60`: 6 predictions

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

- **Record trích xuất:** `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

```json
{
  "sample_id": "kitchen",
  "coco_image_id": 397133,
  "image_width": 640,
  "image_height": 427,
  "task": "object_detection",
  "taxonomy_name": "COCO-80",
  "model_file": "yolo11n.pt",
  "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
  "ultralytics_version": "8.4.145",
  "score_threshold": 0.35,
  "class_id": 0,
  "class_name": "person",
  "score": 0.912625,
  "coordinate_unit": "pixel",
  "bbox_format": "xyxy",
  "bbox_xyxy": [
    385.33,
    69.24,
    498.92,
    348.92
  ],
  "bbox_width": 113.58,
  "bbox_height": 279.68
}
```

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

- **Góc trên-trái:** (x_min, y_min) = (385.33, 69.24) pixel
- **Góc dưới-phải:** (x_max, y_max) = (498.92, 348.92) pixel
- **Kích thước:** Chiều rộng = 113.58 pixel, Chiều cao = 279.68 pixel

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

- **Ngưỡng 0.20:** 17 predictions
- **Ngưỡng 0.60:** 6 predictions

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

- **Trả lời:** Khi hạ ngưỡng (ví dụ từ 0.35 xuống 0.20), số lượng prediction tăng lên (từ 11 lên 17 hộp), dẫn đến xuất hiện thêm nhiều hộp dự đoán sai hoặc hộp rác (False Positives). Lúc này người reviewer (QC) phải tốn thêm nhiều thời gian và công sức để rà soát, lọc bỏ các hộp rác, loại bỏ các hộp thừa hoặc chỉnh sửa các hộp vẽ trùng lặp để bảo đảm chất lượng ground truth.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

- **Quy tắc vẽ box:** Bounding box phải vẽ ôm khít (tight) viền ngoài cùng của vật thể (không vẽ quá rộng bao cả background thừa, không vẽ quá chật cắt mất một phần vật thể). Mỗi lớp đối tượng (class) được phân biệt bằng nhãn và màu sắc quy định.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

- **Trả lời:** Trên ảnh `kitchen`, khu vực bàn tay của người (`person`) bị cắt mép ở cạnh phải của ảnh. Em sẽ viết trường hợp này thành **quy tắc trong guideline** (vì hiện tượng vật thể bị cắt mép ở biên ảnh là rất phổ biến; guideline sẽ quy định rõ: annotator chỉ vẽ bounding box bám sát theo phần nhìn thấy được của vật thể trong khung hình, không tự ý kéo box ra ngoài biên ảnh).

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

- **Record trích xuất:** `instance_id`=kitchen-004 ; `class_name`=potted plant ; `score`=0.632235 ; `polygon_point_count`=100 ; `polygon_xy`=[[1.0, 2.0], [1.0, 148.0], [3.0, 148.0], [3.0, 147.0], [4.0, 146.0], ...]

```json
{
  "sample_id": "kitchen",
  "coco_image_id": 397133,
  "image_width": 640,
  "image_height": 427,
  "task": "instance_segmentation",
  "taxonomy_name": "COCO-80",
  "model_file": "yolo11n-seg.pt",
  "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
  "ultralytics_version": "8.4.145",
  "score_threshold": 0.35,
  "instance_id": "kitchen-004",
  "class_id": 58,
  "class_name": "potted plant",
  "score": 0.632235,
  "coordinate_unit": "pixel",
  "bbox_format": "xyxy",
  "bbox_xyxy": [
    0.05,
    1.4,
    63.45,
    148.89
  ],
  "polygon_point_count": 100,
  "polygon_xy": [
    [1.0, 2.0],
    [1.0, 148.0],
    [3.0, 148.0],
    [3.0, 147.0],
    [4.0, 146.0]
  ]
}
```

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

- **Trả lời:** So với bounding box (chỉ cung cấp 4 tọa độ của một hình chữ nhật bao quanh vật thể), polygon cung cấp một chuỗi các tọa độ điểm nối liên tiếp ôm sát theo đường viền thực tế của vật thể. Điều này giúp loại bỏ hoàn toàn các vùng pixel nền không thuộc về vật thể và mô tả chính xác hình dạng hình học phức tạp (cong, lồi lõm) của đối tượng.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

- **Trả lời:** 
  - `instance_id` dùng để phân biệt các cá thể riêng biệt trong cùng một bức ảnh tĩnh (ví dụ tách riêng Người 1, Người 2 dù cùng chung `class_id` là "person").
  - Nó không phải `class_id` vì `class_id` chỉ gọi tên loại đối tượng chung chung.
  - Nó không phải `tracking_id` vì `tracking_id` có tính lịch sử/thời gian (giữ nguyên ID của cùng một vật thể xuyên suốt các frame trong video). Còn `instance_id` chỉ có giá trị duy nhất trong một bức ảnh tĩnh, sang ảnh khác sẽ được reset và đánh số lại từ đầu.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

- **Quy tắc vẽ biên mask:**
  - **Bám sát viền thực:** Biên polygon phải nằm chính xác tại pixel giao thoa giữa vật thể và nền, không bao thừa nền và không cắt lẹm vào vật thể.
  - **Xử lý che khuất:** Thống nhất chỉ chấm điểm theo phần nhìn thấy được (visible part) của vật thể.
  - **Luôn khép kín:** Biên phải là một đa giác đóng hoàn chỉnh để tạo thành mặt nạ (mask) tô màu được, không bị đứt gãy.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

- **Vị trí cụ thể:** Khu vực các chiếc bát chồng lên/chạm nhau ở góc dưới cùng bên trái ảnh (gồm các nhãn bowl 0.70, bowl 0.55, bowl 0.74). Các vật thể này đè lên nhau, tạo ra ranh giới chồng lấn phức tạp và mờ giữa các viền bát.
- **Lựa chọn:** Em sẽ dùng **guideline** vì hiện tượng đồ vật xếp chồng, che khuất một phần lên nhau là hiện tượng rất phổ biến trong các dataset thị giác máy tính. Guideline cần hướng dẫn rõ: vẽ mask riêng cho phần nhìn thấy của từng chiếc bát, không gộp chung thành một mảng.

---

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn mức ảnh (Image-level label) | Ảnh có nhiều đối tượng gây nhiễu, chủ thể bị mờ, hoặc đối tượng nằm ở ranh giới giữa 2 lớp dễ nhầm. | Quan sát tổng thể bức ảnh và gán 1 (hoặc một tập) class ID tương ứng cho cả file. | Kiểm tra nhãn có đúng ngữ cảnh chủ đạo không; rà soát các ca mập mờ có theo đúng guideline không. |
| Phát hiện vật thể | Hộp giới hạn và nhãn lớp (Bounding Box + Class Label) | Đối tượng bị che khuất một phần (occlusion), bị cắt viền (truncation), hoặc hộp vẽ quá rộng / quá chật. | Kéo bounding box bám sát biên ngoài của từng vật thể và chọn nhãn lớp tương ứng. | Kiểm tra IoU (hộp có ôm khít không), phát hiện sót vật thể (missed objects) hoặc vẽ trùng lặp. |
| Instance segmentation | Mặt nạ điểm ảnh từng cá thể và nhãn lớp (Pixel-level Instance Mask + Class Label) | Ranh giới phức tạp (tóc, tán cây, viền mỏng), hai cá thể cùng loài đè lên nhau khó tách rời. | Chấm polygon hoặc dùng cọ vẽ chính xác từng pixel thuộc về từng cá thể riêng biệt kèm class ID. | Soi kỹ đường viền ranh giới (boundary accuracy), kiểm tra các phần bị che khuất và xem các cá thể dính liền đã được tách đúng chưa. |

---

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

- **Trả lời:** Chỉ sử dụng 3 ảnh mẫu COCO công khai có sẵn theo quy định của bài lab; tuyệt đối không đưa ảnh cá nhân, dữ liệu nội bộ/nhạy cảm hoặc thông tin định danh cá nhân (PII như họ tên, MSSV, email) vào Colab và các file báo cáo/output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

- **Trả lời:** Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi (dữ liệu nhạy cảm, vi phạm bản quyền hoặc nằm ngoài phạm vi bài học), em sẽ lập tức dừng thao tác (không tự ý xử lý, gán nhãn hay lan truyền) và thực hiện **escalation**: báo ngay cho **Lab Coach / Giảng viên (mentor) phụ trách** để nhận hướng dẫn xử lý.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
