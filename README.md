# VehicleAccessVerifier

VehicleAccessVerifier là bản demo kiểm soát ra vào tại điểm xác thực bằng cách kết hợp:

- nhận dạng biển số xe
- đối sánh khuôn mặt

## Tổng Quan Hóa

Đầu vào:
- ảnh hoặc video của biển số xe, khuôn mặt người điều khiển
- dữ liệu người dùng trong `data/users/users.csv`

Đầu ra:
- một file `result.json` cho mỗi điểm xác thực
- các ảnh tiền xử lý thể hiện vùng mặt và vùng biển số đã được phát hiện

Quyết định:
- quyết định xem người đó được phép đi tiếp hay vào vùng hỗ trợ dựa vào phần đầu ra `result.json`

## Quy Trình Xác Thực

Với mỗi điểm xác thực (checkpoint) như `CP1`, hệ thống sẽ:

1. đọc dữ liệu từ `data/incoming/CP1/face/` và `data/incoming/CP1/plate/`
2. phát hiện vùng biển số
3. chạy OCR trên biển số đã cắt và giữ lại kết quả đọc tốt nhất
4. tra cứu người dùng tương ứng trong `data/users/users.csv` dựa vào biển số vừa đọc được
5. tìm dữ liệu khuôn mặt tốt nhất từ dữ liệu điểm xác thực
6. chỉ so sánh khuôn mặt đó với thư mục khuôn mặt của người dùng đã khớp biển số
7. ghi kết quả cuối cùng vào `data/incoming/CP1/result.json`

## Các Mô Hình Được Sử Dụng

- `models/detector/yolo_detector_model.pt (YOLOv5)`: dùng để phát hiện vùng biển số trong ảnh xe.
Lý do lựa chọn: YOLOv5 là mô hình phát hiện đối tượng có tốc độ xử lý nhanh và độ ổn định cao, phù hợp với bài toán cần xác định đúng vị trí biển số trước khi đưa sang bước OCR. Việc tách chính xác vùng biển số giúp tăng đáng kể chất lượng nhận dạng ký tự ở bước sau.

- `models/recognizer/license_plates_ocr_model.onnx`: dùng để đọc ký tự từ vùng biển số đã cắt.
Lý do lựa chọn: mô hình ở định dạng ONNX có ưu điểm gọn nhẹ, dễ triển khai và chạy hiệu quả trên CPU. Và vì biển số có chuỗi ký tự ngắn và có cấu trúc tương đối rõ ràng, nên mô hình này đáp ứng tốt cả về tốc độ lẫn độ phù hợp thực tế.

- `models/face/det_10g.onnx (SCRFD)`: dùng để phát hiện khuôn mặt trong ảnh hoặc video.
Lý do lựa chọn: SCRFD cho khả năng phát hiện khuôn mặt tốt trong nhiều điều kiện thực tế như thay đổi góc mặt, ánh sáng hoặc chất lượng khung hình từ video. 

- `models/face/face_recognition_sface_2021dec.onnx (SFace)`: dùng để trích xuất vector đặc trưng khuôn mặt để so khớp.
Lý do lựa chọn: SFace có khả năng biểu diễn khuôn mặt dưới dạng vector đặc trưng gọn và hiệu quả, giúp việc so sánh khuôn mặt bằng độ tương đồng cosine diễn ra nhanh chóng.

## File Users CSV

Cấu trúc:

```csv
user_id,registered_plate,face_dir
HE211206,14-N1 135.62,faces\HE211206
HE210000,00A-000.00,faces\HE210000
```

Ý nghĩa:
- `user_id`: mã định danh duy nhất của người dùng
- `registered_plate`: biển số xe đã đăng ký cho người dùng
- `face_dir`: thư mục chứa ảnh hoặc video khuôn mặt đã đăng ký

## Vì Sao Thiết Kế Như Vậy

- Biển số trước, khuôn mặt sau: tránh việc phải so sánh một khuôn mặt checkpoint với toàn bộ cơ sở dữ liệu.
- Hỗ trợ video: cả đầu vào checkpoint và dữ liệu đăng ký đều có thể dùng video ngắn, không chỉ ảnh.
- Có bằng chứng trực quan: thư mục tiền xử lý giúp ban giám khảo kiểm tra hệ thống đã phát hiện gì.
- Kết quả rõ ràng: mỗi checkpoint luôn tạo ra một kết quả JSON cuối cùng kèm lý do.

## Chạy Demo

Từ thư mục dự án:

```powershell
.\VehicleAccessVerifier\venv\Scripts\python.exe .\VehicleAccessVerifier\src\verify_access.py --checkpoint-id CP1
```

Để chạy cổng đăng ký khách trên trình duyệt:

```powershell
.\VehicleAccessVerifier\venv\Scripts\python.exe .\VehicleAccessVerifier\src\main.py
```

Sau đó mở `http://localhost:8000`.

## Cấu Trúc Dữ Liệu Đầu Vào

```text
data/
  incoming/
    CP1/
      face/
        1.jpg
        2.jpg
        rider.mp4
      plate/
        1.jpg
        2.jpg
        vehicle.mp4
      preprocessing/
        face_bbox/
        plate/
          bbox/
          cropped/
      result.json
  users/
    users.csv
    faces/
      HE211206/
        1.jpg
        2.jpg
        helmet.mp4
      HE210000/
```

Ghi chú:
- `face/` chứa ảnh hoặc video khuôn mặt tại checkpoint
- `plate/` chứa ảnh hoặc video biển số xe
- `users.csv` lưu biển số đã đăng ký và thư mục khuôn mặt của từng người dùng
- `faces/<user_id>/` chứa ảnh hoặc video khuôn mặt đã đăng ký của người dùng đó

## Đăng Ký Khách

Cổng web chỉ dùng để đăng ký khách.

Người dùng nhập:
- biển số xe
- ảnh hoặc video khuôn mặt tải lên, hoặc video quay nhanh từ webcam

Hệ thống sẽ:
- tự tạo mã khách như `guest_000001`
- lưu dữ liệu khuôn mặt vào thư mục của khách đó
- ghi bản ghi khách vào `data/users/users.csv`

Chạy cổng web:

```powershell
venv\Scripts\python.exe src\main.py
```

## Đăng Ký Người Dùng Bằng Dòng Lệnh

Đăng ký bằng ảnh khuôn mặt:

```powershell
venv\Scripts\python.exe src\enroll_user.py --user-id HE211206 --plate "14-N1 135.62" --face-image data\incoming\CP1\face\1.jpg
```

Đăng ký bằng video khuôn mặt:

```powershell
venv\Scripts\python.exe src\enroll_user.py --user-id HE211206 --plate "14-N1 135.62" --face-video data\incoming\CP1\face\rider.mp4
```

Đăng ký khách và để hệ thống tự tạo mã:

```powershell
venv\Scripts\python.exe src\enroll_user.py --guest --plate "14-N1 135.62" --face-image data\incoming\CP1\face\1.jpg
```

## Kết Quả Đầu Ra

Ví dụ `result.json`:

```json
{
  "checkpoint_id": "CP1",
  "success": true,
  "reason": "Scanned plate matched a registered user and the face matched that user's face folder.",
  "best_match": {
    "scanned_plate": "14-N1 135.62",
    "plate_score": 0.9306633472442627,
    "plate_matches": true,
    "plate_image": "plate/2.jpg",
    "user_id": "HE211206",
    "face_similarity": 0.9999999412054876,
    "registered_plate": "14-N1 135.62",
    "face_image": "face/1.jpg"
  }
}
```

Nếu xác thực thất bại, hệ thống vẫn ghi `result.json` với:
- `success: false`
- `reason` mô tả rõ lý do thất bại

Thư mục tiền xử lý cũng lưu bằng chứng trực quan:
- `preprocessing/face_bbox/`: ảnh khuôn mặt có vẽ khung mặt
- `preprocessing/plate/bbox/`: ảnh xe có vẽ khung biển số
- `preprocessing/plate/cropped/`: vùng biển số đã cắt để OCR

Với đầu vào video, nguồn thắng cuối cùng có thể được hiển thị như:

```text
face/rider.mp4#frame=73
```

## Các File Chính

- `src/verify_access.py`: chạy xác thực checkpoint và ghi `result.json`
- `src/enroll_user.py`: đăng ký người dùng hoặc khách vào cơ sở dữ liệu
- `src/main.py`: chạy cổng web đăng ký khách
- `classes/access_verifier.py`: kết hợp OCR biển số và đối sánh khuôn mặt để đưa ra quyết định
- `classes/face_recognizer.py`: phân tích khuôn mặt, chấm chất lượng, tạo embedding và đối sánh
- `classes/detector.py`: phát hiện biển số và cắt vùng biển số
- `classes/recognizer.py`: OCR biển số và hậu xử lý biển số Việt Nam
