# K4-Track4-Capstone-Team-Nhunaocungduoc

Hệ thống cảnh báo tài xế buồn ngủ (Driver Monitoring System) chạy trên webcam. Ứng dụng nhận diện
khuôn mặt bằng dlib (HOG + 68 landmark), tính độ mở mắt (EAR), độ mở miệng (MAR) và góc cúi đầu, rồi
phát cảnh báo bằng âm thanh và hiển thị trên HUD khi phát hiện:

- **Buồn ngủ:** nhắm mắt liên tục quá 1.5 s.
- **Ngáp nhiều:** 3 lần ngáp (miệng mở rộng trên 1 s) trong vòng 60 s.
- **PERCLOS cao:** tỉ lệ thời gian nhắm mắt trong 60 s vượt 15%.

Ngưỡng EAR được hiệu chỉnh riêng cho từng tài xế trong 4 giây đầu (nhìn thẳng, mở mắt).

## Cấu trúc

| Đường dẫn | Vai trò |
|---|---|
| `face_core.py` | Lõi nhận diện (Người 1): dlib HOG, 68 landmark, EAR/MAR/pitch, hiệu chỉnh theo người |
| `download_model.py` | Tải `shape_predictor_68_face_landmarks.dat` (~100 MB) vào `models/` |
| `drowsiness/` | Logic cảnh báo, HUD, âm thanh, simulator, vòng lặp camera (Người 2). Chi tiết: [drowsiness/README.md](drowsiness/README.md) |
| `drowsiness/tests/` | Unit test, chạy không cần camera |
| `requirements.txt` | `dlib-bin`, `opencv-python`, `numpy`, `pytest` |

## Cài đặt (một lần)

Yêu cầu Python 3.10+ (đã chạy thử với 3.11). Chạy mọi lệnh từ thư mục gốc của repo.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python download_model.py            # tải model dlib vào models/
python -m pytest drowsiness/tests   # 35 test, không cần camera
```

`download_model.py` tự tải tiếp khi mạng bị đứt; nếu vẫn hết lượt thử thì chạy lại, nó sẽ tải tiếp từ
chỗ dừng. Cách khác: tải tay file
[shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2),
giải nén và đặt vào `models/shape_predictor_68_face_landmarks.dat`.

## Chạy

```powershell
python -m drowsiness.main --sim              # demo 60 s bằng dữ liệu giả, không cần camera
python -m drowsiness.main                    # webcam
python -m drowsiness.main --draw-landmarks   # webcam, vẽ thêm điểm mắt/miệng
python -m drowsiness.main --video clip.mp4   # chạy trên video quay sẵn
python -m drowsiness.simulate                # in timeline cảnh báo ra terminal
python face_core.py                          # chạy riêng lõi nhận diện
```

Phải chạy ứng dụng dưới dạng module (`python -m drowsiness.main`). Chạy `python drowsiness/main.py`
sẽ khiến `drowsiness/types.py` che mất module `types` của Python.

**Phím tắt:** `q` hoặc `Esc` để thoát, `c` để hiệu chỉnh lại (khi đổi tài xế).

**Tùy chọn:** `--camera N` (chọn camera), `--no-mirror`, `--mute` (tắt âm), `--sound-file alert.mp3`
(cần `pygame`), `--model mediapipe` (model thay thế, cần Python ≤ 3.12).

## Cách dùng với webcam

Ngồi cách camera 50-80 cm, nhìn thẳng, ánh sáng đều.

1. Nhìn thẳng, mở mắt trong 4 s đầu: HUD hiện `CALIBRATING x%`, xong sẽ hiện ngưỡng EAR. Trong lúc này
   chưa có cảnh báo.
2. Chớp mắt bình thường: không cảnh báo.
3. Nhắm mắt 2 s: viền đỏ, `DROWSY! WAKE UP` và tiếng bíp.
4. Ngáp 3 lần trong 1 phút: `FREQUENT YAWNING`.
5. Lim dim mắt liên tục 20-30 s: `HIGH PERCLOS - TAKE A BREAK`.
6. Quay đi hoặc che camera: `NO FACE`, bộ đếm tạm dừng 1 s rồi reset.

## Tinh chỉnh

Các ngưỡng cảnh báo nằm trong [drowsiness/config.py](drowsiness/config.py) (thời gian nhắm mắt,
`mar_threshold`, số lần ngáp, PERCLOS). Tỉ lệ hiệu chỉnh EAR (`EAR_RATIO = 0.72`) nằm trong
[face_core.py](face_core.py). Dùng `python -m drowsiness.landmarks --video clip.mp4` để xem dải giá trị
EAR/MAR thực tế trên video.

## License

Xem [LICENSE](LICENSE).
