"""Lõi nhận diện (Người 1): dlib HOG + 68 landmark -> EAR, MAR, pitch + hiệu chỉnh theo từng người.

Pipeline tham khảo từ github.com/Gagandeep-2003/driver-drowsiness-detection-system (blinkDetect.py):
resize -> xám -> cân bằng histogram -> dlib HOG face detector (trên ảnh thu nhỏ) -> shape predictor 68 điểm.
Khác bản gốc: ngưỡng EAR tự hiệu chỉnh thay vì cố định 0.27, thêm MAR và góc cúi đầu.

Cần file models/shape_predictor_68_face_landmarks.dat (chạy: python download_model.py).

Dùng trong code khác:
    core = FaceCore()
    m = core.process(frame)        # FaceMetrics hoặc None nếu không thấy mặt
    core.draw(frame, m)
Chạy thử độc lập:  python face_core.py   (q để thoát, c để hiệu chỉnh lại)
"""
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import dlib
import numpy as np

PREDICTOR_PATH = Path(__file__).parent / "models" / "shape_predictor_68_face_landmarks.dat"

# Chỉ số landmark dlib 68 điểm, thứ tự p1..p6 theo công thức EAR (Soukupová & Čech, 2016)
RIGHT_EYE = [36, 37, 38, 39, 40, 41]
LEFT_EYE = [42, 43, 44, 45, 46, 47]
# Miệng: 2 khóe môi trong + 3 cặp môi trong (trên, dưới)
MOUTH_CORNERS = (60, 64)
MOUTH_PAIRS = [(61, 67), (62, 66), (63, 65)]

# Điểm dùng cho solvePnP và mô hình 3D mặt chung (đơn vị tùy ý)
POSE_IDS = [30, 8, 36, 45, 48, 54]
POSE_MODEL = np.array([
    (0.0, 0.0, 0.0),          # mũi
    (0.0, -330.0, -65.0),     # cằm
    (-225.0, 170.0, -135.0),  # khóe mắt ngoài (bên trái ảnh)
    (225.0, 170.0, -135.0),   # khóe mắt ngoài (bên phải ảnh)
    (-150.0, -150.0, -125.0), # khóe miệng
    (150.0, -150.0, -125.0),
])

FRAME_HEIGHT = 460             # như bản gốc: resize trước khi xử lý
FACE_DOWNSAMPLE_RATIO = 1.5    # detect mặt trên ảnh nhỏ hơn cho nhanh
CALIB_SECONDS = 4.0
EAR_RATIO = 0.72  # ngưỡng nhắm = 72% EAR khi mở mắt


@dataclass
class FaceMetrics:
    ear: float                 # EAR trung bình hai mắt
    mar: float
    pitch: float | None        # độ, so với tư thế lúc hiệu chỉnh (âm = cúi xuống)
    eyes_closed: bool | None   # None khi đang hiệu chỉnh
    calibrated: bool
    landmarks: np.ndarray      # (68, 2) tọa độ pixel trên frame gốc


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts, ids):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in ids)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)


def mouth_aspect_ratio(pts):
    width = _dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
    height = sum(_dist(pts[a], pts[b]) for a, b in MOUTH_PAIRS) / len(MOUTH_PAIRS)
    return height / (width + 1e-6)


def head_pitch(pts, w, h):
    image_pts = np.array([pts[i] for i in POSE_IDS], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)  # f ~ chiều rộng ảnh
    ok, rvec, _ = cv2.solvePnP(POSE_MODEL, image_pts, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rot)[0]
    pitch = angles[0]
    # chuẩn hóa về [-90, 90] vì RQDecomp có thể trả ±180
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return float(pitch)


class FaceCore:
    def __init__(self, calib_seconds=CALIB_SECONDS, ear_ratio=EAR_RATIO, predictor_path=PREDICTOR_PATH):
        if not Path(predictor_path).exists():
            raise FileNotFoundError(f"Thiếu {predictor_path}. Chạy: python download_model.py")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.calib_seconds = calib_seconds
        self.ear_ratio = ear_ratio
        self.recalibrate()

    def recalibrate(self):
        self._calib_start = None
        self._ear_samples, self._pitch_samples = [], []
        self.ear_open = None
        self.ear_threshold = None
        self.pitch_baseline = 0.0

    @property
    def calibrated(self):
        return self.ear_threshold is not None

    def calib_progress(self):
        if self.calibrated:
            return 1.0
        if self._calib_start is None:
            return 0.0
        return min(1.0, (time.time() - self._calib_start) / self.calib_seconds)

    def _update_calibration(self, ear, pitch):
        if self._calib_start is None:
            self._calib_start = time.time()
        self._ear_samples.append(ear)
        if pitch is not None:
            self._pitch_samples.append(pitch)
        if time.time() - self._calib_start >= self.calib_seconds and len(self._ear_samples) >= 10:
            # median để bỏ qua vài lần chớp mắt trong lúc hiệu chỉnh
            self.ear_open = median(self._ear_samples)
            self.ear_threshold = self.ear_ratio * self.ear_open
            if self._pitch_samples:
                self.pitch_baseline = median(self._pitch_samples)

    def _landmarks(self, frame_bgr):
        """Trả về (68, 2) tọa độ trên frame gốc, hoặc None."""
        h, w = frame_bgr.shape[:2]
        scale = FRAME_HEIGHT / h
        small = cv2.resize(frame_bgr, (int(w * scale), FRAME_HEIGHT))
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))  # đỡ thiếu sáng

        tiny = cv2.resize(gray, None, fx=1 / FACE_DOWNSAMPLE_RATIO, fy=1 / FACE_DOWNSAMPLE_RATIO)
        faces = self.detector(tiny, 0)
        if not faces:
            return None
        f = max(faces, key=lambda r: r.area())  # mặt lớn nhất = tài xế
        r = FACE_DOWNSAMPLE_RATIO
        rect = dlib.rectangle(int(f.left() * r), int(f.top() * r), int(f.right() * r), int(f.bottom() * r))
        shape = self.predictor(gray, rect)
        pts = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float64)
        return pts / scale

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        pts = self._landmarks(frame_bgr)
        if pts is None:
            return None

        ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
        mar = mouth_aspect_ratio(pts)
        raw_pitch = head_pitch(pts, w, h)

        if not self.calibrated:
            self._update_calibration(ear, raw_pitch)

        pitch = None if raw_pitch is None else raw_pitch - self.pitch_baseline
        closed = (ear < self.ear_threshold) if self.calibrated else None
        return FaceMetrics(ear, mar, pitch, closed, self.calibrated, pts)

    def draw(self, frame, m):
        if m is None:
            cv2.putText(frame, "Khong thay mat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        eye_color = (0, 0, 255) if m.eyes_closed else (0, 255, 0)
        for ids in (LEFT_EYE, RIGHT_EYE):
            cv2.polylines(frame, [m.landmarks[ids].astype(np.int32)], True, eye_color, 1)
            for i in ids:
                cv2.circle(frame, tuple(m.landmarks[i].astype(int)), 2, eye_color, -1)
        mouth_ids = list(MOUTH_CORNERS) + [i for pair in MOUTH_PAIRS for i in pair]
        for i in mouth_ids:
            cv2.circle(frame, tuple(m.landmarks[i].astype(int)), 2, (255, 200, 0), -1)

        lines = [f"EAR {m.ear:.3f}", f"MAR {m.mar:.3f}"]
        if m.pitch is not None:
            lines.append(f"Pitch {m.pitch:+.1f} deg")
        if self.calibrated:
            lines.append(f"Nguong EAR {self.ear_threshold:.3f} (mo {self.ear_open:.3f})")
            if m.eyes_closed:
                lines.append("MAT NHAM")
        else:
            lines.append(f"Hieu chinh: mo mat, nhin thang {self.calib_progress() * 100:.0f}%")
        for k, text in enumerate(lines):
            cv2.putText(frame, text, (10, 30 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame


def main():
    cap = cv2.VideoCapture(0)
    core = FaceCore()
    t_prev = time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        core.draw(frame, core.process(frame))
        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("Face core", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            core.recalibrate()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
