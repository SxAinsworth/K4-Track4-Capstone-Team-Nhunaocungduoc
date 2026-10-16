"""Adapter: Person 1's FaceCore (face_core.py, dlib 68 points) -> FrameMetrics.

face_core.py is used as-is. It needs dlib (pip install -r requirements.txt) and
models/shape_predictor_68_face_landmarks.dat (python download_model.py).

Besides the FrameMetrics contract this exposes what FaceCore adds on top:
  calibrating    True during FaceCore's first seconds (main.py holds alerts back)
  ear_threshold  per-driver EAR threshold after calibration (None before)
  status_lines() calibration progress / threshold / head pitch for the HUD
  recalibrate()  restart calibration (bound to the 'c' key in main.py)
"""
import time

from .landmarks import DLIB68_LAYOUT
from .types import FrameMetrics


class FaceCoreExtractor:
    def __init__(self, core=None):
        if core is None:
            try:
                from face_core import FaceCore  # Person 1, repo root
            except ImportError as e:
                raise ImportError(f"{e}. Install Person 1's requirements: "
                                  "pip install -r requirements.txt") from e
            core = FaceCore()  # FileNotFoundError if the .dat model is missing
        self.core = core
        self.layout = DLIB68_LAYOUT   # same ids as face_core.py, for --draw-landmarks
        self.last_points = None
        self.last = None              # last FaceCore FaceMetrics (or None)

    def __call__(self, frame) -> FrameMetrics:
        t = time.monotonic()  # stamp before inference
        fm = self.core.process(frame)
        self.last = fm
        self.last_points = None if fm is None else fm.landmarks
        if fm is None:
            return FrameMetrics(t=t, ear=None, mar=None, face_found=False)
        return FrameMetrics(t=t, ear=float(fm.ear), mar=float(fm.mar), face_found=True)

    @property
    def calibrating(self) -> bool:
        """True until FaceCore has its per-driver threshold; alerts are held back meanwhile."""
        return not self.core.calibrated

    @property
    def ear_threshold(self) -> float | None:
        return self.core.ear_threshold if self.core.calibrated else None

    def recalibrate(self) -> None:
        self.core.recalibrate()

    def status_lines(self) -> list[str]:
        if not self.core.calibrated:
            return [f"CALIBRATING {self.core.calib_progress():.0%}: eyes open, look straight"]
        lines = [f"EAR thr {self.core.ear_threshold:.3f} (open {self.core.ear_open:.3f})"]
        if self.last is not None and self.last.pitch is not None:
            lines.append(f"Head pitch: {self.last.pitch:+.1f} deg")
        return lines

    def close(self) -> None:
        pass
