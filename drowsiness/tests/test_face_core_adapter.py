"""FaceCoreExtractor tests with a fake FaceCore (no dlib, camera or model file)."""
from types import SimpleNamespace

import pytest

from drowsiness.face_core_adapter import FaceCoreExtractor
from drowsiness.landmarks import DLIB68_LAYOUT, contract_problems, model_ear_threshold


class FakeCore:
    """Mimics face_core.FaceCore: process() -> FaceMetrics-like object or None."""

    def __init__(self, results):
        self.results = list(results)
        self.ear_threshold = None
        self.ear_open = None
        self.recalibrated = 0

    @property
    def calibrated(self):
        return self.ear_threshold is not None

    def calib_progress(self):
        return 0.5

    def recalibrate(self):
        self.ear_threshold = self.ear_open = None
        self.recalibrated += 1

    def process(self, frame):
        return self.results.pop(0)


def face(ear=0.3, mar=0.2, pitch=-5.0):
    return SimpleNamespace(ear=ear, mar=mar, pitch=pitch, eyes_closed=None,
                           calibrated=False, landmarks=[(float(i), 0.0) for i in range(68)])


def test_maps_face_and_no_face_to_frame_metrics():
    ex = FaceCoreExtractor(FakeCore([face(0.31, 0.45), None]))
    m = ex(object())
    assert (m.face_found, m.ear, m.mar) == (True, pytest.approx(0.31), pytest.approx(0.45))
    assert ex.last_points is not None and ex.layout is DLIB68_LAYOUT
    assert contract_problems(m, None) == []
    m2 = ex(object())
    assert (m2.face_found, m2.ear, m2.mar) == (False, None, None)
    assert ex.last_points is None
    assert contract_problems(m2, m.t) == []


def test_calibrated_threshold_overrides_config_default():
    core = FakeCore([face()])
    ex = FaceCoreExtractor(core)
    assert ex.ear_threshold is None
    assert model_ear_threshold(ex, default=0.22) == 0.22
    assert ex.status_lines()[0].startswith("CALIBRATING 50%")

    core.ear_open, core.ear_threshold = 0.30, 0.216
    ex(object())
    assert model_ear_threshold(ex, default=0.22) == pytest.approx(0.216)
    assert model_ear_threshold(ex, default=0.22, use_model=False) == 0.22
    assert ex.status_lines() == ["EAR thr 0.216 (open 0.300)", "Head pitch: -5.0 deg"]

    ex.recalibrate()
    assert core.recalibrated == 1
    assert model_ear_threshold(ex, default=0.22) == 0.22


def test_models_without_calibration_use_config():
    assert model_ear_threshold(object(), default=0.22) == 0.22
    assert model_ear_threshold(None, default=0.22) == 0.22
