"""EAR/MAR geometry and model plug-in tests (no camera, no real model)."""
import pytest

from drowsiness.landmarks import (
    DLIB68_LAYOUT, MEDIAPIPE_LAYOUT, LandmarkExtractor, contract_problems,
    eye_aspect_ratio, load_extractor, mouth_aspect_ratio,
)
from drowsiness.types import FrameMetrics


def eye(half_open: float, x0: float = 0.0, width: float = 10.0):
    """p1..p6 for an eye `width` wide whose lids are `half_open` px from the centre line."""
    a, b = x0 + width * 0.3, x0 + width * 0.7
    return [(x0, 0.0), (a, -half_open), (b, -half_open), (x0 + width, 0.0),
            (b, half_open), (a, half_open)]


def dlib_face(ear: float, mar: float):
    """68 points whose eyes have the given EAR and mouth the given MAR."""
    pts = [(0.0, 0.0)] * 68
    h = ear * 10.0 / 2.0                    # EAR = 4h / (2 * 10)
    for ids, x0 in ((DLIB68_LAYOUT.right_eye, 0.0), (DLIB68_LAYOUT.left_eye, 30.0)):
        for i, p in zip(ids, eye(h, x0)):
            pts[i] = p
    left, right = DLIB68_LAYOUT.mouth_corners
    pts[left], pts[right] = (0.0, 50.0), (20.0, 50.0)
    for k, (top, bottom) in enumerate(DLIB68_LAYOUT.mouth_pairs):
        x = 5.0 + 5.0 * k
        pts[top], pts[bottom] = (x, 50.0 - mar * 10.0), (x, 50.0 + mar * 10.0)
    return pts


def test_eye_aspect_ratio_open_and_closed():
    assert eye_aspect_ratio(eye(1.5)) == pytest.approx(0.30)
    assert eye_aspect_ratio(eye(0.25)) == pytest.approx(0.05)
    assert eye_aspect_ratio([(0, 0)] * 6) is None


def test_mouth_aspect_ratio():
    pairs = [((5, -4), (5, 4)), ((10, -5), (10, 5)), ((15, -3), (15, 3))]
    assert mouth_aspect_ratio((0, 0), (20, 0), pairs) == pytest.approx(0.4)
    assert mouth_aspect_ratio((0, 0), (0, 0), pairs) is None


def test_layout_ids_fit_their_models():
    assert max(MEDIAPIPE_LAYOUT.ids) < 468
    assert max(DLIB68_LAYOUT.ids) < 68


def test_landmark_extractor_computes_metrics():
    extract = LandmarkExtractor(lambda frame: dlib_face(ear=0.28, mar=0.7), DLIB68_LAYOUT)
    m = extract(object())
    assert m.face_found
    assert m.ear == pytest.approx(0.28)
    assert m.mar == pytest.approx(0.7)
    assert contract_problems(m, None) == []


def test_landmark_extractor_no_face():
    extract = LandmarkExtractor(lambda frame: None, DLIB68_LAYOUT)
    m = extract(object())
    assert (m.face_found, m.ear, m.mar) == (False, None, None)
    assert contract_problems(m, None) == []


# Plug-in targets used by load_extractor below.

class FakeExtractor(LandmarkExtractor):
    def __init__(self):
        super().__init__(lambda frame: dlib_face(0.3, 0.2), DLIB68_LAYOUT)


def extract_metrics(frame) -> FrameMetrics:
    return FrameMetrics(t=1.0, ear=0.3, mar=0.2, face_found=True)


def test_load_extractor_class_function_and_default_name():
    mod = "drowsiness.tests.test_landmarks"
    assert isinstance(load_extractor(f"{mod}:FakeExtractor"), FakeExtractor)
    assert load_extractor(f"{mod}:extract_metrics") is extract_metrics
    assert load_extractor(mod) is extract_metrics


def test_load_extractor_errors():
    with pytest.raises(ImportError):
        load_extractor("drowsiness.no_such_module")
    with pytest.raises(ImportError):
        load_extractor("drowsiness.tests.test_landmarks:NoSuchName")


def test_template_loads():
    extractor = load_extractor("drowsiness.person1_template:Person1Extractor")
    assert extractor.layout is DLIB68_LAYOUT


def test_contract_problems_catches_mistakes():
    assert contract_problems((0.3, 0.2), None)
    assert contract_problems(FrameMetrics(0.5, 0.3, 0.2, True), prev_t=1.0)     # went backwards
    assert contract_problems(FrameMetrics(2000.0, 0.3, 0.2, True), prev_t=1966.0)  # ms, not s
    assert contract_problems(FrameMetrics(1.0, None, None, True), None)
    assert contract_problems(FrameMetrics(1.0, 0.0, 0.0, False), None)
