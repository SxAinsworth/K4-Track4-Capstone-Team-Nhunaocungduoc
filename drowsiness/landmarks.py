"""Person 1 side: camera frame -> FrameMetrics (EAR, MAR).

Any model plugs in through one contract: a callable ``frame -> FrameMetrics``.
``load_extractor(spec)`` builds one from a name:

    "dlib"                           Person 1's face_core.py (default)
    "mediapipe"                      built-in MediaPipe Face Mesh
    "pkg.module"                     uses pkg.module.extract_metrics
    "pkg.module:Name"                Name is a class (instantiated with no args)
                                     or a function, called as Name(frame)

If a model outputs landmark points, wrap it in ``LandmarkExtractor`` with a
``FaceLayout`` and the EAR/MAR math below is reused. See person1_template.py.

Check a model against the contract (no alert logic involved):
    python -m drowsiness.landmarks                              # webcam, default model
    python -m drowsiness.landmarks --model ... --video clip.mp4
"""
import importlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .types import FrameMetrics

Extractor = Callable[[object], FrameMetrics]
Points = Sequence[Sequence[float]]  # (x, y) in pixels, indexed by landmark id

_EPS = 1e-6


# -- EAR / MAR geometry ---------------------------------------------------------

def eye_aspect_ratio(p: Sequence[Sequence[float]]) -> float | None:
    """EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|), points p1..p6 around one eye
    (p1/p4 corners, p2/p3 upper lid, p5/p6 lower lid). None if degenerate."""
    width = math.dist(p[0], p[3])
    if width < _EPS:
        return None
    return (math.dist(p[1], p[5]) + math.dist(p[2], p[4])) / (2.0 * width)


def mouth_aspect_ratio(left: Sequence[float], right: Sequence[float],
                       pairs: Sequence[tuple[Sequence[float], Sequence[float]]]) -> float | None:
    """MAR = mean inner-lip opening (top/bottom pairs) / mouth width. None if degenerate."""
    width = math.dist(left, right)
    if width < _EPS or not pairs:
        return None
    return sum(math.dist(top, bottom) for top, bottom in pairs) / (len(pairs) * width)


@dataclass(frozen=True)
class FaceLayout:
    """Which landmark ids a model uses for the eyes and the mouth."""
    left_eye: tuple[int, int, int, int, int, int]    # p1..p6
    right_eye: tuple[int, int, int, int, int, int]
    mouth_corners: tuple[int, int]                   # inner lip corners
    mouth_pairs: tuple[tuple[int, int], ...]         # (upper inner lip, lower inner lip)

    @property
    def ids(self) -> tuple[int, ...]:
        pair_ids = tuple(i for pair in self.mouth_pairs for i in pair)
        return self.left_eye + self.right_eye + self.mouth_corners + pair_ids


MEDIAPIPE_LAYOUT = FaceLayout(           # MediaPipe Face Mesh, 468/478 points
    left_eye=(362, 385, 387, 263, 373, 380),
    right_eye=(33, 160, 158, 133, 153, 144),
    mouth_corners=(78, 308),
    mouth_pairs=((81, 178), (13, 14), (311, 402)),
)

DLIB68_LAYOUT = FaceLayout(              # dlib / iBUG 68-point layout
    left_eye=(42, 43, 44, 45, 46, 47),
    right_eye=(36, 37, 38, 39, 40, 41),
    mouth_corners=(60, 64),
    mouth_pairs=((61, 67), (62, 66), (63, 65)),
)


def face_metrics(points: Points, layout: FaceLayout) -> tuple[float, float] | None:
    """(EAR averaged over both eyes, MAR), or None if the geometry is degenerate."""
    left = eye_aspect_ratio([points[i] for i in layout.left_eye])
    right = eye_aspect_ratio([points[i] for i in layout.right_eye])
    mar = mouth_aspect_ratio(points[layout.mouth_corners[0]], points[layout.mouth_corners[1]],
                             [(points[a], points[b]) for a, b in layout.mouth_pairs])
    if left is None or right is None or mar is None:
        return None
    return (left + right) / 2.0, mar


class LandmarkExtractor:
    """Turns a landmark model into an extractor.

    ``model(frame)`` must return the face's landmark points in pixel
    coordinates (indexable by the ids in ``layout``), or None if no face.
    """

    def __init__(self, model: Callable[[object], Points | None], layout: FaceLayout):
        self.model = model
        self.layout = layout
        self.last_points: Points | None = None  # for --draw-landmarks

    def __call__(self, frame) -> FrameMetrics:
        t = time.monotonic()  # stamp before inference: when the frame was seen
        points = self.model(frame)
        self.last_points = points
        values = face_metrics(points, self.layout) if points is not None else None
        if values is None:
            return FrameMetrics(t=t, ear=None, mar=None, face_found=False)
        ear, mar = values
        return FrameMetrics(t=t, ear=ear, mar=mar, face_found=True)

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if close:
            close()


# -- Built-in model: MediaPipe Face Mesh ----------------------------------------

class MediaPipeFaceMesh:
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError("MediaPipe is not installed: pip install mediapipe") from e
        if not hasattr(mp, "solutions"):
            raise ImportError("This MediaPipe version has no Face Mesh solutions API: "
                              "pip install \"mediapipe<0.10.15\"")
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(self, frame) -> list[tuple[float, float]] | None:
        import cv2
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        # Pixel coordinates: EAR/MAR need equal x/y scale, normalized coords don't have it.
        return [(lm.x * w, lm.y * h) for lm in result.multi_face_landmarks[0].landmark]

    def close(self) -> None:
        self._mesh.close()


def _face_core() -> Extractor:
    from .face_core_adapter import FaceCoreExtractor
    return FaceCoreExtractor()


BUILTIN_MODELS: dict[str, Callable[[], Extractor]] = {
    "dlib": _face_core,  # Person 1's face_core.py (team model)
    "mediapipe": lambda: LandmarkExtractor(MediaPipeFaceMesh(), MEDIAPIPE_LAYOUT),
}
DEFAULT_MODEL = "dlib"


def model_ear_threshold(extractor: object, default: float, use_model: bool = True) -> float:
    """EAR threshold to use now: the model's per-driver calibrated threshold if it
    exposes one (``extractor.ear_threshold``, None until calibrated), else ``default``."""
    threshold = getattr(extractor, "ear_threshold", None) if use_model else None
    return default if threshold is None else float(threshold)


def load_extractor(spec: str = DEFAULT_MODEL) -> Extractor:
    """Build an extractor from a built-in name or "pkg.module[:Name]" (see module doc)."""
    if spec in BUILTIN_MODELS:
        return BUILTIN_MODELS[spec]()
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import model module {module_name!r}: {e}") from e
    obj = getattr(module, attr or "extract_metrics", None)
    if obj is None:
        raise ImportError(f"{module_name!r} has no {attr or 'extract_metrics'!r}")
    extractor = obj() if isinstance(obj, type) else obj
    if not callable(extractor):
        raise TypeError(f"{spec!r} is not callable as extractor(frame) -> FrameMetrics")
    return extractor


_default: Extractor | None = None


def extract_metrics(frame) -> FrameMetrics:
    """Spec contract (section 3), backed by the default model."""
    global _default
    if _default is None:
        _default = load_extractor()
    return _default(frame)


# -- Contract check for the merge checklist (spec section 10) -------------------

def contract_problems(m: object, prev_t: float | None) -> list[str]:
    """Everything wrong with one extractor output; empty list if it's fine."""
    if not isinstance(m, FrameMetrics):
        return [f"returned {type(m).__name__}, expected FrameMetrics"]
    problems = []
    if not isinstance(m.t, (int, float)):
        problems.append(f"t is {type(m.t).__name__}, expected float seconds")
    elif prev_t is not None and m.t < prev_t:
        problems.append(f"t went backwards ({m.t} after {prev_t}); use time.monotonic()")
    elif prev_t is not None and m.t - prev_t > 10:
        problems.append(f"t jumped {m.t - prev_t:.0f}s between frames; use seconds, not ms")
    if m.face_found and (m.ear is None or m.mar is None):
        problems.append("face_found=True but ear/mar is None")
    if not m.face_found and (m.ear is not None or m.mar is not None):
        problems.append("face_found=False but ear/mar is set (should be None)")
    for name in ("ear", "mar"):
        v = getattr(m, name)
        if v is not None and not (0.0 <= v < 5.0):
            problems.append(f"{name}={v:.3f} is outside the plausible range")
    return problems


def _check(spec: str, source: int | str, frames: int) -> None:
    import cv2

    extractor = load_extractor(spec)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {source!r}")
    prev_t, ears, mars, found, n, bad, dup = None, [], [], 0, 0, 0, 0
    start = time.monotonic()
    try:
        while n < frames:
            ok, frame = cap.read()
            if not ok:
                break
            m = extractor(frame)
            n += 1
            problems = contract_problems(m, prev_t)
            if problems:
                bad += 1
                if bad <= 10:
                    print(f"frame {n}: " + "; ".join(problems))
                if not isinstance(m, FrameMetrics):
                    continue
            dup += prev_t is not None and m.t == prev_t
            prev_t = m.t
            if m.face_found and m.ear is not None:
                found += 1
                ears.append(m.ear)
                mars.append(m.mar)
    finally:
        cap.release()
        close = getattr(extractor, "close", None)
        if close:
            close()

    elapsed = max(time.monotonic() - start, 1e-6)
    print(f"\n{n} frames, {n / elapsed:.1f} FPS, face found in {found}/{n}, "
          f"{bad} frames with contract problems")
    if dup:
        print(f"note: {dup} frames repeated the previous timestamp (faster than the clock "
              "resolution, ~16 ms on Windows); the monitor skips those frames")
    if ears:
        print(f"EAR min/mean/max: {min(ears):.3f} / {sum(ears) / len(ears):.3f} / {max(ears):.3f}")
        print(f"MAR min/mean/max: {min(mars):.3f} / {sum(mars) / len(mars):.3f} / {max(mars):.3f}")
    print("OK: contract satisfied" if n and not bad else "FAILED: fix the problems above")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check a model against the FrameMetrics contract")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help='"dlib" (default), "mediapipe" or "pkg.module[:Name]"')
    parser.add_argument("--video", help="video file (default: webcam)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300, help="frames to check (default 300)")
    args = parser.parse_args()
    _check(args.model, args.video if args.video else args.camera, args.frames)
