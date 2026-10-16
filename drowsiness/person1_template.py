"""Template for Person 1: plug your face/landmark model into the app.

1. Copy this file to drowsiness/person1_model.py and fill in the TODOs.
2. Check the contract:  python -m drowsiness.landmarks --model drowsiness.person1_model:Person1Extractor
3. Run the app:         python -m drowsiness.main --model drowsiness.person1_model:Person1Extractor

Pick ONE option:
  A. Your model outputs face landmarks -> fill in MyLandmarkModel and MY_LAYOUT,
     and the shared EAR/MAR math in landmarks.py does the rest.
  B. Your model computes EAR/MAR itself -> write extract_metrics() at the bottom and run
     with --model drowsiness.person1_model
"""
import time
from pathlib import Path

from drowsiness.landmarks import DLIB68_LAYOUT, FaceLayout, LandmarkExtractor
from drowsiness.types import FrameMetrics

WEIGHTS = Path(__file__).parent / "weights" / "my_model.onnx"  # TODO: your weights file


# -- Option A: landmark model ---------------------------------------------------

class MyLandmarkModel:
    def __init__(self, weights: Path = WEIGHTS):
        # TODO: load the model ONCE here (not per frame), e.g.
        #   self.net = onnxruntime.InferenceSession(str(weights))
        self.weights = weights

    def __call__(self, frame):
        """frame: BGR uint8 image (H, W, 3) straight from cv2.VideoCapture.

        Return the landmarks of ONE face as a sequence of (x, y) in PIXELS of
        this frame (not normalized 0-1, not of a cropped face), or None if no face.
        """
        raise NotImplementedError("TODO: run face detection + landmarks on frame")

    def close(self) -> None:
        pass  # TODO: release resources if needed


# TODO: the landmark ids your model uses. DLIB68_LAYOUT fits any 68-point model;
# for another layout list p1..p6 per eye (corners p1/p4, upper p2/p3, lower p5/p6)
# and the inner-lip corners and upper/lower pairs, e.g.
#   MY_LAYOUT = FaceLayout(left_eye=(...), right_eye=(...),
#                          mouth_corners=(l, r), mouth_pairs=((top, bottom), ...))
MY_LAYOUT: FaceLayout = DLIB68_LAYOUT


class Person1Extractor(LandmarkExtractor):
    """--model drowsiness.person1_model:Person1Extractor"""

    def __init__(self):
        super().__init__(MyLandmarkModel(), MY_LAYOUT)


# -- Option B: EAR/MAR computed by your own code --------------------------------

def extract_metrics(frame) -> FrameMetrics:
    """--model drowsiness.person1_model"""
    t = time.monotonic()  # same clock as the alert logic; stamp before inference
    ear, mar = None, None  # TODO: compute from frame; leave None if no face
    face_found = ear is not None and mar is not None
    return FrameMetrics(t=t, ear=ear, mar=mar, face_found=face_found)
