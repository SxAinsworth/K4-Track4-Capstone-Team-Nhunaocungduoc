"""Interface contract between Person 1 (landmarks) and Person 2 (alert logic)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameMetrics:
    t: float            # timestamp in seconds (time.monotonic())
    ear: float | None   # None if no face detected
    mar: float | None
    face_found: bool


@dataclass
class MonitorState:
    ear: float | None
    mar: float | None
    perclos: float            # 0.0 - 1.0
    yawn_count_60s: int
    eyes_closed_s: float      # current continuous closed duration
    drowsy_alert: bool
    yawn_alert: bool
    face_found: bool
    # Extensions (Person 2 only; Person 1 never builds MonitorState)
    perclos_alert: bool = False       # perclos > cfg.perclos_warn (after warm-up)
    perclos_warming_up: bool = True   # window still too short to trust PERCLOS
