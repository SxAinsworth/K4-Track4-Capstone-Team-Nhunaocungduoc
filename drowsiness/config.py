"""All tunable thresholds live here. Tuning should only touch this file."""
from dataclasses import dataclass


@dataclass
class Config:
    ear_threshold: float = 0.22       # tune with Person 1/3
    use_model_ear_threshold: bool = True  # use the model's per-driver calibrated
                                          # threshold when it has one (face_core)
    mar_threshold: float = 0.60       # tune
    drowsy_seconds: float = 1.5       # R1
    yawn_min_seconds: float = 1.0     # MAR must stay high this long to count
    yawn_alert_count: int = 3         # R2
    yawn_window_s: float = 60.0
    perclos_window_s: float = 60.0    # R3
    perclos_warn: float | None = 0.15 # optional extra alert, None to disable
    perclos_warmup_s: float = 10.0    # PERCLOS is unreliable before this much data
    alert_cooldown_s: float = 2.0     # don't spam sound
    no_face_grace_s: float = 1.0      # pause timers briefly if face is lost
    max_gap_s: float = 0.5            # bigger dt = frozen camera: reset timers
    sound_file: str | None = None     # mp3 path; None -> winsound beep
