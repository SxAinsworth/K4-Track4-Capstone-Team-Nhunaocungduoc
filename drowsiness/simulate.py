"""Scripted fake FrameMetrics so alerts and HUD work without a camera.

Timeline (spec section 9):
    0-5 s    normal, EAR ~0.30, 0.15 s blinks       -> no alerts
    5-7 s    eyes closed for 2.0 s                  -> drowsy alert at ~6.5 s
    7-20 s   normal                                 -> alert clears
    20-50 s  3 yawns (MAR 0.8, 1.5 s, 8 s apart)    -> yawn alert at the 3rd yawn
    50-60 s  eyes half-closed intermittently        -> PERCLOS rises

Run ``python -m drowsiness.simulate`` to print the alert timeline without a
camera, OpenCV or sound.
"""
import random
import time
from collections.abc import Iterator

from .config import Config
from .monitor import DrowsinessMonitor
from .types import FrameMetrics, MonitorState

SIM_DURATION_S = 60.0

EAR_OPEN = 0.30
EAR_BLINK = 0.08
EAR_CLOSED = 0.12
EAR_HALF_CLOSED = 0.17
MAR_NORMAL = 0.25
MAR_YAWN = 0.80

BLINK_S = 0.15
BLINK_TIMES = (1.5, 3.2, 4.5, 9.0, 12.5, 16.0, 19.0, 24.5, 28.0, 33.5, 36.0, 42.0, 45.5, 48.5)
DROWSY_CLOSURE = (5.0, 7.0)
YAWN_STARTS = (22.0, 30.0, 38.0)
YAWN_S = 1.5
HALF_CLOSED_START = 50.0
HALF_CLOSED_CYCLE = (0.9, 0.6)  # (seconds half-closed, seconds open), repeating


def scripted_values(t: float) -> tuple[float, float]:
    """Noise-free (EAR, MAR) at ``t`` seconds into the timeline."""
    ear, mar = EAR_OPEN, MAR_NORMAL
    if any(b <= t < b + BLINK_S for b in BLINK_TIMES):
        ear = EAR_BLINK
    if DROWSY_CLOSURE[0] <= t < DROWSY_CLOSURE[1]:
        ear = EAR_CLOSED
    if any(y <= t < y + YAWN_S for y in YAWN_STARTS):
        mar = MAR_YAWN
    if t >= HALF_CLOSED_START:
        closed_s, open_s = HALF_CLOSED_CYCLE
        if (t - HALF_CLOSED_START) % (closed_s + open_s) < closed_s:
            ear = EAR_HALF_CLOSED
    return ear, mar


class Simulator:
    """Iterates FrameMetrics along the scripted timeline.

    realtime=True:  paced at ``fps`` with ``t = time.monotonic()`` (same clock
                    as the camera pipeline), for the live demo.
    realtime=False: as fast as possible with ``t = i / fps``, for tests.
    """

    def __init__(self, fps: float = 30.0, realtime: bool = True,
                 duration_s: float = SIM_DURATION_S, noise: float = 0.01, seed: int = 0):
        self.fps = fps
        self.realtime = realtime
        self.duration_s = duration_s
        self.noise = noise
        self._rng = random.Random(seed)

    def _metrics(self, t: float, elapsed: float) -> FrameMetrics:
        ear, mar = scripted_values(elapsed)
        if self.noise:
            ear += self._rng.gauss(0.0, self.noise)
            mar += self._rng.gauss(0.0, self.noise)
        return FrameMetrics(t=t, ear=max(ear, 0.0), mar=max(mar, 0.0), face_found=True)

    def __iter__(self) -> Iterator[FrameMetrics]:
        if not self.realtime:
            i = 0
            while (t := i / self.fps) <= self.duration_s:
                yield self._metrics(t, t)
                i += 1
            return

        t0 = time.monotonic()
        i = 0
        while True:
            delay = t0 + i / self.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            now = time.monotonic()
            elapsed = now - t0
            if elapsed > self.duration_s:
                return
            yield self._metrics(now, elapsed)
            # Skip frames we fell behind on instead of bursting to catch up.
            i = max(i + 1, int(elapsed * self.fps) + 1)


def blank_canvas(width: int = 640, height: int = 480):
    """Dark BGR image to draw the simulated face and HUD on."""
    import numpy as np
    return np.full((height, width, 3), 40, dtype=np.uint8)


def draw_fake_face(canvas, m: FrameMetrics, cfg: Config | None = None) -> None:
    """Cartoon face whose eye/mouth opening follows EAR/MAR."""
    import cv2

    cfg = cfg or Config()
    h, w = canvas.shape[:2]
    cx, cy, r = w // 2, h // 2 + 20, min(w, h) // 3
    cv2.circle(canvas, (cx, cy), r, (150, 170, 190), -1, cv2.LINE_AA)
    if not m.face_found or m.ear is None:
        return
    eye_h = max(1, int(m.ear / EAR_OPEN * r * 0.12))
    eye_color = (0, 0, 200) if m.ear < cfg.ear_threshold else (60, 40, 30)
    for dx in (-r // 3, r // 3):
        cv2.ellipse(canvas, (cx + dx, cy - r // 4), (r // 6, eye_h), 0, 0, 360,
                    eye_color, -1, cv2.LINE_AA)
    mouth_h = max(2, int((m.mar or 0.0) * r * 0.35))
    mouth_color = (0, 140, 255) if (m.mar or 0.0) > cfg.mar_threshold else (60, 60, 140)
    cv2.ellipse(canvas, (cx, cy + r // 2), (r // 4, mouth_h), 0, 0, 360,
                mouth_color, -1, cv2.LINE_AA)


def alert_events(states: list[tuple[float, MonitorState]]) -> list[tuple[float, str, bool]]:
    """(t, alert_name, on) for every rising/falling edge of each alert flag."""
    names = ("drowsy_alert", "yawn_alert", "perclos_alert")
    prev = dict.fromkeys(names, False)
    events = []
    for t, s in states:
        for name in names:
            value = getattr(s, name)
            if value != prev[name]:
                events.append((t, name, value))
                prev[name] = value
    return events


def run_offline(fps: float = 30.0, cfg: Config | None = None,
                noise: float = 0.01) -> list[tuple[float, MonitorState]]:
    monitor = DrowsinessMonitor(cfg or Config())
    return [(m.t, monitor.update(m))
            for m in Simulator(fps=fps, realtime=False, noise=noise)]


if __name__ == "__main__":
    for fps in (30.0, 15.0):
        states = run_offline(fps)
        print(f"--- {fps:.0f} FPS ---")
        for t, name, on in alert_events(states):
            print(f"{t:6.2f}s  {name:<14} {'ON' if on else 'off'}")
        last = states[-1][1]
        print(f"end: PERCLOS={last.perclos:.1%}  yawns(60s)={last.yawn_count_60s}")
