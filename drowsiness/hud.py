"""On-screen HUD (FPS, EAR, MAR, PERCLOS, yawns) and visual alerts."""
import time

import cv2

from .config import Config
from .types import MonitorState

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
GREEN = (0, 220, 0)
RED = (0, 0, 255)
BAD = (70, 70, 255)     # brighter red, readable as small text on the dark panel
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)


class FpsMeter:
    """Exponential moving average of the frame rate: fps = 0.9*fps + 0.1*instant."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self._last: float | None = None

    def tick(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        if self._last is not None:
            instant = 1.0 / max(now - self._last, 1e-6)
            # Start from the first measurement instead of ramping up from 0.
            self.fps = instant if self.fps == 0.0 else (1 - self.alpha) * self.fps + self.alpha * instant
        self._last = now
        return self.fps


def draw_hud(frame, state: MonitorState, fps: float, cfg: Config | None = None,
             extra_lines: list[str] | None = None) -> None:
    """Draw the metrics panel and any active alerts onto ``frame`` (BGR, in place).

    ``extra_lines`` are model status lines (e.g. calibration) shown under the metrics;
    lines starting with "CALIBRATING" are highlighted.
    """
    cfg = cfg or Config()
    scale = max(frame.shape[1] / 640, 0.6)
    lines = _panel_lines(state, fps, cfg)
    for text in extra_lines or ():
        lines.append([(text, YELLOW if text.startswith("CALIBRATING") else GRAY)])
    _draw_panel(frame, lines, scale)
    _draw_alerts(frame, state, scale)


def draw_landmarks(frame, points, layout) -> None:
    """Draw the eye/mouth points the EAR/MAR math uses (to check a model's ids)."""
    if points is None or layout is None:
        return
    for i in layout.ids:
        x, y = points[i][0], points[i][1]
        cv2.circle(frame, (int(x), int(y)), 2, YELLOW, -1, cv2.LINE_AA)


def _ok(good: bool) -> tuple[int, int, int]:
    return GREEN if good else BAD


def _panel_lines(state: MonitorState, fps: float, cfg: Config):
    """Each line is a list of (text, color) segments."""
    if state.ear is None:
        ear_seg = ("--", GRAY)
    else:
        ear_seg = (f"{state.ear:.2f}", _ok(state.ear >= cfg.ear_threshold))
    if state.mar is None:
        mar_seg = ("--", GRAY)
    else:
        mar_seg = (f"{state.mar:.2f}", _ok(state.mar <= cfg.mar_threshold))

    perclos_ok = cfg.perclos_warn is None or state.perclos <= cfg.perclos_warn
    perclos_line = [("PERCLOS: ", WHITE), (f"{state.perclos:.1%}", _ok(perclos_ok))]
    if state.perclos_warming_up:
        perclos_line.append(("  warming up", GRAY))

    return [
        [("FPS: ", WHITE), (f"{fps:.1f}", WHITE)],
        [("EAR: ", WHITE), ear_seg, ("   MAR: ", WHITE), mar_seg],
        perclos_line,
        [("Yawns (60s): ", WHITE),
         (str(state.yawn_count_60s), _ok(state.yawn_count_60s < cfg.yawn_alert_count))],
        [("Eyes closed: ", WHITE),
         (f"{state.eyes_closed_s:.1f}s", _ok(state.eyes_closed_s < cfg.drowsy_seconds))],
    ]


def _draw_panel(frame, lines, scale: float) -> None:
    font_scale, thickness = 0.55 * scale, max(1, round(scale))
    pad, x0, y0 = int(8 * scale), int(10 * scale), int(10 * scale)
    (_, text_h), _ = cv2.getTextSize("Ag", FONT, font_scale, thickness)
    line_h = int(text_h * 1.9)

    widths = [sum(cv2.getTextSize(t, FONT, font_scale, thickness)[0][0] for t, _ in line)
              for line in lines]
    h, w = frame.shape[:2]
    x1 = min(w, x0 + max(widths) + 2 * pad)
    y1 = min(h, y0 + line_h * len(lines) + pad)

    # Semi-transparent dark box so the text stays readable on any background.
    roi = frame[y0:y1, x0:x1]
    roi[:] = cv2.convertScaleAbs(roi, alpha=0.2)

    y = y0 + pad + text_h
    for line in lines:
        x = x0 + pad
        for text, color in line:
            cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)
            x += cv2.getTextSize(text, FONT, font_scale, thickness)[0][0]
        y += line_h


def _put_centered(frame, text: str, y: int, font_scale: float, color, thickness: int) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    x = max(0, (frame.shape[1] - tw) // 2)
    # Black outline first so the text reads on light backgrounds too.
    cv2.putText(frame, text, (x, y), FONT, font_scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_alerts(frame, state: MonitorState, scale: float) -> None:
    h, w = frame.shape[:2]
    if state.drowsy_alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, int(12 * scale))
        _put_centered(frame, "DROWSY! WAKE UP", h // 2, 1.6 * scale, RED, max(2, int(4 * scale)))

    # Secondary messages stacked from the bottom up.
    y = h - int(25 * scale)
    if state.yawn_alert:
        _put_centered(frame, "FREQUENT YAWNING", y, 1.0 * scale, ORANGE, max(2, int(2 * scale)))
        y -= int(45 * scale)
    if state.perclos_alert:
        _put_centered(frame, "HIGH PERCLOS - TAKE A BREAK", y, 0.8 * scale, ORANGE,
                      max(2, int(2 * scale)))
        y -= int(40 * scale)
    if not state.face_found:
        _put_centered(frame, "NO FACE", int(h * 0.3), 1.2 * scale, YELLOW, max(2, int(3 * scale)))
