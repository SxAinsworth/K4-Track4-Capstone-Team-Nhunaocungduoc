"""Core alert logic: (EAR, MAR) stream -> drowsiness, yawning and PERCLOS.

No OpenCV and no audio here, so everything is unit-testable with fake
timestamps. All durations are measured with timestamps, never frame counts,
so the alerts fire at the same real times at any FPS.
"""
from collections import deque

from .config import Config
from .types import FrameMetrics, MonitorState

_EPS = 1e-9


class DrowsinessMonitor:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.reset()

    def reset(self) -> None:
        self._start_t: float | None = None
        self._last_t: float | None = None
        self._last_closed: bool | None = None  # eye state of previous sample; None = no face
        # R1: drowsiness
        self._closed_since: float | None = None
        # R2: yawns
        self._mouth_open_since: float | None = None
        self._yawn_counted = False             # already counted this mouth-open episode
        self._yawn_times: deque[float] = deque()
        # R3: PERCLOS, as (t_start, t_end, closed) intervals between consecutive samples
        self._intervals: deque[tuple[float, float, bool]] = deque()
        self._total_s = 0.0
        self._closed_s = 0.0
        # Edge case: face lost
        self._face_lost_since: float | None = None
        self._state = MonitorState(
            ear=None, mar=None, perclos=0.0, yawn_count_60s=0, eyes_closed_s=0.0,
            drowsy_alert=False, yawn_alert=False, face_found=False,
        )

    @property
    def state(self) -> MonitorState:
        return self._state

    def update(self, m: FrameMetrics) -> MonitorState:
        cfg = self.cfg

        # Out-of-order or duplicate timestamps: ignore the sample.
        if self._last_t is not None and m.t <= self._last_t:
            return self._state
        if self._start_t is None:
            self._start_t = m.t

        # Large gap (frozen camera): don't interpolate, reset continuous timers.
        if self._last_t is not None and m.t - self._last_t > cfg.max_gap_s:
            self._reset_timers()
        elif self._last_closed is not None:
            # Time-weighted PERCLOS: the interval belongs to the earlier sample.
            self._add_interval(self._last_t, m.t, self._last_closed)

        face = m.face_found and m.ear is not None
        if face:
            closed = m.ear < cfg.ear_threshold
            eyes_closed_s = self._update_face(m, closed)
        else:
            closed = None
            eyes_closed_s = self._update_no_face(m)

        # Expire old yawns (also while the face is lost).
        while self._yawn_times and m.t - self._yawn_times[0] > cfg.yawn_window_s:
            self._yawn_times.popleft()
        yawn_count = len(self._yawn_times)

        perclos = self._perclos(m.t)
        warming_up = m.t - self._start_t < cfg.perclos_warmup_s

        self._state = MonitorState(
            ear=m.ear if face else None,
            mar=m.mar if face else None,
            perclos=perclos,
            yawn_count_60s=yawn_count,
            eyes_closed_s=eyes_closed_s,
            drowsy_alert=eyes_closed_s + _EPS >= cfg.drowsy_seconds,
            yawn_alert=yawn_count >= cfg.yawn_alert_count,
            face_found=face,
            perclos_alert=(cfg.perclos_warn is not None and not warming_up
                           and perclos > cfg.perclos_warn),
            perclos_warming_up=warming_up,
        )
        self._last_t = m.t
        self._last_closed = closed
        return self._state

    # -- face present / absent ------------------------------------------------

    def _update_face(self, m: FrameMetrics, closed: bool) -> float:
        cfg = self.cfg

        # Face is back within the grace period: resume the paused timers by
        # shifting their start forward by the time the face was missing.
        if self._face_lost_since is not None:
            paused = m.t - self._face_lost_since
            if self._closed_since is not None:
                self._closed_since += paused
            if self._mouth_open_since is not None:
                self._mouth_open_since += paused
            self._face_lost_since = None

        # R1: continuous eye closure
        if closed:
            if self._closed_since is None:
                self._closed_since = m.t
            eyes_closed_s = m.t - self._closed_since
        else:
            self._closed_since = None
            eyes_closed_s = 0.0

        # R2: one yawn per mouth-open episode lasting >= yawn_min_seconds
        if m.mar is not None and m.mar > cfg.mar_threshold:
            if self._mouth_open_since is None:
                self._mouth_open_since = m.t
            if (not self._yawn_counted
                    and m.t - self._mouth_open_since + _EPS >= cfg.yawn_min_seconds):
                self._yawn_times.append(m.t)
                self._yawn_counted = True
        else:
            self._mouth_open_since = None
            self._yawn_counted = False

        return eyes_closed_s

    def _update_no_face(self, m: FrameMetrics) -> float:
        if self._face_lost_since is None:
            self._face_lost_since = m.t
        if m.t - self._face_lost_since > self.cfg.no_face_grace_s:
            # Lost too long: a missing face must not count as closed eyes.
            self._reset_timers()
            return 0.0
        # Within grace: timers are paused, keep showing the frozen value.
        return self._state.eyes_closed_s if self._closed_since is not None else 0.0

    def _reset_timers(self) -> None:
        self._closed_since = None
        self._mouth_open_since = None

    # -- PERCLOS ------------------------------------------------------------------

    def _add_interval(self, t0: float, t1: float, closed: bool) -> None:
        self._intervals.append((t0, t1, closed))
        self._total_s += t1 - t0
        if closed:
            self._closed_s += t1 - t0

    def _perclos(self, now: float) -> float:
        cutoff = now - self.cfg.perclos_window_s
        while self._intervals and self._intervals[0][1] <= cutoff:
            t0, t1, closed = self._intervals.popleft()
            self._total_s -= t1 - t0
            if closed:
                self._closed_s -= t1 - t0
        if not self._intervals:
            self._total_s = self._closed_s = 0.0  # also clears float drift
            return 0.0

        total, closed_s = self._total_s, self._closed_s
        t0, _, closed = self._intervals[0]
        if t0 < cutoff:  # clip the part of the oldest interval outside the window
            total -= cutoff - t0
            if closed:
                closed_s -= cutoff - t0
        if total <= _EPS:
            return 0.0
        return min(max(closed_s / total, 0.0), 1.0)
