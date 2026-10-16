"""Unit tests with fake timestamps: no camera, OpenCV or audio needed.

Run from the repo root: python -m pytest drowsiness/tests
"""
import threading
import time

import pytest

from drowsiness.alert import AlertPlayer
from drowsiness.config import Config
from drowsiness.monitor import DrowsinessMonitor
from drowsiness.simulate import alert_events, run_offline
from drowsiness.types import FrameMetrics

OPEN, CLOSED = 0.30, 0.10
MOUTH, YAWN = 0.20, 0.80


def feed(monitor, t0, t1, fps=30.0, ear=OPEN, mar=MOUTH, face=True):
    """Feed samples on [t0, t1] (both ends included) and return the states."""
    states = []
    n = round((t1 - t0) * fps)
    for i in range(n + 1):
        t = t0 + i / fps
        m = FrameMetrics(t=t, ear=ear if face else None, mar=mar if face else None,
                         face_found=face)
        states.append(monitor.update(m))
    return states


def first_time(states_with_t, attr):
    return next((t for t, s in states_with_t if getattr(s, attr)), None)


# -- R1: drowsiness -----------------------------------------------------------

def test_eyes_closed_1_4s_then_open_no_alert():
    mon = DrowsinessMonitor(Config())
    states = feed(mon, 0.0, 1.0)
    states += feed(mon, 1.0 + 1 / 30, 2.4, ear=CLOSED)
    states += feed(mon, 2.4 + 1 / 30, 4.0)
    assert not any(s.drowsy_alert for s in states)


def test_eyes_closed_1_6s_alerts():
    mon = DrowsinessMonitor(Config())
    states = feed(mon, 0.0, 1.6, ear=CLOSED)
    assert states[-1].drowsy_alert
    assert states[-1].eyes_closed_s == pytest.approx(1.6)


def test_normal_blinks_no_alert():
    mon = DrowsinessMonitor(Config())
    states, t = [], 0.0
    for _ in range(10):
        states += feed(mon, t, t + 2.0)                     # open
        states += feed(mon, t + 2.0 + 1 / 30, t + 2.2, ear=CLOSED)  # 0.2 s blink
        t += 2.2 + 1 / 30
    assert not any(s.drowsy_alert for s in states)


def test_alert_clears_when_eyes_open():
    mon = DrowsinessMonitor(Config())
    assert feed(mon, 0.0, 2.0, ear=CLOSED)[-1].drowsy_alert
    assert not feed(mon, 2.0 + 1 / 30, 2.1)[-1].drowsy_alert


# -- R2: yawns ------------------------------------------------------------------

def yawn(mon, t, length=1.5):
    feed(mon, t, t + length, mar=YAWN)
    return feed(mon, t + length + 1 / 30, t + length + 0.5)[-1]


def test_mouth_open_0_5s_not_a_yawn():
    mon = DrowsinessMonitor(Config())
    state = yawn(mon, 0.0, length=0.5)
    assert state.yawn_count_60s == 0


def test_long_yawn_counted_once():
    mon = DrowsinessMonitor(Config())
    state = yawn(mon, 0.0, length=5.0)
    assert state.yawn_count_60s == 1


def test_three_yawns_within_60s_alert():
    mon = DrowsinessMonitor(Config())
    assert not yawn(mon, 0.0).yawn_alert
    assert not yawn(mon, 10.0).yawn_alert
    state = yawn(mon, 20.0)
    assert state.yawn_count_60s == 3
    assert state.yawn_alert


def test_three_yawns_over_more_than_60s_no_alert():
    mon = DrowsinessMonitor(Config())
    yawn(mon, 0.0)
    feed(mon, 2.0 + 1 / 30, 34.0)
    yawn(mon, 34.0)
    feed(mon, 36.0 + 1 / 30, 68.0)
    state = yawn(mon, 68.0)  # first yawn (counted at t=1.0) has expired
    assert state.yawn_count_60s == 2
    assert not state.yawn_alert


# -- R3: PERCLOS ----------------------------------------------------------------

def test_perclos_half_closed():
    mon = DrowsinessMonitor(Config())
    feed(mon, 0.0, 30.0, ear=CLOSED)
    state = feed(mon, 30.0 + 1 / 30, 60.0)[-1]
    assert state.perclos == pytest.approx(0.5, abs=0.01)


def test_perclos_sliding_window_forgets_old_closure():
    mon = DrowsinessMonitor(Config())
    feed(mon, 0.0, 30.0, ear=CLOSED)
    state = feed(mon, 30.0 + 1 / 30, 120.0)[-1]
    assert state.perclos == pytest.approx(0.0, abs=0.01)


def test_perclos_empty_and_warming_up():
    mon = DrowsinessMonitor(Config())
    state = mon.update(FrameMetrics(t=0.0, ear=OPEN, mar=MOUTH, face_found=True))
    assert state.perclos == 0.0
    assert state.perclos_warming_up
    state = feed(mon, 1 / 30, 11.0)[-1]
    assert not state.perclos_warming_up


def test_perclos_alert_respects_threshold_and_disable():
    mon = DrowsinessMonitor(Config(drowsy_seconds=100.0))
    feed(mon, 0.0, 20.0)
    assert feed(mon, 20.0 + 1 / 30, 30.0, ear=CLOSED)[-1].perclos_alert
    mon = DrowsinessMonitor(Config(drowsy_seconds=100.0, perclos_warn=None))
    feed(mon, 0.0, 20.0)
    assert not feed(mon, 20.0 + 1 / 30, 30.0, ear=CLOSED)[-1].perclos_alert


# -- Edge cases -----------------------------------------------------------------

def test_face_lost_within_grace_pauses_timer():
    mon = DrowsinessMonitor(Config())
    feed(mon, 0.0, 1.0, ear=CLOSED)
    lost = feed(mon, 1.0 + 1 / 30, 1.5, face=False)        # 0.5 s < grace
    assert all(not s.face_found for s in lost)
    assert lost[-1].eyes_closed_s == pytest.approx(1.0)       # frozen, not growing
    back = feed(mon, 1.5 + 1 / 30, 1.9, ear=CLOSED)
    assert back[0].eyes_closed_s == pytest.approx(1.0 + 1 / 30, abs=0.04)
    assert not back[-1].drowsy_alert                          # ~1.4 s of real closure
    assert feed(mon, 1.9 + 1 / 30, 2.1, ear=CLOSED)[-1].drowsy_alert


def test_face_lost_beyond_grace_resets_timer():
    mon = DrowsinessMonitor(Config())
    feed(mon, 0.0, 1.0, ear=CLOSED)
    lost = feed(mon, 1.0 + 1 / 30, 3.0, face=False)         # 2 s > grace
    assert lost[-1].eyes_closed_s == 0.0
    assert not lost[-1].drowsy_alert
    back = feed(mon, 3.0 + 1 / 30, 3.5, ear=CLOSED)
    assert back[-1].eyes_closed_s == pytest.approx(0.5 - 1 / 30)


def test_no_face_not_counted_as_closed_in_perclos():
    mon = DrowsinessMonitor(Config())
    feed(mon, 0.0, 10.0)
    state = feed(mon, 10.0 + 1 / 30, 40.0, face=False)[-1]
    assert state.perclos == pytest.approx(0.0, abs=0.01)


def test_out_of_order_and_duplicate_timestamps_ignored():
    mon = DrowsinessMonitor(Config())
    feed(mon, 0.0, 1.0, ear=CLOSED)
    before = mon.state
    assert mon.update(FrameMetrics(t=1.0, ear=OPEN, mar=MOUTH, face_found=True)) is before
    assert mon.update(FrameMetrics(t=0.5, ear=OPEN, mar=MOUTH, face_found=True)) is before
    assert mon.state.eyes_closed_s == pytest.approx(1.0)


def test_large_gap_resets_continuous_timers():
    mon = DrowsinessMonitor(Config())
    feed(mon, 0.0, 1.2, ear=CLOSED)
    state = mon.update(FrameMetrics(t=3.0, ear=CLOSED, mar=MOUTH, face_found=True))
    assert state.eyes_closed_s == 0.0   # frozen camera, don't assume 3 s of closure
    assert not state.drowsy_alert


@pytest.mark.parametrize("fps", [15.0, 30.0])
def test_variable_fps_same_alert_times(fps):
    mon = DrowsinessMonitor(Config())
    states = []
    for t0, t1, ear in [(0.0, 2.0, OPEN), (2.0, 5.0, CLOSED)]:
        n = round((t1 - t0) * fps)
        for i in range(n):
            t = t0 + i / fps
            states.append((t, mon.update(FrameMetrics(t, ear, MOUTH, True))))
    assert first_time(states, "drowsy_alert") == pytest.approx(3.5, abs=1 / fps + 1e-6)


# -- Section 9 simulated scenario, within +-0.2 s -------------------------------

@pytest.mark.parametrize("fps", [15.0, 30.0])
def test_simulated_timeline(fps):
    states = run_offline(fps=fps)
    events = alert_events(states)
    on = lambda name: [t for t, n, v in events if n == name and v]
    off = lambda name: [t for t, n, v in events if n == name and not v]

    assert first_time(states, "drowsy_alert") == pytest.approx(6.5, abs=0.2)
    assert len(on("drowsy_alert")) == 1               # blinks/half-closures never trigger it
    assert off("drowsy_alert")[0] < 8.0               # clears in the normal segment
    assert on("yawn_alert") == [pytest.approx(39.0, abs=0.2)]  # 3rd yawn (38 s + 1 s)
    perclos_at = lambda t: next(s.perclos for ts, s in states if ts >= t)
    assert perclos_at(60.0) > perclos_at(50.0)        # rises in the half-closed segment


# -- Alert sound (fake backend, no audio hardware) ------------------------------

def test_alert_player_non_blocking_with_cooldown():
    played, release = [], threading.Event()

    def slow_backend(kind):
        played.append(kind)
        release.wait(2.0)

    player = AlertPlayer(Config(alert_cooldown_s=2.0), backend=slow_backend)
    start = time.monotonic()
    assert player.trigger("drowsy", now=0.0)
    assert time.monotonic() - start < 0.1      # returns while the sound "plays"
    assert not player.trigger("drowsy", now=0.5)  # busy / cooldown
    release.set()
    player._thread.join(1.0)
    assert not player.trigger("yawn", now=1.9)    # still within cooldown
    assert player.trigger("yawn", now=2.1)
    player._thread.join(1.0)
    assert played == ["drowsy", "yawn"]


def test_alert_player_muted():
    player = AlertPlayer(Config(), enabled=False, backend=lambda kind: None)
    assert not player.trigger("drowsy", now=0.0)
