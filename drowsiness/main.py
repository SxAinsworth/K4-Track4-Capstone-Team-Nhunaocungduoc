"""Camera / simulator loop gluing Person 1 (metrics) and Person 2 (alerts, HUD).

Run from the repo root:
    python -m drowsiness.main --sim                       # scripted data, no camera needed
    python -m drowsiness.main                             # webcam + Person 1's dlib model
    python -m drowsiness.main --video clip.mp4            # Person 3's test clips
    python -m drowsiness.main --model mediapipe           # another landmark model
Keys: q / Esc quit, c recalibrate (dlib model).
"""
import argparse
import dataclasses
import sys
import time
from collections.abc import Iterator

if __package__ in (None, ""):
    sys.exit("Run as a module from the repo root: python -m drowsiness.main [--sim]")

import cv2

from .alert import AlertPlayer
from .config import Config
from .hud import FpsMeter, draw_hud, draw_landmarks
from .landmarks import DEFAULT_MODEL, Extractor, load_extractor, model_ear_threshold
from .monitor import DrowsinessMonitor
from .simulate import Simulator, blank_canvas, draw_fake_face
from .types import FrameMetrics, MonitorState

WINDOW = "DMS"


def video_frames(source: int | str, extract: Extractor,
                 mirror: bool = False) -> Iterator[tuple[object, FrameMetrics]]:
    """Webcam (int index) or video file (path). For files, ``t`` is the clip's
    own time and playback is paced to real time, so alerts match the video."""
    is_file = isinstance(source, str)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open {'video' if is_file else 'camera'} {source!r}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, i = time.monotonic(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            m = extract(frame)
            if is_file:
                t = i / video_fps
                time.sleep(max(0.0, t - (time.monotonic() - start)))
                m = dataclasses.replace(m, t=t)
            i += 1
            yield frame, m
    finally:
        cap.release()


def sim_frames(fps: float, cfg: Config) -> Iterator[tuple[object, FrameMetrics]]:
    for m in Simulator(fps=fps, realtime=True):
        canvas = blank_canvas()
        draw_fake_face(canvas, m, cfg)
        yield canvas, m


def update_state(m: FrameMetrics, monitor: DrowsinessMonitor, cfg: Config,
                 extractor: object, default_ear: float) -> MonitorState:
    """monitor.update plus the model's calibration (face_core):

    - its per-driver EAR threshold replaces cfg.ear_threshold (cfg is shared by
      monitor and HUD); the monitor is reset so PERCLOS doesn't keep samples
      judged with the old threshold;
    - while it calibrates (driver looks straight, eyes open) alerts are held back
      and closure/PERCLOS are blanked: they use the placeholder threshold and are
      discarded when calibration ends.
    """
    threshold = model_ear_threshold(extractor, default_ear, cfg.use_model_ear_threshold)
    if threshold != cfg.ear_threshold:
        print(f"[main] EAR threshold {cfg.ear_threshold:.3f} -> {threshold:.3f}")
        cfg.ear_threshold = threshold
        monitor.reset()
    state = monitor.update(m)
    if getattr(extractor, "calibrating", False):
        state = dataclasses.replace(state, drowsy_alert=False, yawn_alert=False,
                                    perclos_alert=False, perclos=0.0, eyes_closed_s=0.0,
                                    perclos_warming_up=True)
    return state


def run(args: argparse.Namespace) -> None:
    cfg = Config()
    if args.sound_file:
        cfg = dataclasses.replace(cfg, sound_file=args.sound_file)
    default_ear = cfg.ear_threshold
    monitor = DrowsinessMonitor(cfg)
    player = AlertPlayer(cfg, enabled=not args.mute)
    fps_meter = FpsMeter()

    extractor = None
    if args.sim:
        frames = sim_frames(args.fps, cfg)
    else:
        try:
            extractor = load_extractor(args.model)
        except (ImportError, TypeError, OSError) as e:
            sys.exit(f"Cannot load model {args.model!r}: {e}\n(or run with --sim)")
        source = args.video if args.video else args.camera
        frames = video_frames(source, extractor, mirror=not args.video and not args.no_mirror)

    cv2.namedWindow(WINDOW)
    frame = None
    try:
        for frame, m in frames:
            state = update_state(m, monitor, cfg, extractor, default_ear)
            if state.drowsy_alert:
                player.trigger("drowsy", m.t)
            elif state.yawn_alert:
                player.trigger("yawn", m.t)
            elif state.perclos_alert:
                player.trigger("perclos", m.t)

            if args.draw_landmarks and extractor is not None:
                draw_landmarks(frame, getattr(extractor, "last_points", None),
                               getattr(extractor, "layout", None))
            status = getattr(extractor, "status_lines", None)
            draw_hud(frame, state, fps_meter.tick(), cfg, status() if status else None)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and hasattr(extractor, "recalibrate"):
                extractor.recalibrate()
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            if frame is not None:  # simulation / clip finished: hold the last frame
                cv2.putText(frame, "Finished - press any key", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, frame)
                cv2.waitKey(0)
    finally:
        close = getattr(extractor, "close", None)
        if close:
            close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Driver drowsiness monitor")
    parser.add_argument("--sim", action="store_true", help="use the scripted simulator instead of the camera")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help='"dlib" = Person 1\'s face_core (default), "mediapipe" or "pkg.module[:Name]"')
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--no-mirror", action="store_true", help="don't mirror the webcam image")
    parser.add_argument("--video", help="video file instead of the camera")
    parser.add_argument("--draw-landmarks", action="store_true", help="draw the eye/mouth points used")
    parser.add_argument("--fps", type=float, default=30.0, help="simulator frame rate (default 30)")
    parser.add_argument("--mute", action="store_true", help="disable warning sounds")
    parser.add_argument("--sound-file", help="mp3 to play instead of the beep (needs pygame)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
