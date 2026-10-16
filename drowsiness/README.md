# drowsiness: driver drowsiness monitor

The app combines Person 1's face core (`face_core.py`: dlib HOG face detector, 68 landmarks, EAR/MAR/head
pitch, per-driver calibration) with the alert logic and HUD in this package.

## Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python download_model.py            # dlib 68-point model, about 100 MB, into models/
python -m pytest drowsiness/tests   # should pass with no camera
```

Run every command from the repo root, and run the app as a module (`python -m drowsiness.main`).
Running `python drowsiness/main.py` directly would let `drowsiness/types.py` shadow the standard-library
`types` module.

## Commands

```powershell
python -m drowsiness.main --sim                    # 60 s scripted demo, no camera
python -m drowsiness.main                          # webcam + Person 1's dlib model
python -m drowsiness.main --draw-landmarks         # also show the eye/mouth points
python -m drowsiness.main --video clip.mp4         # recorded clip (Person 3's test data)
python -m drowsiness.landmarks                     # check the model only: FPS, EAR/MAR ranges
python -m drowsiness.simulate                      # print the simulated alert timeline
```

Keys: `q`/`Esc` quit, `c` recalibrate (for example when the driver changes).
Options: `--camera N`, `--no-mirror`, `--mute`, `--sound-file alert.mp3` (needs pygame),
`--model mediapipe` (optional alternative model).

## Demo script (about 4 minutes)

1. **Simulator, 1 min.** `python -m drowsiness.main --sim` shows every alert without a face:
   drowsy at about 6.5 s, yawn alert at the third yawn (about 39 s), then PERCLOS rising from 50 s.
2. **Live webcam.** `python -m drowsiness.main --draw-landmarks`. Sit 50-80 cm from the camera,
   facing it, in even light.
   | Do | You should see |
   |---|---|
   | Look straight with eyes open for the first 4 s | `CALIBRATING x%`, then `EAR thr ... (open ...)`. Alerts are held back until then |
   | Blink normally | Nothing; blinks are under 0.4 s |
   | Close your eyes for 2 s | Red border, `DROWSY! WAKE UP` and a beep after 1.5 s |
   | Yawn 3 times within a minute, each with the mouth wide for over 1 s | `FREQUENT YAWNING` and a lower beep |
   | Keep half-closing your eyes for 20-30 s | PERCLOS turns red past 15%, then `HIGH PERCLOS - TAKE A BREAK` |
   | Turn away or cover the camera | `NO FACE`; timers pause for 1 s, then reset |
   | Press `c` | Calibration starts again |
3. **Recorded clip.** `python -m drowsiness.main --video clip.mp4`. The clip's own time is used, so
   alerts match the video. Calibration runs on the first 4 s, so clips must start with the driver
   looking straight with eyes open.

## How the pieces connect

```
frame ─► FaceCoreExtractor ─► FrameMetrics ─► DrowsinessMonitor ─► AlertPlayer + draw_hud
         (face_core_adapter.py wraps            (monitor.py)        (alert.py, hud.py)
          Person 1's face_core.FaceCore)
```

- `face_core.py` is used unchanged. `face_core_adapter.py` turns its `FaceMetrics` (or `None`) into
  the agreed `FrameMetrics(t, ear, mar, face_found)`.
- **Calibration:** FaceCore sets a per-driver EAR threshold (72% of the eyes-open EAR). `main.py` uses it
  instead of `Config.ear_threshold` (disable with `use_model_ear_threshold = False` in `config.py`).
  While calibrating, alerts are held back. When the threshold arrives, the monitor is reset so PERCLOS
  doesn't keep samples judged with the placeholder threshold.
- Head pitch from FaceCore is shown on the HUD but doesn't raise an alert yet.

## Tuning

Alert thresholds live in `config.py`: drowsy duration, `mar_threshold`, yawn count and window, PERCLOS.
Person 1's calibration ratio (`EAR_RATIO = 0.72`) lives in `face_core.py`. Use
`python -m drowsiness.landmarks --video clip.mp4` on Person 3's clips to see the EAR/MAR ranges.
The HUD shows each value in green, or in red once it crosses its threshold.

## Other models

`--model` also accepts `"mediapipe"` or `"pkg.module[:Name]"` (a class instantiated with no arguments,
or a function `frame -> FrameMetrics`). `person1_template.py` is a starting point for a new model.

| Module | Role |
|---|---|
| `types.py` | `FrameMetrics` (input), `MonitorState` (output) |
| `face_core_adapter.py` | Person 1's `FaceCore` -> `FrameMetrics`, calibration status |
| `landmarks.py` | EAR/MAR geometry, model registry (`dlib`, `mediapipe`), contract checker |
| `monitor.py` | `DrowsinessMonitor`: drowsy (R1), yawns (R2), PERCLOS (R3), timestamp-based |
| `alert.py` | `AlertPlayer`: sound on a daemon thread with cooldown (winsound / pygame mp3 / bell) |
| `hud.py` | `draw_hud`, `draw_landmarks`, `FpsMeter` |
| `simulate.py` | Scripted timeline and cartoon face for `--sim` |
| `main.py` | Camera / video / simulator loop |
