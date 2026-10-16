"""Warning sounds, played on a background thread so the video loop never blocks."""
import threading
from collections.abc import Callable

from .config import Config

try:
    import winsound
except ImportError:  # macOS / Linux
    winsound = None

# (frequency Hz, duration ms); drowsy is higher and longer than the others.
TONES = {
    "drowsy": (2500, 800),
    "perclos": (1600, 400),
    "yawn": (1000, 300),
}


class AlertPlayer:
    def __init__(self, cfg: Config, enabled: bool = True,
                 backend: Callable[[str], None] | None = None):
        """``backend(kind)`` plays one sound (blocking is fine, it runs in a
        thread). Defaults to the mp3 in ``cfg.sound_file``, else winsound,
        else the terminal bell. Pass a fake backend in tests."""
        self.cfg = cfg
        self.enabled = enabled
        self._play = backend or self._default_backend()
        self._last_play: float | None = None
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def trigger(self, kind: str, now: float) -> bool:
        """Play sound if cooldown elapsed. kind: 'drowsy', 'yawn' or 'perclos'.

        Returns True if a sound was started. Never blocks.
        """
        if not self.enabled or self.busy:
            return False
        if self._last_play is not None and now - self._last_play < self.cfg.alert_cooldown_s:
            return False
        self._last_play = now
        self._thread = threading.Thread(target=self._safe_play, args=(kind,), daemon=True)
        self._thread.start()
        return True

    def _safe_play(self, kind: str) -> None:
        try:
            self._play(kind)
        except Exception as e:  # a broken audio device must not kill the app
            print(f"[alert] sound failed: {e}")

    def _default_backend(self) -> Callable[[str], None]:
        if self.cfg.sound_file:
            backend = _pygame_backend(self.cfg.sound_file)
            if backend is not None:
                return backend
        if winsound is not None:
            return lambda kind: winsound.Beep(*TONES.get(kind, TONES["drowsy"]))
        return lambda kind: print("\a", end="", flush=True)


def _pygame_backend(path: str) -> Callable[[str], None] | None:
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(path)
    except Exception as e:
        print(f"[alert] cannot use sound file {path!r} ({e}); falling back to beep")
        return None

    def play(kind: str) -> None:
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # keep `busy` true while it plays
            pygame.time.wait(50)

    return play
