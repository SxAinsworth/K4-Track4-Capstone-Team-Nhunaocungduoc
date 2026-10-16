"""Tải dlib shape_predictor_68_face_landmarks.dat (~100 MB sau giải nén) vào models/.

Tải theo từng khúc vào file .part; nếu mạng đứt giữa chừng thì tự tải tiếp từ chỗ dừng
(HTTP Range), chạy lại script cũng tải tiếp chứ không tải lại từ đầu.
"""
import bz2
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
OUT = Path(__file__).parent / "models" / "shape_predictor_68_face_landmarks.dat"
PART = OUT.with_name(OUT.name + ".bz2.part")
RETRIES = 20
CHUNK = 1 << 16


def download(url: str, dest: Path) -> None:
    total = None
    for attempt in range(1, RETRIES + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have >= total:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if have and resp.status != 206:  # server bỏ qua Range -> tải lại từ đầu
                    have = 0
                length = int(resp.headers.get("Content-Length", 0))
                total = have + length if length else None
                with open(dest, "ab" if have else "wb") as f:
                    while chunk := resp.read(CHUNK):
                        f.write(chunk)
                        have += len(chunk)
                        if total:
                            print(f"\r  {have / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
            print()
            if total is None or have >= total:
                return
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # file .part đã đủ từ lần chạy trước
                return
            raise
        except OSError as e:  # URLError, IncompleteRead, timeout, reset...
            print(f"\n  Mất kết nối ({e.__class__.__name__}), thử lại {attempt}/{RETRIES}...")
            time.sleep(2)
    sys.exit(f"Tải không xong sau {RETRIES} lần thử. Chạy lại script để tải tiếp.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8")
    if OUT.exists():
        print(f"Đã có {OUT}")
    else:
        OUT.parent.mkdir(exist_ok=True)
        print(f"Đang tải {URL} ...")
        download(URL, PART)
        OUT.write_bytes(bz2.decompress(PART.read_bytes()))
        PART.unlink()
        print(f"Xong: {OUT}")
