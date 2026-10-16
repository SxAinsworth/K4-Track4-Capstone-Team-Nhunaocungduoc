"""Tải dlib shape_predictor_68_face_landmarks.dat (~100 MB sau giải nén) vào models/."""
import bz2
import urllib.request
from pathlib import Path

URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
OUT = Path(__file__).parent / "models" / "shape_predictor_68_face_landmarks.dat"

if __name__ == "__main__":
    if OUT.exists():
        print(f"Đã có {OUT}")
    else:
        OUT.parent.mkdir(exist_ok=True)
        print(f"Đang tải {URL} ...")
        data = urllib.request.urlopen(URL).read()
        OUT.write_bytes(bz2.decompress(data))
        print(f"Xong: {OUT}")
