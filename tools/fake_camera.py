"""A fake IP camera for testing: serves a video file as a password-protected MJPEG stream over HTTP.

Loops the file in real time, like a live camera. Each frame carries its frame number as a small
black/white bar code in the top-left corner, so a client can measure how far behind live it is.

Usage:
  python tools/fake_camera.py --video office.mp4 --user admin --password secret --port 8090
  # then, in .env:  CAM1_URL=http://admin:secret@127.0.0.1:8090/stream.mjpg
"""
from __future__ import annotations

import argparse
import base64
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BITS, CELL = 20, 12   # frame-number bar code: 20 bits, 12 px cells


def stamp(frame: np.ndarray, n: int) -> None:
    for i in range(BITS):
        v = 255 if (n >> i) & 1 else 0
        frame[0:CELL, i * CELL:(i + 1) * CELL] = v


def read_stamp(frame: np.ndarray) -> int:
    n = 0
    for i in range(BITS):
        if frame[CELL // 4:CELL * 3 // 4, i * CELL + CELL // 4:(i + 1) * CELL - CELL // 4].mean() > 127:
            n |= 1 << i
    return n


class Camera:
    """Produces the 'live' frame: the file played in real time, looping."""

    def __init__(self, video: str, quality: int = 80):
        self.video, self.quality = video, quality
        self.jpeg, self.number = None, 0
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            cap = cv2.VideoCapture(self.video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            t_next = time.monotonic()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.number += 1
                stamp(frame, self.number)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                with self.cond:
                    self.jpeg = buf.tobytes()
                    self.cond.notify_all()
                t_next += 1 / fps
                time.sleep(max(0, t_next - time.monotonic()))
            cap.release()


def make_handler(cam: Camera, user: str, password: str):
    token = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.headers.get("Authorization") != token:
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="camera"')
                self.end_headers()
                return
            if self.path == "/now":                    # current live frame number (for lag tests)
                body = str(cam.number).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    with cam.cond:
                        cam.cond.wait_for(lambda: cam.jpeg is not last, timeout=5)
                        jpeg = last = cam.jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    cam = Camera(args.video)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(cam, args.user, args.password))
    print(f"fake camera on http://{args.host}:{args.port}/stream.mjpg", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
