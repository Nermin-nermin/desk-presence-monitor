"""Video sources: webcam index, video file, or RTSP/HTTP stream.

Credentials
  Never put a camera password on the command line or in config files that go to git. Put the URL in a
  `.env` file (git-ignored) and reference it:  --source '${CAM1_URL}'.  Anything printed or saved goes
  through redact(), which masks `user:password@` in URLs.

Live streams
  LiveReader decodes the stream in a background thread and keeps only the newest frame, so processing
  never falls behind real time (OpenCV's own RTSP buffer would otherwise grow a backlog). If the stream
  drops, it reconnects automatically with exponential backoff.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2

_CRED = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")
_UNRESOLVED = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def redact(text: object) -> str:
    """Mask credentials in any URL inside `text`: rtsp://user:pass@host -> rtsp://***@host."""
    return _CRED.sub(r"\1***@", str(text))


def load_env_file(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (existing variables win)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def resolve_source(raw: str) -> str | int:
    """Expand ${VARS} (from the environment / .env) and turn '0', '1', ... into webcam indexes."""
    value = os.path.expandvars(raw)
    missing = _UNRESOLVED.findall(value)
    if missing:
        sys.exit(f"Source refers to undefined variable(s): {', '.join(missing)} - set them in .env or the environment.")
    return int(value) if value.isdigit() else value


# Backwards-compatible name used by older scripts
parse_source = resolve_source


def is_live(source: str | int) -> bool:
    """Webcams and network streams run on wall-clock time; files run on video time."""
    if isinstance(source, int):
        return True
    return source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")) or not os.path.exists(source)


def _capture(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://")):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def open_source(source: str | int) -> cv2.VideoCapture:
    cap = _capture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {redact(source)}")
    return cap


class LiveReader:
    """Background reader for live sources: newest-frame-only, auto-reconnect."""

    def __init__(self, source: str | int, max_backoff: float = 30.0, log=print):
        self.source = source
        self.max_backoff = max_backoff
        self.log = log
        self._frame = None
        self._frame_id = 0
        self._last_returned = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self.connected = False
        self.reconnects = 0
        self.fps = 25.0
        self._thread = threading.Thread(target=self._run, name="LiveReader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        backoff = 1.0
        first = True
        while not self._stop.is_set():
            cap = _capture(self.source)
            if not cap.isOpened():
                cap.release()
                self.log(f"[stream] cannot open {redact(self.source)}; retrying in {backoff:.0f}s")
                self._stop.wait(backoff)
                backoff = min(self.max_backoff, backoff * 2)
                continue
            self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            if not first:
                self.reconnects += 1
                self.log(f"[stream] reconnected to {redact(self.source)} (reconnect #{self.reconnects})")
            first = False
            backoff = 1.0
            self.connected = True
            failures = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures >= 25:      # ~1 s of failed reads at 25 fps: treat as disconnected
                        break
                    time.sleep(0.04)
                    continue
                failures = 0
                with self._cond:
                    self._frame = frame
                    self._frame_id += 1
                    self._cond.notify_all()
            cap.release()
            if not self._stop.is_set():
                self.connected = False
                self.log(f"[stream] lost {redact(self.source)}; reconnecting")

    def read(self, timeout: float = 5.0):
        """Return the newest frame not returned before, waiting up to `timeout` s. None if nothing arrived."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame_id > self._last_returned or self._stop.is_set(),
                                       timeout=timeout):
                return None
            if self._stop.is_set():
                return None
            self._last_returned = self._frame_id
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5)
