"""Draw desk zones on the first frame of a camera / video and save them to zones.json.

Controls (in the window):
  Left click   add a corner point to the current desk
  Right click  undo the last point
  N / Enter    finish the current desk (needs >= 3 points)
  D            delete the last finished desk
  S            save to zones.json and quit
  Q / Esc      quit without saving

Tip: draw each zone around the chair + front edge of the desk - the area a
seated person's body actually covers in the image - not the whole desktop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from video_source import load_env_file, open_source, redact, resolve_source

WINDOW = "Desk zone setup"
COLORS = [(255, 170, 0), (0, 200, 255), (200, 80, 255), (0, 220, 120),
          (255, 90, 90), (90, 160, 255), (255, 220, 60), (160, 255, 160)]


def grab_frame(source: str | int, skip_seconds: float) -> np.ndarray:
    cap = open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame = None
    for _ in range(max(1, int(skip_seconds * fps))):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    if frame is None:
        sys.exit(f"Could not read a frame from {redact(source)}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, webcam index (0), or '${CAM1_URL}' from .env")
    ap.add_argument("--image", help="use this still image instead of grabbing a frame from --source")
    ap.add_argument("--out", default="zones.json")
    ap.add_argument("--skip", type=float, default=1.0, help="seconds to skip before grabbing the frame")
    ap.add_argument("--env-file", default=".env", help="file with camera URLs/passwords (see .env.example)")
    args = ap.parse_args()

    load_env_file(args.env_file)
    frame = cv2.imread(args.image) if args.image else grab_frame(resolve_source(args.source), args.skip)
    h, w = frame.shape[:2]
    scale = min(1.0, 1400 / w)  # fit large frames on screen; points are stored in original pixels

    desks: list[dict] = []
    current: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((int(x / scale), int(y / scale)))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        canvas = frame.copy()
        for i, d in enumerate(desks):
            pts = np.array(d["polygon"], np.int32)
            color = COLORS[i % len(COLORS)]
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
            cv2.polylines(canvas, [pts], True, color, 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(canvas, d["name"], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if current:
            pts = np.array(current, np.int32)
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)
            for p in current:
                cv2.circle(canvas, p, 4, (0, 255, 255), -1)
        help_txt = f"Desks: {len(desks)} | click=point  right=undo  N=finish desk  D=delete last  S=save  Q=quit"
        cv2.rectangle(canvas, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(canvas, help_txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        shown = cv2.resize(canvas, None, fx=scale, fy=scale) if scale < 1 else canvas
        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("n"), ord("N"), 13):
            if len(current) >= 3:
                desks.append({"name": f"Desk {len(desks) + 1}", "polygon": list(current)})
                current.clear()
            else:
                print("A desk needs at least 3 points.")
        elif key in (ord("d"), ord("D")) and desks:
            desks.pop()
        elif key in (ord("s"), ord("S")):
            if not desks:
                print("Nothing to save - draw at least one desk.")
                continue
            ref_name = Path(args.out).with_suffix(".ref.jpg").name
            cv2.imwrite(str(Path(args.out).with_name(ref_name)), frame)  # reference view for stabilize: true
            data = {"frame_width": w, "frame_height": h, "reference_image": ref_name, "desks": desks}
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(desks)} desk(s) to {args.out}. Rename desks (e.g. 'Finance - Aysel') in that file if you like.")
            break
        elif key in (ord("q"), ord("Q"), 27):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
