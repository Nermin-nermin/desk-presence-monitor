"""Build a scripted test video + zones from the YOLO sample image, so the expected result is known.

Schedule (125 s video):
  Desk 1  present the whole time                          -> ~100%, 0 absences
  Desk 2  away 20s-75s                                    -> AWAY at 20, ALERT at 50, back at 75
  Desk 3  away 40s-46s (short, ignored), then away 90s+   -> 1 absence, ALERT at 120
  Desk 4  nobody ever there                               -> 0%
"""
import json
import os
import sys

import cv2
import ultralytics

out_dir = sys.argv[1] if len(sys.argv) > 1 else "test_data"
os.makedirs(out_dir, exist_ok=True)
img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
h, w = img.shape[:2]
FPS, SECONDS = 10, 125

people = {"P1": (47, 400, 239, 904), "P2": (223, 409, 344, 860), "P3": (671, 395, 810, 879)}
hidden = {"P2": [(20, 75)], "P3": [(40, 46), (90, SECONDS)]}

writer = cv2.VideoWriter(os.path.join(out_dir, "test_office.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
for i in range(FPS * SECONDS):
    t = i / FPS
    frame = img.copy()
    for pid, spans in hidden.items():
        if any(a <= t < b for a, b in spans):
            x1, y1, x2, y2 = people[pid]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), -1)
    writer.write(frame)
writer.release()

rect = lambda x1, y1, x2, y2: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
zones = {"frame_width": w, "frame_height": h, "desks": [
    {"name": "Desk 1", "polygon": rect(40, 390, 232, 910)},
    {"name": "Desk 2", "polygon": rect(228, 400, 350, 870)},
    {"name": "Desk 3", "polygon": rect(660, 390, 810, 890)},
    {"name": "Desk 4", "polygon": rect(100, 150, 640, 350)},
]}
with open(os.path.join(out_dir, "zones.json"), "w") as f:
    json.dump(zones, f, indent=2)
print("Wrote", out_dir)
