"""Person detection: full frame, or only around desk zones (ROI mode).

Why ROI mode: most of a CCTV frame is floor, walls and corridors we don't care about. Cropping a
generous region around each desk and running the crops through YOLO in one batch:
  - is much cheaper than running at a high resolution on the whole frame, and
  - enlarges small, distant people (a 120 px desk crop is upscaled to `roi_imgsz`), which helps recall.
Boxes are mapped back to frame coordinates and de-duplicated with NMS.
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from occupancy import DeskZone


class PersonDetector:
    def __init__(self, model: str, conf: float, imgsz: int,
                 roi_zones: list[DeskZone] | None = None, roi_imgsz: int = 320,
                 roi_margin: float = 0.6, frame_shape: tuple[int, int] | None = None):
        self.model = YOLO(model)
        self.conf, self.imgsz, self.roi_imgsz = conf, imgsz, roi_imgsz
        self.crops: list[tuple[int, int, int, int]] = []
        if roi_zones:
            assert frame_shape is not None
            self.crops = self._make_crops(roi_zones, frame_shape, roi_margin)

    @staticmethod
    def _make_crops(zones, frame_shape, margin):
        """Square-ish crop around each zone, extended upwards (heads/torsos sit above the chair area)."""
        h, w = frame_shape[:2]
        crops = []
        for z in zones:
            x1, y1 = z.polygon.min(axis=0)
            x2, y2 = z.polygon.max(axis=0)
            zw, zh = x2 - x1, y2 - y1
            side = max(zw, zh) * (1 + margin)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - zh * margin * 0.25
            crops.append((int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
                          int(min(w, cx + side / 2)), int(min(h, cy + side / 2))))
        return crops

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """Return person boxes [x1, y1, x2, y2] in frame coordinates."""
        if not self.crops:
            r = self.model.predict(frame, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy().tolist()

        images = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in self.crops]
        results = self.model.predict(images, classes=[0], conf=self.conf, imgsz=self.roi_imgsz, verbose=False)
        boxes, scores = [], []
        for (ox, oy, _, _), r in zip(self.crops, results):
            for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                boxes.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy])
                scores.append(float(s))
        if not boxes:
            return []
        # Overlapping crops can see the same person twice: keep one box per person.
        xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        keep = cv2.dnn.NMSBoxes(xywh, scores, self.conf, 0.5)
        return [boxes[i] for i in np.array(keep).flatten()]
