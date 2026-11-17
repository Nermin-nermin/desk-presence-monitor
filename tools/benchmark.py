"""Benchmark detector setups on a clip: speed (ms/frame) and agreement of per-desk occupancy with a reference setup.

Usage:
  python tools/benchmark.py --source clip.mp4 --zones zones.json [--stabilize] [--every 6]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector import PersonDetector  # noqa: E402
from monitor import load_zones  # noqa: E402
from occupancy import zones_occupied  # noqa: E402
from stabilizer import Stabilizer  # noqa: E402

SETUPS = [  # (label, model, mode, imgsz)
    ("full yolo11s @1280 (reference)", "yolo11s.pt", "full", 1280),
    ("full yolo11n @1280", "yolo11n.pt", "full", 1280),
    ("full yolo11n @640", "yolo11n.pt", "full", 640),
    ("ROI  yolo11s @320", "yolo11s.pt", "roi", 320),
    ("ROI  yolo11n @320", "yolo11n.pt", "roi", 320),
    ("ROI  yolo11n @256", "yolo11n.pt", "roi", 256),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--zones", required=True)
    ap.add_argument("--stabilize", action="store_true")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--min-overlap", type=float, default=0.25)
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    ok, first = cap.read()
    zones = load_zones(args.zones, first.shape)
    stab = Stabilizer(first) if args.stabilize else None
    frames, i, t_stab = [], 0, []
    while ok:
        if i % args.every == 0:
            t = time.perf_counter()
            frames.append(stab.align(first if i == 0 else frame) if stab else (first if i == 0 else frame))
            t_stab.append(time.perf_counter() - t)
        ok, frame = cap.read()
        i += 1
    print(f"{len(frames)} frames; stabilization {1000 * np.mean(t_stab):.1f} ms/frame")

    results, reference = [], None
    for label, model, mode, imgsz in SETUPS:
        det = PersonDetector(model, 0.3, imgsz, roi_zones=zones if mode == "roi" else None,
                             roi_imgsz=imgsz, frame_shape=first.shape)
        for f in frames[:3]:
            det.detect(f)  # warm-up
        occ, times = [], []
        for f in frames:
            t = time.perf_counter()
            boxes = det.detect(f)
            times.append(time.perf_counter() - t)
            o = zones_occupied(zones, boxes, args.min_overlap)
            occ.append([o[z.name] for z in zones])
        occ = np.array(occ)
        if reference is None:
            reference = occ
        agree = float((occ == reference).mean())
        ms = 1000 * float(np.mean(times))
        results.append({"setup": label, "ms_per_frame": round(ms, 1), "fps": round(1000 / ms, 1),
                        "agreement_with_reference": round(100 * agree, 1), "occupancy": occ.astype(int).tolist()})
        print(f"{label:32s} {ms:7.1f} ms/frame  {1000 / ms:5.1f} FPS   agreement {100 * agree:5.1f}%")
    Path(args.out).write_text(json.dumps({"stabilize_ms": round(1000 * float(np.mean(t_stab)), 1),
                                          "desks": [z.name for z in zones], "results": results}, indent=1))


if __name__ == "__main__":
    main()
