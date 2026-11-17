"""Suggest desk zones automatically: find places where a person stays put for much of the footage.

Why: drawing dozens of zones by hand is slow and misses desks. Seats are where people are
detected in (almost) the same spot again and again; people walking around don't form stable clusters.

Steps
  1. sample frames (aligned to the reference view if --stabilize)
  2. tiled detection: overlapping tiles, each enlarged, so small distant people are found
  3. cluster boxes across frames by IoU; keep clusters seen in >= --min-presence of the samples
  4. zone = median box of the cluster with the top part (head) trimmed -> body + chair area

Usage:
  python tools/discover_seats.py --source clip.mp4 --stabilize --out zones_auto.json [--preview preview.jpg]
Review the preview, delete/rename zones in the JSON if needed (or adjust them in setup_zones.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stabilizer import Stabilizer  # noqa: E402


def tiles(w, h, cols, rows, overlap):
    tw, th = int(w / (cols - (cols - 1) * overlap)), int(h / (rows - (rows - 1) * overlap))
    xs = np.linspace(0, w - tw, cols).astype(int)
    ys = np.linspace(0, h - th, rows).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def tiled_detect(model, frame, grid, conf, imgsz):
    h, w = frame.shape[:2]
    regions = tiles(w, h, *grid, 0.25)
    res = model.predict([frame[y1:y2, x1:x2] for x1, y1, x2, y2 in regions], classes=[0], conf=conf,
                        imgsz=imgsz, verbose=False)
    boxes, scores = [], []
    for (ox, oy, _, _), r in zip(regions, res):
        for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            boxes.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy])
            scores.append(float(s))
    if not boxes:
        return np.zeros((0, 4))
    keep = np.array(cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], scores, conf, 0.5)).flatten()
    return np.array([boxes[i] for i in keep])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--stabilize", action="store_true")
    ap.add_argument("--samples", type=int, default=40, help="number of frames to sample across the clip")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--grid", default="3x2", help="tiles, cols x rows")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-presence", type=float, default=0.25, help="share of samples a seat must be occupied in")
    ap.add_argument("--max-aspect", type=float, default=2.3, help="height/width above this = standing person")
    ap.add_argument("--out", default="zones_auto.json")
    ap.add_argument("--preview")
    ap.add_argument("--dump-detections", help="save per-sample boxes (json) for later evaluation")
    ap.add_argument("--load-detections", help="reuse boxes saved with --dump-detections (skips detection)")
    args = ap.parse_args()
    grid = tuple(int(v) for v in args.grid.lower().split("x"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wanted = set(np.linspace(0, n - 1, args.samples).astype(int))
    ok, ref = cap.read()
    stab = Stabilizer(ref) if args.stabilize else None
    frames_boxes, i, frame = [], 0, ref
    if args.load_detections:
        frames_boxes = [(d["t"], np.array(d["boxes"]).reshape(-1, 4)) for d in json.loads(Path(args.load_detections).read_text())]
        ok = False
    else:
        model = YOLO(args.model)
    while ok:
        aligned = stab.align(frame) if stab else frame
        if i in wanted:
            frames_boxes.append((i / fps, tiled_detect(model, aligned, grid, args.conf, args.imgsz)))
        ok, frame = cap.read()
        i += 1
    print(f"{len(frames_boxes)} samples, {np.mean([len(b) for _, b in frames_boxes]):.1f} people per frame on average")

    # Cluster on the "seat point" (bottom-centre of the person = where the chair is). Distance is
    # relative to person height, so the same rule works for near (big) and far (small) people.
    clusters: list[dict] = []
    for k, (_, boxes) in enumerate(frames_boxes):
        for b in boxes:
            w_, h_ = b[2] - b[0], b[3] - b[1]
            pt = np.array([(b[0] + b[2]) / 2, b[3] - 0.15 * h_])
            best, best_d = None, None
            for c in clusters:
                if c["last"] == k:
                    continue  # one box per frame per cluster
                d = np.linalg.norm(pt - np.median(c["pts"], axis=0)) / np.median(c["h"])
                if d < 0.35 and (best_d is None or d < best_d):
                    best, best_d = c, d
            if best is None:
                clusters.append({"boxes": [b], "pts": [pt], "h": [h_], "ar": [h_ / max(w_, 1)], "last": k, "frames": {k}})
            else:
                best["boxes"].append(b); best["pts"].append(pt); best["h"].append(h_)
                best["ar"].append(h_ / max(w_, 1)); best["last"] = k; best["frames"].add(k)

    n_samples = len(frames_boxes)
    seats = []
    for c in clusters:
        presence = len(c["frames"]) / n_samples
        if presence < args.min_presence:
            continue
        if np.median(c["ar"]) > args.max_aspect:
            continue  # tall and thin = standing person (chatting, waiting), not a seat
        x1, y1, x2, y2 = np.median(c["boxes"], axis=0)
        h = y2 - y1
        zone = [x1, y1 + 0.3 * h, x2, y2]          # drop the head: body + chair area
        seats.append({"zone": zone, "presence": presence, "height": h})
    # Remove near-duplicate seats (same person split into two clusters)
    seats.sort(key=lambda s: -s["presence"])
    final = []
    for s in seats:
        if all(iou(s["zone"], f["zone"]) < 0.3 for f in final):
            final.append(s)
    # Name seats row by row: rows from the front (bottom of image) backwards, left to right within a row
    final.sort(key=lambda s: -s["zone"][3])
    rows, cur = [], []
    for s in final:
        if cur and abs(s["zone"][3] - np.mean([c["zone"][3] for c in cur])) > 0.6 * np.median([c["height"] for c in cur]):
            rows.append(cur)
            cur = []
        cur.append(s)
    if cur:
        rows.append(cur)
    desks = []
    for r_i, row in enumerate(rows):
        for c_i, s in enumerate(sorted(row, key=lambda s: s["zone"][0])):
            x1, y1, x2, y2 = (int(round(v)) for v in s["zone"])
            desks.append({"name": f"{chr(65 + r_i)}{c_i + 1}", "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                          "presence": round(s["presence"], 2), "person_height_px": int(s["height"])})
    h, w = ref.shape[:2]
    ref_name = Path(args.out).with_suffix(".ref.jpg").name
    cv2.imwrite(str(Path(args.out).with_name(ref_name)), ref)
    Path(args.out).write_text(json.dumps({"frame_width": w, "frame_height": h, "reference_image": ref_name,
                                          "desks": desks}, indent=1))
    print(f"{len(desks)} seats in {len(rows)} rows -> {args.out}")
    if args.dump_detections:
        Path(args.dump_detections).write_text(json.dumps([{"t": round(t, 2), "boxes": b.round(1).tolist()}
                                                          for t, b in frames_boxes]))
    if args.preview:
        im = ref.copy()
        for d in desks:
            p = np.array(d["polygon"])
            cv2.rectangle(im, tuple(p[0]), tuple(p[2]), (0, 220, 255), 2)
            cv2.putText(im, d["name"], (int(p[0][0]), int(p[0][1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
        cv2.imwrite(args.preview, im)


if __name__ == "__main__":
    main()
