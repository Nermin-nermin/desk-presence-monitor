"""Score the system against hand-labelled ground truth.

Two levels:
  1. Raw detection, per desk per labelled second: did the detector see someone at the desk (yes/no)?
     Reported for several detector setups, with speed.
  2. System decisions (after debounce), from a monitor run's summary.json:
     - state agreement per labelled second, where the ground truth applies the same policy as the system
       (empty gaps shorter than leave_seconds count as "still present")
     - absence events: real absences >= leave_seconds that were flagged, missed, and false absences/alerts.

Usage:
  python tools/evaluate.py --source clip.mp4 --zones zones.json --gt ground_truth.json \
      --summary output/summary.json --stabilize --out eval.json
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

SETUPS = [
    ("yolo11s full-frame @1280", "yolo11s.pt", "full", 1280),
    ("yolo11n full-frame @1280", "yolo11n.pt", "full", 1280),
    ("yolo11n full-frame @640", "yolo11n.pt", "full", 640),
    ("yolo11s desk-ROI @320", "yolo11s.pt", "roi", 320),
    ("yolo11n desk-ROI @320", "yolo11n.pt", "roi", 320),
    ("yolo11n desk-ROI @256", "yolo11n.pt", "roi", 256),
]


def frames_at_seconds(source, n_seconds, stabilize):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, first = cap.read()
    stab = Stabilizer(first) if stabilize else None
    out, i, frame = {}, 0, first
    while ok and len(out) < n_seconds:
        aligned = stab.align(frame) if stab else frame   # align every frame so tracking stays smooth
        sec = round(i / fps)
        if abs(i / fps - sec) < 0.5 / fps and sec not in out:
            out[sec] = aligned
        ok, frame = cap.read()
        i += 1
    return first, [out[s] for s in sorted(out)]


def policy_truth(labels, leave_s):
    """Apply the system's policy to the ground truth: empty runs shorter than leave_s count as present."""
    y = np.array(labels).copy()
    i = 0
    while i < len(y):
        if y[i] == 0:
            j = i
            while j < len(y) and y[j] == 0:
                j += 1
            if (j - i) < leave_s and i > 0:     # a short gap after someone was there: debounce ignores it
                y[i:j] = 1
            i = j
        else:
            i += 1
    return y


def absences(seq):
    runs, i = [], 0
    while i < len(seq):
        if seq[i] == 0:
            j = i
            while j < len(seq) and seq[j] == 0:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--zones", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--summary", help="summary.json from a monitor run, to score the system decisions")
    ap.add_argument("--stabilize", action="store_true")
    ap.add_argument("--min-overlap", type=float, default=0.25)
    ap.add_argument("--out", default="eval.json")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())["labels"]
    n = len(next(iter(gt.values())))
    first, frames = frames_at_seconds(args.source, n, args.stabilize)
    zones = load_zones(args.zones, first.shape)
    names = [z.name for z in zones]
    truth = np.array([gt[nm] for nm in names]).T          # (seconds, desks)
    report = {"labelled_desk_seconds": int(truth.size), "empty_desk_seconds": int((truth == 0).sum()), "raw": []}

    print(f"Raw detection vs ground truth ({truth.size} desk-seconds, {(truth == 0).sum()} of them empty)")
    for label, model, mode, imgsz in SETUPS:
        det = PersonDetector(model, 0.3, imgsz, roi_zones=zones if mode == "roi" else None,
                             roi_imgsz=imgsz, frame_shape=first.shape)
        det.detect(frames[0])
        pred, times = [], []
        for f in frames:
            t = time.perf_counter()
            boxes = det.detect(f)
            times.append(time.perf_counter() - t)
            o = zones_occupied(zones, boxes, args.min_overlap)
            pred.append([int(o[nm]) for nm in names])
        pred = np.array(pred)
        acc = float((pred == truth).mean())
        # "empty" is the event we care about: precision/recall of detecting an empty desk
        tp = int(((pred == 0) & (truth == 0)).sum())
        fp = int(((pred == 0) & (truth == 1)).sum())
        fn = int(((pred == 1) & (truth == 0)).sum())
        row = {"setup": label, "accuracy": round(100 * acc, 1),
               "empty_precision": round(100 * tp / (tp + fp), 1) if tp + fp else None,
               "empty_recall": round(100 * tp / (tp + fn), 1) if tp + fn else None,
               "ms_per_frame": round(1000 * float(np.mean(times)), 1),
               "errors": [{"desk": names[d], "second": int(s), "truth": int(truth[s, d]), "pred": int(pred[s, d])}
                          for s, d in zip(*np.where(pred != truth))]}
        report["raw"].append(row)
        print(f"  {label:26s} accuracy {row['accuracy']:5.1f}%  empty P/R {row['empty_precision']}/{row['empty_recall']}"
              f"  {row['ms_per_frame']:6.1f} ms/frame")

    if args.summary:
        summ = json.loads(Path(args.summary).read_text())
        leave = summ["settings"]["leave_seconds"]
        alert = summ["settings"]["alert_seconds"]
        rows, flagged_ok, missed, false_abs, false_alerts, alerts_ok, timing = [], 0, [], [], [], 0, []
        state_agree = []
        for d in summ["desks"]:
            y = policy_truth(gt[d["name"]], leave)
            def state_at(t):
                for p in d["periods"]:
                    if p["start"] <= t < p["end"]:
                        return p["state"]
                return d["periods"][-1]["state"]
            sys_seq = [0 if state_at(s + 0.01) in ("AWAY", "ALERT") else (1 if state_at(s + 0.01) == "PRESENT" else -1)
                       for s in range(n)]
            state_agree += [int(a == b) for a, b in zip(sys_seq, y) if a != -1]
            sys_abs = [(p["start"], p["end"]) for p in d["periods"] if p["state"] in ("AWAY",)]
            sys_alerts = [p for p in d["periods"] if p["state"] == "ALERT"]
            real = [(a, b) for a, b in absences(list(y)) if (b - a) >= leave or a == 0]
            for a, b in real:
                hit = [s for s in sys_abs if s[0] < b + 0.5 and s[1] > a - 0.5]
                if hit:
                    flagged_ok += 1
                    if a > 0:
                        timing.append(abs(hit[0][0] - (a - 0.5)))   # truth change lies between a-1 and a
                else:
                    missed.append((d["name"], a, b))
                if (b - a) >= alert:
                    alerts_ok += int(bool([p for p in sys_alerts if p["start"] < b + 0.5 and p["end"] > a]))
            for s in sys_abs:
                if not [r for r in real if s[0] < r[1] + 0.5 and s[1] > r[0] - 0.5]:
                    false_abs.append((d["name"], s))
            for p in sys_alerts:
                if not [r for r in real if p["start"] < r[1] + 0.5 and p["end"] > r[0] - 0.5]:
                    false_alerts.append((d["name"], p["start"]))
        real_alerts = sum(1 for d in summ["desks"]
                          for a, b in absences(list(policy_truth(gt[d["name"]], leave))) if (b - a) >= alert)
        report["system"] = {
            "state_agreement_pct": round(100 * float(np.mean(state_agree)), 1),
            "real_absences": flagged_ok + len(missed), "absences_flagged": flagged_ok, "absences_missed": missed,
            "false_absences": false_abs, "real_alerts": real_alerts, "alerts_correct": alerts_ok,
            "false_alerts": false_alerts,
            "mean_change_timing_error_s": round(float(np.mean(timing)), 2) if timing else None,
            "settings": summ["settings"],
        }
        s = report["system"]
        print(f"System (debounced): state agreement {s['state_agreement_pct']}%  | absences flagged "
              f"{s['absences_flagged']}/{s['real_absences']}  false absences {len(s['false_absences'])}  | alerts "
              f"{s['alerts_correct']}/{s['real_alerts']}  false alerts {len(s['false_alerts'])}  | timing error "
              f"{s['mean_change_timing_error_s']} s")
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
