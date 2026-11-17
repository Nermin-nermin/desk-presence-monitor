"""Desk presence monitor - live view + event log + HTML report.

Usage:
  python monitor.py --source office.mp4                  # video file
  python monitor.py --source 0                           # webcam
  python monitor.py --source "rtsp://user:pass@ip/stream" # IP camera
  python monitor.py --source office.mp4 --save out.mp4 --no-show

Press Q in the window to stop. Results go to ./output/ (events.csv, summary.json, report.html).
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

from occupancy import DeskState, DeskTracker, DeskZone, format_duration, zones_occupied
from detector import PersonDetector
from report import build_report
from stabilizer import Stabilizer
from store import EventStore
from video_source import LiveReader, is_live, load_env_file, open_source, redact, resolve_source

PANEL_W = 360
STATE_COLORS = {  # BGR
    DeskState.UNKNOWN: (150, 150, 150),
    DeskState.PRESENT: (80, 190, 60),
    DeskState.AWAY: (0, 165, 255),
    DeskState.ALERT: (40, 40, 230),
}


def load_zones(path: str, frame_shape: tuple[int, int]) -> list[DeskZone]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    h, w = frame_shape[:2]
    sx = w / data.get("frame_width", w)
    sy = h / data.get("frame_height", h)
    zones = []
    for d in data["desks"]:
        poly = (np.array(d["polygon"], np.float32) * [sx, sy]).astype(np.int32)
        z = DeskZone(d["name"], poly)
        z.build_mask(frame_shape)
        zones.append(z)
    return zones


def pixelate(frame: np.ndarray, box, blocks: int = 12) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def privacy_mode(cfg) -> str:
    """'off' (default), 'frame' (whole picture) or 'people' (detected people only). Booleans kept for old configs."""
    v = cfg.get("privacy_blur", "off")
    if v is True:
        return "frame"
    if v is False:
        return "off"
    return str(v)


def draw_frame(frame, trackers, boxes, cfg, now_s, clock: datetime, blink: bool):
    h = frame.shape[0]
    k = max(1.0, h / 720)          # scale text/lines so labels stay readable on 1080p+ footage
    lw = max(1, round(2 * k))
    font = cv2.FONT_HERSHEY_SIMPLEX

    mode = privacy_mode(cfg)
    if mode == "frame":
        # Anonymise every pixel, so privacy never depends on the detector having found someone.
        block = max(4, frame.shape[1] // 160)
        small = cv2.resize(frame, (frame.shape[1] // block, h // block), interpolation=cv2.INTER_AREA)
        frame[:] = cv2.resize(small, (frame.shape[1], h), interpolation=cv2.INTER_NEAREST)
    elif mode == "people":
        for b in boxes:
            pixelate(frame, b)
    if cfg.get("show_person_boxes", False):
        for b in boxes:
            x1, y1, x2, y2 = (int(v) for v in b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)

    overlay = frame.copy()
    for t in trackers:
        color = STATE_COLORS[t.state]
        if t.state == DeskState.ALERT and not blink:
            color = (20, 20, 120)
        cv2.fillPoly(overlay, [t.zone.polygon], color)
    frame[:] = cv2.addWeighted(overlay, 0.32, frame, 0.68, 0)

    many = len(trackers) > 12   # crowded view: compact labels (name, plus timer only when away)
    for t in trackers:
        color = STATE_COLORS[t.state]
        cv2.polylines(frame, [t.zone.polygon], True, color, lw)
        x, y = int(t.zone.polygon[:, 0].min()), int(t.zone.polygon[:, 1].min())
        away = t.state in (DeskState.AWAY, DeskState.ALERT)
        if many:
            label = t.zone.name + (f" {format_duration(t.current_duration(now_s))}" if away else "")
            fs = 0.42 * k
        else:
            label = f"{t.zone.name}: {t.state.value}" + (f" {format_duration(t.current_duration(now_s))}" if away else "")
            fs = 0.55 * k
        pad = int(4 * k)
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        y = max(th + 2 * pad, y)
        cv2.rectangle(frame, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), color, -1)
        cv2.putText(frame, label, (x + pad, y - pad), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    # Side dashboard
    pw = int(PANEL_W * k)
    panel = np.full((h, pw, 3), (32, 28, 24), np.uint8)

    def put(txt, x, y, s, c=(235, 235, 235), bold=False):
        cv2.putText(panel, txt, (int(x * k), int(y * k)), font, s * k, c, (2 if bold else 1) * max(1, round(k * 0.8)), cv2.LINE_AA)

    put("DESK PRESENCE", 16, 34, 0.75, (255, 255, 255), True)
    put(clock.strftime("%Y-%m-%d  %H:%M:%S"), 16, 60, 0.5, (180, 180, 180))
    present = sum(t.state == DeskState.PRESENT for t in trackers)
    away = [t for t in trackers if t.state in (DeskState.AWAY, DeskState.ALERT)]
    alerts = sum(t.state == DeskState.ALERT for t in trackers)
    put(f"At desk: {present}/{len(trackers)}", 16, 96, 0.65, (120, 220, 120), True)
    put(f"Away: {len(away)}   Alerts: {alerts}", 16, 122, 0.55, (90, 90, 255) if alerts else (0, 165, 255))

    if not many:
        y = 160
        for t in trackers:
            if y * k > h - 20:
                break
            color = STATE_COLORS[t.state]
            cv2.circle(panel, (int(24 * k), int((y - 6) * k)), int(7 * k), color, -1)
            put(t.zone.name[:20], 42, y, 0.62, (245, 245, 245))
            total = sum(t.totals.values()) or 1
            pct = 100 * t.totals[DeskState.PRESENT] / total
            put(f"{t.state.value}  {format_duration(t.current_duration(now_s))}   |   {pct:.0f}% at desk", 42, y + 24, 0.5, color)
            y += 60
        return np.hstack([frame, panel])

    # Crowded view: "needs attention" list, then a compact grid of every desk
    y = 160
    put("NEEDS ATTENTION", 16, y, 0.45, (170, 170, 170))
    y += 26
    if not away:
        put("none", 16, y, 0.5, (120, 220, 120))
        y += 26
    for t in sorted(away, key=lambda t: -t.current_duration(now_s))[:6]:
        color = STATE_COLORS[t.state]
        cv2.circle(panel, (int(22 * k), int((y - 5) * k)), int(6 * k), color, -1)
        put(f"{t.zone.name:<5} {t.state.value:<6} {format_duration(t.current_duration(now_s))}", 36, y, 0.5, color)
        y += 24
    y += 14
    put("ALL DESKS", 16, y, 0.45, (170, 170, 170))
    y += 12
    cols, cw, ch = 5, (PANEL_W - 32) / 5, 30
    for i, t in enumerate(trackers):
        cx, cy = 16 + (i % cols) * cw, y + (i // cols) * ch
        if (cy + ch) * k > h:
            break
        cv2.rectangle(panel, (int(cx * k), int(cy * k)), (int((cx + cw - 5) * k), int((cy + ch - 6) * k)), STATE_COLORS[t.state], -1)
        put(t.zone.name, cx + 7, cy + 17, 0.45, (255, 255, 255))
    return np.hstack([frame, panel])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, webcam index (0), or '${CAM1_URL}' from .env")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--save", help="also write the annotated video to this .mp4 path")
    ap.add_argument("--no-show", action="store_true", help="run without a window (servers / batch)")
    ap.add_argument("--db", help="write live desk states and events to this SQLite file (read by api.py)")
    ap.add_argument("--camera-id", default="cam1", help="camera name stored with each event (for multi-camera setups)")
    ap.add_argument("--env-file", default=".env", help="file with camera URLs/passwords, e.g. CAM1_URL=rtsp://user:pass@ip/...")
    ap.add_argument("--max-seconds", type=float, help="stop after this many seconds (live sources; for tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    load_env_file(args.env_file)
    source = resolve_source(args.source)
    live = is_live(source)
    shown_source = redact(args.source if "${" in args.source else source)

    # Graceful stop on `docker stop` / systemd (SIGTERM): same path as Ctrl+C, so outputs are written.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    reader, cap = None, None
    if live:
        reader = LiveReader(source)
        frame = reader.read(timeout=30)
        if frame is None:
            reader.stop()
            raise SystemExit(f"No frames from {redact(source)} within 30 s (check URL, network and credentials).")
        fps = reader.fps
    else:
        cap = open_source(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Could not read the first frame.")

    zones = load_zones(args.zones, frame.shape)
    stabilizer = None
    if cfg.get("stabilize"):
        # Align every frame to the image the zones were drawn on (saved by setup_zones.py),
        # or to the first frame if that image isn't available.
        ref_path = json.loads(Path(args.zones).read_text(encoding="utf-8")).get("reference_image")
        ref_path = (Path(args.zones).parent / ref_path) if ref_path else None
        ref = cv2.imread(str(ref_path)) if ref_path and ref_path.exists() else frame
        stabilizer = Stabilizer(cv2.resize(ref, (frame.shape[1], frame.shape[0])),
                                every=int(cfg.get("stabilize_every", 5)))
    trackers = [DeskTracker(z, cfg["enter_seconds"], cfg["leave_seconds"], cfg["alert_seconds"]) for z in zones]
    detector = PersonDetector(cfg["model"], cfg["confidence"], cfg["imgsz"],
                              roi_zones=zones if cfg.get("roi_mode") else None,
                              roi_imgsz=int(cfg.get("roi_imgsz", 320)), frame_shape=frame.shape)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    run_start = datetime.now()
    t0 = time.monotonic()
    frame_idx, boxes = 0, []
    store = EventStore(args.db, args.camera_id, run_start) if args.db else None
    headless = args.no_show and not args.save
    last_sync = -1.0
    interval = float(cfg.get("detect_interval_seconds", 1.0))   # live sources: detect on a clock
    last_detect = -1e9
    outage_since = None
    print(f"Monitoring {len(zones)} desk(s) from {shown_source} ({'live' if live else 'file'}). Press Q to stop.")

    def run_detection(img, now_s):
        nonlocal boxes, last_sync
        boxes = detector.detect(img)
        raw = zones_occupied(zones, boxes, cfg["min_overlap"])
        for t in trackers:
            before = t.state
            after = t.update(raw[t.zone.name], now_s)
            if after != before and after == DeskState.ALERT:
                print(f"[ALERT] {t.zone.name} away for over {format_duration(cfg['alert_seconds'])}"
                      f" at {(run_start + timedelta(seconds=now_s)):%H:%M:%S}", flush=True)
        if store and now_s - last_sync >= 1.0:
            store.sync(trackers, now_s)
            last_sync = now_s

    try:
        while True:
            if live:
                now_s = time.monotonic() - t0
                if args.max_seconds and now_s >= args.max_seconds:
                    break
                due = now_s - last_detect >= interval
                if headless and not due:
                    time.sleep(min(0.05, interval - (now_s - last_detect)))
                    continue
                new = reader.read(timeout=5.0)
                now_s = time.monotonic() - t0
                if new is None:
                    # No frames: the camera is down. Freeze the desk timers - an outage is not an absence.
                    if outage_since is None:
                        outage_since = now_s
                        print(f"[stream] no frames since {(run_start + timedelta(seconds=now_s)):%H:%M:%S};"
                              f" desk timers paused", flush=True)
                    continue
                if outage_since is not None:
                    print(f"[stream] frames back after {now_s - outage_since:.0f} s outage", flush=True)
                    for t in trackers:
                        t.skip_gap(now_s)
                    outage_since = None
                frame = new
                if stabilizer is not None:
                    frame = stabilizer.align(frame)
                if due:
                    run_detection(frame, now_s)
                    last_detect = now_s
            else:
                now_s = frame_idx / fps
                if headless and frame_idx % cfg["detect_every"] != 0:
                    # Nothing to show or save: skip alignment/drawing on frames we don't detect on.
                    if not cap.grab():
                        break
                    frame_idx += 1
                    continue
                if frame_idx > 0:          # frame 0 was read before the loop
                    ok, frame = cap.read()
                    if not ok:
                        break
                if stabilizer is not None:
                    frame = stabilizer.align(frame)
                if frame_idx % cfg["detect_every"] == 0:
                    run_detection(frame, now_s)
                frame_idx += 1

            if headless:
                continue
            clock = run_start + timedelta(seconds=now_s)
            view = draw_frame(frame.copy(), trackers, boxes, cfg, now_s, clock, blink=int(now_s * 2) % 2 == 0)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)
            if not args.no_show:
                scale = cfg["display_width"] / view.shape[1]
                cv2.imshow("Desk presence monitor", cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        end_s = (time.monotonic() - t0) if live else frame_idx / fps
        if reader:
            reader.stop()
        if cap:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    for t in trackers:
        t.close(end_s)
    if store:
        store.sync(trackers, end_s)
        store.close()
    write_outputs(trackers, run_start, end_s, out_dir, args, cfg, shown_source)


def write_outputs(trackers, run_start: datetime, end_s: float, out_dir: Path, args, cfg, shown_source: str) -> None:
    ts = lambda s: (run_start + timedelta(seconds=s)).strftime("%Y-%m-%d %H:%M:%S")

    with open(out_dir / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["desk", "state", "start", "end", "duration_seconds"])
        for t in trackers:
            for p in t.periods:
                w.writerow([p.desk, p.state.value, ts(p.start), ts(p.end), round(p.duration, 1)])

    desks = []
    for t in trackers:
        away_periods = [p for p in t.periods if p.state in (DeskState.AWAY, DeskState.ALERT)]
        # Merge AWAY + following ALERT into single absences for the stats.
        absences, cur = [], None
        for p in t.periods:
            if p.state in (DeskState.AWAY, DeskState.ALERT):
                cur = [p.start, p.end] if cur is None else [cur[0], p.end]
            elif cur is not None:
                absences.append(cur)
                cur = None
        if cur is not None:
            absences.append(cur)
        known = t.totals[DeskState.PRESENT] + t.totals[DeskState.AWAY] + t.totals[DeskState.ALERT]
        desks.append({
            "name": t.zone.name,
            "present_seconds": round(t.totals[DeskState.PRESENT], 1),
            "away_seconds": round(t.totals[DeskState.AWAY] + t.totals[DeskState.ALERT], 1),
            "occupancy_pct": round(100 * t.totals[DeskState.PRESENT] / known, 1) if known else 0.0,
            "absences": len(absences),
            "longest_absence_seconds": round(max((b - a for a, b in absences), default=0), 1),
            "alerts": sum(p.state == DeskState.ALERT for p in away_periods),
            "periods": [{"state": p.state.value, "start": round(p.start, 2), "end": round(p.end, 2)}
                        for p in t.periods],
        })

    summary = {
        "source": shown_source,
        "started": run_start.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(end_s, 1),
        "settings": {k: cfg[k] for k in ("enter_seconds", "leave_seconds", "alert_seconds", "min_overlap")},
        "desks": desks,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_report(summary, out_dir / "report.html")
    print(f"Done. Wrote {out_dir/'events.csv'}, {out_dir/'summary.json'}, {out_dir/'report.html'}")


if __name__ == "__main__":
    main()
