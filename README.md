# Desk Presence Monitor (demo)

Shows, live from an office camera, whether each desk is **occupied** or **empty**, and produces a report
of occupancy per desk over the monitored period.

Inspired by Huseyn Hasanov's [AI Employee Monitoring Using Computer Vision](https://www.linkedin.com/feed/update/urn:li:activity:7525847005479399425/)
post (YOLO + OpenCV, working / absent / phone states). This project narrows the idea to desk presence only and
monitors desks rather than individual people.

## Why it works this way

- **Desks, not people.** The system watches drawn desk zones and asks "is someone sitting here?". It never
  identifies or follows individuals. That makes it simpler and more robust (no ID switches when people cross paths),
  and much easier to defend on privacy grounds.
- **Debounced states.** Raw detections flicker: YOLO misses a frame, or a colleague walks past a desk. Each desk has a
  small state machine, so a desk only becomes PRESENT after 3 s of continuous presence and AWAY after 10 s of
  continuous absence. Transitions are backdated to when the change really happened, so the times stay accurate.
- **Escalation.** AWAY longer than `alert_seconds` (default 5 min) becomes ALERT (red, blinking, logged).

```
 Camera / video file / RTSP
          │
          ▼
 ┌──────────────────┐   every Nth frame
 │ YOLO person det. │──────────────┐
 └──────────────────┘              ▼
                         ┌───────────────────────┐   zones.json
                         │ Desk zone matcher     │◀── (drawn once with
                         │ person box ∩ desk poly│     setup_zones.py)
                         └──────────┬────────────┘
                                    ▼  raw: occupied yes/no per desk
                         ┌───────────────────────┐
                         │ Per-desk state machine│
                         │ PRESENT ⇄ AWAY → ALERT│
                         └──────────┬────────────┘
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          Live overlay +      events.csv          report.html
          side dashboard     (state periods)    (summary for mgmt)
```

## Files

| File | Purpose |
|---|---|
| `setup_zones.py` | Draw desk zones on a camera frame, saves `zones.json` |
| `monitor.py` | Main program: detection, live view, outputs |
| `occupancy.py` | Zone matching + per-desk state machine (no video code, unit-tested) |
| `report.py` | Builds `report.html` from `summary.json` |
| `video_source.py` | Webcam / file / RTSP, `.env` credentials, redaction, live reader with reconnect |
| `stabilizer.py` | Optional camera-motion compensation (`stabilize: true`) |
| `detector.py` | Person detection, full frame or desk-ROI crops (`roi_mode: true`) |
| `store.py` / `api.py` | SQLite event store + FastAPI REST API and live dashboard |
| `tools/benchmark.py` | Speed / agreement of detector setups on a clip |
| `tools/evaluate.py` | Score against hand-labelled ground truth (raw detection + system decisions) |
| `tools/discover_seats.py` | Suggest desk zones automatically from footage (review them before use) |
| `tools/fake_camera.py` | Password-protected fake live camera for testing |
| `config.yaml` | All thresholds and settings |
| `tests/` | Unit tests + a scripted test video generator |

## Run it

**1. Install** (Python 3.10+)
```bash
pip install -r requirements.txt
```
Check: `python -c "import ultralytics, cv2; print('ok')"` prints `ok`. The YOLO model downloads automatically on first run.

**2. Draw the desks** (once per camera)

Fastest: let the tool suggest zones, then review them:
```bash
python tools/discover_seats.py --source office.mp4 --stabilize --out zones.json --preview zones_preview.jpg
```
Open `zones_preview.jpg` and delete any zone that is not a real seat (standing spots, chairs hidden behind monitors,
zones squeezed between neighbours). In our test, reviewing 27 suggestions took overall agreement from 84% to 95%.

Or draw them by hand:
```bash
python setup_zones.py --source office.mp4          # or --source 0 for a webcam, or an rtsp:// URL
```
Click corners around each desk's **chair + front edge** (where a seated body appears in the image), press **N** after
each desk and **S** to save. Rename desks in `zones.json` if you like ("Finance 1", ...).
Check: `zones.json` exists and lists your desks.

**3. Run the monitor**
```bash
python monitor.py --source office.mp4
python monitor.py --source office.mp4 --save demo_out.mp4      # also save the annotated video
python monitor.py --source "rtsp://user:pass@CAMERA_IP:554/stream1"
```
Press **Q** to stop. Check: `output/` contains `events.csv`, `summary.json` and `report.html`.

**4. Tests**
```bash
python -m pytest tests -q                 # state machine unit tests
python tests/make_test_video.py test_data # scripted video with known absences
python monitor.py --source test_data/test_office.mp4 --zones test_data/zones.json --no-show
```

## Tuning

| Symptom | Change in `config.yaml` |
|---|---|
| Seated people sometimes not detected (desk flips to AWAY) | lower `confidence` (0.25), raise `imgsz` (960), use `yolo11s.pt`, or raise `leave_seconds` |
| Desk marked occupied when someone stands next to it | tighten the zone around the chair, raise `min_overlap` (0.4) |
| Too slow on CPU | raise `detect_every` (5), keep `yolo11n.pt`, lower `imgsz` |

| Camera pans, tilts, zooms or gets bumped (zones drift off desks) | set `stabilize: true`: frames are aligned to the reference image saved by `setup_zones.py` |

**Camera placement matters most.** A high, angled view where each chair is visible and people don't hide behind
monitors gives far better results than a low or side-on view.

## Recommended production settings

Measured on a hand-labelled open-plan clip (see `examples/open_plan_demo/`):

| Setup | Raw accuracy | Detection time (2 vCPU, no GPU) |
|---|---|---|
| yolo11s full frame @1280 | 94.6% | ~470 ms |
| **yolo11n desk-ROI @320** (`roi_mode: true`) | **97.0%** | **~160 ms** |

After debounce: 97% state agreement, 0 false alerts. With `detect_every` set to about 1 check per second and `--no-show`,
one camera costs ~0.44 CPU-seconds per second of video on a 2-vCPU VM.

## API

```bash
python monitor.py --source rtsp://... --db desk_monitor.db --camera-id floor2 --no-show
DESK_DB=desk_monitor.db uvicorn api:app --host 0.0.0.0 --port 8000
# http://server:8000/            live dashboard
# http://server:8000/api/desks   current states (also /api/events, /api/summary, /api/health)
```

## Live CCTV cameras

**Credentials never go on the command line or into git.** Copy `.env.example` to `.env` (git-ignored), put the camera URL
there and refer to it by name:

```bash
cp .env.example .env          # then edit: CAM1_URL=rtsp://USER:PASSWORD@IP:554/Streaming/Channels/102
python setup_zones.py --source '${CAM1_URL}'
python monitor.py --source '${CAM1_URL}' --db desk_monitor.db --camera-id floor1 --no-show
```

Anything printed or saved (console, `summary.json`, `report.html`, database) shows URLs as `rtsp://***@host/...`.
Use single quotes so your shell doesn't expand `${CAM1_URL}` itself.

Live sources are handled differently from files:

- **No lag build-up:** a background thread decodes the stream and keeps only the newest frame. In a test with a slow
  consumer, plain OpenCV capture fell 9 s behind live within 20 s, while this stayed at 0 s.
- **Detection on a clock:** `detect_interval_seconds` (default 1 s) instead of every Nth frame.
- **Auto-reconnect:** when the stream drops, it retries with backoff (1, 2, 4 ... 30 s). Desk timers are paused during
  the outage, so a dead camera never shows up as "everyone left".
- **Clean shutdown:** Ctrl+C or SIGTERM (`docker stop`, systemd) still writes the outputs.
- Prefer the camera's **sub-stream** (e.g. 720p): plenty for desk detection and much cheaper to decode.
- For seat discovery, record a few minutes first (`ffmpeg -rtsp_transport tcp -i "$CAM1_URL" -t 300 -c copy sample.mp4`)
  and run `tools/discover_seats.py` on the file.

**Test without a real camera:** `tools/fake_camera.py` serves a video as a password-protected live stream.

```bash
python tools/fake_camera.py --video office.mp4 --password secret --port 8090
# .env:  CAM1_URL=http://admin:secret@127.0.0.1:8090/stream.mjpg
```

## Reproduce the open-plan demo

The footage is not in this repo (third-party stock video). Download it yourself:
[“Open office space”, Mixkit](https://mixkit.co/free-stock-video/open-office-space-914/) (Mixkit Stock Video Free License),
720p version, saved as `office.mp4`.

```bash
# A) 8 front desks: hand-drawn zones, labelled every second -> score it
python monitor.py --source office.mp4 --zones examples/open_plan_demo/zones.json \
    --config examples/open_plan_demo/config_demo.yaml --out-dir output_8 --no-show
python tools/evaluate.py --source office.mp4 --zones examples/open_plan_demo/zones.json \
    --gt examples/open_plan_demo/ground_truth.json --summary output_8/summary.json --stabilize

# B) whole room, 24 desks: zones auto-suggested then reviewed
python monitor.py --source office.mp4 --zones examples/open_plan_demo/zones_all_desks.json \
    --config examples/open_plan_demo/config_all_desks.yaml --out-dir output_24 --save demo_out.mp4
```

The example zone files name a `reference_image` that is not included (it would be a frame of the clip). The monitor then
uses the first frame of the video as the reference, which is the frame the zones were drawn on.

## Licence

This project is licensed under the **GNU AGPL-3.0** (see `LICENSE`), matching its detector dependency,
Ultralytics YOLO (AGPL-3.0). If you use it inside a product or service without releasing your source under AGPL-3.0,
Ultralytics requires its Enterprise licence; alternatively swap `detector.py` for a permissively licensed detector
and re-run `tools/evaluate.py`.

## Before using it on real staff

`privacy_blur` (off by default) can pixelate the whole displayed picture (`frame`) or only detected people (`people`). Continuous monitoring of identifiable employees is personal-data processing in most jurisdictions. Agree the purpose
with management/HR/legal first and inform staff. This demo keeps the footprint small: privacy pixelation is on by
default, nothing is identified, and only per-desk occupancy times are stored.
