"""API + store tests on synthetic tracker data (no video / YOLO needed)."""
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occupancy import DeskTracker, DeskZone  # noqa: E402
from store import EventStore  # noqa: E402


def make_tracker(name):
    z = DeskZone(name, np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
    return DeskTracker(z, enter_seconds=3, leave_seconds=10, alert_seconds=60)


def run(trackers, patterns, store, step=0.5):
    """patterns: per tracker list of (occupied, seconds); all the same total length."""
    total = sum(d for _, d in patterns[0])
    ts, last_sync = 0.0, -1
    while ts < total:
        for t, pat in zip(trackers, patterns):
            acc, occ = 0.0, pat[-1][0]
            for o, d in pat:
                if ts < acc + d:
                    occ = o
                    break
                acc += d
            t.update(occ, ts)
        if ts - last_sync >= 1:
            store.sync(trackers, ts)
            last_sync = ts
        ts += step
    for t in trackers:
        t.close(total)
    store.sync(trackers, total)


def test_store_and_api(tmp_path):
    db = tmp_path / "t.db"
    store = EventStore(db, "cam1", datetime(2026, 11, 16, 9, 0, 0))
    a, b = make_tracker("A1"), make_tracker("A2")
    # A1: at desk 20s, away 90s (-> ALERT after 60s), back 30s.  A2: at desk the whole 140s.
    run([a, b], [[(True, 20), (False, 90), (True, 30)], [(True, 140)]], store)
    store.close()

    os.environ["DESK_DB"] = str(db)
    import api  # noqa: E402
    client = TestClient(api.app)

    assert client.get("/api/health").json()["desks"] == 2

    desks = {d["desk"]: d for d in client.get("/api/desks").json()}
    assert desks["A1"]["state"] == "PRESENT" and desks["A2"]["state"] == "PRESENT"
    assert desks["A2"]["occupancy_pct"] == 100.0

    ev = client.get("/api/events", params={"desk": "A1"}).json()
    states = [e["state"] for e in reversed(ev)]            # API returns newest first
    assert states == ["PRESENT", "AWAY", "ALERT", "PRESENT"]
    away = [e for e in ev if e["state"] == "AWAY"][0]
    assert away["started_at"] == "2026-11-16T09:00:20" and away["duration_s"] == 60.0
    assert ev[0]["ended_at"] == "2026-11-16T09:02:20"      # final period closed at the end of the run

    alerts = client.get("/api/events", params={"state": "ALERT"}).json()
    assert len(alerts) == 1 and alerts[0]["desk"] == "A1"

    summ = {s["desk"]: s for s in client.get("/api/summary").json()}
    assert summ["A1"]["absences"] == 1 and summ["A1"]["alerts"] == 1
    assert summ["A1"]["longest_absence_seconds"] == 90.0
    assert summ["A2"]["absences"] == 0

    assert "Desk presence" in client.get("/").text
    assert client.get("/api/events", params={"state": "BOGUS"}).status_code == 422
