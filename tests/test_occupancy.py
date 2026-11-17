"""Run with:  python -m pytest tests -q   (from the project folder)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occupancy import DeskState, DeskTracker, DeskZone, zones_occupied  # noqa: E402


def make_tracker(**kw):
    z = DeskZone("D1", np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    z.build_mask((200, 200))
    return DeskTracker(z, enter_seconds=3, leave_seconds=10, alert_seconds=60, **kw)


def feed(t, pattern, start=0.0, step=0.5):
    """pattern: list of (occupied, seconds)."""
    ts = start
    for occ, dur in pattern:
        for _ in range(int(dur / step)):
            t.update(occ, ts)
            ts += step
    return ts


def test_present_after_enter_delay_and_backdated():
    t = make_tracker()
    end = feed(t, [(True, 5)])
    assert t.state == DeskState.PRESENT
    assert t.state_since == 0.0          # backdated to when they actually sat down
    t.close(end)
    assert abs(t.totals[DeskState.PRESENT] - 5.0) < 1e-6   # full 0-5 s, incl. final slice


def test_short_gap_does_not_trigger_away():
    t = make_tracker()
    feed(t, [(True, 10), (False, 6), (True, 10)])
    assert t.state == DeskState.PRESENT
    assert all(p.state != DeskState.AWAY for p in t.periods)


def test_walk_by_does_not_mark_present():
    t = make_tracker()
    feed(t, [(False, 15), (True, 2), (False, 10)])
    assert t.state == DeskState.AWAY
    assert all(p.state != DeskState.PRESENT for p in t.periods)


def test_away_then_alert_then_return():
    t = make_tracker()
    end = feed(t, [(True, 10), (False, 80), (True, 5)])
    t.close(end)
    states = [p.state for p in t.periods]
    assert states == [DeskState.PRESENT, DeskState.AWAY, DeskState.ALERT, DeskState.PRESENT]
    away, alert = t.periods[1], t.periods[2]
    assert away.start == 10.0 and abs(away.duration - 60) < 1e-6   # alert after exactly 60s away
    assert abs(alert.end - 90.0) < 1e-6                             # back at t=90 (backdated)
    # totals agree with the event log
    for s in DeskState:
        logged = sum(p.duration for p in t.periods if p.state == s)
        assert abs(t.totals[s] - logged) < 1.0, s


def test_person_assigned_to_single_best_desk():
    a = DeskZone("A", np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    b = DeskZone("B", np.array([[100, 0], [200, 0], [200, 100], [100, 100]]))
    for z in (a, b):
        z.build_mask((100, 200))
    res = zones_occupied([a, b], [(60, 0, 120, 100)], min_overlap=0.25)  # 2/3 in A, 1/3 in B
    assert res == {"A": True, "B": False}


def test_alert_timer_shows_whole_absence():
    t = make_tracker()
    end = feed(t, [(True, 10), (False, 75)])
    assert t.state == DeskState.ALERT
    assert abs(t.current_duration(end) - 75) < 1e-6   # since leaving at t=10, not since ALERT began


def test_outage_is_not_counted_as_absence():
    t = make_tracker()
    feed(t, [(True, 10)])                 # present 0-10 s
    t.skip_gap(310.0)                     # camera down for 5 minutes
    end = feed(t, [(True, 5)], start=310.0)
    t.close(end)
    assert t.state == DeskState.PRESENT
    assert all(p.state == DeskState.PRESENT for p in t.periods)
    assert t.totals[DeskState.PRESENT] < 20   # the 300 s outage is not credited to any state
