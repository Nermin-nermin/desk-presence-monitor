"""Core desk-occupancy logic: zone matching + debounced per-desk state machine.

Kept free of video/YOLO code so it can be unit-tested on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class DeskState(str, Enum):
    UNKNOWN = "UNKNOWN"   # not enough evidence yet (start of the run)
    PRESENT = "PRESENT"
    AWAY = "AWAY"
    ALERT = "ALERT"       # away longer than the alert threshold


@dataclass
class DeskZone:
    """A desk area drawn as a polygon on the camera image."""
    name: str
    polygon: np.ndarray            # (N, 2) int32, image pixel coordinates
    mask: np.ndarray | None = None  # filled binary mask, built lazily

    def build_mask(self, frame_shape: tuple[int, int]) -> None:
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon.astype(np.int32)], 1)
        self.mask = mask

    def overlap_ratio(self, box: tuple[float, float, float, float]) -> float:
        """Fraction of a person's bounding box that lies inside this desk zone."""
        assert self.mask is not None, "call build_mask() first"
        h, w = self.mask.shape
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 0.0
        return float(self.mask[y1:y2, x1:x2].sum()) / area


@dataclass
class StatePeriod:
    """One continuous period a desk spent in a given state (for the event log)."""
    desk: str
    state: DeskState
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DeskTracker:
    """Debounced state machine for one desk.

    Raw detections flicker (missed frames, someone walking past), so the
    state only changes after the new evidence has lasted long enough:
      - becomes PRESENT after `enter_seconds` of continuous occupancy
      - becomes AWAY after `leave_seconds` of continuous emptiness
      - escalates AWAY -> ALERT after `alert_seconds` away
    """
    zone: DeskZone
    enter_seconds: float = 3.0
    leave_seconds: float = 10.0
    alert_seconds: float = 300.0

    state: DeskState = DeskState.UNKNOWN
    state_since: float | None = None      # when the current state began
    _pending: bool | None = None          # raw value we are waiting to confirm
    _pending_since: float | None = None
    absence_since: float | None = None    # start of the current AWAY(+ALERT) absence
    periods: list[StatePeriod] = field(default_factory=list)
    totals: dict[DeskState, float] = field(
        default_factory=lambda: {s: 0.0 for s in DeskState})
    _last_ts: float | None = None

    def update(self, occupied: bool, ts: float) -> DeskState:
        """Feed one raw observation (occupied or not) at time `ts` seconds."""
        if self.state_since is None:
            self.state_since = ts
        if self._last_ts is not None:
            self.totals[self.state] += max(0.0, ts - self._last_ts)
        self._last_ts = ts

        # Track how long the current raw value has been stable.
        if occupied != self._pending:
            self._pending = occupied
            self._pending_since = ts
        stable_for = ts - self._pending_since

        if occupied:
            if self.state != DeskState.PRESENT and stable_for >= self.enter_seconds:
                # Backdate the change to when the person actually sat down.
                self._transition(DeskState.PRESENT, self._pending_since)
        else:
            if self.state in (DeskState.PRESENT, DeskState.UNKNOWN) \
                    and stable_for >= self.leave_seconds:
                self._transition(DeskState.AWAY, self._pending_since)
            if self.state == DeskState.AWAY \
                    and ts - self.state_since >= self.alert_seconds:
                self._transition(DeskState.ALERT, self.state_since + self.alert_seconds)
        return self.state

    def _transition(self, new_state: DeskState, at: float) -> None:
        # Move the backdated slice of time from the old state to the new one
        # so totals match the event log.
        if self._last_ts is not None and at < self._last_ts:
            moved = self._last_ts - max(at, self.state_since)
            self.totals[self.state] -= moved
            self.totals[new_state] += moved
        if self.state_since is not None and at > self.state_since:
            self.periods.append(StatePeriod(self.zone.name, self.state, self.state_since, at))
        if new_state == DeskState.AWAY:
            self.absence_since = at
        elif new_state == DeskState.PRESENT:
            self.absence_since = None
        self.state = new_state
        self.state_since = at

    def skip_gap(self, ts: float) -> None:
        """Resume after a camera outage: don't count the outage as time in any state,
        and restart the evidence timer so stale observations can't trigger a change."""
        self._last_ts = ts
        self._pending = None
        self._pending_since = ts

    def current_duration(self, now: float) -> float:
        """Time in the current state; for AWAY/ALERT, the whole absence so far."""
        if self.state in (DeskState.AWAY, DeskState.ALERT) and self.absence_since is not None:
            return now - self.absence_since
        return 0.0 if self.state_since is None else now - self.state_since

    def close(self, ts: float) -> None:
        """Flush the final open period (call once at the end of the run)."""
        if self._last_ts is not None and ts > self._last_ts:
            self.totals[self.state] += ts - self._last_ts
            self._last_ts = ts
        if self.state_since is not None and ts > self.state_since:
            self.periods.append(StatePeriod(self.zone.name, self.state, self.state_since, ts))
            self.state_since = ts


def zones_occupied(zones: list[DeskZone],
                   person_boxes: list[tuple[float, float, float, float]],
                   min_overlap: float) -> dict[str, bool]:
    """For each desk, is at least `min_overlap` of some person's box inside it?

    Each person is assigned to at most one desk (the one they overlap most),
    so a person sitting between two desks doesn't mark both as occupied.
    """
    result = {z.name: False for z in zones}
    for box in person_boxes:
        best_name, best_ratio = None, 0.0
        for z in zones:
            r = z.overlap_ratio(box)
            if r > best_ratio:
                best_name, best_ratio = z.name, r
        if best_name is not None and best_ratio >= min_overlap:
            result[best_name] = True
    return result


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
