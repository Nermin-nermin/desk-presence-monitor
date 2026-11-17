"""SQLite event store shared by monitor.py (writer) and api.py (reader).

Tables
  desk_state  one row per (camera, desk): current state, since when, accumulated present/away seconds
  events      one row per state period (PRESENT / AWAY / ALERT); the current one has ended_at = NULL

WAL mode lets the API read while a monitor writes. Several monitors (one per camera) can share one file.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from occupancy import DeskState, DeskTracker

SCHEMA = """
CREATE TABLE IF NOT EXISTS desk_state (
    camera_id       TEXT NOT NULL,
    desk            TEXT NOT NULL,
    state           TEXT NOT NULL,
    since           TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    present_seconds REAL NOT NULL DEFAULT 0,
    away_seconds    REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (camera_id, desk)
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    desk        TEXT NOT NULL,
    state       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL
);
CREATE INDEX IF NOT EXISTS ix_events_desk_time ON events (camera_id, desk, started_at);
CREATE INDEX IF NOT EXISTS ix_events_state_time ON events (state, started_at);
"""

ISO = "%Y-%m-%dT%H:%M:%S"


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


class EventStore:
    """Mirrors DeskTracker state into SQLite. Call sync() periodically and once more at the end."""

    def __init__(self, path: str | Path, camera_id: str, run_start: datetime):
        self.conn = connect(path)
        self.camera_id = camera_id
        self.run_start = run_start
        self._closed: dict[str, int] = {}        # how many tracker periods are already stored
        self._open: dict[str, int | None] = {}   # row id of the desk's current (open) event

    def _ts(self, seconds: float) -> str:
        return (self.run_start + timedelta(seconds=seconds)).strftime(ISO)

    def sync(self, trackers: list[DeskTracker], now_s: float) -> None:
        cur = self.conn.cursor()
        for t in trackers:
            desk = t.zone.name
            # 1. periods that ended since the last sync
            for p in t.periods[self._closed.get(desk, 0):]:
                if p.state == DeskState.UNKNOWN:
                    continue
                row_id = self._open.get(desk)
                values = (p.state.value, self._ts(p.start), self._ts(p.end), round(p.duration, 1))
                if row_id is not None:
                    cur.execute("UPDATE events SET state=?, started_at=?, ended_at=?, duration_s=? WHERE id=?",
                                (*values, row_id))
                    self._open[desk] = None
                else:
                    cur.execute("INSERT INTO events (camera_id, desk, state, started_at, ended_at, duration_s) "
                                "VALUES (?,?,?,?,?,?)", (self.camera_id, desk, *values))
            self._closed[desk] = len(t.periods)
            # 2. the current, still-open period
            if t.state != DeskState.UNKNOWN and t.state_since is not None and t.state_since < now_s:
                if self._open.get(desk) is None:
                    cur.execute("INSERT INTO events (camera_id, desk, state, started_at) VALUES (?,?,?,?)",
                                (self.camera_id, desk, t.state.value, self._ts(t.state_since)))
                    self._open[desk] = cur.lastrowid
                else:
                    cur.execute("UPDATE events SET state=?, started_at=? WHERE id=?",
                                (t.state.value, self._ts(t.state_since), self._open[desk]))
            # 3. current state snapshot
            since = t.absence_since if t.state in (DeskState.AWAY, DeskState.ALERT) and t.absence_since is not None \
                else (t.state_since or 0.0)
            cur.execute(
                "INSERT INTO desk_state (camera_id, desk, state, since, updated_at, present_seconds, away_seconds) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(camera_id, desk) DO UPDATE SET state=excluded.state, "
                "since=excluded.since, updated_at=excluded.updated_at, present_seconds=excluded.present_seconds, "
                "away_seconds=excluded.away_seconds",
                (self.camera_id, desk, t.state.value, self._ts(since), self._ts(now_s),
                 round(t.totals[DeskState.PRESENT], 1),
                 round(t.totals[DeskState.AWAY] + t.totals[DeskState.ALERT], 1)))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
