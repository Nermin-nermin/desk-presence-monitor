"""REST API + live dashboard over the SQLite store written by monitor.py.

Run:
  python monitor.py --source rtsp://... --db desk_monitor.db --camera-id floor2 --no-show
  DESK_DB=desk_monitor.db uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints
  GET /api/health                       store reachable, number of desks, last update
  GET /api/desks[?camera_id=]           current state of every desk
  GET /api/events[?camera_id=&desk=&state=&since=&until=&limit=]   state periods, newest first
  GET /api/summary[?camera_id=]         per-desk totals: occupancy %, absences, longest absence, alerts
  GET /                                 live dashboard (polls /api/desks and /api/events)
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from store import ISO, connect

app = FastAPI(title="Desk Presence API", version="1.0")


def db():
    path = os.environ.get("DESK_DB", "desk_monitor.db")
    if not Path(path).exists():
        raise HTTPException(503, f"store not found: {path} (start monitor.py with --db)")
    return connect(path)


def _where(filters: dict) -> tuple[str, list]:
    clauses, params = [], []
    for col, op, val in filters.values():
        if val is not None:
            clauses.append(f"{col} {op} ?")
            params.append(val)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


@app.get("/api/health")
def health():
    conn = db()
    row = conn.execute("SELECT COUNT(*) AS desks, MAX(updated_at) AS last_update FROM desk_state").fetchone()
    return {"status": "ok", "desks": row["desks"], "last_update": row["last_update"]}


@app.get("/api/desks")
def desks(camera_id: str | None = None):
    conn = db()
    where, params = _where({"c": ("camera_id", "=", camera_id)})
    rows = conn.execute(f"SELECT * FROM desk_state{where} ORDER BY camera_id, desk", params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["current_seconds"] = (datetime.strptime(r["updated_at"], ISO) - datetime.strptime(r["since"], ISO)).total_seconds()
        known = r["present_seconds"] + r["away_seconds"]
        d["occupancy_pct"] = round(100 * r["present_seconds"] / known, 1) if known else None
        out.append(d)
    return out


@app.get("/api/events")
def events(camera_id: str | None = None, desk: str | None = None,
           state: str | None = Query(None, pattern="^(PRESENT|AWAY|ALERT)$"),
           since: str | None = None, until: str | None = None, limit: int = Query(200, le=5000)):
    conn = db()
    where, params = _where({"c": ("camera_id", "=", camera_id), "d": ("desk", "=", desk),
                            "s": ("state", "=", state), "a": ("started_at", ">=", since),
                            "b": ("started_at", "<=", until)})
    rows = conn.execute(f"SELECT * FROM events{where} ORDER BY started_at DESC, id DESC LIMIT ?",
                        [*params, limit]).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/summary")
def summary(camera_id: str | None = None):
    conn = db()
    where, params = _where({"c": ("camera_id", "=", camera_id)})
    states = {(r["camera_id"], r["desk"]): dict(r) for r in conn.execute(f"SELECT * FROM desk_state{where}", params)}
    out = []
    for (cam, desk), s in sorted(states.items()):
        ev = conn.execute("SELECT state, started_at, ended_at, duration_s FROM events WHERE camera_id=? AND desk=? "
                          "ORDER BY started_at, id", (cam, desk)).fetchall()
        # an absence = consecutive AWAY/ALERT periods
        absences, cur = [], None
        for e in ev:
            end = e["ended_at"] or s["updated_at"]
            if e["state"] in ("AWAY", "ALERT"):
                cur = [cur[0] if cur else e["started_at"], end]
            elif cur:
                absences.append(cur)
                cur = None
        if cur:
            absences.append(cur)
        dur = [(datetime.strptime(b, ISO) - datetime.strptime(a, ISO)).total_seconds() for a, b in absences]
        known = s["present_seconds"] + s["away_seconds"]
        out.append({"camera_id": cam, "desk": desk, "state": s["state"],
                    "occupancy_pct": round(100 * s["present_seconds"] / known, 1) if known else None,
                    "present_seconds": s["present_seconds"], "away_seconds": s["away_seconds"],
                    "absences": len(absences), "longest_absence_seconds": max(dur, default=0),
                    "alerts": sum(1 for e in ev if e["state"] == "ALERT")})
    return out


DASHBOARD = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Desk Presence – Live</title><style>
body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f6f8;color:#14181f}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px}h1{font-size:22px;margin:0 0 4px}.sub{color:#667085;font-size:13px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:22px}
.desk{background:#fff;border:1px solid #e4e7ec;border-left:6px solid #c9ccd1;border-radius:10px;padding:12px 14px}
.desk b{display:block;font-size:16px}.desk span{font-size:12.5px;color:#667085;font-variant-numeric:tabular-nums}
.PRESENT{border-left-color:#3fa34d}.AWAY{border-left-color:#f0a020}.ALERT{border-left-color:#d64545;background:#fdf1f1}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e4e7ec;border-radius:10px;overflow:hidden;font-size:14px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #eef0f3}th{font-size:11px;text-transform:uppercase;color:#667085}
.pill{font-size:11px;font-weight:700;color:#fff;border-radius:5px;padding:2px 7px}
</style></head><body><div class="wrap"><h1>Desk presence – live</h1><div class="sub" id="sub">connecting…</div>
<div class="grid" id="desks"></div><table><thead><tr><th>Time</th><th>Desk</th><th>State</th><th>Duration</th></tr></thead><tbody id="ev"></tbody></table></div>
<script>
const C={PRESENT:'#3fa34d',AWAY:'#f0a020',ALERT:'#d64545'};
const fmt=s=>{s=Math.round(s||0);const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return (h?h+':':'')+String(m).padStart(2,'0')+':'+String(x).padStart(2,'0')};
async function tick(){try{
 const d=await (await fetch('api/desks')).json(), e=await (await fetch('api/events?limit=25')).json();
 const here=d.filter(x=>x.state==='PRESENT').length, al=d.filter(x=>x.state==='ALERT').length;
 document.getElementById('sub').textContent=`At desk ${here}/${d.length}`+(al?` · ${al} alert(s)`:'')+` · updated ${d.length?d[0].updated_at.replace('T',' '):'-'}`;
 document.getElementById('desks').innerHTML=d.map(x=>`<div class="desk ${x.state}"><b>${x.desk}</b><span>${x.state} ${fmt(x.current_seconds)} · ${x.occupancy_pct??'–'}% at desk</span></div>`).join('');
 document.getElementById('ev').innerHTML=e.map(x=>`<tr><td>${x.started_at.replace('T',' ')}</td><td>${x.desk}</td><td><span class="pill" style="background:${C[x.state]}">${x.state}</span></td><td>${x.ended_at?fmt(x.duration_s):'ongoing'}</td></tr>`).join('');
}catch(err){document.getElementById('sub').textContent='API not reachable: '+err}}
tick();setInterval(tick,2000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD
