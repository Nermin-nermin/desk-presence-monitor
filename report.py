"""Build a self-contained HTML report from summary.json.

Usage (normally called automatically by monitor.py):
  python report.py output/summary.json output/report.html
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta
from pathlib import Path

from occupancy import format_duration

COLORS = {"PRESENT": "#3fa34d", "AWAY": "#f0a020", "ALERT": "#d64545", "UNKNOWN": "#c9ccd1"}


def _timeline(periods: list[dict], total: float) -> str:
    if total <= 0:
        return ""
    segs = []
    for p in periods:
        width = 100 * (p["end"] - p["start"]) / total
        left = 100 * p["start"] / total
        tip = f'{p["state"]}: {format_duration(p["start"])} - {format_duration(p["end"])}'
        segs.append(f'<div class="seg" title="{tip}" style="left:{left:.3f}%;width:{width:.3f}%;'
                    f'background:{COLORS.get(p["state"], "#999")}"></div>')
    return f'<div class="track">{"".join(segs)}</div>'


def build_report(summary: dict, out_path: str | Path) -> None:
    desks = summary["desks"]
    total = summary["duration_seconds"]
    started = datetime.strptime(summary["started"], "%Y-%m-%d %H:%M:%S")
    ended = started + timedelta(seconds=total)
    avg_occ = sum(d["occupancy_pct"] for d in desks) / len(desks) if desks else 0
    total_alerts = sum(d["alerts"] for d in desks)
    s = summary["settings"]

    rows = "".join(f"""
      <tr>
        <td class="name">{html.escape(d['name'])}</td>
        <td><div class="bar"><div style="width:{d['occupancy_pct']}%"></div></div>{d['occupancy_pct']:.0f}%</td>
        <td>{format_duration(d['present_seconds'])}</td>
        <td>{format_duration(d['away_seconds'])}</td>
        <td>{d['absences']}</td>
        <td>{format_duration(d['longest_absence_seconds'])}</td>
        <td class="{'alert' if d['alerts'] else ''}">{d['alerts']}</td>
      </tr>""" for d in desks)

    timelines = "".join(f"""
      <div class="tl-row"><div class="tl-name">{html.escape(d['name'])}</div>{_timeline(d['periods'], total)}</div>"""
                        for d in desks)
    ticks = "".join(f'<span style="left:{100 * i / 4}%">{(started + timedelta(seconds=total * i / 4)):%H:%M:%S}</span>'
                    for i in range(5))

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Desk Presence Report - {started:%Y-%m-%d}</title>
<style>
  :root {{ --ink:#1d2330; --muted:#6b7280; --line:#e6e8ec; --bg:#f6f7f9; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--ink); background:var(--bg); }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:40px 24px 64px; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); margin-bottom:28px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:28px; }}
  .kpi {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
  .kpi .v {{ font-size:28px; font-weight:650; font-variant-numeric:tabular-nums; }}
  .kpi .l {{ color:var(--muted); font-size:13px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:20px 22px; margin-bottom:22px; }}
  h2 {{ font-size:16px; margin:0 0 14px; }}
  table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
  th {{ text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; padding:8px 10px; border-bottom:1px solid var(--line); }}
  td {{ padding:10px; border-bottom:1px solid var(--line); }}
  td.name {{ font-weight:600; }}
  td.alert {{ color:#d64545; font-weight:700; }}
  .bar {{ display:inline-block; width:90px; height:8px; background:#eef0f3; border-radius:4px; margin-right:8px; vertical-align:middle; overflow:hidden; }}
  .bar div {{ height:100%; background:{COLORS['PRESENT']}; }}
  .tl-row {{ display:flex; align-items:center; gap:12px; margin:8px 0; }}
  .tl-name {{ width:140px; flex:none; font-weight:600; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .track {{ position:relative; flex:1; height:22px; background:#eef0f3; border-radius:5px; overflow:hidden; }}
  .seg {{ position:absolute; top:0; bottom:0; }}
  .ticks {{ position:relative; margin-left:152px; height:18px; font-size:11px; color:var(--muted); }}
  .ticks span {{ position:absolute; transform:translateX(-50%); }}
  .ticks span:first-child {{ transform:none; }} .ticks span:last-child {{ transform:translateX(-100%); }}
  .legend {{ display:flex; gap:18px; font-size:13px; color:var(--muted); margin-top:12px; }}
  .legend i {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
  .note {{ font-size:13px; color:var(--muted); }}
  @media (max-width:700px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
</style></head><body><div class="wrap">
  <h1>Desk Presence Report</h1>
  <div class="sub">{started:%d %b %Y, %H:%M:%S} – {ended:%H:%M:%S} &middot; {format_duration(total)} monitored &middot; source: {html.escape(summary['source'])}</div>

  <div class="kpis">
    <div class="kpi"><div class="v">{len(desks)}</div><div class="l">Desks monitored</div></div>
    <div class="kpi"><div class="v">{avg_occ:.0f}%</div><div class="l">Average desk occupancy</div></div>
    <div class="kpi"><div class="v">{sum(d['absences'] for d in desks)}</div><div class="l">Absences detected</div></div>
    <div class="kpi"><div class="v" style="color:{'#d64545' if total_alerts else 'inherit'}">{total_alerts}</div><div class="l">Long-absence alerts (&gt; {format_duration(s['alert_seconds'])})</div></div>
  </div>

  <div class="card"><h2>Timeline</h2>
    {timelines}
    <div class="ticks">{ticks}</div>
    <div class="legend"><span><i style="background:{COLORS['PRESENT']}"></i>At desk</span><span><i style="background:{COLORS['AWAY']}"></i>Away</span><span><i style="background:{COLORS['ALERT']}"></i>Away &gt; alert limit</span><span><i style="background:{COLORS['UNKNOWN']}"></i>Warming up</span></div>
  </div>

  <div class="card"><h2>Per desk</h2>
    <table><thead><tr><th>Desk</th><th>Occupancy</th><th>At desk</th><th>Away</th><th>Absences</th><th>Longest absence</th><th>Alerts</th></tr></thead>
    <tbody>{rows}</tbody></table>
  </div>

  <p class="note">Method: people are detected with YOLO and matched to drawn desk zones. A desk counts as occupied after
  {s['enter_seconds']}s of continuous presence and as away after {s['leave_seconds']}s of continuous absence, so walk-bys and brief
  detection gaps are ignored. Only desk occupancy is measured. No faces are identified and no individuals are tracked.</p>
</div></body></html>"""
    Path(out_path).write_text(page, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build report.html from a monitor run's summary.json")
    ap.add_argument("summary", help="path to summary.json")
    ap.add_argument("out", help="path of the HTML report to write")
    args = ap.parse_args()
    build_report(json.loads(Path(args.summary).read_text(encoding="utf-8")), args.out)
