"""Coach-facing report: heatmaps, possession %, shot map — one HTML page.

Not a data dump: a single summary page a grassroots coach can read in a
minute. Honest about coverage ("ball tracked for N% of match time").
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibrate import PITCH_LENGTH, PITCH_WIDTH


def _draw_pitch(ax):
    L, W = PITCH_LENGTH, W_
    ax.set_xlim(-L / 2 - 2, L / 2 + 2)
    ax.set_ylim(-W / 2 - 2, W / 2 + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    c = "#3a7d44"
    ax.add_patch(plt.Rectangle((-L / 2, -W / 2), L, W, fc="#4c9a52", ec="white"))
    ax.plot([-L / 2, L / 2], [0, 0], color="white", lw=1)
    ax.add_patch(plt.Circle((0, 0), 9.15, fc="none", ec="white"))
    for gx in (-L / 2, L / 2):
        s = 1 if gx < 0 else -1
        ax.add_patch(plt.Rectangle((gx, -20.16), s * 16.5, 40.32,
                                   fc="none", ec="white"))
        ax.add_patch(plt.Rectangle((gx, -9.16), s * 5.5, 18.32,
                                   fc="none", ec="white"))


W_ = PITCH_WIDTH


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def heatmap_fig(points, title):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    _draw_pitch(ax)
    if points:
        xs, ys = zip(*points)
        ax.hist2d(xs, ys, bins=[21, 14],
                  range=[[-PITCH_LENGTH / 2, PITCH_LENGTH / 2],
                         [-PITCH_WIDTH / 2, PITCH_WIDTH / 2]],
                  cmap="YlOrRd", alpha=0.85)
    ax.set_title(title, fontsize=11)
    return fig


def shotmap_fig(shots):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    _draw_pitch(ax)
    style = {"goal": ("gold", "*", 220, "Goal"),
             "on_target": ("deepskyblue", "o", 120, "On target"),
             "off_target": ("lightgrey", "x", 90, "Off target")}
    seen = set()
    for s in shots:
        col, mk, sz, lab = style[s["outcome"]]
        ax.scatter([s["x"]], [s["y"]], c=col, marker=mk, s=sz,
                   edgecolors="black", linewidths=0.7,
                   label=lab if lab not in seen else None)
        seen.add(lab)
    if seen:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                  fontsize=9)
    ax.set_title(f"Shot map ({len(shots)} shots)", fontsize=11)
    return fig


def possession_fig(pct):
    fig, ax = plt.subplots(figsize=(5, 1.6))
    ax.barh(["Team A"], [pct["A"]], color="#d94f3d", height=0.45)
    ax.barh(["Team B"], [pct["B"]], color="#3d6fd9", height=0.45,
            left=[pct["A"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Possession %")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    return fig


def shape_fig(shape):
    """Average positions per track — the team's shape."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    _draw_pitch(ax)
    cols = {"A": "#d94f3d", "B": "#3d6fd9"}
    for team in ("A", "B"):
        pts = shape["positions"][team]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=cols[team], s=130, edgecolors="white",
                       linewidths=1.2, label=f"Team {team}", zorder=3)
        cx, cy = shape["centroids"][team]
        ax.scatter([cx], [cy], c=cols[team], s=320, marker="X",
                   edgecolors="black", linewidths=1, zorder=4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              fontsize=9)
    ax.set_title("Average shape (X = team centroid)", fontsize=11)
    return fig


def build_report(summary, out_path, title="Match Report", clips=None):
    poss = summary["possession_pct"]
    shots = summary["shots"]
    passes = summary.get("passes", {"completed": {"A": 0, "B": 0},
                                    "turnovers": {"A": 0, "B": 0}, "list": []})
    dist = summary.get("distance_m", {"A": 0.0, "B": 0.0})
    hm_a = _fig_to_b64(heatmap_fig(summary["heatmaps"]["A"],
                                 "Team A — where they played"))
    hm_b = _fig_to_b64(heatmap_fig(summary["heatmaps"]["B"],
                                   "Team B — where they played"))
    sm = _fig_to_b64(shotmap_fig(shots))
    pb = _fig_to_b64(possession_fig(poss))
    sh = _fig_to_b64(shape_fig(summary["shape"])) if summary.get("shape") else ""
    rows = "".join(
        f"<tr><td>{s['t']:.1f}s</td><td>{s['x']:.0f}m, {s['y']:.0f}m</td>"
        f"<td>{s['outcome'].replace('_', ' ')}</td>"
        f"<td>{s['speed']:.0f} m/s</td><td>{s.get('xg', 0):.2f}</td></tr>"
        for s in shots)
    txg = summary.get("total_xg", {"A": 0, "B": 0})
    seq = summary.get("sequences", {})
    throw_ins = summary.get("throw_ins", [])
    cgk = summary.get("corners_goal_kicks", [])
    n_corners = sum(1 for e in cgk if e["type"] == "corner")
    n_gk = sum(1 for e in cgk if e["type"] == "goal_kick")
    clip_html = ""
    if clips:
        items = "".join(
            f'<li><a href="{c["name"]}">{c["outcome"].replace("_", " ")} '
            f'at {c["t"]:.0f}s</a></li>' for c in clips)
        clip_html = (f'<div class="card"><h2>🎬 Highlight clips</h2>'
                     f'<ul>{items}</ul></div>')
    goals = summary.get("goals", [])
    tackles = summary.get("tackles", [])
    oop = summary.get("out_of_play", [])
    goal_rows = "".join(
        f"<tr><td>{g['t']:.1f}s</td><td>{g['x']:.0f}m, {g['y']:.0f}m</td>"
        f"<td>{g['goal']} goal</td></tr>" for g in goals)
    tackle_rows = "".join(
        f"<tr><td>{t['t']:.1f}s</td><td>Team {t['team']}</td>"
        f"<td>{t['x']:.0f}m, {t['y']:.0f}m</td></tr>" for t in tackles[:10])
    events_card = f"""<div class="card"><h2>Goals — {len(goals)}</h2>
<table><tr><th>Time</th><th>Where</th><th></th></tr>
{goal_rows or '<tr><td colspan="3">No goals detected.</td></tr>'}</table></div>
<div class="card"><h2>Tackles — {len(tackles)}</h2>
<table><tr><th>Time</th><th>By</th><th>Where</th></tr>
{tackle_rows or '<tr><td colspan="3">No tackles detected.</td></tr>'}</table>
<p class="note">Ball out of play {len(oop)} times.</p></div>"""
    html = f"""<!doctype html><html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;
padding:16px;background:#faf6ed;color:#2b2620}}h1{{font-size:1.3rem}}
.card{{background:#fff;border:1px solid #e8dfcc;border-radius:12px;
padding:14px;margin:12px 0}}img{{max-width:100%}}table{{width:100%;
border-collapse:collapse;font-size:.85rem}}td,th{{padding:6px;
border-bottom:1px solid #e8dfcc;text-align:left}}.note{{font-size:.8rem;
color:#6b6156}}.stat{{display:inline-block;margin:0 18px 6px 0}}
.stat b{{font-size:1.25rem}}</style></head><body>
<h1>⚽ {title}</h1>
<div class="card"><h2>Possession</h2><img src="data:image/png;base64,{pb}">
<p>Team A {poss['A']:.0f}% · Team B {poss['B']:.0f}% ·
based on {summary['possession_frames']} tracked frames.</p></div>
<div class="card"><h2>Key numbers</h2>
<span class="stat"><b>{passes['completed']['A'] + passes['completed']['B']}</b><br>completed passes</span>
<span class="stat"><b>{passes['turnovers']['A'] + passes['turnovers']['B']}</b><br>turnovers</span>
<span class="stat"><b>{dist['A'] + dist['B']:.0f} m</b><br>distance covered</span>
<span class="stat"><b>{len(shots)}</b><br>shots ({len(goals)} goals)</span>
<p class="note">Passes — Team A: {passes['completed']['A']} completed,
{passes['turnovers']['A']} turnovers · Team B: {passes['completed']['B']}
completed, {passes['turnovers']['B']} turnovers.<br>
Distance — Team A {dist['A']:.0f} m · Team B {dist['B']:.0f} m.</p></div>
<div class="card"><h2>Team shape</h2><img src="data:image/png;base64,{sh}">
<p class="note">Average positions across the match. Higher centroid = team
played further up the pitch.</p></div>
<div class="card"><h2>Heatmaps</h2>
<img src="data:image/png;base64,{hm_a}">
<img src="data:image/png;base64,{hm_b}"></div>
<div class="card"><h2>Shots — {len(shots)} ({len(goals)} goals)</h2>
<img src="data:image/png;base64,{sm}">
<p>xG — Team A {txg['A']:.2f} · Team B {txg['B']:.2f}
<span class="note">(geometric heuristic, not a fitted model)</span></p>
<table><tr><th>Time</th><th>From</th><th>Outcome</th><th>Speed</th><th>xG</th></tr>
{rows or '<tr><td colspan="5">No shots detected.</td></tr>'}</table></div>
{events_card}
<div class="card"><h2>Possession sequences</h2>
<table><tr><th></th><th>Sequences</th><th>Longest</th><th>Average</th></tr>
<tr><td>Team A</td><td>{seq.get('A', {}).get('count', 0)}</td>
<td>{seq.get('A', {}).get('longest_s', 0)}s</td>
<td>{seq.get('A', {}).get('avg_s', 0)}s</td></tr>
<tr><td>Team B</td><td>{seq.get('B', {}).get('count', 0)}</td>
<td>{seq.get('B', {}).get('longest_s', 0)}s</td>
<td>{seq.get('B', {}).get('avg_s', 0)}s</td></tr></table>
<p class="note">Throw-ins: {len(throw_ins)} · Corners: {n_corners} ·
Goal kicks: {n_gk}.</p></div>
{clip_html}
<p class="note">Coverage: ball tracked for {summary['ball_coverage']:.0%}
of the match ({summary['n_frames']} frames analysed). Stats reflect what the
single camera saw — treat them as approximate, not broadcast-grade.</p>
</body></html>"""
    Path(out_path).write_text(html, encoding="utf-8")
    return str(out_path)


def build_pdf(summary, out_path, title="Match Report"):
    """PDF version of the report (spec: exportable as PDF) — same figures
    plus a text summary page, via matplotlib only (no extra deps)."""
    from matplotlib.backends.backend_pdf import PdfPages
    poss = summary["possession_pct"]
    passes = summary.get("passes", {"completed": {"A": 0, "B": 0},
                                    "turnovers": {"A": 0, "B": 0}})
    dist = summary.get("distance_m", {"A": 0.0, "B": 0.0})
    shots = summary["shots"]
    goals = summary.get("goals", [])
    tackles = summary.get("tackles", [])
    txg = summary.get("total_xg", {"A": 0, "B": 0})
    seq = summary.get("sequences", {})

    with PdfPages(out_path) as pdf:
        # page 1: key numbers
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = [
            title, "",
            f"Possession — Team A {poss['A']:.0f}%   Team B {poss['B']:.0f}%",
            f"Shots — {len(shots)}   Goals — {len(goals)}   "
            f"Tackles — {len(tackles)}",
            f"xG — Team A {txg['A']:.2f}   Team B {txg['B']:.2f}  (heuristic)",
            f"Completed passes — A {passes['completed']['A']}   "
            f"B {passes['completed']['B']}",
            f"Turnovers — A {passes['turnovers']['A']}   "
            f"B {passes['turnovers']['B']}",
            f"Distance — A {dist['A']:.0f} m   B {dist['B']:.0f} m",
            f"Possession sequences — A {seq.get('A', {}).get('count', 0)} "
            f"(longest {seq.get('A', {}).get('longest_s', 0)}s)   "
            f"B {seq.get('B', {}).get('count', 0)} "
            f"(longest {seq.get('B', {}).get('longest_s', 0)}s)",
            "",
            "Goals:",
        ]
        for g in goals:
            lines.append(f"  {g['t']:.0f}s — {g['x']:.0f}m, {g['y']:.0f}m "
                         f"({g['goal']} goal)")
        lines += ["", "Shots:"]
        for s in shots:
            lines.append(f"  {s['t']:.0f}s — {s['outcome'].replace('_', ' ')}, "
                         f"{s['speed']:.0f} m/s from {s['x']:.0f}m, {s['y']:.0f}m, "
                         f"xG {s.get('xg', 0):.2f}")
        lines += ["",
                  f"Coverage: ball tracked for {summary['ball_coverage']:.0%} "
                  f"of the match. Single-camera stats are approximate."]
        ax.text(0.08, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=11, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        for fig in (possession_fig(poss),
                    heatmap_fig(summary["heatmaps"]["A"], "Team A heatmap"),
                    heatmap_fig(summary["heatmaps"]["B"], "Team B heatmap"),
                    shotmap_fig(shots),
                    shape_fig(summary["shape"])):
            pdf.savefig(fig)
            plt.close(fig)
    return str(out_path)
