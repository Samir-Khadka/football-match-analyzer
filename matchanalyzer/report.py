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

TEAM_COLORS = {"A": "#3b82f6", "B": "#ef4444"}


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


def network_fig(network):
    """Pass network: nodes at average positions, sized by involvement;
    edges = passes between players."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _draw_pitch(ax)
    if not network or not network.get("nodes"):
        ax.set_title("Pass network — no completed passes")
        fig.tight_layout()
        return fig
    nodes = {n["id"]: n for n in network["nodes"]}
    for e in network["edges"]:
        a, b = nodes.get(e["from"]), nodes.get(e["to"])
        if not a or not b:
            continue
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="0.35",
                lw=0.5 + e["count"], alpha=0.6, zorder=2)
    for n in network["nodes"]:
        col = TEAM_COLORS.get(n["team"], "gray")
        sz = 60 + n["passes"] * 40
        ax.scatter([n["x"]], [n["y"]], s=sz, c=[col], zorder=3,
                   edgecolors="white", linewidths=1)
    ax.set_title("Pass network (node size = involvement)")
    fig.tight_layout()
    return fig


def momentum_fig(shots):
    """Cumulative xG race chart — who was on top, when."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    # split by team via stored team if present, else all in one
    for team, col in (("A", TEAM_COLORS["A"]), ("B", TEAM_COLORS["B"])):
        ts = sorted((s["t"], s.get("xg", 0)) for s in shots
                    if s.get("team") == team)
        if not ts:
            continue
        xs, cum, run = [0], [0], 0.0
        for t, xg in ts:
            xs += [t, t]
            cum += [run, run + xg]
            run += xg
        ax.step(xs, cum, where="post", label=f"Team {team}", color=col, lw=2)
    ax.set_xlabel("Match time (s)")
    ax.set_ylabel("Cumulative xG")
    ax.set_title("Momentum — cumulative expected goals")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def build_report(summary, out_path, title="Match Report", clips=None,
                 labels=None):
    """labels: {track_id: "Name #7"} — replaces #id in player tables."""
    labels = labels or {}
    def _name(tid):
        return labels.get(str(tid), labels.get(tid, f"#{tid}"))
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
    nw = _fig_to_b64(network_fig(summary.get("pass_network"))) if summary.get("pass_network") else ""
    mo = _fig_to_b64(momentum_fig(shots))
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
    n_prog = summary.get("n_progressive", {"A": 0, "B": 0})
    carries = summary.get("carry_m", {"A": 0, "B": 0})
    players = summary.get("players", {})
    # top 8 by touches for the table
    top = sorted(players.items(), key=lambda kv: kv[1]["touches"],
                 reverse=True)[:8]
    prow = "".join(
        f"<tr><td>{_name(tid)}</td><td>{s['team']}</td><td>{s['touches']}</td>"
        f"<td>{s['distance_m']:.0f} m</td><td>{s['carry_m']:.0f} m</td>"
        f"<td>{s['passes_made']}</td><td>{s['passes_received']}</td></tr>"
        for tid, s in top)
    # tactical analysis (v1.0)
    from . import tactics as tactics_mod
    tprof = tactics_mod.tactical_profile(summary)
    tvulns = tactics_mod.find_vulnerabilities(tprof)
    trows = "".join(
        f"<tr><td>{v['severity'].upper()}</td><td>Team {v['team']}</td>"
        f"<td>{v['finding']}<br><b>→ {v['exploit']}</b></td></tr>"
        for v in tvulns)
    _, _, plan_b = tactics_mod.anti_game_plan(summary, vs_team="B")
    _, _, plan_a = tactics_mod.anti_game_plan(summary, vs_team="A")
    plan_html = "".join(f"<li>{p}</li>" for p in plan_b)
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
<div class="card"><h2>Top performers</h2>
<table><tr><th>Player</th><th>Team</th><th>Touches</th><th>Distance</th>
<th>Carries</th><th>Passes</th><th>Received</th></tr>{prow}</table>
<p class="note">Track IDs can split on occlusions — one real player may appear
as several rows. Check the pass network map to merge.</p></div>
<div class="card"><h2>⚔️ Tactical vulnerabilities</h2>
<p>Formations — Team A: <b>{tprof['A'].get('formation', '–')}</b> ·
Team B: <b>{tprof['B'].get('formation', '–')}</b></p>
<table><tr><th>Severity</th><th>Team</th><th>Finding → Exploit</th></tr>{trows}</table>
<p class="note">Heuristic analysis from the tracked data — not a fitted model.
Verify against video before basing a game plan on it.</p></div>
<div class="card"><h2>📋 Anti-game-plan vs Team B</h2><ul>{plan_html}</ul></div>
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
Distance — Team A {dist['A']:.0f} m · Team B {dist['B']:.0f} m.<br>
Carries (with ball) — A {carries['A']:.0f} m · B {carries['B']:.0f} m.<br>
Progressive passes — A {n_prog['A']} · B {n_prog['B']}.<br>
Crosses — A {summary.get('n_crosses', {}).get('A', 0)} · B {summary.get('n_crosses', {}).get('B', 0)}.<br>
Interceptions — A {summary.get('n_interceptions', {}).get('A', 0)} · B {summary.get('n_interceptions', {}).get('B', 0)}.<br>
PPDA (press intensity, lower = hotter) — A {summary.get('ppda', {}).get('A', '–')} · B {summary.get('ppda', {}).get('B', '–')}.</p></div>
<div class="card"><h2>Team shape</h2><img src="data:image/png;base64,{sh}">
<p class="note">Average positions across the match. Higher centroid = team
played further up the pitch.</p></div>
<div class="card"><h2>Pass network</h2><img src="data:image/png;base64,{nw}">
<p class="note">Nodes at average positions (size = pass involvement);
lines = passes between players.</p></div>
<div class="card"><h2>Momentum</h2><img src="data:image/png;base64,{mo}"></div>
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


def build_pdf(summary, out_path, title="Match Report", labels=None):
    labels = labels or {}
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
            f"Carries — A {summary.get('carry_m', {}).get('A', 0):.0f} m   "
            f"B {summary.get('carry_m', {}).get('B', 0):.0f} m",
            f"Progressive passes — A {summary.get('n_progressive', {}).get('A', 0)}   "
            f"B {summary.get('n_progressive', {}).get('B', 0)}",
            f"Crosses — A {summary.get('n_crosses', {}).get('A', 0)}   "
            f"B {summary.get('n_crosses', {}).get('B', 0)}",
            f"Interceptions — A {summary.get('n_interceptions', {}).get('A', 0)}   "
            f"B {summary.get('n_interceptions', {}).get('B', 0)}",
            f"PPDA — A {summary.get('ppda', {}).get('A', '–')}   "
            f"B {summary.get('ppda', {}).get('B', '–')}",
            f"Possession sequences — A {seq.get('A', {}).get('count', 0)} "
            f"(longest {seq.get('A', {}).get('longest_s', 0)}s)   "
            f"B {seq.get('B', {}).get('count', 0)} "
            f"(longest {seq.get('B', {}).get('longest_s', 0)}s)",
            "",
            "Goals:",
        ]
        players = summary.get("players", {})
        if players:
            lines += ["", "Top performers (touches):"]
            top = sorted(players.items(),
                         key=lambda kv: kv[1]["touches"], reverse=True)[:5]
            for tid, s in top:
                nm = labels.get(str(tid), f"#{tid}") if labels else f"#{tid}"
                lines.append(
                    f"  {nm} ({s['team']}): {s['touches']} touches, "
                    f"{s['distance_m']:.0f} m, {s['passes_made']} passes")
            lines.append("")
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
                    momentum_fig(shots),
                    network_fig(summary.get("pass_network")),
                    shape_fig(summary["shape"])):
            pdf.savefig(fig)
            plt.close(fig)
    return str(out_path)
