"""Team assignment (jersey-colour clustering) + event detection.

v1 events per the spec's build step 4: possession (nearest player to the
ball) and shots (fast ball toward goal). Pass completion etc. are later.

Conventions: pitch coords in metres, origin centre spot, +x = right goal.
Teams: "A" attacks +x in the demo; "B" attacks -x.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .calibrate import PITCH_LENGTH, PITCH_WIDTH

POSSESSION_RADIUS_M = 1.5
# Real shots travel ~17-36 m/s; hard passes top out ~15. v1 threshold: 16.
SHOT_SPEED_MPS = 16.0
GOAL_HALF_WIDTH_M = 3.66


def _kmeans_2d(points, seed=0, iters=20):
    """Tiny 2-cluster k-means (avoids an sklearn dependency)."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=float)
    centers = pts[rng.choice(len(pts), 2, replace=False)]
    for _ in range(iters):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        new = np.array([pts[labels == k].mean(0) if (labels == k).any()
                        else centers[k] for k in range(2)])
        if np.allclose(new, centers):
            break
        centers = new
    return labels


def assign_teams(tracks_by_id, sample_colors):
    """Cluster player tracks into two teams by jersey colour.

    tracks_by_id: {id: {"cx"..}}  sample_colors: {id: (h, s, v)} mean HSV
    from the player's box. Returns {id: "A" | "B"}.
    """
    ids = [i for i in tracks_by_id if i in sample_colors]
    if len(ids) < 2:
        return {i: "A" for i in ids}
    labels = _kmeans_2d([sample_colors[i] for i in ids])
    return {i: ("A" if labels[k] == 0 else "B") for k, i in enumerate(ids)}


def mean_hsv(frame_bgr, x1, y1, x2, y2):
    """Mean HSV of the box CENTRE (torso) — the full box is mostly grass
    for small/far players, which corrupts jersey-colour clustering."""
    import cv2
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return (0.0, 0.0, 0.0)
    # centre 40% x 60% — jersey torso, not background
    cx1 = int(x1 + (x2 - x1) * 0.3)
    cx2 = int(x1 + (x2 - x1) * 0.7)
    cy1 = int(y1 + (y2 - y1) * 0.2)
    cy2 = int(y1 + (y2 - y1) * 0.8)
    if cx2 <= cx1 or cy2 <= cy1:
        return (0.0, 0.0, 0.0)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = hsv.reshape(-1, 3).mean(0)
    return float(m[0]), float(m[1]), float(m[2])


def possession_per_frame(ball_xy, players_xy):
    """players_xy: {id: (x_m, y_m)}. Returns possessing player id or None."""
    if ball_xy is None:
        return None
    best, best_d = None, POSSESSION_RADIUS_M
    for pid, (px, py) in players_xy.items():
        d = math.hypot(px - ball_xy[0], py - ball_xy[1])
        if d < best_d:
            best, best_d = pid, d
    return best


def detect_shots(ball_traj, fps):
    """ball_traj: list of (t_s, x_m, y_m) | None per frame.

    A shot = ball speed > SHOT_SPEED_MPS while moving toward a goal from
    inside THAT goal's attacking third. Returns [{"t", "x", "y", "outcome"}],
    outcome in {"goal", "on_target", "off_target"}.
    """
    shots = []
    n = len(ball_traj)
    third = PITCH_LENGTH / 2 - PITCH_LENGTH / 3  # |x| beyond this = attacking
    for i in range(1, n - 1):
        cur, nxt = ball_traj[i], ball_traj[i + 1]
        if cur is None or nxt is None:
            continue
        t, x, y = cur
        if _ball_out((x, y)):
            continue  # already out of play — not a new shot
        dt = nxt[0] - t
        if dt <= 0:
            continue
        vx, vy = (nxt[1] - x) / dt, (nxt[2] - y) / dt
        speed = math.hypot(vx, vy)
        if speed < SHOT_SPEED_MPS or abs(vx) < 1e-6:
            continue
        goal_x = PITCH_LENGTH / 2 if vx > 0 else -PITCH_LENGTH / 2
        # must be in the attacking third of the goal it's moving toward
        if vx > 0 and x < third:
            continue
        if vx < 0 and x > -third:
            continue
        # one shot per 2s window (don't re-fire every frame of the strike)
        if shots and t - shots[-1]["t"] < 2.0:
            continue
        # outcome: prefer the OBSERVED goal-line crossing (robust to noisy
        # per-frame velocity); fall back to projection only if the ball
        # never reaches the line within 3 s (e.g. end of footage).
        outcome = "off_target"
        crossed_y = None
        for j in range(i + 1, n):
            pt = ball_traj[j]
            if pt is None:
                continue  # missed detection: keep scanning, don't give up
            tj, xj, yj = pt
            if tj - t > 3.0:
                break
            if (goal_x > 0 and xj >= goal_x) or (goal_x < 0 and xj <= goal_x):
                crossed_y = yj
                break
        if crossed_y is not None:
            outcome = "goal" if abs(crossed_y) <= GOAL_HALF_WIDTH_M else "off_target"
        else:
            t_goal = (goal_x - x) / vx
            if 0 < t_goal < 3.0 and abs(y + vy * t_goal) <= GOAL_HALF_WIDTH_M:
                outcome = "on_target"
        shots.append({"t": t, "x": x, "y": y, "vx": vx, "vy": vy,
                      "speed": speed, "outcome": outcome,
                      "goal": "+x" if goal_x > 0 else "-x"})
    return shots


def detect_tackles(frames_data, teams, duel_radius_m: float = 2.0,
                   max_gap_s: float = 3.0):
    """A tackle = possession changes teams in a physical duel: at the
    change, BOTH the previous carrier and the new possessor are within
    duel_radius_m of the ball. A turnover where the ball travelled to a
    distant player is an interception/pickup, not a tackle.
    Returns [{"t", "tackler", "team", "x", "y"}]."""
    tackles = []
    last = None  # (pid, team, t)
    for f in frames_data:
        if _ball_out(f["ball"]):
            last = None
            continue
        pid = possession_per_frame(f["ball"], f["players"])
        if pid is None or pid not in teams or f["ball"] is None:
            if last and f["t"] - last[2] > max_gap_s:
                last = None
            continue
        team = teams[pid]
        if last is not None:
            lpid, lteam, lt = last
            bx, by = f["ball"]
            if pid != lpid and team != lteam and f["t"] - lt <= max_gap_s:
                p_new = f["players"].get(pid)
                p_old = f["players"].get(lpid)
                if p_new and p_old:
                    d_new = math.hypot(bx - p_new[0], by - p_new[1])
                    d_old = math.hypot(bx - p_old[0], by - p_old[1])
                    if d_new <= duel_radius_m and d_old <= duel_radius_m:
                        tackles.append({"t": f["t"], "tackler": pid,
                                        "team": team, "x": round(bx, 1),
                                        "y": round(by, 1)})
        last = (pid, team, f["t"])
    return tackles


def detect_out_of_play(ball_traj):
    """Ball leaves the pitch rectangle. Returns [{"t", "x", "y"}]."""
    events = []
    was_in = None
    for pt in ball_traj:
        if pt is None:
            continue
        t, x, y = pt
        inside = abs(x) <= PITCH_LENGTH / 2 and abs(y) <= PITCH_WIDTH / 2
        if was_in and not inside:
            events.append({"t": t, "x": round(x, 1), "y": round(y, 1)})
        was_in = inside
    return events


def detect_goals(ball_traj):
    """Any crossing of a goal line between the posts — including slow
    rollers the shot detector (speed-gated) would miss."""
    goals = []
    prev = None
    for pt in ball_traj:
        if pt is None:
            prev = None
            continue
        t, x, y = pt
        if prev is not None:
            pt0, px, py = prev
            for goal_x in (PITCH_LENGTH / 2, -PITCH_LENGTH / 2):
                # sign change (or exact touch) relative to the goal line;
                # must be moving INTO the goal, not a clearance out
                into_goal = (x - px) > 0 if goal_x > 0 else (x - px) < 0
                touched = (px - goal_x) * (x - goal_x) <= 0 and x != px
                if touched and into_goal and abs(y) <= GOAL_HALF_WIDTH_M:
                    if not goals or t - goals[-1]["t"] > 3.0:
                        goals.append({"t": t, "x": round(x, 1),
                                      "y": round(y, 1),
                                      "goal": "+x" if goal_x > 0 else "-x"})
        prev = pt
    return goals


def shot_xg(x, y):
    """Expected goals for a shot from (x, y) — a transparent GEOMETRIC
    HEURISTIC (distance bands x centrality), not a fitted model. Real xG
    needs a trained model; this gives coaches a sensible 0-1 number."""
    d = math.hypot(PITCH_LENGTH / 2 - abs(x), y)  # dist to nearest goal centre
    centrality = max(0.2, 1.0 - abs(y) / 30.0)
    if d <= 6:
        base = 0.45
    elif d <= 11:
        base = 0.25
    elif d <= 18:
        base = 0.12
    elif d <= 25:
        base = 0.05
    else:
        base = 0.02
    return round(min(0.9, base * centrality), 3)


def possession_sequences(frames_data, teams):
    """Unbroken possession spells per team. A spell survives brief None gaps
    (ball in transit during a pass) — it ends only on out-of-play or the
    other team gaining possession. Returns
    [{"team", "t_start", "t_end", "duration_s"}]."""
    seqs = []
    cur_team, cur_start, last_t = None, None, None
    for f in frames_data:
        if _ball_out(f["ball"]):
            if cur_team is not None:
                seqs.append({"team": cur_team, "t_start": cur_start,
                             "t_end": last_t,
                             "duration_s": round(last_t - cur_start, 1)})
                cur_team, cur_start = None, None
            continue
        pid = possession_per_frame(f["ball"], f["players"])
        team = teams.get(pid) if pid else None
        if team is not None and team != cur_team:
            if cur_team is not None:
                seqs.append({"team": cur_team, "t_start": cur_start,
                             "t_end": last_t,
                             "duration_s": round(last_t - cur_start, 1)})
            cur_team, cur_start = team, f["t"]
        # team None -> ball in transit, spell continues
        last_t = f["t"]
    if cur_team is not None:
        seqs.append({"team": cur_team, "t_start": cur_start, "t_end": last_t,
                     "duration_s": round(last_t - cur_start, 1)})
    return seqs


def detect_throw_ins(frames_data, teams):
    """Sideline exits -> throw-in to the opponent of the last touch.
    (Byline exits need attack-direction inference for corner/goal-kick;
    v0.5.) Returns [{"t", "x", "y", "team"}]."""
    throw_ins = []
    last_team = None
    was_out = False
    for f in frames_data:
        out = _ball_out(f["ball"])
        if out and not was_out and f["ball"] is not None:
            x, y = f["ball"]
            if abs(y) > PITCH_WIDTH / 2 and abs(x) <= PITCH_LENGTH / 2:
                if last_team is not None:
                    awarded = "B" if last_team == "A" else "A"
                    throw_ins.append({"t": f["t"], "x": round(x, 1),
                                      "y": round(y, 1), "team": awarded})
        if out:
            last_team = None
        else:
            pid = possession_per_frame(f["ball"], f["players"])
            if pid is not None and pid in teams:
                last_team = teams[pid]
        was_out = out
    return throw_ins


def _shot_team(shot, teams, frames_data):
    """Team that took the shot: possession at shot time, else nearest player."""
    best = None
    for f in frames_data:
        if f["t"] <= shot["t"]:
            best = f
        else:
            break
    if best is None or best["ball"] is None:
        return None
    pid = possession_per_frame(best["ball"], best["players"])
    if pid is not None and pid in teams:
        return teams[pid]
    bx, by = best["ball"]
    nearest, nd = None, 1e9
    for pid, (px, py) in best["players"].items():
        d = math.hypot(bx - px, by - py)
        if d < nd:
            nearest, nd = pid, d
    return teams.get(nearest) if nearest else None


def summarize(frames_data, teams, fps):
    """frames_data: list of {"t", "ball": (x,y)|None, "players": {id: (x,y)}}.

    Returns possession %, per-team heatmap points, shots, passes, distance,
    average shape, coverage stats.
    """
    poss_counts = defaultdict(int)
    poss_frames = 0
    heat = {"A": [], "B": []}
    ball_traj = []
    ball_seen = 0
    for f in frames_data:
        ball_traj.append((f["t"],) + f["ball"] if f["ball"] else None)
        if f["ball"] is not None:
            ball_seen += 1
        pid = possession_per_frame(f["ball"], f["players"])
        if pid is not None and pid in teams:
            poss_counts[teams[pid]] += 1
            poss_frames += 1
        for pid2, (px, py) in f["players"].items():
            if pid2 in teams:
                heat[teams[pid2]].append((px, py))
    total = poss_counts["A"] + poss_counts["B"]
    possession_pct = {k: (poss_counts[k] / total * 100 if total else 0.0)
                      for k in ("A", "B")}
    shots = detect_shots(ball_traj, fps)
    for s in shots:
        s["xg"] = shot_xg(s["x"], s["y"])
    passes = detect_passes(frames_data, teams)
    distance = distance_covered(frames_data, teams)
    shape = average_shape(frames_data, teams)
    tackles = detect_tackles(frames_data, teams)
    out_of_play = detect_out_of_play(ball_traj)
    goals = detect_goals(ball_traj)
    throw_ins = detect_throw_ins(frames_data, teams)
    sequences = possession_sequences(frames_data, teams)
    seq_stats = {}
    for team in ("A", "B"):
        tsegs = [q for q in sequences if q["team"] == team]
        seq_stats[team] = {
            "count": len(tsegs),
            "longest_s": round(max([q["duration_s"] for q in tsegs] + [0]), 1),
            "avg_s": round(sum(q["duration_s"] for q in tsegs) /
                           len(tsegs), 1) if tsegs else 0.0,
        }
    return {
        "possession_pct": possession_pct,
        "possession_frames": poss_frames,
        "heatmaps": heat,
        "shots": shots,
        "n_shots": len(shots),
        "total_xg": {t: round(sum(s["xg"] for s in shots
                                  if _shot_team(s, teams, frames_data) == t), 3)
                     for t in ("A", "B")},
        "passes": passes,
        "distance_m": distance,
        "shape": shape,
        "tackles": tackles,
        "n_tackles": len(tackles),
        "out_of_play": out_of_play,
        "throw_ins": throw_ins,
        "goals": goals,
        "n_goals": len(goals),
        "sequences": seq_stats,
        "ball_coverage": ball_seen / len(frames_data) if frames_data else 0.0,
        "n_frames": len(frames_data),
    }


def _ball_out(ball_xy):
    """True when the ball has left the pitch (restarts break chains)."""
    if ball_xy is None:
        return False
    return abs(ball_xy[0]) > PITCH_LENGTH / 2 or abs(ball_xy[1]) > PITCH_WIDTH / 2


def detect_passes(frames_data, teams, max_gap_s: float = 3.0,
                  min_dist_m: float = 2.0):
    """Completed pass = possession moves between teammates within max_gap_s
    with the ball travelling > min_dist_m. Returns per-team counts plus a
    pass list [{"t", "from", "to", "team", "dist_m"}] and turnovers."""
    passes = {"A": 0, "B": 0}
    turnovers = {"A": 0, "B": 0}
    pass_list = []
    last = None  # (pid, team, t, ball_xy)
    for f in frames_data:
        if _ball_out(f["ball"]):
            last = None  # restart breaks the chain (no phantom turnovers)
            continue
        pid = possession_per_frame(f["ball"], f["players"])
        if pid is None or pid not in teams:
            # ball loose: keep last possessor until the gap expires
            if last and f["t"] - last[2] > max_gap_s:
                last = None
            continue
        team = teams[pid]
        if last is not None:
            lpid, lteam, lt, lball = last
            if pid != lpid and f["t"] - lt <= max_gap_s and f["ball"] and lball:
                dist = math.hypot(f["ball"][0] - lball[0],
                                  f["ball"][1] - lball[1])
                if dist >= min_dist_m:
                    if team == lteam:
                        passes[team] += 1
                        pass_list.append({"t": f["t"], "from": lpid,
                                          "to": pid, "team": team,
                                          "dist_m": round(dist, 1)})
                    else:
                        turnovers[lteam] += 1
        last = (pid, team, f["t"], f["ball"])
    return {"completed": passes, "turnovers": turnovers, "list": pass_list}


def distance_covered(frames_data, teams):
    """Metres travelled per team, from pitch-coordinate trajectories."""
    dist = {"A": 0.0, "B": 0.0}
    prev = {}
    for f in frames_data:
        for pid, (px, py) in f["players"].items():
            if pid not in teams:
                continue
            if pid in prev:
                ox, oy = prev[pid]
                dist[teams[pid]] += math.hypot(px - ox, py - oy)
            prev[pid] = (px, py)
    return {k: round(v, 1) for k, v in dist.items()}


def average_shape(frames_data, teams):
    """Mean position per track -> {"A": [(x, y)...], "B": [...]} plus each
    team's centroid (tells a coach how high/low the block sat)."""
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for f in frames_data:
        for pid, (px, py) in f["players"].items():
            if pid in teams:
                a = acc[(teams[pid], pid)]
                a[0] += px
                a[1] += py
                a[2] += 1
    shape = {"A": [], "B": []}
    for (team, _pid), (sx, sy, n) in acc.items():
        if n:
            shape[team].append((round(sx / n, 1), round(sy / n, 1)))
    centroids = {}
    for team in ("A", "B"):
        pts = shape[team]
        centroids[team] = (round(sum(p[0] for p in pts) / len(pts), 1),
                           round(sum(p[1] for p in pts) / len(pts), 1)) if pts else (0.0, 0.0)
    return {"positions": shape, "centroids": centroids}
