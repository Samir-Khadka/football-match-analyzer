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
    import cv2
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return (0.0, 0.0, 0.0)
    crop = frame_bgr[y1:y2, x1:x2]
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


def summarize(frames_data, teams, fps):
    """frames_data: list of {"t", "ball": (x,y)|None, "players": {id: (x,y)}}.

    Returns possession %, per-team heatmap points, shots, coverage stats.
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
    return {
        "possession_pct": possession_pct,
        "possession_frames": poss_frames,
        "heatmaps": heat,
        "shots": shots,
        "n_shots": len(shots),
        "ball_coverage": ball_seen / len(frames_data) if frames_data else 0.0,
        "n_frames": len(frames_data),
    }
