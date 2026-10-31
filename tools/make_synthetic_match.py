"""Synthetic match video generator (OpenCV) with ground truth.

Renders a 30s 5v5 match at 640x360/10fps with a SCRIPTED ball path:
  0-8s   Team A possession, midfield passing
  ~9s    Team A shot at +x goal -> OFF TARGET (wide right)
  10-18s Team B possession, build-up
  ~19s   Team B shot at -x goal -> GOAL
  20-30s Team A possession, midfield

Writes: match.mp4 + ground_truth.json
  {"fps", "frames": [{"players": [[x, y, "A"|"B"]...], "ball": [x, y]|None}],
   "pitch_rect": [x1, y1, x2, y2] (pixel corners of the pitch),
   "shots": [{"t", "outcome"}]  (scripted, for scoring the event layer)}

Players are simple humanoids (body rect + head circle) in team colours;
the ball is a white circle. This validates tracking -> events -> report.
YOLO detection on these shapes is NOT expected to work like real footage
(see README); the pipeline test uses GroundTruthDetector.
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360
FPS = 10
DURATION = 30
N_FRAMES = FPS * DURATION

# pitch rectangle in pixels (margins for "stands")
PITCH_RECT = (40, 30, 600, 330)  # x1, y1, x2, y2
PL, PW = 105.0, 68.0  # metres


def pitch_to_px(xm, ym):
    x1, y1, x2, y2 = PITCH_RECT
    px = x1 + (xm + PL / 2) / PL * (x2 - x1)
    py = y1 + (ym + PW / 2) / PW * (y2 - y1)
    return px, py


def interp_waypoints(waypoints, t):
    """waypoints: [(t_s, xm, ym)] -> (xm, ym) at time t."""
    if t <= waypoints[0][0]:
        return waypoints[0][1], waypoints[0][2]
    for (t0, x0, y0), (t1, x1, y1) in zip(waypoints, waypoints[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0)
    return waypoints[-1][1], waypoints[-1][2]


BALL_SCRIPT = [
    (0, -10, 5), (2, -5, -8), (4, -12, 10), (6, -2, 0), (8, 15, -5),
    # A shot at +x goal, wide right (|y| > 3.66 at the line) ~20 m/s
    (8.6, 35, -2), (9.2, 52.5, -6.5), (10.0, 58, -9),
    # B collects, builds up toward -x
    (11, 30, 5), (13, 10, -12), (15, -8, 8), (17, -25, -4),
    # B shot at -x goal -> GOAL (|y| < 3.66) ~20 m/s
    (18.2, -38, 1), (18.9, -52.5, 0.5), (19.6, -58, 0.5),
    # A possession, midfield passing to the end
    (21, -20, -6), (23, -5, 10), (25, 8, -4), (27, -2, 6), (30, 5, 0),
]

SCRIPTED_SHOTS = [
    {"t": 8.6, "outcome": "off_target"},
    {"t": 18.2, "outcome": "goal"},
]

FORMATION_A = [(-30, 0), (-15, -15), (-15, 15), (-5, -8), (-5, 8)]
FORMATION_B = [(30, 0), (15, -15), (15, 15), (5, -8), (5, 8)]

TEAM_COLORS = {"A": (60, 60, 220), "B": (220, 60, 60)}  # BGR: red-ish, blue-ish
SKIN = (200, 180, 160)


def draw_pitch(img):
    img[:] = (45, 110, 45)
    x1, y1, x2, y2 = PITCH_RECT
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.line(img, ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2), (255, 255, 255), 2)
    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 30, (255, 255, 255), 2)
    # penalty boxes (simple)
    bw = int((x2 - x1) * 16.5 / PL)
    bh = int((y2 - y1) * 40.32 / PW)
    cv2.rectangle(img, (x1, (y1 + y2) // 2 - bh // 2),
                  (x1 + bw, (y1 + y2) // 2 + bh // 2), (255, 255, 255), 2)
    cv2.rectangle(img, (x2 - bw, (y1 + y2) // 2 - bh // 2),
                  (x2, (y1 + y2) // 2 + bh // 2), (255, 255, 255), 2)
    # goals
    gh = int((y2 - y1) * 7.32 / PW)
    cv2.rectangle(img, (x1 - 8, (y1 + y2) // 2 - gh // 2),
                  (x1, (y1 + y2) // 2 + gh // 2), (255, 255, 255), 2)
    cv2.rectangle(img, (x2, (y1 + y2) // 2 - gh // 2),
                  (x2 + 8, (y1 + y2) // 2 + gh // 2), (255, 255, 255), 2)


def draw_player(img, px, py, team):
    body_w, body_h = 10, 18
    x1, y1 = int(px - body_w / 2), int(py - body_h)
    cv2.rectangle(img, (x1, y1), (x1 + body_w, int(py)), TEAM_COLORS[team], -1)
    cv2.circle(img, (int(px), y1 - 4), 4, SKIN, -1)


def make_match(outdir, seed=7):
    rng = random.Random(seed)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(outdir / "match.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt_frames = []
    # per-player wander offsets
    wander = {(team, i): (rng.uniform(-6, 6), rng.uniform(-6, 6))
              for team in "AB" for i in range(5)}

    for f in range(N_FRAMES):
        t = f / FPS
        bxm, bym = interp_waypoints(BALL_SCRIPT, t)
        bpx, bpy = pitch_to_px(bxm, bym)
        img = np.zeros((H, W, 3), np.uint8)
        draw_pitch(img)

        players = []
        for team, formation in (("A", FORMATION_A), ("B", FORMATION_B)):
            for i, (ax, ay) in enumerate(formation):
                wx, wy = wander[(team, i)]
                # drift toward ball a little + wander
                pxm = ax + (bxm - ax) * 0.25 + wx * math.sin(t * 0.7 + i)
                pym = ay + (bym - ay) * 0.25 + wy * math.cos(t * 0.6 + i * 2)
                pxm = max(-PL / 2 + 2, min(PL / 2 - 2, pxm))
                pym = max(-PW / 2 + 2, min(PW / 2 - 2, pym))
                px, py = pitch_to_px(pxm, pym)
                draw_player(img, px, py, team)
                players.append([px, py, team])

        cv2.circle(img, (int(bpx), int(bpy)), 4, (255, 255, 255), -1)
        cv2.circle(img, (int(bpx), int(bpy)), 4, (60, 60, 60), 1)
        writer.write(img)
        gt_frames.append({"players": players, "ball": [bpx, bpy]})
    writer.release()

    gt = {"fps": FPS, "frames": gt_frames, "pitch_rect": list(PITCH_RECT),
          "shots": SCRIPTED_SHOTS, "pitch_size_m": [PL, PW]}
    (outdir / "ground_truth.json").write_text(json.dumps(gt))
    return outdir / "match.mp4", outdir / "ground_truth.json"


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tools/synth_out")
    v, g = make_match(d)
    print("wrote", v, "and", g)
