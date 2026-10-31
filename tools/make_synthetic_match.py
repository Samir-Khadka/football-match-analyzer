"""Synthetic match video generator (OpenCV) with ground truth.

Possession-based model: the ball is glued to a carrier's feet while
dribbling, flies between players on passes, rockets on shots. This makes
end-to-end validation of possession/passes/tackles MEANINGFUL (the v0.1
script moved the ball at 7 m/s while players ambled — possession was None
95% of the time).

Script (30 s, 5v5, 640x360 @ 10 fps):
  0-8    A2 dribbles up the left
  t=8    A2 SHOT at +x goal -> OFF TARGET (wide)
  10-15  B2 dribbles up the left (after restart)
  t=15   B2 SHOT at -x goal -> GOAL
  18-24  A5 dribbles up the right (after restart)
  t=24   PASS A5 -> A4 (long ball)
  26-28  A4 dribbles
  t=28   TACKLE: B5 takes it off A4
  t=28.2 PASS B5 -> B3

Writes: match.mp4 + ground_truth.json (same format as before).
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360
FPS = 10
DURATION = 32
N_FRAMES = FPS * DURATION

PITCH_RECT = (40, 30, 600, 330)
PL, PW = 105.0, 68.0

PLAYERS = {
    "A1": ("A", (-30, 0)), "A2": ("A", (-12, -6)), "A3": ("A", (-12, 6)),
    "A4": ("A", (-4, -10)), "A5": ("A", (-4, 10)),
    "B1": ("B", (30, 0)), "B2": ("B", (12, -6)), "B3": ("B", (12, 6)),
    "B4": ("B", (4, -10)), "B5": ("B", (4, 10)),
}
ATTACK = {"A": 1.0, "B": -1.0}  # sign of x the team attacks

PASS_SPEED = 12.0   # m/s
SHOT_SPEED = 25.0
DRIBBLE_OFFSET = 0.5  # ball ahead of carrier, metres


def _stable_hash(s):
    """Deterministic hash (Python's hash() is salted per process)."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def pitch_to_px(xm, ym):
    x1, y1, x2, y2 = PITCH_RECT
    return (x1 + (xm + PL / 2) / PL * (x2 - x1),
            y1 + (ym + PW / 2) / PW * (y2 - y1))


def lerp(a, b, f):
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def draw_pitch(img):
    img[:] = (45, 110, 45)
    x1, y1, x2, y2 = PITCH_RECT
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.line(img, ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2), (255, 255, 255), 2)
    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 30, (255, 255, 255), 2)
    bw = int((x2 - x1) * 16.5 / PL)
    bh = int((y2 - y1) * 40.32 / PW)
    cv2.rectangle(img, (x1, (y1 + y2) // 2 - bh // 2),
                  (x1 + bw, (y1 + y2) // 2 + bh // 2), (255, 255, 255), 2)
    cv2.rectangle(img, (x2 - bw, (y1 + y2) // 2 - bh // 2),
                  (x2, (y1 + y2) // 2 + bh // 2), (255, 255, 255), 2)
    gh = int((y2 - y1) * 7.32 / PW)
    cv2.rectangle(img, (x1 - 8, (y1 + y2) // 2 - gh // 2),
                  (x1, (y1 + y2) // 2 + gh // 2), (255, 255, 255), 2)
    cv2.rectangle(img, (x2, (y1 + y2) // 2 - gh // 2),
                  (x2 + 8, (y1 + y2) // 2 + gh // 2), (255, 255, 255), 2)


TEAM_COLORS = {"A": (60, 60, 220), "B": (220, 60, 60)}
SKIN = (200, 180, 160)


def draw_player(img, px, py, team):
    x1, y1 = int(px - 5), int(py - 18)
    cv2.rectangle(img, (x1, y1), (x1 + 10, int(py)), TEAM_COLORS[team], -1)
    cv2.circle(img, (int(px), y1 - 4), 4, SKIN, -1)


# Event list in time order. Positions in metres, times in seconds.
# All dribbles capped at 4 m/s (faster breaks ByteTrack's IoU matching —
# realistic anyway; nobody dribbles at 7 m/s).
#   ("dribble", pid, t0, t1, (x0,y0), (x1,y1))
#   ("pass", frm, to, t0)
#   ("shot", pid, t0, (tx,ty))   # target point (beyond the goal)
#   ("restart", pid, t0)          # ball placed at pid's feet
#   ("tackle", defender, victim, t0)
#   ("move", pid, t0, t1, p0, p1) # scripted off-ball run
SCRIPT = [
    ("dribble", "A2", 0.0, 8.0, (-12, -6), (20, -6)),
    ("shot", "A2", 8.0, (58, -8)),          # wide right -> out
    ("restart", "B2", 10.0),
    ("dribble", "B2", 10.0, 17.5, (12, -6), (-18, -6)),
    ("shot", "B2", 17.5, (-58, 0.5)),       # GOAL
    ("restart", "A5", 20.0),
    ("dribble", "A5", 20.0, 25.0, (-4, 10), (15, 10)),
    ("pass", "A5", "A4", 25.0),
    ("dribble", "A4", 27.3, 29.0, (-4, -10), (2.8, -10)),
    ("move", "B5", 26.0, 29.0, (4, 10), (5.3, -10)),  # tracks back to duel
    ("tackle", "B5", "A4", 29.0),           # B5 takes it
    ("dribble", "B5", 29.0, 30.5, (4.3, -10), (8, -6)),  # breaks away
    ("pass", "B5", "B3", 30.5),
    ("dribble", "B3", 31.6, 32.0, (12, 6), (10, 6)),
]

EXPECTED = {
    "shots": [{"t": 8.0, "outcome": "off_target"},
              {"t": 17.5, "outcome": "goal"}],
    "goals": [{"t": 18.7, "goal": "-x"}],
    "passes": 2,    # A5->A4, B5->B3
    "tackles": 1,
}


def make_match(outdir, seed=7):
    rng = random.Random(seed)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(outdir / "match.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    anchors = {pid: list(info[1]) for pid, info in PLAYERS.items()}
    wander = {pid: (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
              for pid in PLAYERS}
    gt_frames = []
    poses = {pid: tuple(anchors[pid]) for pid in PLAYERS}

    def base_pos(pid, t):
        ax, ay = anchors[pid]
        wx, wy = wander[pid]
        return (ax + wx * math.sin(t * 0.7 + _stable_hash(pid) % 7),
                ay + wy * math.cos(t * 0.6 + _stable_hash(pid) % 5))

    # event-driven ball state machine
    ev_idx = 0
    carrier = None
    dribble = None      # (pid, t0, t1, p0, p1)
    pass_flight = None  # (p_from, p_to, t0, dur, to_pid)
    shot_flight = None  # (p_from, dx, dy, t0)
    ball = None
    exempt_until = {}   # pid -> t: skip repulsion (scripted duel only)
    moving = {}         # pid -> (t0, t1, p0, p1): scripted off-ball run

    for f in range(N_FRAMES):
        t = f / FPS

        # apply due events
        while ev_idx < len(SCRIPT):
            ev = SCRIPT[ev_idx]
            t_ev = ev[2] if ev[0] in ("dribble", "restart", "shot", "move") else ev[3]
            if t_ev > t:
                break
            kind = ev[0]
            if kind == "dribble":
                _, pid, t0, t1, p0, p1 = ev
                carrier, dribble = pid, (pid, t0, t1, p0, p1)
                pass_flight = shot_flight = None
                poses[pid] = p0
            elif kind == "pass":
                _, frm, to, t0 = ev
                anchors[frm] = list(poses[frm])  # passer stays (no teleport)
                p_from = poses[frm]
                p_to = base_pos(to, t0)
                dist = math.hypot(p_to[0] - p_from[0], p_to[1] - p_from[1])
                dur = max(0.25, dist / PASS_SPEED)
                pass_flight = (p_from, p_to, t0, dur, to)
                carrier, dribble, shot_flight = None, None, None
                exempt_until[to] = t0 + dur + 0.5  # don't repel the receiver
            elif kind == "shot":
                _, pid, t0, target = ev
                anchors[pid] = list(poses[pid])  # shooter stays (no teleport)
                p_from = poses[pid]
                d = math.hypot(target[0] - p_from[0], target[1] - p_from[1])
                dx, dy = (target[0] - p_from[0]) / d, (target[1] - p_from[1]) / d
                shot_flight = (p_from, dx, dy, t0)
                carrier, dribble, pass_flight = None, None, None
            elif kind == "restart":
                _, pid, t0 = ev
                carrier = pid
                dribble, pass_flight, shot_flight = None, None, None
                poses[pid] = tuple(anchors[pid])
                ball = (poses[pid][0] +
                        ATTACK[PLAYERS[pid][0]] * DRIBBLE_OFFSET,
                        poses[pid][1])
            elif kind == "move":
                _, pid, t0, t1, p0, p1 = ev
                moving[pid] = (t0, t1, p0, p1)
                exempt_until[pid] = t1 + 0.5  # don't repel the run
            elif kind == "tackle":
                _, defender, victim, t0 = ev
                duel = list(poses[victim])
                # standing tackle: defender stops 1.5 m off (not full overlap —
                # full occlusions merge tracks; the unit tests cover the logic)
                dpos = [duel[0] + 1.5, duel[1]]
                anchors[defender] = dpos
                anchors[victim] = list(duel)    # victim stays down
                poses[defender] = list(dpos)
                carrier = defender
                dribble, pass_flight, shot_flight = None, None, None
                ball = (dpos[0] + ATTACK[PLAYERS[defender][0]] * DRIBBLE_OFFSET,
                        dpos[1])
                # duel pair: defender (now carrier) keeps the ball; victim stays
                # for the tackle frame then retreats (avoids possession flicker)
                exempt_until[defender] = t0 + 2.0
                exempt_until[victim] = t0 + 0.3
            ev_idx += 1

        # ball position from state
        if dribble is not None:
            pid, t0, t1, p0, p1 = dribble
            f_ = min(1.0, max(0.0, (t - t0) / (t1 - t0))) if t1 > t0 else 1.0
            pos = lerp(p0, p1, f_)
            poses[pid] = pos
            ball = (pos[0] + ATTACK[PLAYERS[pid][0]] * DRIBBLE_OFFSET, pos[1])
        elif pass_flight is not None:
            p_from, p_to, t0, dur, to_pid = pass_flight
            f_ = (t - t0) / dur
            if f_ >= 1.0:
                carrier = to_pid
                poses[to_pid] = p_to
                pass_flight = None
                ball = (p_to[0] + ATTACK[PLAYERS[to_pid][0]] * DRIBBLE_OFFSET,
                        p_to[1])
            else:
                ball = lerp(p_from, p_to, max(0.0, f_))
        elif shot_flight is not None:
            p_from, dx, dy, t0 = shot_flight
            dt = t - t0
            nb = (p_from[0] + dx * SHOT_SPEED * dt,
                  p_from[1] + dy * SHOT_SPEED * dt)
            # freeze once clearly out (net / stands) — capped, so the
            # out-of-play event is observable instead of a phantom freeze
            if abs(nb[0]) > PL / 2 + 2 or abs(nb[1]) > PW / 2 + 2:
                shot_flight = None
                ball = (max(-PL / 2 - 5, min(PL / 2 + 5, nb[0])),
                        max(-PW / 2 - 5, min(PW / 2 + 5, nb[1])))
            else:
                ball = nb
        # else: dead ball (after shot out / goal) — stays where it was

        # base positions for non-carriers (scripted moves override anchors)
        for pid in PLAYERS:
            if pid != carrier:
                if pid in moving:
                    t0, t1, p0, p1 = moving[pid]
                    f_ = min(1.0, max(0.0, (t - t0) / (t1 - t0))) if t1 > t0 else 1.0
                    poses[pid] = lerp(p0, p1, f_)
                    if t >= t1:
                        anchors[pid] = list(p1)
                        del moving[pid]
                else:
                    poses[pid] = base_pos(pid, t)

        # repulsion: non-carriers keep 4 m clear of the ball (no accidental
        # possession). The scripted duel pair is exempt for 2 s.
        if ball is not None:
            for pid in PLAYERS:
                if pid == carrier or t < exempt_until.get(pid, -1):
                    continue
                px, py = poses[pid]
                d = math.hypot(px - ball[0], py - ball[1])
                if d < 4.0 and d > 1e-6:
                    push = (4.0 - d) / d
                    poses[pid] = (px + (px - ball[0]) * push,
                                  py + (py - ball[1]) * push)

        # render
        img = np.zeros((H, W, 3), np.uint8)
        draw_pitch(img)
        plist = []
        for pid, (team, _) in PLAYERS.items():
            px, py = pitch_to_px(*poses[pid])
            draw_player(img, px, py, team)
            plist.append([px, py, team])
        bpx, bpy = pitch_to_px(*ball) if ball else (None, None)
        if ball is not None:
            cv2.circle(img, (int(bpx), int(bpy)), 4, (255, 255, 255), -1)
            cv2.circle(img, (int(bpx), int(bpy)), 4, (60, 60, 60), 1)
        writer.write(img)
        gt_frames.append({"players": plist,
                          "ball": [bpx, bpy] if ball else None})
    writer.release()

    gt = {"fps": FPS, "frames": gt_frames, "pitch_rect": list(PITCH_RECT),
          "shots": EXPECTED["shots"], "pitch_size_m": [PL, PW],
          "expected": EXPECTED}
    (outdir / "ground_truth.json").write_text(json.dumps(gt))
    return outdir / "match.mp4", outdir / "ground_truth.json"


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tools/synth_out")
    v, g = make_match(d)
    print("wrote", v, "and", g)
