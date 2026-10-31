"""Annotated video export (v1.3).

Draws tracking boxes (team colors + IDs), ball marker, and event callouts
(SHOT, GOAL, TACKLE, …) onto the video. Very demoable.
"""
from __future__ import annotations

import cv2

TEAM_COLORS = {"A": (255, 120, 40), "B": (60, 60, 230)}  # BGR
BALL_COLOR = (0, 255, 255)
EVENT_COLOR = (0, 255, 0)


def _event_markers(summary):
    """Map timestamp -> label for on-screen callouts."""
    markers = {}
    for s in summary.get("shots", []):
        markers.setdefault(round(s["t"], 1), []).append("SHOT")
    for g in summary.get("goals", []):
        markers.setdefault(round(g["t"], 1), []).append("GOAL!")
    for t in summary.get("tackles", []):
        markers.setdefault(round(t["t"], 1), []).append("TACKLE")
    for c in summary.get("corners_goal_kicks", []):
        label = "CORNER" if c["type"] == "corner" else "GOAL KICK"
        markers.setdefault(round(c["t"], 1), []).append(label)
    return markers


def annotate_video(video_path, output_path, pixel_frames, teams, summary,
                   fps=10.0, stride=1):
    """Write an annotated MP4.

    pixel_frames: [{"t", "players": [{id,x1,y1,x2,y2}], "ball": {cx,cy}|None}]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path),
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    markers = _event_markers(summary)
    # index pixel frames by time for lookup
    pf_by_t = {round(pf["t"], 1): pf for pf in pixel_frames}

    frame_idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            t = kept * stride / fps
            pf = pf_by_t.get(round(t, 1))
            if pf:
                for p in pf["players"]:
                    pid = p["id"]
                    team = teams.get(pid)
                    if not team:
                        continue
                    col = TEAM_COLORS.get(team, (200, 200, 200))
                    x1, y1, x2, y2 = (int(p["x1"]), int(p["y1"]),
                                      int(p["x2"]), int(p["y2"]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(frame, f"#{pid}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                if pf["ball"]:
                    bx, by = int(pf["ball"]["cx"]), int(pf["ball"]["cy"])
                    cv2.circle(frame, (bx, by), 6, BALL_COLOR, -1)
                    cv2.circle(frame, (bx, by), 9, BALL_COLOR, 2)
            # event callouts (show for ~1s around the event)
            for mt, labels in markers.items():
                if abs(t - mt) < 0.6:
                    txt = " ".join(labels)
                    cv2.putText(frame, txt, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                EVENT_COLOR, 3)
            out.write(frame)
            kept += 1
        frame_idx += 1
    cap.release()
    out.release()
    return str(output_path)
