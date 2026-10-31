"""Frame-to-frame tracking via ByteTrack (supervision).

Keeps player identity consistent across frames — the spec's core hard
problem on single-camera footage. Ball is tracked separately by nearest
detection (one ball, no ID switches to manage).
"""
from __future__ import annotations

import math


class Tracker:
    def __init__(self):
        import supervision as sv
        self._sv = sv
        self.byte_track = sv.ByteTrack()

    def update(self, detections):
        """detections: list of (x1, y1, x2, y2, cls, conf).

        Returns (players, ball):
          players: list of {"id": int, "x1".. "y2", "cx", "cy", "conf"}
          ball: {"cx", "cy", "conf"} | None (highest-conf ball detection)
        """
        sv = self._sv
        import numpy as np
        players, balls = [], []
        for d in detections:
            (balls if d[4] == "ball" else players).append(d)

        tracked = []
        if players:
            arr = np.array([[d[0], d[1], d[2], d[3]] for d in players])
            conf = np.array([d[5] for d in players])
            cls = np.zeros(len(players), dtype=int)
            det = sv.Detections(xyxy=arr, confidence=conf, class_id=cls)
            det = self.byte_track.update_with_detections(det)
            for i in range(len(det)):
                x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
                tid = det.tracker_id[i]
                tracked.append({
                    "id": int(tid) if tid is not None else -1,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1 + x2) / 2, "cy": y2,  # feet ~ bottom centre
                    "conf": float(det.confidence[i]),
                })
        ball = None
        if balls:
            b = max(balls, key=lambda d: d[5])
            ball = {"cx": (b[0] + b[2]) / 2, "cy": (b[1] + b[3]) / 2,
                    "conf": b[5]}
        return tracked, ball


def count_id_switches(tracked_ids_per_frame, max_players):
    """How many distinct IDs appeared beyond the true player count —
    a rough tracking-quality signal (0 = perfect on synthetic data)."""
    seen = set()
    for ids in tracked_ids_per_frame:
        seen.update(ids)
    return max(0, len(seen) - max_players)
