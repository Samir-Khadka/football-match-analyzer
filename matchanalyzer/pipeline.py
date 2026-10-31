"""Pipeline orchestration: video -> detections -> tracks -> events -> report.

Batch job (upload-and-wait), per the spec — not real-time for v1.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import events
from .calibrate import PitchCalibrator
from .track import Tracker


class Pipeline:
    def __init__(self, detector, calibrator: PitchCalibrator, fps: float,
                 frame_stride: int = 1, max_frames: int | None = None,
                 collect_colors: bool = True):
        self.detector = detector
        self.calibrator = calibrator
        self.fps = fps
        self.stride = max(1, frame_stride)
        self.max_frames = max_frames
        self.collect_colors = collect_colors
        self.tracker = Tracker()

    def run(self, video_path, report_path="report.html"):
        from . import report as report_mod
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")

        tracks_by_id: dict[int, dict] = {}
        colors: dict[int, list] = {}
        ball_px: list = []
        frame_idx = 0
        kept = 0
        ids_per_frame = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % self.stride != 0:
                frame_idx += 1
                continue
            if self.max_frames and kept >= self.max_frames:
                break
            t = kept * self.stride / self.fps

            dets = self.detector.detect(frame)
            players, ball = self.tracker.update(dets)

            ids_per_frame.append(sorted(p["id"] for p in players))
            for p in players:
                tracks_by_id.setdefault(p["id"], {})[kept] = (t, p["cx"], p["cy"])
                if self.collect_colors:
                    hsv = events.mean_hsv(frame, p["x1"], p["y1"],
                                          p["x2"], p["y2"])
                    colors.setdefault(p["id"], []).append(hsv)
            ball_px.append(ball)
            kept += 1
            frame_idx += 1
        cap.release()

        # team assignment from jersey colours
        mean_colors = {i: tuple(np.mean(v, axis=0)) for i, v in colors.items()
                       if v}
        teams = events.assign_teams(tracks_by_id, mean_colors)

        # to pitch coordinates
        frames_data = []
        for k in range(kept):
            t = k * self.stride / self.fps
            players_xy = {}
            for pid, traj in tracks_by_id.items():
                if k in traj:
                    _, cx, cy = traj[k]
                    players_xy[pid] = self.calibrator.to_pitch(cx, cy)
            b = ball_px[k] if k < len(ball_px) else None
            ball_xy = self.calibrator.to_pitch(b["cx"], b["cy"]) if b else None
            frames_data.append({"t": t, "ball": ball_xy,
                                "players": players_xy})

        summary = events.summarize(frames_data, teams, self.fps / self.stride)
        summary["teams"] = teams
        summary["n_tracks"] = len(tracks_by_id)
        report_mod.build_report(summary, report_path)
        return summary
