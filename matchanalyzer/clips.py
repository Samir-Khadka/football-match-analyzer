"""Highlight clips: cut short mp4s around key events (shots, goals).

A coach reviews moments, not 90 minutes — this is the spec's clip library.
"""
from __future__ import annotations

from pathlib import Path

import cv2


def make_clips(video_path, events, outdir, pre_s: float = 4.0,
               post_s: float = 4.0, max_clips: int = 20):
    """events: [{"t": seconds, "outcome": ...}]. Returns [clip paths]."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clips = []
    for i, ev in enumerate(events[:max_clips]):
        t = ev["t"]
        start = max(0, int((t - pre_s) * fps))
        end = min(n_frames - 1, int((t + post_s) * fps))
        name = f"clip_{i:02d}_{ev.get('outcome', 'event')}_{t:.0f}s.mp4"
        path = outdir / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        clips.append({"path": path, "t": t,
                      "outcome": ev.get("outcome", "event"), "name": name})
    cap.release()
    return clips
