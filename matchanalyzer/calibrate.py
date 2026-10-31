"""Pitch calibration: image pixels -> real pitch coordinates (metres).

Heatmaps and distances mean nothing until positions are mapped to the
pitch. Standard pitch: 105 x 68 m, origin at the centre spot, +x toward
the right goal.

Usage:
    cal = PitchCalibrator()
    cal.set_points(image_pts=[(x, y)] * 4, pitch_pts=[(x_m, y_m)] * 4)
    x_m, y_m = cal.to_pitch(px, py)
"""
from __future__ import annotations

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


class PitchCalibrator:
    def __init__(self):
        self.H = None  # 3x3 homography, image -> pitch

    def set_points(self, image_pts, pitch_pts):
        """>= 4 correspondences. Raises ValueError if unsolvable."""
        import cv2
        if len(image_pts) < 4 or len(pitch_pts) < 4:
            raise ValueError("need at least 4 calibration points")
        H, mask = cv2.findHomography(np.float32(image_pts),
                                     np.float32(pitch_pts), 0)
        if H is None:
            raise ValueError("homography estimation failed")
        self.H = H

    def set_from_corners(self, top_left, top_right, bottom_right, bottom_left):
        """Full-pitch view: image corners of the pitch rectangle."""
        self.set_points(
            [top_left, top_right, bottom_right, bottom_left],
            [(-PITCH_LENGTH / 2, -PITCH_WIDTH / 2),
             (PITCH_LENGTH / 2, -PITCH_WIDTH / 2),
             (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
             (-PITCH_LENGTH / 2, PITCH_WIDTH / 2)])

    def to_pitch(self, x, y):
        if self.H is None:
            raise ValueError("calibrator has no homography yet")
        p = self.H @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_pitch_many(self, pts):
        return [self.to_pitch(x, y) for x, y in pts]
