"""Tests for the football match analyzer pipeline.

End-to-end runs on the synthetic match (ground truth boxes with noise),
proving tracking -> calibration -> events -> report. Real-footage accuracy
(SoccerNet fine-tuning) is out of scope for CPU v0.1 by design.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from matchanalyzer import calibrate, events  # noqa: E402
from matchanalyzer.calibrate import PitchCalibrator  # noqa: E402
from matchanalyzer.detect import GroundTruthDetector  # noqa: E402
from matchanalyzer.pipeline import Pipeline  # noqa: E402
from make_synthetic_match import make_match, PITCH_RECT  # noqa: E402

OUT = ROOT / "tests" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def calibrator_for_synth():
    x1, y1, x2, y2 = PITCH_RECT
    cal = PitchCalibrator()
    cal.set_from_corners((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    return cal


class TestCalibration(unittest.TestCase):
    def test_corners_map_to_pitch_extents(self):
        cal = calibrator_for_synth()
        x1, y1, x2, y2 = PITCH_RECT
        for (px, py), (ex, ey) in [
                ((x1, y1), (-52.5, -34.0)), ((x2, y1), (52.5, -34.0)),
                ((x2, y2), (52.5, 34.0)), ((x1, y1 + 0), (-52.5, -34.0))]:
            gx, gy = cal.to_pitch(px, py)
            self.assertAlmostEqual(gx, ex, places=4)
            self.assertAlmostEqual(gy, ey, places=4)

    def test_centre_spot(self):
        cal = calibrator_for_synth()
        x1, y1, x2, y2 = PITCH_RECT
        gx, gy = cal.to_pitch((x1 + x2) / 2, (y1 + y2) / 2)
        self.assertAlmostEqual(gx, 0.0, places=4)
        self.assertAlmostEqual(gy, 0.0, places=4)

    def test_needs_four_points(self):
        cal = PitchCalibrator()
        with self.assertRaises(ValueError):
            cal.set_points([(0, 0)] * 3, [(0, 0)] * 3)


class TestTeams(unittest.TestCase):
    def test_red_vs_blue_cluster(self):
        # OpenCV HSV: red H~0/179, blue H~110
        colors = {1: (2.0, 200.0, 200.0), 2: (1.0, 210.0, 190.0),
                  3: (110.0, 200.0, 200.0), 4: (112.0, 190.0, 210.0)}
        tracks = {i: {} for i in colors}
        teams = events.assign_teams(tracks, colors)
        self.assertEqual(teams[1], teams[2])
        self.assertEqual(teams[3], teams[4])
        self.assertNotEqual(teams[1], teams[3])


class TestEvents(unittest.TestCase):
    def test_shot_detection_and_outcome(self):
        # ball screaming toward +x goal, crosses line at |y| < 3.66 -> goal
        traj = [(i * 0.1, 30 + i * 2.0, 1.0) for i in range(15)]
        traj = [(t, x, y) for t, x, y in traj]
        shots = events.detect_shots([(t, x, y) for t, x, y in traj], 10)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0]["outcome"], "goal")
        self.assertGreater(shots[0]["speed"], 12.0)

    def test_slow_ball_is_not_a_shot(self):
        traj = [(i * 0.1, i * 0.5, 0.0) for i in range(30)]
        self.assertEqual(events.detect_shots(traj, 10), [])

    def test_possession_nearest_player(self):
        ball = (0.0, 0.0)
        players = {1: (0.5, 0.0), 2: (10.0, 0.0)}
        self.assertEqual(events.possession_per_frame(ball, players), 1)
        self.assertIsNone(events.possession_per_frame(None, players))
        far = {1: (50.0, 0.0)}
        self.assertIsNone(events.possession_per_frame(ball, far))


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vpath, cls.gpath = make_match(OUT / "synth", seed=7)
        gt = json.loads(cls.gpath.read_text())
        cls.gt = gt
        det = GroundTruthDetector(gt["frames"], miss_rate=0.02, jitter=0.5)
        pipe = Pipeline(det, calibrator_for_synth(), fps=gt["fps"])
        cls.summary = pipe.run(cls.vpath, report_path=OUT / "report.html")

    def test_report_written(self):
        html = (OUT / "report.html").read_text()
        self.assertIn("Possession", html)
        self.assertIn("Shots", html)
        self.assertIn("Coverage", html)

    def test_two_shots_detected(self):
        shots = self.summary["shots"]
        self.assertEqual(len(shots), 2,
                         f"expected 2 scripted shots, got {len(shots)}")

    def test_shot_outcomes_match_script(self):
        outcomes = sorted(s["outcome"] for s in self.summary["shots"])
        self.assertEqual(outcomes, ["goal", "off_target"])

    def test_possession_sane(self):
        poss = self.summary["possession_pct"]
        # scripted: one team has the ball ~20s of 30s; cluster labels are
        # arbitrary, so check the split, not which side is "A"
        split = sorted(poss.values())
        self.assertAlmostEqual(sum(split), 100.0, places=6)
        self.assertGreater(split[1], 55.0)
        self.assertLess(abs(split[1] - 66.7), 15.0)

    def test_ball_coverage_high(self):
        self.assertGreater(self.summary["ball_coverage"], 0.95)

    def test_ten_tracks(self):
        # noisy detections fragment a few tracks; the pipeline must stay
        # near the true count, not explode
        self.assertGreaterEqual(self.summary["n_tracks"], 10)
        self.assertLessEqual(self.summary["n_tracks"], 14)


class TestYoloLoads(unittest.TestCase):
    def test_yolo_detector_imports(self):
        from matchanalyzer.detect import YoloDetector  # noqa
        self.assertTrue(True)  # model download happens on first use


if __name__ == "__main__":
    unittest.main(verbosity=1)
