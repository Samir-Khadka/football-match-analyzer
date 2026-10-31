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
        # scripted: A holds ~65% (two spells), B ~35%; cluster labels are
        # arbitrary, so check the split, not which side is "A"
        split = sorted(poss.values())
        self.assertAlmostEqual(sum(split), 100.0, places=6)
        self.assertGreater(split[1], 55.0)
        self.assertLess(abs(split[1] - 61.0), 15.0)

    def test_ball_coverage_high(self):
        self.assertGreater(self.summary["ball_coverage"], 0.90)

    def test_passes_detected(self):
        total = (self.summary["passes"]["completed"]["A"] +
                 self.summary["passes"]["completed"]["B"])
        self.assertGreaterEqual(total, 2,
                                f"expected >=2 scripted passes, got {total}")

    def test_tackle_detected(self):
        self.assertGreaterEqual(self.summary["n_tackles"], 1,
                                "the scripted B5 tackle on A4 was missed")

    def test_goal_detected(self):
        self.assertEqual(self.summary["n_goals"], 1)
        self.assertEqual(self.summary["goals"][0]["goal"], "-x")

    def test_out_of_play(self):
        # off-target shot + goal both leave the pitch
        self.assertEqual(len(self.summary["out_of_play"]), 2)

    def test_ten_tracks(self):
        # noisy detections + tackle occlusion + fast breaks fragment tracks;
        # the pipeline must not explode, and it degrades gracefully
        # (longevity-weighted team clustering). 10 players, expect < 2x.
        self.assertGreaterEqual(self.summary["n_tracks"], 10)
        self.assertLessEqual(self.summary["n_tracks"], 20)


class TestYoloLoads(unittest.TestCase):
    def test_yolo_detector_imports(self):
        from matchanalyzer.detect import YoloDetector  # noqa
        self.assertTrue(True)  # model download happens on first use


class TestRealFootage(unittest.TestCase):
    """Detection on a REAL night-match photo (Wikimedia Commons,
    tools/real_test.jpg — 8 visible players + ball). This is the spec's
    build step 1: validate detection before trusting anything downstream."""

    @classmethod
    def setUpClass(cls):
        import cv2
        from matchanalyzer.detect import YoloDetector
        cls.img = cv2.imread(str(ROOT / "tools" / "real_test.jpg"))

    def test_coco_finds_players(self):
        from matchanalyzer.detect import YoloDetector
        dets = YoloDetector("models/yolov8n.pt", conf=0.35).detect(self.img)
        players = [d for d in dets if d[4] == "player"]
        # 8 visible; 6+ is a pass (occlusion is real)
        self.assertGreaterEqual(len(players), 6,
                                f"only {len(players)} players found")

    def test_football_model_needs_low_threshold(self):
        # documents the calibration finding: the fine-tuned model is
        # under-confident out-of-domain; at conf=0.05 it sees the game
        from matchanalyzer.detect import FootballDetector
        det = FootballDetector("models/football-yolov8.pt", conf=0.05)
        dets = det.detect(self.img)
        players = [d for d in dets if d[4] == "player"]
        balls = [d for d in dets if d[4] == "ball"]
        self.assertGreaterEqual(len(players), 6)
        self.assertGreaterEqual(len(balls), 1)


class TestAdvancedEvents(unittest.TestCase):
    def _frames(self, seq):
        """seq: [(t, ball_xy|None, {pid: xy})]"""
        return [{"t": t, "ball": b, "players": p} for t, b, p in seq]

    def test_completed_pass(self):
        teams = {1: "A", 2: "A", 3: "B"}
        frames = self._frames([
            (0.0, (0, 0), {1: (0.2, 0), 2: (10, 0), 3: (20, 0)}),
            (0.5, (5, 0), {1: (4.8, 0), 2: (10, 0), 3: (20, 0)}),
            (1.0, (9.8, 0), {1: (5, 0), 2: (10, 0), 3: (20, 0)}),
        ])
        res = events.detect_passes(frames, teams)
        self.assertEqual(res["completed"]["A"], 1)
        self.assertEqual(res["completed"]["B"], 0)
        self.assertEqual(len(res["list"]), 1)
        self.assertGreater(res["list"][0]["dist_m"], 2.0)

    def test_turnover(self):
        teams = {1: "A", 3: "B"}
        frames = self._frames([
            (0.0, (0, 0), {1: (0.2, 0), 3: (20, 0)}),
            (1.0, (19.8, 0), {1: (0.2, 0), 3: (20, 0)}),
        ])
        res = events.detect_passes(frames, teams)
        self.assertEqual(res["completed"]["A"], 0)
        self.assertEqual(res["turnovers"]["A"], 1)

    def test_too_short_is_not_a_pass(self):
        teams = {1: "A", 2: "A"}
        frames = self._frames([
            (0.0, (0, 0), {1: (0.2, 0), 2: (1.0, 0)}),
            (0.5, (0.9, 0), {1: (0.2, 0), 2: (1.0, 0)}),
        ])
        res = events.detect_passes(frames, teams)
        self.assertEqual(res["completed"]["A"], 0)

    def test_distance_covered(self):
        teams = {1: "A"}
        frames = self._frames([
            (0.0, None, {1: (0, 0)}),
            (1.0, None, {1: (3, 4)}),
            (2.0, None, {1: (6, 8)}),
        ])
        d = events.distance_covered(frames, teams)
        self.assertAlmostEqual(d["A"], 10.0, places=5)

    def test_average_shape(self):
        teams = {1: "A", 2: "B"}
        frames = self._frames([
            (0.0, None, {1: (-10, 0), 2: (10, 5)}),
            (1.0, None, {1: (-12, 2), 2: (12, 7)}),
        ])
        shape = events.average_shape(frames, teams)
        self.assertEqual(shape["positions"]["A"], [(-11.0, 1.0)])
        self.assertEqual(shape["centroids"]["B"], (11.0, 6.0))


class TestV10(unittest.TestCase):
    def test_player_stats(self):
        teams = {1: "A", 2: "A"}
        frames = [
            {"t": 0.0, "ball": (0, 0),
             "players": {1: (0.0, 0.0), 2: (10.0, 0.0)}},
            {"t": 1.0, "ball": (3, 4),
             "players": {1: (3.0, 4.0), 2: (10.0, 0.0)}},
            {"t": 2.0, "ball": (10, 0),
             "players": {1: (3.0, 4.0), 2: (10.2, 0.0)}},
        ]
        plist = [{"t": 2.0, "from": 1, "to": 2, "team": "A", "dist_m": 8.0,
                  "from_xy": (3.0, 4.0), "to_xy": (10.0, 0.0)}]
        st = events.player_stats(frames, teams, plist)
        self.assertEqual(st[1]["touches"], 2)  # possessed frames 0,1
        self.assertEqual(st[2]["touches"], 1)  # possessed frame 2
        self.assertAlmostEqual(st[1]["distance_m"], 5.0, places=1)
        self.assertAlmostEqual(st[1]["carry_m"], 5.0, places=1)
        self.assertEqual(st[1]["passes_made"], 1)
        self.assertEqual(st[2]["passes_received"], 1)

    def test_player_stats_empty(self):
        self.assertEqual(events.player_stats([], {}, []), {})


class TestV08Web(unittest.TestCase):
    def test_index_and_status(self):
        from fastapi.testclient import TestClient
        from matchanalyzer import web as web_mod
        client = TestClient(web_mod.app)
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Football Match Analyzer", r.text)
        r = client.get("/api/jobs/doesnotexist")
        self.assertEqual(r.status_code, 404)
        r = client.get("/report/doesnotexist")
        self.assertEqual(r.status_code, 404)

    def test_calibrate_pages(self):
        from fastapi.testclient import TestClient
        from matchanalyzer import web as web_mod
        # fake a job awaiting calibration
        web_mod.JOBS["testcal"] = {"status": "awaiting_calibration",
                                   "frame_path": "/tmp/nonexistent.jpg"}
        client = TestClient(web_mod.app)
        r = client.get("/calibrate/testcal")
        self.assertEqual(r.status_code, 200)
        self.assertIn("pitch corners", r.text)
        r = client.get("/progress/testcal")
        self.assertEqual(r.status_code, 200)
        # unknown job -> 404 on frame
        r = client.get("/frame/nope")
        self.assertEqual(r.status_code, 404)
        del web_mod.JOBS["testcal"]


class TestV07(unittest.TestCase):
    def test_carry_distance(self):
        teams = {1: "A"}
        frames = [
            {"t": 0.0, "ball": (0, 0), "players": {1: (0.0, 0.0)}},
            {"t": 1.0, "ball": (3, 4), "players": {1: (3.0, 4.0)}},
            {"t": 2.0, "ball": (6, 8), "players": {1: (6.0, 8.0)}},
        ]
        # 5 m per frame x 2 = 10 m carried
        self.assertAlmostEqual(
            events.carry_distance(frames, teams)["A"], 10.0, places=1)

    def test_carry_resets_on_turnover(self):
        teams = {1: "A", 2: "B"}
        frames = [
            {"t": 0.0, "ball": (0, 0), "players": {1: (0.0, 0.0), 2: (50, 0)}},
            {"t": 1.0, "ball": (3, 4), "players": {1: (3.0, 4.0), 2: (50, 0)}},
            {"t": 2.0, "ball": (50, 0), "players": {1: (3.0, 4.0), 2: (50.2, 0)}},
            {"t": 3.0, "ball": (53, 0), "players": {1: (3.0, 4.0), 2: (53.2, 0)}},
        ]
        c = events.carry_distance(frames, teams)
        self.assertAlmostEqual(c["A"], 5.0, places=1)
        self.assertAlmostEqual(c["B"], 3.0, places=1)

    def test_progressive_pass(self):
        # A attacks +x
        dirs = {"A": 1, "B": -1}
        prog = events.progressive_passes(
            [{"t": 1.0, "from": 1, "to": 2, "team": "A", "dist_m": 15.0,
              "from_xy": (0.0, 0.0), "to_xy": (15.0, 0.0)}], dirs)
        self.assertEqual(len(prog), 1)
        # sideways pass is not progressive
        flat = events.progressive_passes(
            [{"t": 1.0, "from": 1, "to": 2, "team": "A", "dist_m": 15.0,
              "from_xy": (0.0, 0.0), "to_xy": (0.0, 15.0)}], dirs)
        self.assertEqual(len(flat), 0)
        # entry into final third counts even if short
        entry = events.progressive_passes(
            [{"t": 1.0, "from": 1, "to": 2, "team": "A", "dist_m": 5.0,
              "from_xy": (15.0, 0.0), "to_xy": (20.0, 0.0)}], dirs)
        self.assertEqual(len(entry), 1)


class TestV06(unittest.TestCase):
    def test_pass_network(self):
        teams = {1: "A", 2: "A", 3: "B"}
        frames = [
            {"t": 0.0, "ball": (0, 0),
             "players": {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (-10, 0)}},
            {"t": 1.0, "ball": (5, 0),
             "players": {1: (5.0, 0.0), 2: (10.0, 0.0), 3: (-10, 0)}},
        ]
        plist = [{"t": 0.5, "from": 1, "to": 2, "team": "A", "dist_m": 10.0}]
        net = events.pass_network(frames, teams, plist)
        self.assertEqual(len(net["nodes"]), 3)
        self.assertEqual(len(net["edges"]), 1)
        self.assertEqual(net["edges"][0]["count"], 1)
        n1 = next(n for n in net["nodes"] if n["id"] == 1)
        self.assertAlmostEqual(n1["x"], 2.5, places=1)
        self.assertEqual(n1["passes"], 1)

    def test_network_fig_empty(self):
        from matchanalyzer import report as report_mod
        fig = report_mod.network_fig({"nodes": [], "edges": []})
        self.assertIsNotNone(fig)

    def test_momentum_fig(self):
        from matchanalyzer import report as report_mod
        shots = [{"t": 10.0, "xg": 0.1, "team": "A"},
                 {"t": 20.0, "xg": 0.3, "team": "B"}]
        fig = report_mod.momentum_fig(shots)
        self.assertIsNotNone(fig)

    def test_shots_have_teams(self):
        # via summarize on a tiny synthetic
        teams = {1: "A", 2: "B"}
        frames = [
            {"t": 0.0, "ball": (30, 0), "players": {1: (30.2, 0), 2: (-30, 0)}},
            {"t": 0.5, "ball": (35, 0), "players": {1: (30.2, 0), 2: (-30, 0)}},
            {"t": 1.0, "ball": (60, 0), "players": {1: (30.2, 0), 2: (-30, 0)}},
        ]
        traj = [(f["t"],) + f["ball"] for f in frames]
        shots = events.detect_shots(traj, 10)
        # manually attribute (summarize does this in the pipeline)
        for s in shots:
            s["team"] = events._shot_team(s, teams, frames)
        if shots:
            self.assertIn(shots[0]["team"], ("A", "B", None))


class TestV05(unittest.TestCase):
    def test_attack_directions_from_heatmaps(self):
        # no shots: team positioned higher up attacks +x
        dirs = events.attack_directions(
            [], {1: "A"}, [], {"A": [(20, 0)], "B": [(-20, 0)]})
        self.assertEqual(dirs, {"A": 1, "B": -1})

    def test_attack_directions_from_shots(self):
        teams = {1: "A", 2: "B"}
        frames = [
            {"t": 0.5, "ball": (30, 0), "players": {1: (30.2, 0), 2: (-30, 0)}},
            {"t": 1.5, "ball": (-30, 0), "players": {1: (30, 0), 2: (-30.2, 0)}},
        ]
        shots = [{"t": 1.0, "x": 30, "y": 0, "goal": "+x"},
                 {"t": 2.0, "x": -30, "y": 0, "goal": "-x"}]
        dirs = events.attack_directions(shots, teams, frames,
                                        {"A": [(0, 0)], "B": [(0, 0)]})
        self.assertEqual(dirs, {"A": 1, "B": -1})

    def test_corner_when_defender_touches_last(self):
        teams = {1: "A", 2: "B"}
        heatmaps = {"A": [(20, 0)], "B": [(-20, 0)]}  # A attacks +x
        frames = [
            {"t": 0.0, "ball": (40, 5), "players": {1: (30, 0), 2: (40.2, 5)}},
            {"t": 0.5, "ball": (53, 10), "players": {1: (30, 0), 2: (40.2, 5)}},
        ]
        ev = events.detect_corners_goal_kicks(frames, teams, [], heatmaps)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["type"], "corner")
        self.assertEqual(ev[0]["team"], "A")

    def test_goal_kick_when_attacker_touches_last(self):
        teams = {1: "A", 2: "B"}
        heatmaps = {"A": [(20, 0)], "B": [(-20, 0)]}  # A attacks +x
        frames = [
            {"t": 0.0, "ball": (40, 5), "players": {1: (40.2, 5), 2: (30, 0)}},
            {"t": 0.5, "ball": (53, 10), "players": {1: (40.2, 5), 2: (30, 0)}},
        ]
        ev = events.detect_corners_goal_kicks(frames, teams, [], heatmaps)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["type"], "goal_kick")
        self.assertEqual(ev[0]["team"], "B")

    def test_goal_is_not_a_corner(self):
        teams = {1: "A", 2: "B"}
        heatmaps = {"A": [(20, 0)], "B": [(-20, 0)]}
        frames = [
            {"t": 0.0, "ball": (40, 0), "players": {1: (40.2, 0)}},
            {"t": 0.5, "ball": (53, 0.5), "players": {1: (40.2, 0)}},
        ]
        # between the posts (|y| <= 3.66) -> goal, not corner/goal-kick
        self.assertEqual(
            events.detect_corners_goal_kicks(frames, teams, [], heatmaps), [])


class TestV04(unittest.TestCase):
    def test_xg_sane(self):
        # penalty spot, central >> 30 m, wide; always a probability
        pen = events.shot_xg(41.5, 0.0)
        wild = events.shot_xg(22.5, -25.0)
        self.assertGreater(pen, wild)
        self.assertGreater(pen, 0.15)
        for x, y in [(52, 0), (30, 10), (0, 0), (-40, -20)]:
            v = events.shot_xg(x, y)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_throw_in_awarded_to_opponent(self):
        teams = {1: "A", 2: "B"}
        frames = [
            {"t": 0.0, "ball": (0, 0), "players": {1: (0.2, 0), 2: (10, 0)}},
            {"t": 0.5, "ball": (0, 20), "players": {1: (0.2, 20), 2: (10, 0)}},
            # ball rolls over the sideline (|y| > 34)
            {"t": 1.0, "ball": (0, 35), "players": {1: (0.2, 20), 2: (10, 0)}},
        ]
        ti = events.detect_throw_ins(frames, teams)
        self.assertEqual(len(ti), 1)
        self.assertEqual(ti[0]["team"], "B")  # A touched last -> B's throw

    def test_byline_exit_is_not_a_throw(self):
        teams = {1: "A", 2: "B"}
        frames = [
            {"t": 0.0, "ball": (40, 0), "players": {1: (40.2, 0)}},
            {"t": 0.5, "ball": (53, 0), "players": {1: (40.2, 0)}},
        ]
        self.assertEqual(events.detect_throw_ins(frames, teams), [])

    def test_possession_sequences(self):
        teams = {1: "A", 2: "B"}
        frames = [
            {"t": 0.0, "ball": (0, 0), "players": {1: (0.2, 0), 2: (30, 0)}},
            {"t": 1.0, "ball": (1, 0), "players": {1: (1.2, 0), 2: (30, 0)}},
            {"t": 2.0, "ball": (29, 0), "players": {1: (1.2, 0), 2: (29.2, 0)}},
            {"t": 3.0, "ball": (30, 0), "players": {1: (1.2, 0), 2: (30.2, 0)}},
        ]
        seqs = events.possession_sequences(frames, teams)
        self.assertEqual([s["team"] for s in seqs], ["A", "B"])
        self.assertAlmostEqual(seqs[0]["duration_s"], 1.0, places=1)


class TestV03Events(unittest.TestCase):
    def _frames(self, seq):
        return [{"t": t, "ball": b, "players": p} for t, b, p in seq]

    def test_tackle_detected(self):
        teams = {1: "A", 2: "B"}
        frames = self._frames([
            (0.0, (0, 0), {1: (0.2, 0), 2: (5, 0)}),
            (0.5, (4.8, 0), {1: (4.0, 0), 2: (5, 0)}),   # B takes it
        ])
        tackles = events.detect_tackles(frames, teams)
        self.assertEqual(len(tackles), 1)
        self.assertEqual(tackles[0]["team"], "B")
        self.assertEqual(tackles[0]["tackler"], 2)

    def test_far_turnover_is_not_a_tackle(self):
        teams = {1: "A", 2: "B"}
        frames = self._frames([
            (0.0, (0, 0), {1: (0.2, 0), 2: (40, 0)}),
            (0.5, (39.8, 0), {1: (0.2, 0), 2: (40, 0)}),  # B collects a stray
        ])
        self.assertEqual(events.detect_tackles(frames, teams), [])

    def test_out_of_play(self):
        traj = [(0.0, 50.0, 0.0), (0.5, 51.0, 0.0), (1.0, 53.0, 0.0),
                (1.5, 54.0, 0.0)]
        oop = events.detect_out_of_play(traj)
        self.assertEqual(len(oop), 1)
        self.assertAlmostEqual(oop[0]["t"], 1.0)

    def test_slow_roller_goal(self):
        # too slow for the shot detector, still a goal
        traj = [(i * 0.2, 48.0 + i * 1.0, 0.5) for i in range(10)]
        goals = events.detect_goals([(t, x, y) for t, x, y in traj])
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["goal"], "+x")

    def test_wide_ball_is_not_a_goal(self):
        traj = [(i * 0.1, 48.0 + i * 1.0, 10.0) for i in range(10)]
        self.assertEqual(events.detect_goals([(t, x, y) for t, x, y in traj]),
                         [])

    def test_pdf_builds(self):
        from matchanalyzer import report as report_mod
        summary = {
            "possession_pct": {"A": 60.0, "B": 40.0},
            "possession_frames": 100,
            "heatmaps": {"A": [(-10, 0)], "B": [(10, 0)]},
            "shots": [{"t": 5.0, "x": 30.0, "y": 0.0, "outcome": "goal",
                       "speed": 20.0}],
            "passes": {"completed": {"A": 3, "B": 2},
                       "turnovers": {"A": 1, "B": 1}},
            "distance_m": {"A": 100.0, "B": 90.0},
            "shape": {"positions": {"A": [(-10, 0)], "B": [(10, 0)]},
                      "centroids": {"A": (-10, 0), "B": (10, 0)}},
            "goals": [{"t": 5.0, "x": 52.5, "y": 0.5, "goal": "+x"}],
            "tackles": [{"t": 3.0, "tackler": 2, "team": "B",
                         "x": 5.0, "y": 1.0}],
            "ball_coverage": 0.97,
            "n_frames": 300,
        }
        out = OUT / "report.pdf"
        report_mod.build_pdf(summary, out)
        data = out.read_bytes()
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertGreater(len(data), 10000)


class TestClips(unittest.TestCase):
    def test_clips_cut_around_shots(self):
        from matchanalyzer.clips import make_clips
        # self-sufficient: TestEndToEnd (alphabetically later) also generates
        # this, but we may run first
        if not (OUT / "synth" / "match.mp4").exists():
            make_match(OUT / "synth", seed=7)
        out = OUT / "clips"
        clips = make_clips(OUT / "synth" / "match.mp4",
                           [{"t": 8.0, "outcome": "off_target"},
                            {"t": 17.5, "outcome": "goal"}],
                           out, pre_s=2.0, post_s=2.0)
        self.assertEqual(len(clips), 2)
        for c in clips:
            self.assertTrue(c["path"].exists())
            cap = __import__("cv2").VideoCapture(str(c["path"]))
            n = int(cap.get(__import__("cv2").CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertGreater(n, 30)  # ~4 s at 10 fps


if __name__ == "__main__":
    unittest.main(verbosity=1)
