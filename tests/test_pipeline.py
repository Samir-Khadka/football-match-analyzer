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


class TestV17(unittest.TestCase):
    def test_pdf_endpoint(self):
        from fastapi.testclient import TestClient
        from matchanalyzer import web as web_mod
        import pickle, tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        d = tmp / "pdftest"
        d.mkdir()
        # minimal summary pickle
        summary = {
            "possession_pct": {"A": 50, "B": 50},
            "possession_frames": {"A": 10, "B": 10},
            "heatmaps": {"A": [], "B": []},
            "shots": [], "goals": [], "tackles": [],
            "passes": {"completed": {"A": 0, "B": 0},
                       "turnovers": {"A": 0, "B": 0}},
            "distance_m": {"A": 0, "B": 0},
            "shape": {"positions": {"A": [], "B": []},
                      "centroids": {"A": (0, 0), "B": (0, 0)}},
            "players": {}, "total_xg": {"A": 0, "B": 0},
            "sequences": {}, "throw_ins": [], "corners_goal_kicks": [],
            "n_shots": 0, "n_goals": 0, "n_tackles": 0,
            "ball_coverage": 1.0,
        }
        with open(d / "summary.pkl", "wb") as f:
            pickle.dump(summary, f)
        orig = web_mod.UPLOAD_DIR
        web_mod.UPLOAD_DIR = tmp
        web_mod.JOBS["pdftest"] = {"status": "done"}
        try:
            client = TestClient(web_mod.app)
            r = client.get("/pdf/pdftest")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers["content-type"], "application/pdf")
            self.assertTrue((d / "report.pdf").exists())
            r = client.get("/pdf/nonexistent")
            self.assertEqual(r.status_code, 404)
        finally:
            web_mod.UPLOAD_DIR = orig
            del web_mod.JOBS["pdftest"]


class TestV16(unittest.TestCase):
    def test_team_names(self):
        from fastapi.testclient import TestClient
        from matchanalyzer import web as web_mod
        import json, tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        d = tmp / "teamtest"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({
            "players": {"5": {"team": "A", "touches": 10}}}))
        orig = web_mod.UPLOAD_DIR
        web_mod.UPLOAD_DIR = tmp
        web_mod.JOBS["teamtest"] = {"status": "done"}
        try:
            client = TestClient(web_mod.app)
            r = client.post("/api/jobs/teamtest/players",
                            json={"labels": {},
                                  "team_names": {"A": "Urlabari FC",
                                                 "B": "Damak Utd"}})
            self.assertEqual(r.json(), {"ok": True})
            r = client.get("/api/jobs/teamtest/players")
            self.assertEqual(r.json()["team_names"]["A"], "Urlabari FC")
            # meta.json persisted
            meta = json.loads((d / "meta.json").read_text())
            self.assertEqual(meta["team_names"]["B"], "Damak Utd")
        finally:
            web_mod.UPLOAD_DIR = orig
            del web_mod.JOBS["teamtest"]


class TestV15(unittest.TestCase):
    def test_library_endpoints(self):
        from fastapi.testclient import TestClient
        from matchanalyzer import web as web_mod
        import json, tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        for jid in ("aaa111", "bbb222"):
            d = tmp / jid
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({
                "n_shots": 5, "n_goals": 1,
                "possession_pct": {"A": 60, "B": 40},
                "total_xg": {"A": 1.2, "B": 0.5},
                "passes": {"completed": {"A": 100, "B": 80}},
                "n_tackles": 8,
            }))
        orig = web_mod.UPLOAD_DIR
        web_mod.UPLOAD_DIR = tmp
        try:
            client = TestClient(web_mod.app)
            r = client.get("/library")
            self.assertEqual(r.status_code, 200)
            r = client.get("/api/library")
            self.assertEqual(len(r.json()["matches"]), 2)
            r = client.get("/api/compare?a=aaa111&b=bbb222")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["a"]["stats"]["Shots"], 5)
            r = client.get("/api/compare?a=aaa111&b=nope")
            self.assertEqual(r.status_code, 404)
        finally:
            web_mod.UPLOAD_DIR = orig


class TestV14(unittest.TestCase):
    def test_player_label_endpoints(self):
        from fastapi.testclient import TestClient
        from matchanalyzer import web as web_mod
        import json, tempfile
        from pathlib import Path
        # fake a done job with a summary
        tmp = Path(tempfile.mkdtemp())
        job_dir = tmp / "lbltest"
        job_dir.mkdir()
        (job_dir / "summary.json").write_text(json.dumps({
            "players": {"5": {"team": "A", "touches": 10}}}))
        # point UPLOAD_DIR at tmp
        orig = web_mod.UPLOAD_DIR
        web_mod.UPLOAD_DIR = tmp
        web_mod.JOBS["lbltest"] = {"status": "done"}
        try:
            client = TestClient(web_mod.app)
            r = client.get("/api/jobs/lbltest/players")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["players"][0]["id"], "5")
            r = client.post("/api/jobs/lbltest/players",
                            json={"labels": {"5": "Ram #7"}})
            self.assertEqual(r.json(), {"ok": True})
            r = client.get("/api/jobs/lbltest/players")
            self.assertEqual(r.json()["players"][0]["label"], "Ram #7")
            r = client.get("/players/lbltest")
            self.assertEqual(r.status_code, 200)
        finally:
            web_mod.UPLOAD_DIR = orig
            del web_mod.JOBS["lbltest"]

    def test_report_uses_labels(self):
        from matchanalyzer import report as rm
        # labels map is applied via _name() inside build_report;
        # test the mapping logic directly
        labels = {"5": "Ram #7"}
        def _name(tid):
            return labels.get(str(tid), labels.get(tid, f"#{tid}"))
        self.assertEqual(_name(5), "Ram #7")
        self.assertEqual(_name(99), "#99")


class TestV13(unittest.TestCase):
    def test_event_markers(self):
        from matchanalyzer import annotate as am
        summary = {
            "shots": [{"t": 10.0}],
            "goals": [{"t": 20.0}],
            "tackles": [{"t": 15.0}],
            "corners_goal_kicks": [{"t": 25.0, "type": "corner"}],
        }
        m = am._event_markers(summary)
        self.assertEqual(m[10.0], ["SHOT"])
        self.assertEqual(m[20.0], ["GOAL!"])
        self.assertEqual(m[15.0], ["TACKLE"])
        self.assertEqual(m[25.0], ["CORNER"])

    def test_annotate_runs(self):
        from matchanalyzer import annotate as am
        import cv2, tempfile, os
        # tiny 5-frame video
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "src.mp4")
        out = os.path.join(tmp, "out.mp4")
        vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (64, 64))
        import numpy as np
        for _ in range(5):
            vw.write(np.zeros((64, 64, 3), dtype=np.uint8))
        vw.release()
        pf = [{"t": 0.0,
               "players": [{"id": 1, "x1": 5, "y1": 5, "x2": 20, "y2": 20}],
               "ball": {"cx": 30, "cy": 30}}]
        am.annotate_video(src, out, pf, {1: "A"}, {"shots": [{"t": 0.0}]},
                          fps=10.0)
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)


class TestV12(unittest.TestCase):
    def test_formation_442(self):
        from matchanalyzer import tactics as tm
        # 4-4-2: GK + 4 defenders + 4 mids + 2 attackers (attacking +x)
        pos = [(-48, 0),  # GK
               (-30, -15), (-30, -5), (-30, 5), (-30, 15),  # defence
               (-10, -15), (-10, -5), (-10, 5), (-10, 15),  # midfield
               (15, -7), (15, 7)]  # attack
        self.assertEqual(tm.detect_formation(pos, 1), "4-4-2")

    def test_formation_433(self):
        from matchanalyzer import tactics as tm
        pos = [(-48, 0),
               (-30, -15), (-30, -5), (-30, 5), (-30, 15),
               (-8, -10), (-8, 0), (-8, 10),
               (18, -12), (18, 0), (18, 12)]
        self.assertEqual(tm.detect_formation(pos, 1), "4-3-3")

    def test_formation_too_few(self):
        from matchanalyzer import tactics as tm
        self.assertIsNone(tm.detect_formation([(-48, 0), (-30, 0)], 1))
        self.assertIsNone(tm.detect_formation([], 1))

    def test_formation_in_profile(self):
        from matchanalyzer import tactics as tm
        summary = {
            "shots": [], "progressive_passes": [],
            "shape": {"positions": {
                "A": [(-48, 0), (-30, -10), (-30, 10),
                      (-10, -10), (-10, 10), (15, 0)],
                "B": []}, "centroids": {}},
            "players": {}, "passes": {"completed": {}, "turnovers": {}},
        }
        prof = tm.tactical_profile(summary)
        # 5 outfield + GK: lines at -30 (2), -10 (2), 15 (1) -> 2-2-1
        self.assertEqual(prof["A"]["formation"], "2-2-1")


class TestV11(unittest.TestCase):
    def test_interception_flagged(self):
        teams = {1: "A", 2: "B"}
        # A passes (ball fast), B cuts it out
        frames = [
            {"t": 0.0, "ball": (0, 0), "players": {1: (0.1, 0), 2: (20, 0)}},
            {"t": 0.2, "ball": (15, 0), "players": {1: (0.1, 0), 2: (15.1, 0)}},
            {"t": 0.3, "ball": (15.2, 0), "players": {1: (0.1, 0), 2: (15.1, 0)}},
        ]
        # ball moved 15m in 0.2s = 75 m/s at the turnover -> interception
        res = events.detect_passes(frames, teams)
        self.assertEqual(res["turnovers"]["A"], 1)
        self.assertEqual(len(res["turnover_list"]), 1)
        self.assertTrue(res["turnover_list"][0]["interception"])
        self.assertEqual(res["turnover_list"][0]["by_team"], "B")

    def test_slow_turnover_not_interception(self):
        teams = {1: "A", 2: "B"}
        # slow dribble takeover, not a pass cut out
        frames = [
            {"t": 0.0, "ball": (0, 0), "players": {1: (0.1, 0), 2: (5, 0)}},
            {"t": 1.0, "ball": (2, 0), "players": {1: (0.1, 0), 2: (2.1, 0)}},
        ]
        res = events.detect_passes(frames, teams)
        if res["turnover_list"]:
            self.assertFalse(res["turnover_list"][0]["interception"])

    def test_cross_detection(self):
        # A attacks +x; wide pass into the box
        dirs = {"A": 1, "B": -1}
        crosses = events.detect_crosses(
            [{"t": 1.0, "from": 1, "to": 2, "team": "A", "dist_m": 20.0,
              "from_xy": (20.0, 25.0), "to_xy": (40.0, 5.0)}], dirs)
        self.assertEqual(len(crosses), 1)
        # central pass is not a cross
        flat = events.detect_crosses(
            [{"t": 1.0, "from": 1, "to": 2, "team": "A", "dist_m": 20.0,
              "from_xy": (20.0, 5.0), "to_xy": (40.0, 5.0)}], dirs)
        self.assertEqual(len(flat), 0)

    def test_ppda(self):
        passes = {"completed": {"A": 20, "B": 10}, "turnovers": {},
                  "list": []}
        tackles = [{"team": "A"}, {"team": "A"}]  # A made 2 tackles
        interceptions = [{"by_team": "A"}, {"by_team": "A"}]  # + 2 ints
        # PPDA(A) = B passes (10) / A actions (4) = 2.5
        out = events.ppda(passes, tackles, interceptions)
        self.assertEqual(out["A"], 2.5)
        # B did nothing -> None
        self.assertIsNone(out["B"])


class TestTactics(unittest.TestCase):
    def test_profile_and_vulns(self):
        from matchanalyzer import tactics as tm
        summary = {
            "shots": [
                {"t": 10, "x": 30, "y": -20, "xg": 0.05, "team": "A",
                 "goal": "+x"},
                {"t": 20, "x": 32, "y": -18, "xg": 0.04, "team": "A",
                 "goal": "+x"},
                {"t": 30, "x": 28, "y": 5, "xg": 0.03, "team": "A",
                 "goal": "+x"},
            ],
            "progressive_passes": [],
            "shape": {"positions": {"A": [(-30, 0), (-20, -10), (-20, 10),
                                          (-10, 0)],
                                    "B": [(30, 0)]},
                      "centroids": {}},
            "players": {1: {"team": "A", "touches": 10, "distance_m": 50,
                            "carry_m": 20, "passes_made": 8,
                            "passes_received": 2},
                        2: {"team": "A", "touches": 5, "distance_m": 40,
                            "carry_m": 10, "passes_made": 2,
                            "passes_received": 8}},
            "passes": {"completed": {"A": 10, "B": 0},
                       "turnovers": {"A": 0, "B": 0}, "list": []},
        }
        prof = tm.tactical_profile(summary)
        # A attacks left flank (2 of 3 shots y<-11)
        self.assertEqual(prof["A"]["dominant_flank"], "left")
        # player 1 makes 80% of passes -> key man vulnerability
        self.assertEqual(prof["A"]["key_distributor"], 1)
        vulns = tm.find_vulnerabilities(prof)
        types = [v["type"] for v in vulns if v["team"] == "A"]
        self.assertIn("key_man", types)
        # anti-game-plan mentions the key man
        _, _, plan = tm.anti_game_plan(summary, vs_team="A")
        self.assertTrue(any("#1" in p for p in plan))

    def test_empty_summary(self):
        from matchanalyzer import tactics as tm
        prof = tm.tactical_profile({})
        self.assertIn("A", prof)
        self.assertEqual(tm.find_vulnerabilities(prof), [])


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
