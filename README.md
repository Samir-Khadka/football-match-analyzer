# Football Match Analyzer — v0.2

Single-camera, post-match football analysis for grassroots clubs: upload a
match video, get player/ball tracking, possession %, heatmaps, shot map and a
coach-readable report. No manual tagging.

## What v0.2 adds

- **Pass detection** — completed passes + turnovers from possession changes
  between teammates (`detect_passes`).
- **Distance covered** per team, from pitch-coordinate trajectories.
- **Team shape** — average positions per track + team centroid ("how high
  the block sat"), plotted on the report.
- **Highlight clips** — mp4 cut ±4 s around every shot, linked in the
  report's clip library (`matchanalyzer/clips.py`).
- **Football-tuned detector** — `FootballDetector` wraps a YOLOv8 model
  fine-tuned on football footage (ball/goalkeeper/player/referee classes,
  mAP@0.5 0.785 on its validation set; `models/football-yolov8.pt`, ~22 MB
  from HuggingFace).
- **Real-footage detection tests** — `tools/real_test.jpg` (Wikimedia
  Commons night match, 8 players + ball visible); the suite asserts
  detection recall on it. This is the spec's build step 1, automated.

## A real finding, kept honest

Measuring the fine-tuned model on that night-match photo (2026-10-31):
it is **under-confident out of its training domain** — it needed
`conf=0.05` to find 7 players + the ball, while off-the-shelf COCO YOLOv8n
found 7 players at `conf=0.35`. So the **default detector stays COCO**;
use `FootballDetector` (default `conf=0.1`, tune per footage) when you
need its ball/goalkeeper/referee classes. A quoted mAP does not survive
contact with amateur night footage — validate per ground, per camera.

## What v0.1 built

A complete pipeline that runs end-to-end on CPU, validated against a
synthetic match with known ground truth:

```
video -> YOLOv8 detect -> ByteTrack -> homography -> events -> HTML report
```

- `matchanalyzer/detect.py` — detector interface; `YoloDetector` (ultralytics
  YOLOv8n, COCO person + sports ball, off-the-shelf) and
  `GroundTruthDetector` (replays scripted boxes with noise, for validation).
- `matchanalyzer/track.py` — ByteTrack via `supervision`; ball by nearest
  detection.
- `matchanalyzer/calibrate.py` — homography: image pixels -> pitch metres
  (105×68 m). Nothing downstream means anything without this.
- `matchanalyzer/events.py` — jersey-colour clustering (k-means, k=2) for
  team assignment; possession = nearest player within 1.5 m of the ball;
  shots = ball > 12 m/s toward goal from the attacking third, outcome from
  where it crosses the goal line.
- `matchanalyzer/report.py` — one-page HTML: possession bar, per-team
  heatmaps, shot map, shot table, honest coverage note.
- `matchanalyzer/pipeline.py` — batch orchestration (upload-and-wait, not
  real-time).
- `tools/make_synthetic_match.py` — renders a scripted 30 s 5v5 match
  (2 scripted shots: 1 goal, 1 off target) + ground truth JSON.

## Run it

```bash
pip install -r requirements.txt   # torch CPU recommended, no GPU needed
python tools/make_synthetic_match.py tools/synth_out
python -m unittest discover -s tests   # 22 tests: synthetic e2e + real photo
```

To use the football-tuned detector, download once (~22 MB):
```bash
curl -L -o models/football-yolov8.pt \
  https://huggingface.co/uisikdag/yolo-v8-football-players-detection/resolve/main/best.pt
```

## Honest limits (read before trusting any number)

- **Detection is off-the-shelf YOLOv8n.** It finds people and balls in
  reasonable footage but is NOT fine-tuned on football — the spec's build
  step 1 (fine-tune on SoccerNet) needs a GPU and is the documented next
  step. End-to-end validation uses scripted boxes so tracking/events/report
  are proven independently of detector accuracy.
- **Single-camera amateur footage will never match broadcast accuracy.**
  The report states ball-tracking coverage; treat all stats as approximate.
- **No player identity** (jersey numbers/faces) — team-level only, by shirt
  colour. If identity or biometric tracking is added later, check local
  privacy rules: academy/school football involves minors.
- CPU-only: ~10 fps video processes slower than real time. A GPU box or
  cloud GPU is needed for full 90-minute matches.

## Roadmap (per spec)

1. ✅ Pipeline skeleton + synthetic validation (this)
2. Fine-tune YOLOv8 on SoccerNet (GPU: Kaggle/Colab free tier)
3. Validate detection on real amateur footage before trusting events
4. Pass completion, tackles, out-of-play events
5. Clip export (every shot / final-third tackle as video clips)
6. Pilot with 2–3 real local-club matches, manually verified
7. v2: multi-camera, player ID, live analysis
