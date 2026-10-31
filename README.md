# Football Match Analyzer — v0.1 pipeline skeleton

Single-camera, post-match football analysis for grassroots clubs: upload a
match video, get player/ball tracking, possession %, heatmaps, shot map and a
coach-readable report. No manual tagging.

## What v0.1 actually is

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
python -m unittest discover -s tests   # 15 tests, includes full end-to-end
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
