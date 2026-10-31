"""Player/ball detection.

Detector interface: detect(frame_bgr) -> list of (x1, y1, x2, y2, cls, conf)
with cls in {"player", "ball"}.

Two implementations:
- YoloDetector: ultralytics YOLOv8, COCO-pretrained (person + sports ball).
  Off-the-shelf — no training. Good enough to prove the pipeline; the spec's
  build step 1 (fine-tune on SoccerNet) is the documented GPU step.
- GroundTruthDetector: replays scripted boxes (for synthetic validation),
  with optional miss-rate and jitter to simulate a noisy detector.
"""
from __future__ import annotations

import random

PERSON_CLASS = 0
BALL_CLASS = 32


class YoloDetector:
    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.35,
                 imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(model)  # downloads on first use (~6 MB)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame_bgr):
        res = self.model.predict(frame_bgr, conf=self.conf,
                                 imgsz=self.imgsz, verbose=False)[0]
        out = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            if cls_id == PERSON_CLASS:
                cls = "player"
            elif cls_id == BALL_CLASS:
                cls = "ball"
            else:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append((x1, y1, x2, y2, cls, float(box.conf[0])))
        return out


class FootballDetector:
    """Football-fine-tuned YOLOv8 (uisikdag/yolo-v8-football-players-detection,
    mAP@0.5 0.785 on its own validation set).

    Classes: ball / goalkeeper / player / referee. Goalkeepers and referees
    are tracked as players (v0.2 simplification — documented in README).

    Honest calibration note (measured 2026-10-31 on a real night-match
    photo, tools/real_test.jpg): this model is UNDER-CONFIDENT out of its
    training domain — it needed conf=0.05 to find 7 players + the ball,
    while COCO yolov8n found 7 players at conf=0.35. Default conf=0.1 is a
    starting point; tune per footage. Prefer YoloDetector unless you need
    the ball/goalkeeper/referee classes.
    model_path: local .pt (download once from HuggingFace, ~22 MB).
    """

    def __init__(self, model_path: str, conf: float = 0.1, imgsz: int = 960):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz  # higher res: players are small on wide shots

    def detect(self, frame_bgr):
        res = self.model.predict(frame_bgr, conf=self.conf,
                                 imgsz=self.imgsz, verbose=False)[0]
        names = self.model.names
        out = []
        for box in res.boxes:
            label = names[int(box.cls[0])]
            cls = "ball" if label == "ball" else "player"
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append((x1, y1, x2, y2, cls, float(box.conf[0])))
        return out


class GroundTruthDetector:
    """Replays per-frame ground truth boxes.

    frames: list of {"players": [(x, y, team)], "ball": (x, y) or None}
    in PIXEL coordinates. box_size: (w, h) stamped around each point.
    miss_rate / jitter simulate detector noise.
    """

    def __init__(self, frames, box_size=(18, 34), ball_box=10,
                 miss_rate: float = 0.0, jitter: float = 0.0, seed: int = 1):
        self.frames = frames
        self.bw, self.bh = box_size
        self.ball_box = ball_box
        self.miss_rate = miss_rate
        self.jitter = jitter
        self.rng = random.Random(seed)
        self._i = 0

    def reset(self):
        self._i = 0

    def _j(self, v):
        return v + self.rng.uniform(-self.jitter, self.jitter) if self.jitter else v

    def detect(self, frame_bgr):
        gt = self.frames[self._i] if self._i < len(self.frames) else None
        self._i += 1
        if not gt:
            return []
        out = []
        for (x, y, _team) in gt["players"]:
            if self.rng.random() < self.miss_rate:
                continue
            x, y = self._j(x), self._j(y)
            out.append((x - self.bw / 2, y - self.bh, x + self.bw / 2, y,
                        "player", 0.99))
        if gt["ball"] is not None:
            if self.rng.random() >= self.miss_rate:
                x, y = self._j(gt["ball"][0]), self._j(gt["ball"][1])
                b = self.ball_box
                out.append((x - b / 2, y - b / 2, x + b / 2, y + b / 2,
                            "ball", 0.99))
        return out
