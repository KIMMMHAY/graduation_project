"""MediaPipe PoseLandmarker 추정 결과를 JSON으로 만든다 (정확도 비교용).

팀 환경에는 mediapipe를 넣지 않는다(numpy·OpenCV 버전 충돌). 별도 가상환경에서만 실행한다:

  python -m venv .venv-mediapipe
  .venv-mediapipe\\Scripts\\pip install mediapipe==1.0.1
  .venv-mediapipe\\Scripts\\python tools\\mediapipe_predict.py --out mediapipe_heavy.json
  python predict_poses.py --import mediapipe_heavy.json          # (팀 환경에서) DB에 저장

필요한 것은 mediapipe(+ 함께 설치되는 opencv, numpy)뿐이다. 모델 파일은 처음 한 번 내려받는다(Apache-2.0).
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.pose_models import MEDIAPIPE_TO_OURS, to_keypoints  # noqa: E402  (numpy만 필요)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/"
             "pose_landmarker_{v}.task")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="heavy", choices=["lite", "full", "heavy"])
    ap.add_argument("--images", default=str(ROOT / "drawing_ref_test" / "images"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_path = ROOT / "drawing_ref_test" / "cache" / f"pose_landmarker_{args.variant}.task"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print("모델 내려받는 중...", flush=True)
        urllib.request.urlretrieve(MODEL_URL.format(v=args.variant), model_path)
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.IMAGE, num_poses=1))

    paths = sorted(Path(args.images).glob("*.jpg"))
    preds = {}
    for n, p in enumerate(paths, start=1):
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        r = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        kps = None
        if r.pose_landmarks:
            lm = r.pose_landmarks[0]
            xy = np.array([[lm[i].x * w, lm[i].y * h] for i in MEDIAPIPE_TO_OURS])
            conf = np.array([lm[i].visibility for i in MEDIAPIPE_TO_OURS])
            kps = to_keypoints(xy, conf, w, h)
        preds[p.stem] = {"keypoints": kps}
        print(f"\r{n}/{len(paths)}", end="", flush=True)
    Path(args.out).write_text(json.dumps({"model": f"mediapipe-{args.variant}", "predictions": preds}), encoding="utf-8")
    print(f"\n저장: {args.out} · 검출 {sum(v['keypoints'] is not None for v in preds.values())}/{len(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
