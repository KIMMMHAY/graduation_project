"""사전학습 포즈 추정 모델 → 우리 13관절 변환.

- 기본 모델: DWPose (rtmlib, onnxruntime CPU). 일러스트 30장 시험에서 30/30 검출 (MediaPipe는 5~7/30).
- MediaPipe는 팀 환경에 넣지 않는다 (numpy·OpenCV 버전 충돌). 비교용 결과는 tools/mediapipe_predict.py를
  별도 가상환경에서 돌려 JSON으로 만든 뒤 `python predict_poses.py --import` 로 DB에 넣는다.
- 모델 결과는 poses 테이블에 source='model', annotator=모델 이름으로 저장한다.
  검출 실패는 status='skipped'(관절 전부 '없음')로 남겨 검출률 계산에 쓴다.
"""
from dataclasses import dataclass

import numpy as np

from core import pose as P

# 관절 신뢰도 → 상태. DWPose 점수 분포(일러스트 15장): 하위 5% 0.41, 중앙값 0.73
VISIBLE_MIN_CONF = 0.5
OCCLUDED_MIN_CONF = 0.3

# COCO 17관절 순서: 0 코, 1·2 눈, 3·4 귀, 5·6 어깨, 7·8 팔꿈치, 9·10 손목, 11·12 골반, 13·14 무릎, 15·16 발목 (사람 기준 좌우)
COCO_TO_OURS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
COCO_FACE = [0, 1, 2, 3, 4]
# MediaPipe 33관절 → 우리 13관절
MEDIAPIPE_TO_OURS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def state_of(conf: float) -> str:
    if conf >= VISIBLE_MIN_CONF:
        return P.VISIBLE
    if conf >= OCCLUDED_MIN_CONF:
        return P.OCCLUDED
    return P.ABSENT


def to_keypoints(xy: np.ndarray, conf: np.ndarray, width: int, height: int) -> list[dict]:
    """픽셀 좌표 13개 + 신뢰도 13개 → 우리 형식 (0~1 비율, 상태, 신뢰도)."""
    return P.validate_keypoints([
        {"x": float(x) / width, "y": float(y) / height, "state": state_of(float(c)), "conf": float(c)}
        for (x, y), c in zip(xy, conf)
    ])


def estimate_facing(keypoints: list[dict], face_conf: float | None = None) -> str | None:
    """몸이 향한 방향 추정 (라벨링 기본값 제안용, 정답 아님).

    캐릭터 기준 좌우라서 정면이면 왼쪽 어깨가 화면 오른쪽에 있다. 어깨 폭이 몸통 길이에 비해 좁으면 측면.
    얼굴 관절 신뢰도가 낮으면 뒷모습일 가능성이 높다.
    """
    ls, rs = keypoints[P.J["l_shoulder"]], keypoints[P.J["r_shoulder"]]
    lh, rh = keypoints[P.J["l_hip"]], keypoints[P.J["r_hip"]]
    if P.ABSENT in (ls["state"], rs["state"], lh["state"], rh["state"]):
        return None
    width = ls["x"] - rs["x"]
    torso = abs((ls["y"] + rs["y"]) / 2 - (lh["y"] + rh["y"]) / 2) or 1e-6
    if abs(width) < 0.25 * torso:
        return "side"
    if width < 0 or (face_conf is not None and face_conf < OCCLUDED_MIN_CONF):
        return "back"
    return "front"


@dataclass(frozen=True)
class Prediction:
    keypoints: list[dict]
    mean_conf: float
    facing: str | None


class DWPoseModel:
    """rtmlib Wholebody (balanced) = 인물 검출 YOLOX-m(Human-Art 학습) + 관절 추정 RTMW-dw-x-l(DWPose 방식 증류).

    처음 만들 때 모델 파일을 ~/.cache/rtmlib 에 내려받는다(약 15초, 이후 캐시).
    """
    name = "dwpose"
    label = "DWPose 계열 (RTMW-dw-x-l)"

    def __init__(self):
        from rtmlib import Wholebody
        self._model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")

    def predict(self, image_path: str) -> Prediction | None:
        import cv2
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        h, w = bgr.shape[:2]
        kps, scores = self._model(bgr)
        if len(kps) == 0:
            return None
        body = scores[:, :17]
        best = int(np.argmax(body.mean(axis=1)))  # 가장 확실한 인물 하나 (여러 인물 라벨링은 범위 밖)
        conf = np.clip(body[best], 0, 1)
        if conf[[5, 6, 11, 12]].max() < OCCLUDED_MIN_CONF:  # 어깨·골반을 하나도 못 찾았으면 검출 실패로 본다
            return None
        ours = to_keypoints(kps[best][COCO_TO_OURS], conf[COCO_TO_OURS], w, h)
        return Prediction(ours, float(conf[COCO_TO_OURS].mean()), estimate_facing(ours, float(conf[COCO_FACE].max())))


MODELS = {DWPoseModel.name: DWPoseModel}
DEFAULT_MODEL = DWPoseModel.name
