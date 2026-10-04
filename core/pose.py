"""포즈 관절 정의, 정규화, 2D 매칭. Streamlit/DB에 의존하지 않는 순수 모듈.

좌표: 이미지 크기에 대한 비율(0~1), y는 아래로 증가. 좌우는 캐릭터 기준(캐릭터의 왼팔 = l_).

매칭 방식 (아이디어는 x6ud/pose-search, MIT 참고 — 코드는 가져오지 않음):
  1. 비율 좌표에 가로세로 비율을 곱해 실제 모양으로 되돌린다 (x * aspect, y)
  2. 뼈마다 방향 단위벡터를 구해 질의/후보 사이의 각도 오차를 잰다 (크기·위치와 무관)
  3. 관절 상태 가중치: 보임 1, 가려짐 0.5, 없음 0(그 관절이 포함된 뼈는 제외)
  4. 허용 범위 컷오프: 비교한 뼈 중 하나라도 각도 오차가 한도를 넘으면 결과에서 뺀다
  5. 점수 = 가중 평균 (1 - 각도오차/180°) × 100
  6. 좌우 반전: 후보를 x 반전 + 좌우 관절 교환한 것과도 비교해 더 나은 쪽을 쓴다 (flip 표시)
"""
from dataclasses import dataclass

import numpy as np

# ---------- 관절 (13개, 순서 고정) ----------
JOINTS: list[tuple[str, str]] = [
    ("head", "머리"),
    ("l_shoulder", "왼쪽 어깨"), ("r_shoulder", "오른쪽 어깨"),
    ("l_elbow", "왼쪽 팔꿈치"), ("r_elbow", "오른쪽 팔꿈치"),
    ("l_wrist", "왼쪽 손목"), ("r_wrist", "오른쪽 손목"),
    ("l_hip", "왼쪽 골반"), ("r_hip", "오른쪽 골반"),
    ("l_knee", "왼쪽 무릎"), ("r_knee", "오른쪽 무릎"),
    ("l_ankle", "왼쪽 발목"), ("r_ankle", "오른쪽 발목"),
]
JOINT_KEYS = [k for k, _ in JOINTS]
N_JOINTS = len(JOINTS)
J = {k: i for i, k in enumerate(JOINT_KEYS)}
MIRROR_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]  # 좌우 반전 시 짝 관절

# 가상 관절 (계산용): 어깨 중점, 골반 중점
SHOULDER_MID, HIP_MID = N_JOINTS, N_JOINTS + 1

VISIBLE, OCCLUDED, ABSENT = "visible", "occluded", "absent"
STATES = {VISIBLE: "보임", OCCLUDED: "가려짐", ABSENT: "없음"}
STATE_WEIGHT = {VISIBLE: 1.0, OCCLUDED: 0.5, ABSENT: 0.0}

FACINGS = {"front": "정면", "side": "측면", "back": "뒷모습"}

MANUAL, MODEL, MODEL_CORRECTED = "manual", "model", "model_corrected"
DRAFT, DONE, SKIPPED = "draft", "done", "skipped"
SEARCHABLE_SOURCES = (MANUAL, MODEL_CORRECTED)

# ---------- 뼈 ----------
# (key, 시작, 끝, 표시 이름, 쪽) — 쪽은 화면 색 구분용 (l / r / c)
BONES: list[tuple[str, int, int, str, str]] = [
    ("neck", SHOULDER_MID, J["head"], "목", "c"),
    ("torso", HIP_MID, SHOULDER_MID, "몸통", "c"),
    ("shoulder_line", J["r_shoulder"], J["l_shoulder"], "어깨선", "c"),  # 몸이 향한 방향(정면/뒷모습) 정보를 담는다
    ("hip_line", J["r_hip"], J["l_hip"], "골반선", "c"),
    ("l_upper_arm", J["l_shoulder"], J["l_elbow"], "왼쪽 위팔", "l"),
    ("l_forearm", J["l_elbow"], J["l_wrist"], "왼쪽 아래팔", "l"),
    ("r_upper_arm", J["r_shoulder"], J["r_elbow"], "오른쪽 위팔", "r"),
    ("r_forearm", J["r_elbow"], J["r_wrist"], "오른쪽 아래팔", "r"),
    ("l_thigh", J["l_hip"], J["l_knee"], "왼쪽 허벅지", "l"),
    ("l_shin", J["l_knee"], J["l_ankle"], "왼쪽 정강이", "l"),
    ("r_thigh", J["r_hip"], J["r_knee"], "오른쪽 허벅지", "r"),
    ("r_shin", J["r_knee"], J["r_ankle"], "오른쪽 정강이", "r"),
]
BONE_KEYS = [b[0] for b in BONES]
_ARMS = ["l_upper_arm", "l_forearm", "r_upper_arm", "r_forearm"]
_LEGS = ["l_thigh", "l_shin", "r_thigh", "r_shin"]
PARTS: dict[str, tuple[str, list[str]]] = {
    "full": ("전신", BONE_KEYS),
    "upper": ("상체", ["neck", "torso", "shoulder_line", *_ARMS]),
    "arms": ("팔", _ARMS),
    "legs": ("다리", ["hip_line", *_LEGS]),
}

DEFAULT_MAX_ANGLE = 60.0     # 허용 범위 기본값(도)
OCCLUDED_ANGLE_FACTOR = 1.5  # 가려진 관절이 포함된 뼈는 위치가 추정값이라 허용 범위를 1.5배로
MIN_COVERAGE = 0.5           # 비교할 부위의 뼈 중 이 비율(가중치 합 기준) 이상을 비교할 수 있어야 결과로 인정

# 표준 자세 (T자에 가까운 정면 서 있는 자세). 정면이므로 캐릭터의 왼쪽이 화면 오른쪽
DEFAULT_POSE: list[dict] = [
    {"x": x, "y": y, "state": VISIBLE} for x, y in [
        (0.50, 0.12),
        (0.58, 0.24), (0.42, 0.24),
        (0.70, 0.30), (0.30, 0.30),
        (0.80, 0.38), (0.20, 0.38),
        (0.55, 0.52), (0.45, 0.52),
        (0.56, 0.71), (0.44, 0.71),
        (0.57, 0.90), (0.43, 0.90),
    ]
]


class PoseError(ValueError):
    """잘못된 포즈 데이터 (사용자에게 보여줄 한국어 메시지)."""


def default_pose() -> list[dict]:
    return [dict(k) for k in DEFAULT_POSE]


def validate_keypoints(keypoints) -> list[dict]:
    """13개 {x, y, state}로 정리. 좌표는 0~1로 자르고 소수 4자리로 반올림."""
    if not isinstance(keypoints, list) or len(keypoints) != N_JOINTS:
        raise PoseError(f"관절은 {N_JOINTS}개여야 합니다.")
    out = []
    for i, k in enumerate(keypoints):
        try:
            x, y = float(k["x"]), float(k["y"])
        except (KeyError, TypeError, ValueError):
            raise PoseError(f"{JOINTS[i][1]} 좌표가 올바르지 않습니다.") from None
        if not (np.isfinite(x) and np.isfinite(y)):
            raise PoseError(f"{JOINTS[i][1]} 좌표가 올바르지 않습니다.")
        state = k.get("state", VISIBLE)
        if state not in STATES:
            raise PoseError(f"{JOINTS[i][1]} 상태 '{state}'는 쓸 수 없습니다.")
        out.append({"x": round(min(max(x, 0.0), 1.0), 4), "y": round(min(max(y, 0.0), 1.0), 4), "state": state})
    return out


def mirror_keypoints(keypoints: list[dict]) -> list[dict]:
    """좌우 반전: x를 뒤집고 왼쪽/오른쪽 관절을 맞바꾼다."""
    return [{**keypoints[MIRROR_INDEX[i]], "x": round(1.0 - keypoints[MIRROR_INDEX[i]]["x"], 4)}
            for i in range(N_JOINTS)]


def _points(keypoints: list[dict], aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """(15x2 좌표, 15 가중치). 가로세로 비율 보정 + 가상 관절(어깨·골반 중점) 포함."""
    pts = np.zeros((N_JOINTS + 2, 2))
    w = np.zeros(N_JOINTS + 2)
    for i, k in enumerate(keypoints):
        pts[i] = (k["x"] * aspect, k["y"])
        w[i] = STATE_WEIGHT[k["state"]]
    for mid, a, b in ((SHOULDER_MID, J["l_shoulder"], J["r_shoulder"]), (HIP_MID, J["l_hip"], J["r_hip"])):
        pts[mid] = (pts[a] + pts[b]) / 2
        w[mid] = min(w[a], w[b])
    return pts, w


def _bone_dirs(keypoints: list[dict], aspect: float, bone_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pts, w = _points(keypoints, aspect)
    idx = [BONE_KEYS.index(k) for k in bone_keys]
    a = np.array([BONES[i][1] for i in idx])
    b = np.array([BONES[i][2] for i in idx])
    vec = pts[b] - pts[a]
    length = np.linalg.norm(vec, axis=1)
    weight = np.minimum(w[a], w[b])
    weight[length < 1e-6] = 0.0  # 두 관절이 겹치면 방향을 정할 수 없다
    dirs = np.divide(vec, length[:, None], out=np.zeros_like(vec), where=length[:, None] > 1e-6)
    return dirs, weight


@dataclass(frozen=True)
class Match:
    score: float            # 0~100, 높을수록 비슷함
    mean_angle: float       # 가중 평균 각도 오차(도)
    max_angle: float        # 비교한 뼈 중 가장 큰 각도 오차(도)
    n_bones: int            # 비교에 쓴 뼈 수
    flip: bool = False      # 좌우 반전해서 맞았는지


def compare(query: list[dict], query_aspect: float, cand: list[dict], cand_aspect: float,
            part: str = "full", max_angle: float = DEFAULT_MAX_ANGLE) -> Match | None:
    """두 포즈의 유사도. 비교할 뼈가 부족하거나 허용 범위를 넘으면 None."""
    bone_keys = PARTS[part][1]
    qd, qw = _bone_dirs(query, query_aspect, bone_keys)
    cd, cw = _bone_dirs(cand, cand_aspect, bone_keys)
    w = np.minimum(qw, cw)
    used = w > 0
    if w.sum() < MIN_COVERAGE * len(bone_keys):
        return None
    angle = np.degrees(np.arccos(np.clip((qd * cd).sum(axis=1), -1.0, 1.0)))
    limit = np.where(w >= 1.0, max_angle, min(max_angle * OCCLUDED_ANGLE_FACTOR, 180.0))
    if (angle[used] > limit[used]).any():
        return None
    return Match(score=float(100 * (w * (1 - angle / 180)).sum() / w.sum()),
                 mean_angle=float((w * angle).sum() / w.sum()),
                 max_angle=float(angle[used].max()), n_bones=int(used.sum()))


def match(query: list[dict], query_aspect: float, cand: list[dict], cand_aspect: float,
          part: str = "full", allow_mirror: bool = True, max_angle: float = DEFAULT_MAX_ANGLE) -> Match | None:
    direct = compare(query, query_aspect, cand, cand_aspect, part, max_angle)
    if not allow_mirror:
        return direct
    m = compare(query, query_aspect, mirror_keypoints(cand), cand_aspect, part, max_angle)
    if m is not None and (direct is None or m.score > direct.score):
        return Match(m.score, m.mean_angle, m.max_angle, m.n_bones, flip=True)
    return direct


@dataclass(frozen=True)
class Candidate:
    image_id: str
    keypoints: list[dict]
    aspect: float
    facing: str | None = None
    annotator: str = ""


@dataclass(frozen=True)
class SearchHit:
    candidate: Candidate
    match: Match


def search(query: list[dict], query_aspect: float, candidates: list[Candidate], part: str = "full",
           allow_mirror: bool = True, facings: list[str] | None = None, max_angle: float = DEFAULT_MAX_ANGLE,
           top_k: int = 12) -> list[SearchHit]:
    """가까운 순 top_k. 한 이미지에 여러 작성자의 포즈가 있으면 가장 잘 맞는 것 하나만 쓴다."""
    best: dict[str, SearchHit] = {}
    for c in candidates:
        if facings and c.facing not in facings:
            continue
        m = match(query, query_aspect, c.keypoints, c.aspect, part, allow_mirror, max_angle)
        if m is not None and (c.image_id not in best or m.score > best[c.image_id].match.score):
            best[c.image_id] = SearchHit(c, m)
    return sorted(best.values(), key=lambda h: -h.match.score)[:top_k]


def normalize(keypoints: list[dict], aspect: float) -> np.ndarray | None:
    """골반 중점을 원점, 몸통 길이(골반 중점~어깨 중점)를 1로 맞춘 13x2 좌표. 몸통을 알 수 없으면 None.

    방향 기반 매칭에는 필요 없고(이미 크기·위치 무관), 관절 위치 오차·PCK 계산(단계 2)에 쓴다.
    """
    pts, w = _points(keypoints, aspect)
    if w[SHOULDER_MID] == 0 or w[HIP_MID] == 0:
        return None
    torso = np.linalg.norm(pts[SHOULDER_MID] - pts[HIP_MID])
    if torso < 1e-6:
        return None
    return (pts[:N_JOINTS] - pts[HIP_MID]) / torso
