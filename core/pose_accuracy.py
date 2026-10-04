"""모델 추정 vs 사람이 확정한 포즈 정확도 (순수 pandas/numpy, DB·Streamlit 의존 없음).

- 비교 대상: 사람이 완료(done)한 포즈(manual, model_corrected) × 같은 이미지의 모델 추정
- 오차: 관절 위치 거리 / 사람 포즈의 몸통 길이 (가로세로 비율 보정)
- PCK@0.2: 오차가 몸통 길이의 20% 이내인 비율. 사람이 '없음'으로 표시한 관절은 제외,
  모델이 '없음'으로 낸 관절은 오답으로 센다.
- 검출률: 사람이 인물을 확정한 이미지 중 모델이 인물을 찾은 비율
"""
import numpy as np
import pandas as pd

from core import pose as P

CONF_BINS = [0.0, 0.3, 0.5, 0.7, 1.01]
CONF_LABELS = ["0~0.3", "0.3~0.5", "0.5~0.7", "0.7~1"]


def compare(human: pd.DataFrame, models: pd.DataFrame, aspects: dict[str, float]) -> pd.DataFrame:
    """관절 단위 비교표: model, human_source, image_id, annotator, joint, error, correct, conf."""
    human = human[(human["status"] == P.DONE) & human["source"].isin(P.SEARCHABLE_SOURCES)]
    rows = []
    for m in models.itertuples():
        for h in human[human["image_id"] == m.image_id].itertuples():
            base = {"model": m.annotator, "human_source": h.source, "image_id": h.image_id, "annotator": h.annotator}
            if m.status != P.DONE:  # 모델이 인물을 못 찾음
                rows.append({**base, "detected": False})
                continue
            res = P.joint_errors(P.validate_keypoints(h.keypoints), P.validate_keypoints(m.keypoints),
                                 aspects.get(h.image_id, 0.75))
            if res is None:
                continue
            errors, correct = res
            for j, (key, name) in enumerate(P.JOINTS):
                if np.isnan(correct[j]):
                    continue
                rows.append({**base, "detected": True, "joint": key, "joint_name": name, "error": errors[j],
                             "correct": correct[j], "conf": m.keypoints[j].get("conf")})
    return pd.DataFrame(rows, columns=["model", "human_source", "image_id", "annotator", "detected", "joint",
                                       "joint_name", "error", "correct", "conf"])


def summary(cmp: pd.DataFrame) -> pd.DataFrame:
    """모델별 요약: 비교 이미지 수, 검출률, 전체 PCK, 평균 오차."""
    if cmp.empty:
        return pd.DataFrame()
    per_image = cmp.groupby(["model", "image_id", "annotator"])["detected"].first().reset_index()
    det = per_image.groupby("model")["detected"].agg(["size", "mean"])
    joints = cmp[cmp["detected"] == True]  # noqa: E712
    acc = joints.groupby("model").agg(pck=("correct", "mean"), mean_error=("error", "mean"))
    out = det.join(acc, how="left").rename(columns={"size": "images", "mean": "detection_rate"})
    return out.reset_index()


def per_joint(cmp: pd.DataFrame) -> pd.DataFrame:
    """관절별 평균 오차와 PCK (모델 × 관절)."""
    joints = cmp[cmp["detected"] == True]  # noqa: E712
    if joints.empty:
        return pd.DataFrame()
    t = joints.groupby(["model", "joint"]).agg(mean_error=("error", "mean"), pck=("correct", "mean"),
                                               n=("correct", "size")).reset_index()
    t["order"] = t["joint"].map(P.J)
    t["joint_name"] = t["joint"].map(dict(P.JOINTS))
    return t.sort_values(["model", "order"]).drop(columns="order")


def by_confidence(cmp: pd.DataFrame) -> pd.DataFrame:
    """모델 신뢰도 구간별 PCK — 보임/가려짐/없음 기준값을 조정할 때 본다."""
    joints = cmp[(cmp["detected"] == True) & cmp["conf"].notna()].copy()  # noqa: E712
    if joints.empty:
        return pd.DataFrame()
    joints["conf_bin"] = pd.cut(joints["conf"].astype(float), CONF_BINS, labels=CONF_LABELS, right=False)
    return joints.groupby(["model", "conf_bin"], observed=True).agg(pck=("correct", "mean"), n=("correct", "size")).reset_index()
