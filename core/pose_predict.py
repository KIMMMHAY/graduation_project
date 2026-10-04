"""모델 포즈 추정 실행 + DB 저장 (웹앱 '바로 추정'·'미리 계산'과 predict_poses.py가 함께 쓴다)."""
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from core import pose as P
from core import poses as store
from core.pose_models import DEFAULT_MODEL, MODELS, Prediction, estimate_facing

Progress = Callable[[int, int, str], None]  # (완료 수, 전체 수, 메시지)


def absent_pose() -> list[dict]:
    return [{**k, "state": P.ABSENT} for k in P.default_pose()]


def save_prediction(model_name: str, image_id: str, pred: Prediction | None) -> None:
    """검출 성공은 done, 실패는 skipped(관절 전부 '없음')로 저장 → 검출률 계산에 쓴다."""
    if pred is None:
        store.save_pose(image_id, model_name, absent_pose(), None, P.SKIPPED, source=P.MODEL)
    else:
        store.save_pose(image_id, model_name, pred.keypoints, pred.facing, P.DONE, source=P.MODEL)


def predict_and_save(model, image_id: str, image_path: str) -> Prediction | None:
    pred = model.predict(image_path)
    save_prediction(model.name, image_id, pred)
    return pred


def pending_images(df: pd.DataFrame, model_name: str, overwrite: bool = False) -> pd.DataFrame:
    """아직 이 모델의 추정 결과가 없는 이미지."""
    if overwrite:
        return df
    done = set(store.model_poses(model_name)["image_id"])
    return df[~df["id"].isin(done)]


def run_batch(df: pd.DataFrame, model=None, limit: int | None = None, overwrite: bool = False,
              progress: Progress | None = None) -> dict:
    """미리 계산: 아직 없는 이미지만 추정해서 저장. 중간에 멈춰도 다시 실행하면 이어서 한다."""
    model = model or MODELS[DEFAULT_MODEL]()
    todo = pending_images(df, model.name, overwrite)
    if limit:
        todo = todo.head(limit)
    found = 0
    for n, (image_id, path) in enumerate(zip(todo["id"], todo["img_path"]), start=1):
        if predict_and_save(model, image_id, path) is not None:
            found += 1
        if progress:
            progress(n, len(todo), image_id)
    return {"model": model.name, "processed": len(todo), "detected": found}


def import_predictions(path: str | Path, model_name: str | None = None) -> dict:
    """tools/mediapipe_predict.py 등 다른 환경에서 만든 JSON을 DB에 넣는다.

    형식: {"model": "mediapipe-heavy", "predictions": {image_id: {"keypoints": [13개 {x,y,state,conf}] | null}}}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name = model_name or data["model"]
    known = set(store.load_image_ids())
    n = found = 0
    for image_id, item in data["predictions"].items():
        if image_id not in known:
            continue
        kps = item.get("keypoints")
        pred = None
        if kps:
            kps = P.validate_keypoints(kps)
            pred = Prediction(kps, float(sum(k.get("conf", 0) for k in kps) / len(kps)), estimate_facing(kps))
        save_prediction(name, image_id, pred)
        n += 1
        found += pred is not None
    return {"model": name, "processed": n, "detected": found}
