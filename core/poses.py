"""포즈(poses)·포즈 검색 평가(pose_evals) DB 입출력. 실패하면 db.DBError(한국어 메시지)."""
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from core import db
from core import pose as P

POSE_COLUMNS = "image_id, person_index, annotator, source, facing, keypoints, status, updated_at"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_poses() -> pd.DataFrame:
    """모든 포즈 (작성자·상태 무관). 수백~수천 건 규모라 한 번에 읽어 pandas로 거른다."""
    rows = db.call(db.fetch_all, "poses", POSE_COLUMNS)
    df = pd.DataFrame(rows, columns=[c.strip() for c in POSE_COLUMNS.split(",")])
    return df.astype({"image_id": str})


def get_pose(image_id: str, annotator: str, person_index: int = 0) -> dict | None:
    rows = db.call(lambda: db.get_client().table("poses").select(POSE_COLUMNS)
                   .eq("image_id", image_id).eq("annotator", annotator).eq("person_index", person_index)
                   .limit(1).execute().data)
    return rows[0] if rows else None


def save_pose(image_id: str, annotator: str, keypoints: list[dict], facing: str | None, status: str,
              source: str = P.MANUAL, person_index: int = 0) -> None:
    """같은 (이미지, 인물 번호, 작성자)는 덮어쓴다."""
    if not annotator.strip():
        raise P.PoseError("작성자 이름이 비어 있습니다.")
    if status not in (P.DRAFT, P.DONE, P.SKIPPED):
        raise P.PoseError(f"상태 '{status}'는 쓸 수 없습니다.")
    if facing is not None and facing not in P.FACINGS:
        raise P.PoseError(f"방향 '{facing}'은 쓸 수 없습니다.")
    if status == P.DONE and facing is None and source != P.MODEL:  # AI는 방향을 못 정할 수도 있다
        raise P.PoseError("완료하려면 몸이 향한 방향(정면/측면/뒷모습)을 골라 주세요.")
    row = {"image_id": str(image_id), "person_index": person_index, "annotator": annotator, "source": source,
           "facing": facing, "keypoints": P.validate_keypoints(keypoints), "status": status, "updated_at": _now()}
    db.call(db.upsert, "poses", [row], on_conflict="image_id,person_index,annotator")


def model_poses(model_name: str | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """모델 추정 결과 (source='model'). model_name이 없으면 모든 모델."""
    df = load_poses() if df is None else df
    df = df[df["source"] == P.MODEL]
    return df if model_name is None else df[df["annotator"] == model_name]


def load_image_ids() -> list[str]:
    return [r["id"] for r in db.call(db.fetch_all, "images", "id")]


def searchable_candidates(aspects: dict[str, float]) -> list[P.Candidate]:
    """검색 대상: 완료(done) + 사람이 확정한 포즈(manual / model_corrected). aspects: image_id → 가로/세로."""
    df = load_poses()
    df = df[(df["status"] == P.DONE) & df["source"].isin(P.SEARCHABLE_SOURCES) & df["image_id"].isin(aspects)]
    return [P.Candidate(image_id=r.image_id, keypoints=P.validate_keypoints(r.keypoints), aspect=aspects[r.image_id],
                        facing=r.facing, annotator=r.annotator) for r in df.itertuples()]


def progress(df: pd.DataFrame | None = None) -> dict:
    """완료 이미지 수, 건너뛴 이미지 수, 작성자별 완료 수 (사람이 작성한 것만)."""
    df = load_poses() if df is None else df
    human = df[df["source"] != P.MODEL]
    return {
        "done_images": int(human.loc[human["status"] == P.DONE, "image_id"].nunique()),
        "skipped_images": int(human.loc[human["status"] == P.SKIPPED, "image_id"].nunique()),
        "by_annotator": human[human["status"] == P.DONE].groupby("annotator").size().sort_values(ascending=False),
    }


# ---------- 검색 평가 ----------

def query_hash(query: dict, options: dict) -> str:
    return hashlib.md5(json.dumps({"q": query, "o": options}, sort_keys=True).encode()).hexdigest()


def save_pose_eval(evaluator: str, query: dict, options: dict, result_image_id: str, result_annotator: str,
                   rank: int, score: float, flip: bool, verdict: str) -> None:
    if verdict not in ("similar", "different"):
        raise ValueError(verdict)
    row = {"evaluator": evaluator, "query_hash": query_hash(query, options), "query": query, "options": options,
           "result_image_id": str(result_image_id), "result_annotator": result_annotator, "rank": rank,
           "score": round(float(score), 3), "flip": bool(flip), "verdict": verdict, "updated_at": _now()}
    db.call(db.upsert, "pose_evals", [row], on_conflict="evaluator,query_hash,result_image_id")


def load_pose_evals() -> pd.DataFrame:
    cols = ["evaluator", "query_hash", "options", "result_image_id", "rank", "score", "flip", "verdict", "updated_at"]
    return pd.DataFrame(db.call(db.fetch_all, "pose_evals", ", ".join(cols)), columns=cols)
