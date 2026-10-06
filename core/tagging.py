"""팀 태그(Supabase tags), 라벨(Supabase labels), 예측 결과(predicted_tags.csv) 읽기/쓰기.

태그와 라벨은 모두 Supabase가 기준이다. DB 함수는 실패하면 db.DBError(한국어 메시지)를 던진다.
"""
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from core import db
from core.dataset import DATA_DIR

LABELS_CSV = DATA_DIR / "labels.csv"   # 예전 로컬 라벨. DB 이전(migrate_to_supabase.py)에만 쓴다
TAGS_YAML = DATA_DIR / "tags.yaml"     # 예전 태그 정의. DB 이전에만 쓴다
MODELS_DIR = DATA_DIR / "models"
REPORT_CSV = DATA_DIR / "train_report.csv"
PRED_CSV = DATA_DIR / "predicted_tags.csv"
EMB_NPY = DATA_DIR / "embeddings.npy"
EMB_IDS_CSV = DATA_DIR / "embeddings_ids.csv"

PRED_THRESHOLD = 0.5  # 갤러리에서 예측 태그로 표시할 최소 확률
LABEL_COLUMNS = ["image_id", "tag_key", "value", "labeler", "timestamp"]
POSITIVE, NEGATIVE = 1, 0
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
TAGS_CACHE_SECONDS = 30  # 다른 팀원이 바꾼 태그는 최대 30초 뒤에 보인다 (내가 바꾼 건 즉시)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except db.DBError:
        raise
    except Exception as e:
        raise db.DBError(db.friendly_error(e)) from e


# ---------- 태그 ----------

@dataclass(frozen=True)
class TagDef:
    key: str
    name: str
    group: str = ""
    definition: str = ""
    booru_hint: str | None = None
    active: bool = True


class TagError(ValueError):
    """입력값이 잘못된 경우 (사용자에게 그대로 보여줄 한국어 메시지)."""


_tags_cache: tuple[float, list[TagDef]] | None = None


def invalidate_tags_cache() -> None:
    global _tags_cache
    _tags_cache = None


def load_tags(include_inactive: bool = False) -> list[TagDef]:
    """기본은 '사용 중'인 태그만. 태그 관리 페이지에서는 include_inactive=True."""
    global _tags_cache
    if _tags_cache is None or time.monotonic() - _tags_cache[0] > TAGS_CACHE_SECONDS:
        rows = _db_call(db.fetch_all, "tags", "key, name, tag_group, definition, booru_hint, active, sort_order")
        rows.sort(key=lambda r: (r["sort_order"], r["key"]))
        tags = [TagDef(key=r["key"], name=r["name"], group=r["tag_group"] or "", definition=r["definition"] or "",
                       booru_hint=r["booru_hint"] or None, active=bool(r["active"])) for r in rows]
        _tags_cache = (time.monotonic(), tags)
    tags = _tags_cache[1]
    return tags if include_inactive else [t for t in tags if t.active]


def _clean_hint(booru_hint: str | None) -> str | None:
    hint = (booru_hint or "").strip().lower().replace(" ", "_")
    return hint or None


def add_tag(key: str, name: str, group: str, definition: str, booru_hint: str | None = None) -> None:
    key, name, group, definition = key.strip(), name.strip(), group.strip(), definition.strip()
    if not KEY_PATTERN.match(key):
        raise TagError("key는 영문 소문자로 시작하고 영문 소문자·숫자·밑줄(_)만 쓸 수 있습니다 (최대 40자). 예: low_angle")
    if not name:
        raise TagError("표시 이름을 입력해 주세요.")
    if not group:
        raise TagError("분류 축을 선택하거나 입력해 주세요.")
    existing = load_tags(include_inactive=True)
    if any(t.key == key for t in existing):
        raise TagError(f"key '{key}'는 이미 있습니다. (사용 안 함 태그도 포함)")
    row = {"key": key, "name": name, "tag_group": group, "definition": definition,
           "booru_hint": _clean_hint(booru_hint), "active": True, "sort_order": len(existing)}
    _db_call(lambda: db.get_client().table("tags").insert(row, returning="minimal").execute())
    invalidate_tags_cache()


def update_tag(key: str, name: str, definition: str) -> None:
    name = name.strip()
    if not name:
        raise TagError("표시 이름은 비워 둘 수 없습니다.")
    _db_call(lambda: db.get_client().table("tags")
             .update({"name": name, "definition": definition.strip(), "updated_at": _now()}).eq("key", key).execute())
    invalidate_tags_cache()


def set_tag_active(key: str, active: bool) -> None:
    _db_call(lambda: db.get_client().table("tags").update({"active": active, "updated_at": _now()}).eq("key", key).execute())
    invalidate_tags_cache()


def delete_tag(key: str) -> None:
    """라벨이 하나도 없는 태그만 삭제할 수 있다 (DB 정책과 외래키로도 막혀 있음)."""
    n = _db_call(lambda: db.get_client().table("labels").select("id", count="exact").eq("tag_key", key).limit(1).execute()).count
    if n:
        raise TagError(f"라벨이 {n}건 붙어 있어 삭제할 수 없습니다. '사용 안 함'으로 바꿔 주세요.")
    deleted = _db_call(lambda: db.get_client().table("tags").delete().eq("key", key).execute()).data
    if not deleted:
        raise TagError("삭제되지 않았습니다. 그 사이 라벨이 추가되었거나 권한이 없습니다.")
    invalidate_tags_cache()


# ---------- 라벨 ----------

def load_labels() -> pd.DataFrame:
    """컬럼: image_id, tag_key, value, labeler, timestamp."""
    rows = _db_call(db.fetch_all, "labels", "image_id, tag_key, value, labeler, updated_at")
    df = pd.DataFrame(rows, columns=["image_id", "tag_key", "value", "labeler", "updated_at"])
    return df.rename(columns={"updated_at": "timestamp"}).astype({"image_id": str, "value": int})


def save_label(image_id: str, tag_key: str, value: int, labeler: str) -> None:
    """같은 (image_id, tag_key, labeler)는 덮어쓴다."""
    row = {"image_id": str(image_id), "tag_key": tag_key, "value": int(value), "labeler": labeler, "updated_at": _now()}
    _db_call(db.upsert, "labels", [row], on_conflict="image_id,tag_key,labeler")


def save_labels(image_id: str, values: dict[str, int], labeler: str) -> None:
    """이미지 한 장의 여러 태그 라벨을 한 번에 저장. 같은 (image_id, tag_key, labeler)는 덮어쓴다.

    여러 행을 한 번의 요청으로 보내므로 DB에서 하나의 INSERT ... ON CONFLICT 문으로 실행된다 → 전부 저장되거나 전부 실패.
    """
    if not values:
        return
    if not labeler.strip():
        raise TagError("이름이 비어 있습니다.")
    now = _now()
    rows = [{"image_id": str(image_id), "tag_key": key, "value": int(v), "labeler": labeler, "updated_at": now}
            for key, v in values.items()]
    _db_call(lambda: db.get_client().table("labels")
             .upsert(rows, on_conflict="image_id,tag_key,labeler", returning="minimal").execute())


def load_labels_csv(path=LABELS_CSV) -> pd.DataFrame:
    """예전 로컬 CSV 라벨 (DB 이전용)."""
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS).astype({"value": int})
    return pd.read_csv(path, dtype={"image_id": str, "tag_key": str, "labeler": str}, encoding="utf-8-sig")


def consolidate(labels: pd.DataFrame) -> pd.DataFrame:
    """여러 라벨러의 라벨을 (image_id, tag_key)당 하나로. 다수결, 동률이면 제외."""
    if labels.empty:
        return pd.DataFrame(columns=["image_id", "tag_key", "value"])
    mean = labels.groupby(["image_id", "tag_key"])["value"].mean().reset_index()
    mean = mean[mean["value"] != 0.5]
    mean["value"] = (mean["value"] > 0.5).astype(int)
    return mean


# ---------- 예측 ----------

def read_predictions() -> pd.DataFrame:
    """이 PC에서 마지막으로 학습한 예측 (predicted_tags.csv)."""
    if not PRED_CSV.exists():
        return pd.DataFrame(columns=["image_id", "tag_key", "prob"])
    return pd.read_csv(PRED_CSV, dtype={"image_id": str, "tag_key": str}, encoding="utf-8-sig")


def publish_predictions(preds: pd.DataFrame, model: str) -> int:
    """예측을 DB(predictions)에 올려 팀 전체가 같은 결과를 보게 한다. 올린 행 수를 돌려준다.

    - 같은 실행의 행은 모두 같은 predicted_at을 가진다. 읽을 때는 가장 최근 실행의 행만 쓰므로,
      이번에 학습하지 않은 태그의 예전 예측은 자동으로 빠진다 (팀원 키로는 삭제 권한이 없어서 이 방식을 쓴다).
    - 한 번의 요청으로 보낸다 → 전부 저장되거나 전부 실패 (일부만 바뀐 상태가 최신으로 보이지 않게).
    - DB에 없는 이미지(로컬에만 있는 이미지)는 외래키 오류가 나므로 뺀다.
    """
    known = {r["id"] for r in _db_call(db.fetch_all, "images", "id")}
    p = preds[preds["image_id"].astype(str).isin(known)]
    run_at = _now()
    rows = [{"image_id": str(i), "tag_key": k, "prob": round(min(max(float(v), 0.0), 1.0), 4),
             "model": model, "predicted_at": run_at}
            for i, k, v in zip(p["image_id"], p["tag_key"], p["prob"])]
    if rows:
        _db_call(db.upsert, "predictions", rows, on_conflict="image_id,tag_key", chunk=len(rows))
    return len(rows)


def load_shared_predictions() -> pd.DataFrame:
    """DB의 팀 공유 예측 중 가장 최근 학습 실행의 것. 컬럼: image_id, tag_key, prob, model, predicted_at."""
    cols = ["image_id", "tag_key", "prob", "model", "predicted_at"]
    df = pd.DataFrame(_db_call(db.fetch_all, "predictions", ", ".join(cols)), columns=cols)
    if df.empty:
        return df
    at = pd.to_datetime(df["predicted_at"], utc=True, format="ISO8601")  # 문자열 비교는 소수점 자리수에 따라 틀릴 수 있다
    df = df[at == at.max()].reset_index(drop=True)
    return df.astype({"image_id": str, "tag_key": str, "prob": float})


def file_mtime(path) -> float | None:
    """캐시 키로 쓰기 위한 수정 시각. 파일이 바뀌면 캐시가 자동으로 무효화된다."""
    return path.stat().st_mtime if path.exists() else None
