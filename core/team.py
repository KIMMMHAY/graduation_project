"""팀 활동 집계 (팀 현황 페이지, 사이드바 이름 확인).

집계 함수는 순수 pandas(테스트 가능), fetch_known_names()만 DB를 읽는다.
이름이 '__'로 시작하는 기록(테스트용 `__test__` 등)과 모델 결과(poses.source='model')는 사람 활동이 아니므로 뺀다.
"""
import difflib

import pandas as pd

from core import db
from core import pose as P
from core.image_labeling import finished_images

MEMBER_COLUMNS = ["name", "label_images", "labels", "labels_7d", "pose_done", "pose_draft", "evals", "last_active"]


def _is_person(names: pd.Series) -> pd.Series:
    return names.notna() & ~names.astype(str).str.startswith("__")


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


def _norm(name: str) -> str:
    return "".join(name.split()).lower()


def similar_names(name: str, known) -> list[str]:
    """같은 사람이 다르게 쓴 것 같은 기존 이름 (예: '하영' ↔ '김하영', 'Kim' ↔ 'kim', 긴 이름의 오타)."""
    key = _norm(name)
    out = []
    for k in known:
        nk = _norm(k)
        if k == name or not nk or not key:
            continue
        if nk == key or (min(len(nk), len(key)) >= 2 and (nk in key or key in nk)) \
                or difflib.SequenceMatcher(None, nk, key).ratio() >= 0.75:
            out.append(k)
    return out


def suspicious_pairs(names) -> list[tuple[str, str]]:
    """기존 이름 중 같은 사람일 수 있는 쌍 (팀 현황에서 통일을 권하는 용도)."""
    names = sorted(set(names))
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if b in similar_names(a, [b])]


def fetch_known_names() -> list[str]:
    """DB에 기록이 있는 사람 이름 (라벨러, 포즈 작성자, 포즈 검색 평가자). 없는 테이블은 건너뛴다."""
    names: set[str] = set()
    sources = [("labels", "labeler", None), ("poses", "annotator, source", "annotator"), ("pose_evals", "evaluator", None)]
    for table, cols, col in sources:
        try:
            rows = db.call(db.fetch_all, table, cols)
        except db.DBError:
            continue
        col = col or cols
        names |= {r[col] for r in rows if r.get(col) and r.get("source") != P.MODEL}
    return sorted(n for n in names if not n.startswith("__"))


def member_table(labels: pd.DataFrame, poses: pd.DataFrame | None, evals: pd.DataFrame | None,
                 active_keys: list[str], now: pd.Timestamp | None = None) -> pd.DataFrame:
    """팀원별 활동. 컬럼: MEMBER_COLUMNS (last_active는 UTC 시각, 최근 활동 순)."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    week_ago = now - pd.Timedelta(days=7)
    lab = labels[_is_person(labels["labeler"])]
    hum = pd.DataFrame(columns=["annotator", "source", "status", "updated_at"]) if poses is None else poses
    hum = hum[(hum["source"] != P.MODEL) & _is_person(hum["annotator"])]
    ev = pd.DataFrame(columns=["evaluator", "updated_at"]) if evals is None else evals
    ev = ev[_is_person(ev["evaluator"])]

    rows = []
    for name in sorted(set(lab["labeler"]) | set(hum["annotator"]) | set(ev["evaluator"])):
        mine_l, mine_p, mine_e = lab[lab["labeler"] == name], hum[hum["annotator"] == name], ev[ev["evaluator"] == name]
        label_times = _ts(mine_l["timestamp"])
        times = pd.concat([label_times, _ts(mine_p["updated_at"]), _ts(mine_e["updated_at"])]).dropna()
        rows.append({
            "name": name,
            "label_images": len(finished_images(lab, name, active_keys)),
            "labels": len(mine_l),
            "labels_7d": int((label_times >= week_ago).sum()),
            "pose_done": int((mine_p["status"] == P.DONE).sum()),
            "pose_draft": int((mine_p["status"] == P.DRAFT).sum()),
            "evals": len(mine_e),
            "last_active": times.max() if len(times) else pd.NaT,
        })
    out = pd.DataFrame(rows, columns=MEMBER_COLUMNS)
    return out.sort_values("last_active", ascending=False, na_position="last").reset_index(drop=True)


def labels_since(labels: pd.DataFrame, when: pd.Timestamp | None) -> int | None:
    """마지막 학습 이후 새로 저장·수정된 라벨 수 (사람 라벨만). 학습한 적이 없으면 None."""
    if when is None:
        return None
    lab = labels[_is_person(labels["labeler"])]
    return int((_ts(lab["timestamp"]) > when).sum())
