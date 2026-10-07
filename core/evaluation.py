"""이미지 유사 검색 결과 평가(비슷함/다름). Supabase search_evals 테이블에 저장한다.

예전에는 각자 PC의 eval_{key}.csv에 저장했지만, 배포 서버는 재시작 때 파일이 지워지고 팀원끼리 모이지도 않아
DB로 옮겼다. 예전 CSV는 migrate_to_supabase.py가 올린다 (load_evals_csv, csv_to_rows).
DB 함수는 실패하면 db.DBError(한국어 메시지)를 던진다.
"""
from datetime import datetime, timezone

import pandas as pd

from core import db
from core.dataset import DATA_DIR

TABLE = "search_evals"
COLUMNS = ["timestamp", "evaluator", "method", "query_id", "result_id", "rank", "score", "verdict"]
SIMILAR, DIFFERENT = "similar", "different"
VERDICT_LABEL = {SIMILAR: "비슷함", DIFFERENT: "다름"}
_DB_COLUMNS = "method, evaluator, query_id, result_id, rank, score, verdict, updated_at"


def eval_path(method_key: str):
    """예전 로컬 평가 파일 (DB 이전용)."""
    return DATA_DIR / f"eval_{method_key}.csv"


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[c.strip() for c in _DB_COLUMNS.split(",")]).rename(columns={"updated_at": "timestamp"})
    return df[COLUMNS].astype({"query_id": str, "result_id": str, "evaluator": str})


def load_evals(method_key: str) -> pd.DataFrame:
    """이 검색 방식의 팀 전체 평가. 컬럼: COLUMNS."""
    def fetch():
        rows, start = [], 0
        while True:  # 다른 방식의 평가까지 읽지 않도록 method로 거른 채 페이지를 넘긴다
            page = (db.get_client().table(TABLE).select(_DB_COLUMNS).eq("method", method_key)
                    .order("id").range(start, start + db.PAGE_SIZE - 1).execute().data)
            rows += page
            if len(page) < db.PAGE_SIZE:
                return rows
            start += db.PAGE_SIZE
    return _to_frame(db.call(fetch))


def save_verdict(method_key: str, evaluator: str, query_id: str, result_id: str,
                 rank: int, score: float, verdict: str) -> None:
    """같은 평가자가 같은 (쿼리, 결과)를 다시 누르면 덮어쓴다."""
    if verdict not in VERDICT_LABEL:
        raise ValueError(verdict)
    if not evaluator.strip():
        raise db.DBError("이름이 비어 있습니다.")
    row = {"method": method_key, "evaluator": evaluator, "query_id": str(query_id), "result_id": str(result_id),
           "rank": int(rank), "score": float(score), "verdict": verdict,
           "updated_at": datetime.now(timezone.utc).isoformat()}
    db.call(db.upsert, TABLE, [row], on_conflict="method,evaluator,query_id,result_id")


# ---------- 예전 CSV (DB 이전용) ----------

def load_evals_csv(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"query_id": str, "result_id": str, "evaluator": str}, encoding="utf-8-sig")


def csv_to_rows(df: pd.DataFrame, method_key: str, known_images: set[str]) -> tuple[list[dict], int]:
    """예전 CSV → DB 행. (행 목록, 건너뛴 수). DB에 없는 이미지·이름 없는 평가·알 수 없는 판정은 건너뛴다."""
    df = df.drop_duplicates(subset=["evaluator", "query_id", "result_id"], keep="last")
    ok = (df["query_id"].isin(known_images) & df["result_id"].isin(known_images)
          & df["evaluator"].fillna("").str.strip().ne("") & df["verdict"].isin(VERDICT_LABEL))
    rows = []
    for r in df[ok].itertuples():
        row = {"method": method_key, "evaluator": r.evaluator, "query_id": r.query_id, "result_id": r.result_id,
               "rank": None if pd.isna(r.rank) else int(r.rank), "score": None if pd.isna(r.score) else float(r.score),
               "verdict": r.verdict}
        if isinstance(r.timestamp, str) and r.timestamp:
            # 예전 CSV 시각은 그 PC의 현지 시각(시간대 없음). 팀이 한국에서 썼으므로 KST로 본다
            ts = pd.Timestamp(r.timestamp)
            ts = ts.tz_localize("Asia/Seoul") if ts.tzinfo is None else ts
            row["updated_at"] = row["created_at"] = ts.isoformat()
        rows.append(row)
    return rows, int((~ok).sum())
