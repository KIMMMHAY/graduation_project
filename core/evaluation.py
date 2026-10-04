"""검색 결과 평가(비슷함/다름) 저장. 검색 방식마다 eval_{key}.csv 하나씩 쓴다."""
from datetime import datetime

import pandas as pd

from core.dataset import DATA_DIR

COLUMNS = ["timestamp", "evaluator", "method", "query_id", "result_id", "rank", "score", "verdict"]
SIMILAR, DIFFERENT = "similar", "different"
VERDICT_LABEL = {SIMILAR: "비슷함", DIFFERENT: "다름"}


def eval_path(method_key: str):
    return DATA_DIR / f"eval_{method_key}.csv"


def load_evals(method_key: str) -> pd.DataFrame:
    path = eval_path(method_key)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"query_id": str, "result_id": str, "evaluator": str}, encoding="utf-8-sig")


def save_verdict(method_key: str, evaluator: str, query_id: str, result_id: str,
                 rank: int, score: float, verdict: str) -> None:
    """같은 평가자가 같은 (쿼리, 결과)를 다시 누르면 덮어쓴다."""
    df = load_evals(method_key)
    same = (df["evaluator"] == evaluator) & (df["query_id"] == query_id) & (df["result_id"] == result_id)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "evaluator": evaluator, "method": method_key,
        "query_id": query_id, "result_id": result_id,
        "rank": rank, "score": score, "verdict": verdict,
    }
    df = pd.concat([df[~same], pd.DataFrame([row])], ignore_index=True)
    df.to_csv(eval_path(method_key), index=False, encoding="utf-8-sig")  # 엑셀에서 한글 깨짐 방지
