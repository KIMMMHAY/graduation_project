"""학습한 분류기의 예측 결과(predicted_tags.csv)를 태그 출처로 제공."""
import pandas as pd
import streamlit as st

from core.db import DBError
from core.tagging import PRED_CSV, PRED_THRESHOLD, file_mtime, load_tags, read_predictions
from methods.base import SearchMethod, TagSource


@st.cache_data(show_spinner=False)
def _load_predictions(mtime: float | None) -> pd.DataFrame:
    return read_predictions()


def load_predictions() -> pd.DataFrame:
    """학습을 다시 하면 파일 수정 시각이 바뀌어 자동으로 새로 읽는다."""
    return _load_predictions(file_mtime(PRED_CSV))


def _active_tag_names() -> tuple[tuple[str, str], ...]:
    """사용 중인 태그 (key, 이름). DB에 문제가 있어도 갤러리는 열려야 하므로 실패하면 빈 값."""
    try:
        return tuple((t.key, t.name) for t in load_tags())
    except DBError:
        return ()


@st.cache_data(show_spinner=False)
def _predicted_map(mtime: float | None, threshold: float,
                   tag_names: tuple[tuple[str, str], ...]) -> dict[str, list[tuple[str, float]]]:
    names = dict(tag_names)
    p = _load_predictions(mtime)
    p = p[(p["prob"] >= threshold) & p["tag_key"].isin(names)].sort_values("prob", ascending=False)
    out: dict[str, list[tuple[str, float]]] = {}
    for image_id, key, prob in zip(p["image_id"], p["tag_key"], p["prob"]):
        out.setdefault(image_id, []).append((names[key], float(prob)))
    return out


def predicted_tags_of(image_id: str, threshold: float = PRED_THRESHOLD) -> list[tuple[str, float]]:
    """(태그 표시명, 확률) 목록, 확률 높은 순."""
    mtime = file_mtime(PRED_CSV)
    if mtime is None:
        return []  # 학습 전에는 DB에 물어볼 필요도 없다
    return _predicted_map(mtime, threshold, _active_tag_names()).get(image_id, [])


class PredictedTags(SearchMethod, TagSource):
    key = "predicted"
    name = "예측 태그 (학습 모델)"
    description = f"팀 라벨로 학습한 분류기가 확률 {PRED_THRESHOLD} 이상으로 예측한 태그"

    def tags_of(self, idx: int) -> list[str]:
        return [name for name, _ in predicted_tags_of(self.df["id"].iat[idx])]
