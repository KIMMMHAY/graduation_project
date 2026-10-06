"""학습한 분류기의 예측 결과를 태그 출처로 제공.

출처 우선순위: DB의 팀 공유 예측(가장 최근 학습 실행) → 없으면 이 PC의 predicted_tags.csv.
한 명이 학습하면 다른 팀원도 다시 학습하지 않고 같은 예측(AI 추천, 예측 확인, 갤러리 표시)을 본다.
"""
import pandas as pd
import streamlit as st

from core.db import DBError
from core.tagging import PRED_CSV, PRED_THRESHOLD, file_mtime, load_shared_predictions, load_tags, read_predictions
from methods.base import SearchMethod, TagSource

SHARED_TTL = 60  # 다른 팀원이 학습한 결과는 최대 1분 뒤에 보인다 (내가 학습한 건 즉시)


@st.cache_data(show_spinner=False)
def _load_predictions(mtime: float | None) -> pd.DataFrame:
    return read_predictions()


@st.cache_data(show_spinner=False, ttl=SHARED_TTL)
def _load_shared() -> tuple[str, pd.DataFrame, str] | None:
    """(버전, 예측, 설명). DB 문제나 공유된 예측이 없으면 None → 로컬 파일을 쓴다."""
    try:
        df = load_shared_predictions()
    except DBError:
        return None
    if df.empty:
        return None
    at = pd.to_datetime(df["predicted_at"].iat[0], utc=True).tz_convert("Asia/Seoul")
    return f"shared:{df['predicted_at'].iat[0]}", df[["image_id", "tag_key", "prob"]], \
        f"팀 공유 · {df['model'].iat[0]} · {at:%m/%d %H:%M}"


def _current() -> tuple[str, pd.DataFrame, str]:
    if (shared := _load_shared()) is not None:
        return shared
    mtime = file_mtime(PRED_CSV)
    return f"local:{mtime}", _load_predictions(mtime), "이 PC에서 학습한 결과" if mtime else "없음"


def load_predictions() -> pd.DataFrame:
    """컬럼: image_id, tag_key, prob. 학습을 다시 하면 자동으로 새로 읽는다."""
    return _current()[1]


def predictions_source() -> str:
    """화면 안내용: 지금 보이는 예측이 어디서 왔는지."""
    return _current()[2]


def predictions_time() -> pd.Timestamp | None:
    """지금 보이는 예측을 만든 학습 시각(UTC). 예측이 없으면 None."""
    kind, _, value = _current()[0].partition(":")
    if kind == "shared":
        return pd.Timestamp(value).tz_convert("UTC")
    return None if value in ("", "None") else pd.Timestamp(float(value), unit="s", tz="UTC")


def refresh_predictions() -> None:
    """학습 직후 호출: 공유 예측을 기다리지 않고 바로 다시 읽는다."""
    _load_shared.clear()


def _active_tag_names() -> tuple[tuple[str, str], ...]:
    """사용 중인 태그 (key, 이름). DB에 문제가 있어도 갤러리는 열려야 하므로 실패하면 빈 값."""
    try:
        return tuple((t.key, t.name) for t in load_tags())
    except DBError:
        return ()


@st.cache_data(show_spinner=False)
def _predicted_map(version: str, threshold: float, tag_names: tuple[tuple[str, str], ...],
                   _preds: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    # _preds는 캐시 키에서 빠진다 (이름이 _로 시작). 내용이 바뀌면 version이 바뀐다
    names = dict(tag_names)
    p = _preds[(_preds["prob"] >= threshold) & _preds["tag_key"].isin(names)].sort_values("prob", ascending=False)
    out: dict[str, list[tuple[str, float]]] = {}
    for image_id, key, prob in zip(p["image_id"], p["tag_key"], p["prob"]):
        out.setdefault(image_id, []).append((names[key], float(prob)))
    return out


def predicted_tags_of(image_id: str, threshold: float = PRED_THRESHOLD) -> list[tuple[str, float]]:
    """(태그 표시명, 확률) 목록, 확률 높은 순."""
    version, preds, _ = _current()
    if preds.empty:
        return []  # 학습 전에는 태그 이름을 물어볼 필요도 없다
    return _predicted_map(version, threshold, _active_tag_names(), preds).get(image_id, [])


class PredictedTags(SearchMethod, TagSource):
    key = "predicted"
    name = "예측 태그 (학습 모델)"
    description = f"팀 라벨로 학습한 분류기가 확률 {PRED_THRESHOLD} 이상으로 예측한 태그"

    def tags_of(self, idx: int) -> list[str]:
        return [name for name, _ in predicted_tags_of(self.df["id"].iat[idx])]
