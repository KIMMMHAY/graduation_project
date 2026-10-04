"""이미지 단위 라벨링 규칙 (순수 함수, Streamlit·DB 의존 없음).

- 저장: 화면에 보이는 사용 중 태그 전체를 한 번에. 체크 = 1, 해제 = 0, 보류 = 저장하지 않음(기존 내 라벨 유지).
- 끝낸 이미지: 현재 사용 중인 모든 태그에 내 라벨이 있는 이미지. 나중에 추가된 태그는 행이 없으므로 '미검토'.
"""
import pandas as pd

from core.tagging import NEGATIVE, POSITIVE


def values_to_save(checked: dict[str, bool], held: set[str]) -> dict[str, int]:
    """체크박스 상태 → 저장할 값. 보류한 태그는 체크 여부와 상관없이 뺀다(보류가 우선)."""
    return {key: POSITIVE if on else NEGATIVE for key, on in checked.items() if key not in held}


def labeled_tags(labels: pd.DataFrame, labeler: str, active_keys: list[str]) -> dict[str, set[str]]:
    """image_id → 내가 라벨을 붙인 사용 중 태그 집합."""
    mine = labels[(labels["labeler"] == labeler) & labels["tag_key"].isin(active_keys)]
    return mine.groupby("image_id")["tag_key"].agg(set).to_dict()


def finished_images(labels: pd.DataFrame, labeler: str, active_keys: list[str]) -> set[str]:
    need = set(active_keys)
    return {i for i, keys in labeled_tags(labels, labeler, active_keys).items() if need <= keys}


def partial_images(labels: pd.DataFrame, labeler: str, active_keys: list[str]) -> set[str]:
    """일부 태그만 라벨링한 이미지 (주로 태그가 나중에 추가된 경우)."""
    need = set(active_keys)
    return {i for i, keys in labeled_tags(labels, labeler, active_keys).items() if keys and not need <= keys}


def team_finished_images(labels: pd.DataFrame, active_keys: list[str]) -> set[str]:
    """팀원 중 한 명이라도 끝낸 이미지."""
    done: set[str] = set()
    for labeler in labels["labeler"].dropna().unique():
        done |= finished_images(labels, labeler, active_keys)
    return done


def my_values(labels: pd.DataFrame, labeler: str, image_id: str) -> dict[str, int]:
    mine = labels[(labels["labeler"] == labeler) & (labels["image_id"] == image_id)]
    return dict(zip(mine["tag_key"], mine["value"].astype(int)))
