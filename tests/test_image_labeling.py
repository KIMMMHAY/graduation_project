"""이미지 단위 라벨링 규칙 (core/image_labeling.py) 단위 테스트."""
import pandas as pd

from core import image_labeling as IL


def labels(rows):
    return pd.DataFrame(rows, columns=["image_id", "tag_key", "value", "labeler"])


def test_values_to_save_checked_unchecked_held():
    checked = {"a": True, "b": True, "c": False, "d": False}
    assert IL.values_to_save(checked, held=set()) == {"a": 1, "b": 1, "c": 0, "d": 0}
    # 보류는 체크 여부와 상관없이 저장하지 않는다 (보류가 우선)
    assert IL.values_to_save(checked, held={"b", "c"}) == {"a": 1, "d": 0}
    assert IL.values_to_save(checked, held={"a", "b", "c", "d"}) == {}


def test_finished_partial_and_team():
    active = ["a", "b", "c"]
    df = labels([
        ("img1", "a", 1, "kim"), ("img1", "b", 0, "kim"), ("img1", "c", 0, "kim"),  # kim 끝냄
        ("img2", "a", 1, "kim"),                                                    # kim 일부만
        ("img3", "a", 0, "lee"), ("img3", "b", 0, "lee"), ("img3", "c", 1, "lee"),  # lee 끝냄
        ("img4", "old_tag", 1, "kim"),                                              # 사용 안 하는 태그만
    ])
    assert IL.finished_images(df, "kim", active) == {"img1"}
    assert IL.partial_images(df, "kim", active) == {"img2"}
    assert IL.team_finished_images(df, active) == {"img1", "img3"}


def test_new_tag_makes_finished_image_partial():
    """끝낸 이미지라도 태그가 추가되면 그 태그 행이 없으므로 '일부만 한 이미지'가 된다 (음성으로 기록되지 않음)."""
    df = labels([("img1", "a", 1, "kim"), ("img1", "b", 0, "kim")])
    assert IL.finished_images(df, "kim", ["a", "b"]) == {"img1"}
    assert IL.finished_images(df, "kim", ["a", "b", "new"]) == set()
    assert IL.partial_images(df, "kim", ["a", "b", "new"]) == {"img1"}
    assert "new" not in IL.my_values(df, "kim", "img1")


def test_my_values():
    df = labels([("img1", "a", 1, "kim"), ("img1", "b", 0, "kim"), ("img1", "a", 0, "lee")])
    assert IL.my_values(df, "kim", "img1") == {"a": 1, "b": 0}
    assert IL.my_values(df, "kim", "img9") == {}
