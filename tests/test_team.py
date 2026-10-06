"""팀 활동 집계 (core/team.py) 단위 테스트 (DB 불필요)."""
import pandas as pd

from core import pose as P
from core import team

NOW = pd.Timestamp("2026-10-06T12:00:00Z")


def labels(rows):
    return pd.DataFrame(rows, columns=["image_id", "tag_key", "value", "labeler", "timestamp"])


def test_similar_names():
    known = ["김하영", "이도연", "Kim", "__test__"]
    assert team.similar_names("하영", known) == ["김하영"]          # 성을 빼고 씀
    assert team.similar_names("kim ", known) == ["Kim"]             # 대소문자·공백
    assert team.similar_names("박서준", known) == []
    assert team.similar_names("김하영", known) == []                 # 자기 자신은 제외
    assert team.suspicious_pairs(["김하영", "하영", "이도연"]) == [("김하영", "하영")]


def test_member_table_counts_and_excludes_tests_and_models():
    lab = labels([
        ("1", "a", 1, "kim", "2026-10-06T10:00:00+00:00"), ("1", "b", 0, "kim", "2026-10-06T10:00:00+00:00"),
        ("2", "a", 1, "kim", "2026-09-01T10:00:00+00:00"),                      # 7일 전, 일부만
        ("1", "a", 0, "__test__", "2026-10-06T11:00:00+00:00"),                  # 테스트 이름은 제외
    ])
    poses = pd.DataFrame([
        {"annotator": "lee", "source": P.MANUAL, "status": P.DONE, "updated_at": "2026-10-06T11:30:00+00:00"},
        {"annotator": "lee", "source": P.MANUAL, "status": P.DRAFT, "updated_at": "2026-10-05T11:30:00+00:00"},
        {"annotator": "dwpose", "source": P.MODEL, "status": P.DONE, "updated_at": "2026-10-06T11:59:00+00:00"},
    ])
    evals = pd.DataFrame([{"evaluator": "kim", "updated_at": "2026-10-06T09:00:00+00:00"}])
    t = team.member_table(lab, poses, evals, ["a", "b"], now=NOW).set_index("name")
    assert list(t.index) == ["lee", "kim"]                                       # 최근 활동 순, 모델·테스트 제외
    assert t.loc["kim", ["label_images", "labels", "labels_7d", "evals"]].tolist() == [1, 3, 2, 1]
    assert t.loc["lee", ["pose_done", "pose_draft", "labels"]].tolist() == [1, 1, 0]


def test_member_table_without_pose_tables():
    lab = labels([("1", "a", 1, "kim", "")])                                     # 시각이 비어 있어도 된다
    t = team.member_table(lab, None, None, ["a"], now=NOW)
    assert t["name"].tolist() == ["kim"] and pd.isna(t["last_active"].iat[0])


def test_labels_since():
    lab = labels([("1", "a", 1, "kim", "2026-10-06T10:00:00+00:00"), ("2", "a", 1, "kim", "2026-10-06T08:00:00+00:00"),
                  ("3", "a", 1, "__test__", "2026-10-06T10:00:00+00:00")])
    assert team.labels_since(lab, pd.Timestamp("2026-10-06T09:00:00Z")) == 1
    assert team.labels_since(lab, None) is None
