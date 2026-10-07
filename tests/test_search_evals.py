"""유사 검색 평가(search_evals) 저장·읽기·예전 CSV 변환 단위 테스트 (실제 DB 없이 가짜 클라이언트 사용)."""
import pandas as pd
import pytest

from core import db
from core import evaluation as E


class FakeTable:
    """supabase 클라이언트의 table() 체인 중 evaluation.py가 쓰는 부분만 흉내 낸다."""

    def __init__(self, store: list[dict]):
        self.store, self.filters, self.rng = store, {}, None

    def select(self, _cols):
        return self

    def eq(self, col, value):
        self.filters[col] = value
        return self

    def order(self, _col):
        return self

    def range(self, start, end):
        self.rng = (start, end)
        return self

    def upsert(self, rows, on_conflict, **_):
        keys = on_conflict.split(",")
        for row in rows:
            self.store[:] = [r for r in self.store if any(r[k] != row[k] for k in keys)] + [row]
        return self

    def execute(self):
        rows = [r for r in self.store if all(r.get(k) == v for k, v in self.filters.items())]
        if self.rng:
            rows = rows[self.rng[0]:self.rng[1] + 1]
        return type("Res", (), {"data": rows})()


@pytest.fixture
def fake_db(monkeypatch):
    store: list[dict] = []
    client = type("Client", (), {"table": lambda self, name: FakeTable(store)})()
    monkeypatch.setattr(db, "get_client", lambda: client)
    return store


def test_save_and_reload_overwrites_same_click(fake_db):
    E.save_verdict("phash", "하영", "1", "2", 1, 3.0, E.SIMILAR)
    E.save_verdict("phash", "하영", "1", "2", 1, 3.0, E.DIFFERENT)  # 같은 사람이 다시 누르면 덮어쓰기
    E.save_verdict("phash", "도연", "1", "2", 1, 3.0, E.SIMILAR)
    E.save_verdict("clip", "하영", "1", "2", 1, 0.9, E.SIMILAR)    # 다른 방식은 섞이지 않는다
    ev = E.load_evals("phash")
    assert list(ev.columns) == E.COLUMNS
    assert dict(zip(ev["evaluator"], ev["verdict"])) == {"하영": E.DIFFERENT, "도연": E.SIMILAR}


def test_load_reads_every_page(fake_db, monkeypatch):
    monkeypatch.setattr(db, "PAGE_SIZE", 2)
    for n in range(5):
        E.save_verdict("phash", "하영", "q", str(n), n + 1, n, E.SIMILAR)
    assert len(E.load_evals("phash")) == 5


def test_save_rejects_bad_input(fake_db):
    with pytest.raises(ValueError):
        E.save_verdict("phash", "하영", "1", "2", 1, 0, "maybe")
    with pytest.raises(db.DBError):
        E.save_verdict("phash", "  ", "1", "2", 1, 0, E.SIMILAR)
    assert fake_db == []


def test_old_csv_rows():
    csv = pd.DataFrame([
        {"timestamp": "2026-10-04T21:00:00", "evaluator": "하영", "method": "phash", "query_id": "1",
         "result_id": "2", "rank": 1, "score": 4.0, "verdict": "similar"},
        {"timestamp": "2026-10-04T21:05:00", "evaluator": "하영", "method": "phash", "query_id": "1",
         "result_id": "2", "rank": 1, "score": 4.0, "verdict": "different"},   # 같은 클릭은 마지막 것
        {"timestamp": "2026-10-04T21:06:00", "evaluator": "하영", "method": "phash", "query_id": "1",
         "result_id": "999", "rank": 2, "score": 5.0, "verdict": "similar"},   # DB에 없는 이미지
        {"timestamp": "", "evaluator": None, "method": "phash", "query_id": "1",
         "result_id": "3", "rank": 3, "score": 6.0, "verdict": "similar"},     # 이름 없음
    ])
    rows, skipped = E.csv_to_rows(csv, "phash", {"1", "2", "3"})
    assert skipped == 2
    assert len(rows) == 1 and rows[0]["verdict"] == "different" and rows[0]["method"] == "phash"
    assert rows[0]["updated_at"] == "2026-10-04T21:05:00+09:00"  # 시간대 없는 예전 시각은 KST로
