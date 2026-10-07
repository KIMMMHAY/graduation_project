"""이미지별 라벨링 E2E (실제 Streamlit 서버 + Chromium + Supabase) + 학습 연동.

- 라벨러 `__test__`로만 쓰고, 끝나면 그 행만 지운다.
- 임시 태그(key `zz_test_*`, 이름 `[테스트] ...`)는 테스트가 실패해도 반드시 정리한다.
  테스트 시작 전에도 이전 실행에서 남은 임시 태그가 있으면 먼저 정리한다.
  임시 태그에 붙은 라벨(임시 태그의 것만)을 먼저 지워야 태그를 지울 수 있다(외래키 restrict).
실행: python -m pytest tests/e2e/test_labeling_image_app.py -s
"""
import re
import shutil

import pytest

from core import db
from core.dataset import read_metadata
from core.tagging import EMB_IDS_CSV, EMB_NPY, add_tag, invalidate_tags_cache, save_labels
from tests.e2e.test_pose_app import TEST_USER, browser, login, open_page, pytestmark, server  # noqa: F401

TEST_PREFIX_KEY, TEST_PREFIX_NAME = "zz_test_", "[테스트]"
TEST_TAGS = [("zz_test_a", "[테스트] A", "solo"), ("zz_test_b", "[테스트] B", None),
             ("zz_test_c", "[테스트] C", None), ("zz_test_d", "[테스트] D", None)]


def cleanup_test_tags_and_labels() -> None:
    c = db.get_client()
    keys = [t["key"] for t in c.table("tags").select("key,name").execute().data
            if t["key"].startswith(TEST_PREFIX_KEY) and t["name"].startswith(TEST_PREFIX_NAME)]
    if keys:
        c.table("labels").delete().in_("tag_key", keys).execute()  # 임시 태그의 라벨만
        c.table("tags").delete().in_("key", keys).execute()
    c.table("labels").delete().eq("labeler", TEST_USER).execute()
    invalidate_tags_cache()


@pytest.fixture(scope="module")
def test_tags():
    cleanup_test_tags_and_labels()  # 이전 실행에서 남은 것 먼저 정리
    try:
        for key, name, hint in TEST_TAGS:
            add_tag(key, name, "[테스트]", f"{name} 정의 (자동 테스트용 임시 태그)", hint)
        yield [k for k, _, _ in TEST_TAGS]
    finally:
        cleanup_test_tags_and_labels()  # 실패해도 반드시 정리


def my_rows(image_id=None) -> dict:
    q = db.get_client().table("labels").select("image_id,tag_key,value").eq("labeler", TEST_USER)
    if image_id:
        q = q.eq("image_id", image_id)
    return {(r["image_id"], r["tag_key"]): r["value"] for r in q.execute().data}


def active_tag_keys() -> list[str]:
    return [t["key"] for t in db.get_client().table("tags").select("key,active").execute().data if t["active"]]


def box(page, name):
    return page.locator('label[data-baseweb="checkbox"]').filter(has_text=name).first


def is_checked(page, name) -> bool:
    return box(page, name).locator("input").is_checked()


def click_box(page, name):
    box(page, name).click()
    page.wait_for_timeout(500)


def current_image(page) -> str:
    return page.locator("[data-testid=stMain] a code").first.inner_text()


def wait_image_change(page, old: str, timeout=20000):
    page.wait_for_function(
        """old => { const c = document.querySelector('[data-testid=stMain] a code'); return c && c.innerText !== old; }""",
        arg=old, timeout=timeout)
    page.wait_for_timeout(800)


def remaining(page) -> int:
    t = page.get_by_text(re.compile(r"^남은 \d+장$")).first.inner_text()
    return int(re.search(r"\d+", t).group())


def test_image_labeling_flow(server, browser, test_tags):
    n_active = len(active_tag_keys())
    page = open_page(browser, server + "/labeling")
    box(page, "[테스트] A").wait_for(timeout=60000)

    # 1) AI 추천 표시는 보이지만 자동 체크 안 됨 / "모두 체크"는 직접 체크한 것을 유지
    x = current_image(page)
    assert "원본 태그 일치" in box(page, "[테스트] A").inner_text()  # booru_hint 'solo'는 모든 수집 이미지에 있다
    assert not is_checked(page, "[테스트] A")
    click_box(page, "[테스트] B")
    page.get_by_role("button", name="🤖 AI 추천 모두 체크").click()
    page.wait_for_timeout(800)
    assert is_checked(page, "[테스트] A") and is_checked(page, "[테스트] B") and not is_checked(page, "[테스트] C")
    click_box(page, "[테스트] C")
    # 4) 저장 버튼 근처 안내
    info = page.get_by_text("나머지").first.inner_text()
    assert f"태그 {n_active}개 중 3개 체크" in info and "보류 0개" in info and f"나머지 {n_active - 3}개는 '아님'" in info

    # 2) 저장: 사용 중 태그 수만큼 행, 체크한 3개만 1
    page.get_by_role("button", name="💾 저장하고 다음").click()
    wait_image_change(page, x)
    rows = my_rows(x)
    assert len(rows) == n_active
    assert {k for (_, k), v in rows.items() if v == 1} == {"zz_test_a", "zz_test_b", "zz_test_c"}

    # 4) 다음 이미지는 그 이미지 상태로 초기화 / 저장 안 하고 넘어가면 체크가 남지 않음
    y = current_image(page)
    assert not any(is_checked(page, n) for _, n, _ in TEST_TAGS)
    click_box(page, "[테스트] D")
    page.get_by_role("button", name="다음 ▶").click()
    wait_image_change(page, y)
    assert not is_checked(page, "[테스트] D")
    assert my_rows(y) == {}

    # 3) 다시 열면 저장한 그대로 복원
    page.get_by_role("button", name="↩️ 방금 것 다시 보기").click()
    page.get_by_text("다시 보는 중").wait_for(timeout=15000)
    assert current_image(page) == x
    assert [is_checked(page, n) for _, n, _ in TEST_TAGS] == [True, True, True, False]
    page.get_by_role("button", name="다음 ▶").click()
    wait_image_change(page, x)

    # 요청 1) 보류하고 저장한 이미지는 이번 세션에서 다시 나오지 않는다 (끝낸 이미지가 아니어도)
    w = current_image(page)
    before = remaining(page)
    click_box(page, "[테스트] A")
    hold = page.get_by_role("combobox", name=re.compile("잘 모르겠는 태그"))
    hold.click()
    page.keyboard.type("[테스트] B")
    page.keyboard.press("Enter")
    page.keyboard.press("Escape")
    page.wait_for_timeout(800)
    assert "보류 1개" in page.get_by_text("나머지").first.inner_text()
    page.get_by_role("button", name="💾 저장하고 다음").click()
    wait_image_change(page, w)
    rows = my_rows(w)
    assert len(rows) == n_active - 1 and ("zz_test_b" not in {k for (_, k) in rows}) and rows[(w, "zz_test_a")] == 1
    assert remaining(page) == before - 1  # 숨기기로 목록에서 빠졌다

    # 6) 건너뛰기는 DB에 아무것도 쓰지 않는다
    s = current_image(page)
    snapshot = my_rows()
    page.get_by_role("button", name="⏭️ 건너뛰기").click()
    wait_image_change(page, s)
    assert my_rows() == snapshot
    page.close()

    # 5) 새 태그 추가 → 저장한 이미지에는 행이 없고 '새 태그만 남은 이미지'에서 나온다
    add_tag("zz_test_e", "[테스트] E", "[테스트]", "나중에 추가된 임시 태그")
    page = open_page(browser, server + "/tags_admin")
    page.get_by_role("button", name="🔄 새로고침").click()  # 서버의 태그 캐시(30초)를 바로 갱신
    page.wait_for_timeout(1000)
    page.goto(server + "/labeling")
    login(page)
    box(page, "[테스트] E").wait_for(timeout=60000)
    page.get_by_text("새 태그만 남은 이미지 우선").click()
    page.wait_for_timeout(1500)
    shown = current_image(page)
    assert shown in {x, w}
    assert (x, "zz_test_e") not in my_rows(x)
    assert page.get_by_text("태그만 새로 확인하면 됩니다").count() == 1
    if shown == x:
        assert [is_checked(page, n) for _, n, _ in TEST_TAGS] == [True, True, True, False]
        assert not is_checked(page, "[테스트] E")

    # 1) '태그별' 화면은 기존과 같이 동작한다
    page.get_by_text("태그별 (하나씩)").click()
    page.get_by_role("button", name="⭕ 해당").wait_for(timeout=30000)
    assert page.get_by_role("combobox", name="라벨링할 태그").count() == 1
    before = my_rows()
    page.get_by_role("button", name="⭕ 해당").click()
    page.wait_for_timeout(2000)
    after = my_rows()
    added = {k: v for k, v in after.items() if before.get(k) != v}
    assert len(added) == 1 and list(added.values()) == [1]
    page.close()


def test_training_with_image_labels(test_tags, tmp_path, monkeypatch):
    """이미지별로 저장한 라벨로 학습이 기존과 같이 동작한다 (결과 파일은 임시 폴더에)."""
    import training
    db.get_client().table("labels").delete().eq("labeler", TEST_USER).execute()  # 앞 테스트가 남긴 __test__ 라벨만 비운다
    ids = list(read_metadata()["id"][:40])
    for n, image_id in enumerate(ids):  # 화면과 같은 저장 함수로 40장 (해당 20, 아님 20)
        save_labels(image_id, {"zz_test_a": int(n < 20), "zz_test_b": 0}, TEST_USER)

    for name, path in (("EMB_NPY", EMB_NPY), ("EMB_IDS_CSV", EMB_IDS_CSV)):  # 임베딩 캐시도 복사본을 쓴다
        shutil.copy(path, tmp_path / path.name)
        monkeypatch.setattr(training, name, tmp_path / path.name)
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(training, "REPORT_CSV", tmp_path / "train_report.csv")
    monkeypatch.setattr(training, "PRED_CSV", tmp_path / "predicted_tags.csv")

    def no_embedder():
        raise AssertionError("임베딩 캐시가 있으면 모델을 불러오지 않아야 합니다")

    result = training.train_all(get_embedder=no_embedder, progress=lambda *a: None)
    report = result.report.set_index("tag_key")
    assert "zz_test_a" in report.index
    assert report.at["zz_test_a", "n_pos"] == 20 and report.at["zz_test_a", "n_neg"] == 20
    assert (tmp_path / "models" / "zz_test_a.joblib").exists() and (tmp_path / "predicted_tags.csv").exists()
