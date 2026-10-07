"""포즈 라벨링·검색 앱 전체 흐름 E2E (실제 Streamlit 서버 + Chromium + Supabase).

- DB에 poses/pose_evals 테이블이 없으면 건너뛴다.
- 작성자 이름 `__test__`로만 기록하고, 끝나면 그 행만 지운다 (팀 데이터는 건드리지 않음).
- 입장 화면은 임시 팀원(TEST_EMAIL → `__test__`)으로 통과한다. members에 그 행만 넣었다가 끝나면 지운다 (secret 키 필요).
실행: python -m pytest tests/e2e/test_pose_app.py -s
"""
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from core import db  # noqa: E402
from core import pose as P  # noqa: E402
from core import poses as store  # noqa: E402
from core.dataset import read_metadata  # noqa: E402
from views.pose_common import aspects_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEST_USER = "__test__"
TEST_EMAIL = "__test__@test.invalid"


def _tables_ready() -> bool:
    try:
        for t in ("poses", "pose_evals", "members"):
            db.get_client().table(t).select("*").limit(1).execute()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not (db.is_configured() and _tables_ready()),
                                reason="Supabase poses/pose_evals/members 테이블이 없습니다 (schema.sql 실행 필요)")


def cleanup():
    db.get_client().table("pose_evals").delete().eq("evaluator", TEST_USER).execute()
    db.get_client().table("poses").delete().eq("annotator", TEST_USER).execute()
    db.get_client().table("members").delete().eq("email", TEST_EMAIL).execute()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cleanup()
    db.get_client().table("members").insert({"email": TEST_EMAIL, "name": TEST_USER}).execute()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = open(tmp_path_factory.mktemp("streamlit") / "server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", str(port), "--browser.gatherUsageStats", "false",
                             "--server.fileWatcherType", "none"],  # 테스트 중 파일 변경으로 재시작되지 않게
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    yield url
    alive = proc.poll() is None
    proc.terminate()
    proc.wait(timeout=20)
    log.close()
    cleanup()
    if not alive:
        print("\n[streamlit server log]\n" + Path(log.name).read_text(encoding="utf-8")[-3000:])


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def open_page(browser, url, width=1400):
    page = browser.new_page(viewport={"width": width, "height": 1000})
    page.goto(url)
    login(page)
    return page


def login(page):
    """입장 화면에서 임시 팀원 이메일로 들어간다. 주소를 새로 열(goto) 때마다 세션이 새로 생기므로 다시 불러야 한다."""
    email = page.get_by_role("textbox", name="팀원 이메일")
    email.wait_for(timeout=60000)
    email.fill(TEST_EMAIL)
    page.get_by_role("button", name="팀원으로 입장").click()
    page.get_by_text(f"{TEST_USER} 님").wait_for(timeout=60000)


def wait_editor(page):
    page.locator("circle.pe-pin").first.wait_for(timeout=60000)
    page.wait_for_timeout(500)


def image_box(page) -> dict:
    return page.evaluate("""() => {
        const svg = [...document.querySelectorAll('*')].map(e => e.shadowRoot).filter(Boolean)
            .map(r => r.querySelector('.pe-canvas svg')).find(Boolean) || document.querySelector('.pe-canvas svg');
        const m = svg.getScreenCTM(); const img = svg.querySelector('image');
        const W = +img.getAttribute('width'), H = +img.getAttribute('height');
        return {x: m.e, y: m.f, w: m.a * W, h: m.d * H};
    }""")


def pin_center(page, idx):
    b = page.locator(f'circle.pe-pin[data-idx="{idx}"]').bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def wait_pins_at(page, keypoints, joints=("l_shoulder", "r_hip", "l_knee"), timeout=30.0, tol=0.01):
    """핀이 주어진 위치로 그려질 때까지 기다린다 (재실행 후 편집기가 새로 만들어지는 시간)."""
    deadline = time.time() + timeout
    while True:
        try:
            box = image_box(page)
            ok = all(abs((pin_center(page, P.J[k])[0] - box["x"]) / box["w"] - keypoints[P.J[k]]["x"]) <= tol and
                     abs((pin_center(page, P.J[k])[1] - box["y"]) / box["h"] - keypoints[P.J[k]]["y"]) <= tol
                     for k in joints)
        except Exception:
            ok = False
        if ok:
            return
        if time.time() > deadline:
            raise AssertionError(f"핀이 기대한 위치에 오지 않았습니다: {joints}")
        page.wait_for_timeout(300)


def reset_to_default(page):
    page.get_by_role("button", name="↺ 초기화 (표준 자세로)").click()
    wait_pins_at(page, P.default_pose())


def my_poses():
    return db.get_client().table("poses").select("*").eq("annotator", TEST_USER).execute().data


def test_label_save_and_restore(server, browser):
    page = open_page(browser, server + "/pose_label")
    wait_editor(page)
    reset_to_default(page)  # AI 제안 핀이 있으면 그걸로 시작하므로, 위치를 아는 표준 자세에서 출발한다
    idx = P.J["l_wrist"]
    box = image_box(page)
    x, y = pin_center(page, idx)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 40, y + 25, steps=5)
    page.mouse.up()
    page.wait_for_timeout(1500)  # 상태가 Python으로 전달되고 재실행되기를 기다린다
    page.get_by_text("뒷모습", exact=True).click()
    page.wait_for_timeout(800)
    page.get_by_role("button", name="✅ 완료 (저장 후 다음)").click()
    page.wait_for_timeout(2500)

    rows = my_poses()
    assert len(rows) == 1 and rows[0]["status"] == "done" and rows[0]["facing"] == "back"
    saved = rows[0]["keypoints"]
    default = P.default_pose()[idx]
    assert saved[idx]["x"] == pytest.approx(default["x"] + 40 / box["w"], abs=0.01)
    assert saved[idx]["y"] == pytest.approx(default["y"] + 25 / box["h"], abs=0.01)
    image_id = rows[0]["image_id"]
    page.close()

    # 새 세션에서 링크로 다시 열면 같은 위치에 복원된다. 창 크기를 바꿔도 비율 위치가 유지된다.
    for width in (1400, 900):
        page = open_page(browser, f"{server}/pose_label?image={image_id}", width=width)
        wait_editor(page)
        box = image_box(page)
        x, y = pin_center(page, idx)
        assert (x - box["x"]) / box["w"] == pytest.approx(saved[idx]["x"], abs=0.006)
        assert (y - box["y"]) / box["h"] == pytest.approx(saved[idx]["y"], abs=0.006)
        page.close()


def seed_test_poses(n=12):
    """검색 테스트용: 실제 이미지 n장에 서로 다른 포즈를 __test__ 이름으로 저장."""
    df = read_metadata()
    ids = list(df["id"][:n])
    rnd = random.Random(0)
    for k, image_id in enumerate(ids):
        kps = P.default_pose()
        for j in kps[3:]:
            j["x"] = min(max(j["x"] + rnd.uniform(-0.25, 0.25), 0), 1)
            j["y"] = min(max(j["y"] + rnd.uniform(-0.2, 0.2), 0), 1)
        store.save_pose(image_id, TEST_USER, kps, "front", P.DONE)
    return ids, aspects_of(df)


def test_search_top3_and_mirror(server, browser):
    ids, aspects = seed_test_poses()
    cands = [c for c in store.searchable_candidates(aspects) if c.annotator == TEST_USER]
    assert len(cands) >= 10
    # 라벨된 포즈와 같은 포즈(약간의 손떨림)를 만들어 검색하면 상위 3개 안에 나온다
    for c in cands:
        rnd = random.Random(hash(c.image_id) % 1000)
        query = [{**k, "x": k["x"] + rnd.uniform(-0.01, 0.01), "y": k["y"] + rnd.uniform(-0.01, 0.01)} for k in c.keypoints]
        top3 = [h.candidate.image_id for h in P.search(query, c.aspect, cands, top_k=3)]
        assert c.image_id in top3
    # 좌우 반전 옵션
    target = cands[3]
    mirrored = P.mirror_keypoints(target.keypoints)
    on = P.search(mirrored, target.aspect, cands, allow_mirror=True, top_k=3)
    assert on[0].candidate.image_id == target.image_id and on[0].match.flip
    off = P.search(mirrored, target.aspect, cands, allow_mirror=False, top_k=1)
    assert not off or off[0].candidate.image_id != target.image_id

    # 화면에서: 라벨된 포즈에서 시작해 검색하면 그 이미지가 1위, 비슷함 평가가 DB에 저장된다
    page = open_page(browser, server + "/pose_search")
    wait_editor(page)
    box = page.get_by_role("combobox", name="시작 포즈")
    box.click()
    page.keyboard.type(target.image_id)
    page.keyboard.press("Enter")
    page.wait_for_timeout(1500)
    wait_editor(page)
    page.get_by_role("button", name="🔎 검색").click()
    page.get_by_text("가까운 순").wait_for(timeout=30000)
    first = page.locator("text=#1 ·").first
    first.wait_for()
    assert target.image_id in first.inner_text()
    page.get_by_role("button", name="👍 비슷함").first.click()
    page.wait_for_timeout(2000)
    evals = db.get_client().table("pose_evals").select("*").eq("evaluator", TEST_USER).execute().data
    assert len(evals) == 1 and evals[0]["result_image_id"] == target.image_id and evals[0]["verdict"] == "similar"
    page.close()
