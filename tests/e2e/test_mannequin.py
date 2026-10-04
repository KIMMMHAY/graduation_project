"""3D 마네킹(web/mannequin/mannequin.js)을 Streamlit 없이 Chromium에서 검증.

프로젝트 폴더를 간단한 HTTP 서버로 열고 web/mannequin/demo.html(로컬 three.js 사용)을 띄운다 → 오프라인·독립 동작 확인.
실행: python -m pytest tests/e2e/test_mannequin.py
"""
import functools
import http.server
import math
import threading
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
J = {k: i for i, k in enumerate(["head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
                                 "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"])}


@pytest.fixture(scope="module")
def base_url():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser, base_url):
    pg = browser.new_page(viewport={"width": 900, "height": 800})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    # 외부 네트워크 요청이 하나라도 있으면 실패 → 오프라인 동작 보장
    external = []
    pg.on("request", lambda r: external.append(r.url) if not r.url.startswith(base_url) else None)
    pg.goto(f"{base_url}/web/mannequin/demo.html")
    pg.wait_for_function("window.ready === true", timeout=30000)
    yield pg
    assert errors == [], errors
    assert external == [], external
    pg.close()


def snap(page):
    return page.evaluate("window.mq.snapshot()")


def angle(kps, a, b):
    pa, pb = kps[J[a]], kps[J[b]]
    return math.degrees(math.atan2(pb["y"] - pa["y"], pb["x"] - pa["x"]))


def drag(page, x, y, dx, dy, shift=False):
    if shift:
        page.keyboard.down("Shift")
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 6):
        page.mouse.move(x + dx * i / 5, y + dy * i / 5)
    page.mouse.up()
    if shift:
        page.keyboard.up("Shift")


def screen(page, key):
    p = page.evaluate(f"window.mq.screenPositionOf('{key}')")
    return p["x"], p["y"]


def test_projection_has_13_keypoints_in_frame(page):
    s = snap(page)
    assert len(s["keypoints"]) == 13 and s["aspect"] == 1 and s["facing"] == "front"
    assert all(0.0 <= k["x"] <= 1.0 and 0.0 <= k["y"] <= 1.0 and k["state"] == "visible" for k in s["keypoints"])
    k = s["keypoints"]
    # 정면이면 캐릭터의 왼쪽이 화면 오른쪽, 머리가 위·발목이 아래
    assert k[J["l_shoulder"]]["x"] > k[J["r_shoulder"]]["x"]
    assert k[J["head"]]["y"] < k[J["l_hip"]]["y"] < k[J["l_ankle"]]["y"]


def test_drag_wrist_bends_forearm(page):
    before = snap(page)["keypoints"]
    x, y = screen(page, "l_wrist")
    drag(page, x, y, 0, -120)  # 손목을 위로
    after = snap(page)["keypoints"]
    turned = abs(angle(after, "l_elbow", "l_wrist") - angle(before, "l_elbow", "l_wrist"))
    assert turned > 30
    assert abs(angle(after, "l_shoulder", "l_elbow") - angle(before, "l_shoulder", "l_elbow")) < 1  # 위팔은 그대로
    assert page.evaluate("window.mq.snapshot().state.pose.l_elbow") != [0, 0, 0, 1]


def test_drag_knee_moves_whole_leg(page):
    before = snap(page)["keypoints"]
    x, y = screen(page, "r_knee")
    drag(page, x, y, -100, -60)
    after = snap(page)["keypoints"]
    assert abs(angle(after, "r_hip", "r_knee") - angle(before, "r_hip", "r_knee")) > 20
    assert abs(angle(after, "r_knee", "r_ankle") - angle(before, "r_knee", "r_ankle")) > 20  # 정강이도 함께 따라감


def test_orbit_and_views_change_projection_and_facing(page):
    front = snap(page)
    box = page.locator("canvas.mq-canvas").bounding_box()
    drag(page, box["x"] + 30, box["y"] + 30, 160, 0)  # 빈 곳 드래그 → 카메라 회전
    rotated = snap(page)
    assert rotated["state"]["camera"]["azimuth"] != front["state"]["camera"]["azimuth"]
    assert rotated["keypoints"] != front["keypoints"]
    page.get_by_role("button", name="뒤").click()
    back = snap(page)
    assert back["facing"] == "back"
    assert back["keypoints"][J["l_shoulder"]]["x"] < back["keypoints"][J["r_shoulder"]]["x"]  # 뒤에서 보면 좌우가 뒤집힘
    page.get_by_role("button", name="측면").click()
    assert snap(page)["facing"] == "side"


def test_shift_chest_tilts_whole_body(page):
    x, y = screen(page, "chest")
    drag(page, x, y, 200, 120, shift=True)
    s = snap(page)
    assert s["state"]["pose"]["root"] != [0, 0, 0, 1] and s["state"]["pose"]["waist"] == [0, 0, 0, 1]


def test_state_roundtrip_and_reset(page):
    x, y = screen(page, "r_wrist")
    drag(page, x, y, -80, -100)
    page.get_by_role("button", name="하이앵글").click()
    s = snap(page)
    page.get_by_role("button", name="포즈 초기화").click()
    assert snap(page)["keypoints"] != s["keypoints"]
    page.evaluate(f"window.mq.setState({__import__('json').dumps(s['state'])})")
    assert snap(page)["keypoints"] == s["keypoints"]
