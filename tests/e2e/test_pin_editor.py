"""핀 편집기(web/pose_editor/pin_editor.js)를 실제 Chromium에서 검증. Streamlit·DB 불필요.

실행: python -m pytest tests/e2e/test_pin_editor.py   (개발용: playwright + chromium 필요)
"""
import json
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from core import pose as P  # noqa: E402
from components.pose_editor import BONES_META, JOINTS_META  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
JS = (ROOT / "web" / "pose_editor" / "pin_editor.js").read_text(encoding="utf-8")
ASPECT = 0.6


def harness(width: int = 900) -> str:
    opts = {"image": None, "aspect": ASPECT, "keypoints": P.default_pose(), "joints": JOINTS_META,
            "bones": BONES_META, "height": 600}
    return f"""<!doctype html><html><body style="margin:0">
<div id="host" style="width:{width}px"></div>
<script type="module">
{JS}
const style = document.createElement('style'); style.textContent = PIN_EDITOR_CSS; document.head.appendChild(style);
window.changes = [];
const opts = {json.dumps(opts)};
opts.onChange = (k) => window.changes.push(k);
window.editor = new PinEditor(document.getElementById('host'), opts);
window.ready = true;
</script></body></html>"""


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    pg = browser.new_page(viewport={"width": 1000, "height": 800})
    pg.set_content(harness())
    pg.wait_for_function("window.ready === true")
    yield pg
    pg.close()


def pin_center(page, idx: int) -> tuple[float, float]:
    box = page.locator(f'circle.pe-pin[data-idx="{idx}"]').bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def canvas_box(page) -> dict:
    """이미지(내부 좌표 0~W, 0~H)가 화면에 그려진 영역."""
    return page.evaluate("""() => {
        const svg = document.querySelector('.pe-canvas svg'); const m = svg.getScreenCTM();
        const W = window.editor.W, H = window.editor.H;
        return {x: m.e + m.a * 0 , y: m.f, w: m.a * W, h: m.d * H, a: m.a, e: m.e, f: m.f};
    }""")


def drag(page, x, y, dx, dy):
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx / 2, y + dy / 2)
    page.mouse.move(x + dx, y + dy)
    page.mouse.up()


def test_drag_pin_updates_ratio_coordinates(page):
    idx = P.J["l_wrist"]
    before = page.evaluate("window.editor.getKeypoints()")[idx]
    box = canvas_box(page)
    x, y = pin_center(page, idx)
    drag(page, x, y, 60, -40)
    after = page.evaluate("window.editor.getKeypoints()")[idx]
    assert after["x"] == pytest.approx(before["x"] + 60 / box["w"], abs=0.003)
    assert after["y"] == pytest.approx(before["y"] - 40 / box["h"], abs=0.003)
    assert len(page.evaluate("window.changes")) == 1  # 드래그가 끝날 때 한 번만 알린다


def test_zoom_then_drag_keeps_correct_coordinates(page):
    idx = P.J["r_ankle"]
    x, y = pin_center(page, idx)
    for _ in range(3):
        page.mouse.move(x, y)
        page.mouse.wheel(0, -300)  # 확대
    before = page.evaluate("window.editor.getKeypoints()")[idx]
    x2, y2 = pin_center(page, idx)
    drag(page, x2, y2, 30, 0)
    after = page.evaluate("window.editor.getKeypoints()")[idx]
    scale = page.evaluate("document.querySelector('.pe-canvas svg').getScreenCTM().a") * page.evaluate("window.editor.W")
    assert scale > 1.5 * 600 * ASPECT  # 실제로 확대됐는지 (확대 전 이미지 폭보다 크다)
    assert after["x"] == pytest.approx(before["x"] + 30 / scale, abs=0.002)


def test_pin_position_survives_resize(page):
    idx = P.J["l_knee"]
    k = page.evaluate("window.editor.getKeypoints()")[idx]
    for width in (900, 500):
        page.evaluate(f"document.getElementById('host').style.width = '{width}px'")
        page.wait_for_timeout(100)
        box = canvas_box(page)
        x, y = pin_center(page, idx)
        assert (x - box["x"]) / box["w"] == pytest.approx(k["x"], abs=0.004)
        assert (y - box["y"]) / box["h"] == pytest.approx(k["y"], abs=0.004)
    assert page.evaluate("window.editor.getKeypoints()")[idx] == k  # 창 크기는 좌표에 영향 없음


def test_state_buttons_and_keyboard(page):
    idx = P.J["head"]
    x, y = pin_center(page, idx)
    page.mouse.click(x, y)
    assert "머리" in page.locator(".pe-selected").inner_text()
    page.get_by_role("button", name="가려짐").click()
    assert page.evaluate("window.editor.getKeypoints()")[idx]["state"] == "occluded"
    page.keyboard.press("3")
    assert page.evaluate("window.editor.getKeypoints()")[idx]["state"] == "absent"
    assert len(page.evaluate("window.changes")) == 2


def test_pan_does_not_move_pins(page):
    before = page.evaluate("window.editor.getKeypoints()")
    page.mouse.wheel(0, -300)
    box = canvas_box(page)
    drag(page, box["x"] + 5, box["y"] + 5, 50, 50)  # 핀이 없는 구석을 끈다
    assert page.evaluate("window.editor.getKeypoints()") == before
    assert page.evaluate("window.changes") == []
