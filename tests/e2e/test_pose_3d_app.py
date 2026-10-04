"""단계 3 E2E: 3D 마네킹 검색 (실제 Streamlit 서버 + Chromium + Supabase).

수용 기준: 마네킹 관절을 움직이고 카메라를 돌린 뒤 검색하면 결과가 달라진다.
검색 후보로 `__test__` 포즈 12장을 넣고 끝나면 지운다.
실행: python -m pytest tests/e2e/test_pose_3d_app.py -s
"""
from core import pose as P
from tests.e2e.test_pose_app import TEST_USER, browser, open_page, pytestmark, seed_test_poses, server  # noqa: F401


def results(page) -> list[str]:
    page.get_by_text("가까운 순").first.wait_for(timeout=30000)
    page.wait_for_timeout(800)
    return [t.split("\n")[0] for t in page.locator("[data-testid=stMarkdownContainer]").all_inner_texts() if t.startswith("#")]


def drag(page, x, y, dx, dy):
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 8):
        page.mouse.move(x + dx * i / 7, y + dy * i / 7)
    page.mouse.up()


def snapshot(page):
    return page.evaluate("window.__mannequin.snapshot()")


def test_mannequin_search_changes_with_pose_and_camera(server, browser):
    seed_test_poses()
    page = open_page(browser, server + "/pose_search_3d")
    external = []
    # blob:/data: 는 브라우저 안에서 만든 주소라 외부 요청이 아니다 (Streamlit이 컴포넌트 코드를 blob으로 싣는다)
    page.on("request", lambda r: external.append(r.url)
            if not r.url.startswith((server, "blob:", "data:")) else None)
    page.get_by_text("현재 마네킹 방향").wait_for(timeout=60000)
    page.wait_for_function("window.__mannequin !== undefined", timeout=30000)

    # 허용 범위를 최대로 → 후보 전체가 점수순으로 나온다 (순서 비교)
    thumb = page.get_by_role("slider").bounding_box()
    track = page.locator("[data-testid=stSlider]").bounding_box()
    drag(page, thumb["x"] + thumb["width"] / 2, thumb["y"] + thumb["height"] / 2,
         track["x"] + track["width"] + 50 - thumb["x"], 0)
    page.wait_for_timeout(1000)
    assert page.get_by_role("slider").get_attribute("aria-valuenow") == "180"

    page.get_by_role("button", name="🔎 검색").click()
    first = results(page)
    assert first, "검색 결과가 없습니다"
    before = snapshot(page)

    # 관절 움직이기: 왼쪽 손목을 위로, 오른쪽 무릎을 들어 올린다
    for key, dx, dy in (("l_wrist", 0, -140), ("r_knee", -80, -120)):
        p = page.evaluate(f"window.__mannequin.screenPositionOf('{key}')")
        drag(page, p["x"], p["y"], dx, dy)
        page.wait_for_timeout(600)
    # 카메라 돌리기: 빈 곳을 가로로 끈다
    box = page.locator("canvas.mq-canvas").bounding_box()
    drag(page, box["x"] + 25, box["y"] + 25, 120, 40)
    page.wait_for_timeout(1500)
    after = snapshot(page)
    assert after["state"]["camera"]["azimuth"] != before["state"]["camera"]["azimuth"]
    assert after["keypoints"] != before["keypoints"]
    stale = page.get_by_text("바뀐 모습으로 보려면")
    stale.wait_for(timeout=15000)  # Python이 바뀐 마네킹을 받았다는 표시

    page.get_by_role("button", name="🔎 검색").click()
    stale.wait_for(state="detached", timeout=30000)  # 새 결과로 다시 그려질 때까지
    second = results(page)
    print(f"\n[3D 검색] 처음 상위 3: {first[:3]}\n[3D 검색] 바꾼 뒤 상위 3: {second[:3]}")
    assert second != first  # 결과(순서·점수)가 달라진다
    assert external == [], f"외부 요청 발생: {external[:3]}"  # three.js를 로컬에서 불러온다
    page.close()


def test_projection_matches_mannequin_facing(server, browser):
    """뒤에서 본 마네킹은 'back', 투영 좌표는 좌우가 뒤집힌다 (캐릭터 기준 좌우 확인)."""
    page = open_page(browser, server + "/pose_search_3d")
    page.wait_for_function("window.__mannequin !== undefined", timeout=60000)
    front = snapshot(page)
    page.get_by_role("button", name="뒤", exact=True).click()
    page.wait_for_timeout(1000)
    back = snapshot(page)
    l, r = P.J["l_shoulder"], P.J["r_shoulder"]
    assert front["facing"] == "front" and back["facing"] == "back"
    assert front["keypoints"][l]["x"] > front["keypoints"][r]["x"] and back["keypoints"][l]["x"] < back["keypoints"][r]["x"]
    page.get_by_text("현재 마네킹 방향: 뒷모습").wait_for(timeout=15000)  # Python 쪽에도 전달됐다
    page.close()
