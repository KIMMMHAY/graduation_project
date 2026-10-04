"""단계 2 E2E: AI 제안 핀(바로 추정 / 미리 계산), model_corrected 저장, 정확도 리포트.

- 사람 라벨은 `__test__`로만 만들고 끝나면 지운다.
- 테스트가 만든 DWPose 추정(이미지 2장)은 '미리 계산'과 같은 실제 모델 결과라 남겨 둔다.
실행: python -m pytest tests/e2e/test_pose_step2_app.py -s
"""
import pytest

from core import db
from core import pose as P
from core import poses as store
from core.dataset import read_metadata
from core.pose_models import DEFAULT_MODEL, MODELS
from core.pose_predict import predict_and_save
from tests.e2e.test_pose_app import (TEST_USER, browser, image_box, my_poses, open_page, pin_center,  # noqa: F401
                                     pytestmark, reset_to_default, server, wait_editor, wait_pins_at)


def fresh_images(n: int) -> list[tuple[str, str]]:
    """팀원이 손대지 않은 이미지 (팀 데이터와 겹치지 않게). AI 추정이 없는 이미지를 앞에 둔다."""
    poses = store.load_poses()
    human = set(poses.loc[poses["source"] != P.MODEL, "image_id"])
    modeled = set(store.model_poses(DEFAULT_MODEL, poses)["image_id"])
    df = read_metadata()
    cands = [(i, p) for i, p in zip(df["id"], df["img_path"]) if i not in human]
    return sorted(cands, key=lambda c: c[0] in modeled)[:n]


def model_row(image_id: str) -> dict | None:
    rows = db.get_client().table("poses").select("*").eq("image_id", image_id).eq("annotator", DEFAULT_MODEL).execute().data
    return rows[0] if rows else None


def assert_pins_at(page, keypoints):
    wait_pins_at(page, keypoints)


def test_ai_suggestions_and_accuracy_report(server, browser):
    (live_id, _), (pre_id, pre_path) = fresh_images(2)
    if model_row(live_id) is not None:
        # 모든 이미지가 이미 미리 계산된 상태: 실제 모델 결과를 지우지 않고, 저장된 결과를 쓰는 경로로 확인한다
        print("\n[참고] AI 추정이 없는 이미지가 없어 '바로 추정'은 저장된 결과를 재사용하는 경로로 확인합니다.")

    # 1) 바로 추정: 페이지를 여는 순간 추정하고 DB에도 저장한다 (이미 있으면 그 결과를 쓴다)
    page = open_page(browser, f"{server}/pose_label?image={live_id}")
    page.get_by_text("열 때마다 바로 추정 (약 3초)").click()
    page.get_by_text("AI 제안 핀에서 시작합니다").wait_for(timeout=120000)
    wait_editor(page)
    row = model_row(live_id)
    assert row is not None and row["source"] == P.MODEL and row["status"] == P.DONE
    assert_pins_at(page, row["keypoints"])
    page.get_by_role("button", name="✅ 완료 (저장 후 다음)").click()
    page.wait_for_timeout(2500)
    mine = [r for r in my_poses() if r["image_id"] == live_id]
    assert len(mine) == 1 and mine[0]["source"] == P.MODEL_CORRECTED and mine[0]["status"] == P.DONE
    page.close()

    # 2) 미리 계산: 결과가 이미 있으면 기다리지 않고 그 핀으로 시작한다 (기본 모드)
    if model_row(pre_id) is None:
        predict_and_save(MODELS[DEFAULT_MODEL](), pre_id, pre_path)
    page = open_page(browser, f"{server}/pose_label?image={pre_id}")
    page.get_by_text("AI 제안 핀에서 시작합니다").wait_for(timeout=30000)
    wait_editor(page)
    assert_pins_at(page, model_row(pre_id)["keypoints"])
    # 표준 자세로 초기화하면 직접 찍은 것(manual)으로 저장된다
    reset_to_default(page)
    page.get_by_role("button", name="✅ 완료 (저장 후 다음)").click()
    page.wait_for_timeout(2500)
    mine = [r for r in my_poses() if r["image_id"] == pre_id]
    assert len(mine) == 1 and mine[0]["source"] == P.MANUAL
    page.close()

    # 3) 정확도 리포트: 관절별 표가 그려진다 (st.dataframe은 캔버스라 글자 대신 표 개수를 본다)
    page = open_page(browser, f"{server}/pose_accuracy")
    page.get_by_text("관절별").wait_for(timeout=60000)
    page.wait_for_timeout(1500)
    assert page.locator("[data-testid=stDataFrame]").count() >= 3  # 미리 계산 현황, 요약, 관절별
    page.close()
    # 같은 DB 데이터로 계산한 관절별 수치: 우리 두 이미지의 13관절이 모두 들어 있다
    from core import pose_accuracy as A
    from views.pose_common import aspects_of
    poses = store.load_poses()
    human = poses[(poses["annotator"] == TEST_USER)]
    cmp = A.compare(human, store.model_poses(DEFAULT_MODEL, poses), aspects_of(read_metadata()))
    pj = A.per_joint(cmp)
    assert set(pj["joint"]) >= set(P.JOINT_KEYS) - {"head"}
    summ = A.summary(cmp).set_index("model").loc[DEFAULT_MODEL]
    assert summ["images"] == 2 and 0 <= summ["pck"] <= 1
    print(f"\n[참고] 테스트 2장 기준 PCK {summ['pck']:.0%}, 평균 오차 {summ['mean_error']:.3f}")
