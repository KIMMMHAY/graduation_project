"""단계 2: 모델 출력 변환, 오차·PCK, 정확도 집계, COCO 내보내기 (DB 불필요)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import pose as P
from core import pose_accuracy as A
from core import pose_models as M

ROOT = Path(__file__).resolve().parents[1]


def test_state_thresholds():
    assert M.state_of(0.9) == P.VISIBLE and M.state_of(0.5) == P.VISIBLE
    assert M.state_of(0.4) == P.OCCLUDED and M.state_of(0.29) == P.ABSENT


def test_to_keypoints_ratio_state_conf():
    xy = np.array([[50, 20]] * 13, dtype=float)
    conf = np.array([0.9] * 12 + [0.1])
    kps = M.to_keypoints(xy, conf, width=100, height=200)
    assert kps[0] == {"x": 0.5, "y": 0.1, "state": "visible", "conf": 0.9}
    assert kps[12]["state"] == "absent"


def test_mapping_tables_cover_13_joints():
    assert len(M.COCO_TO_OURS) == len(M.MEDIAPIPE_TO_OURS) == P.N_JOINTS
    assert M.COCO_TO_OURS[P.J["l_shoulder"]] == 5 and M.MEDIAPIPE_TO_OURS[P.J["r_ankle"]] == 28


def test_facing_estimate():
    front = P.default_pose()  # 정면: 캐릭터의 왼쪽 어깨가 화면 오른쪽
    assert M.estimate_facing(front) == "front"
    back = P.mirror_keypoints(front)
    back = [{**k, "x": k["x"]} for k in back]
    back[P.J["l_shoulder"]]["x"], back[P.J["r_shoulder"]]["x"] = 0.42, 0.58  # 좌우가 화면에서 뒤집힘
    assert M.estimate_facing(back) == "back"
    side = [dict(k) for k in front]
    side[P.J["l_shoulder"]]["x"] = side[P.J["r_shoulder"]]["x"] = 0.5
    assert M.estimate_facing(side) == "side"
    assert M.estimate_facing(front, face_conf=0.1) == "back"


def test_joint_errors_and_pck():
    truth = P.default_pose()
    pred = [dict(k) for k in truth]
    pred[P.J["l_wrist"]]["x"] += 0.5          # 크게 틀림
    pred[P.J["r_knee"]]["state"] = P.ABSENT   # 모델이 못 찾음 → 오답
    truth[P.J["head"]]["state"] = P.ABSENT    # 사람이 '없음' → 비교 제외
    errors, correct = P.joint_errors(truth, pred, 0.75)
    assert np.isnan(correct[P.J["head"]]) and np.isnan(errors[P.J["head"]])
    assert correct[P.J["l_wrist"]] == 0 and errors[P.J["l_wrist"]] > P.PCK_THRESHOLD
    assert correct[P.J["r_knee"]] == 0 and np.isnan(errors[P.J["r_knee"]])
    assert correct[P.J["l_hip"]] == 1 and errors[P.J["l_hip"]] == pytest.approx(0)


def _rows(entries):
    return pd.DataFrame([{"image_id": i, "annotator": a, "source": s, "status": st, "facing": "front", "keypoints": k}
                         for i, a, s, st, k in entries])


def test_accuracy_report():
    good = P.default_pose()
    bad = [{**k, "x": min(k["x"] + 0.3, 1)} for k in good]
    human = _rows([("a", "kim", P.MANUAL, P.DONE, good), ("b", "kim", P.MODEL_CORRECTED, P.DONE, good),
                   ("c", "kim", P.MANUAL, P.DONE, good), ("d", "kim", P.MANUAL, P.DRAFT, good)])
    models = _rows([("a", "dwpose", P.MODEL, P.DONE, good), ("b", "dwpose", P.MODEL, P.DONE, bad),
                    ("c", "dwpose", P.MODEL, P.SKIPPED, good), ("d", "dwpose", P.MODEL, P.DONE, good)])
    cmp = A.compare(human, models, {"a": 0.75, "b": 0.75, "c": 0.75, "d": 0.75})
    s = A.summary(cmp).set_index("model").loc["dwpose"]
    assert s["images"] == 3                              # draft인 d는 제외
    assert s["detection_rate"] == pytest.approx(2 / 3)   # c는 검출 실패
    assert s["pck"] == pytest.approx(0.5, abs=0.05)      # a는 전부 정답, b는 대부분 오답
    pj = A.per_joint(cmp)
    assert set(pj["joint"]) == set(P.JOINT_KEYS) and (pj["n"] == 2).all()
    only_manual = A.summary(A.compare(human[human["source"] == P.MANUAL], models, {"a": 0.75, "c": 0.75}))
    assert only_manual.set_index("model").loc["dwpose", "pck"] == pytest.approx(1.0)


def test_coco_export():
    import export_coco
    meta = pd.DataFrame({"id": ["7192608"], "img_path": [str(ROOT / "drawing_ref_test/images/7192608.jpg")]})
    if not Path(meta["img_path"][0]).exists():
        pytest.skip("로컬 이미지 없음")
    kps = P.default_pose()
    kps[0]["state"] = P.ABSENT
    kps[1]["state"] = P.OCCLUDED
    poses = _rows([("7192608", "kim", P.MANUAL, P.DONE, kps), ("7192608", "dwpose", P.MODEL, P.DONE, kps)])
    poses["updated_at"] = "2026-10-04"
    coco = export_coco.build(poses, meta)
    assert len(coco["images"]) == len(coco["annotations"]) == 1  # 모델 결과는 내보내지 않는다
    ann = coco["annotations"][0]
    assert ann["keypoints"][:3] == [0.0, 0.0, 0] and ann["keypoints"][5] == 1 and ann["num_keypoints"] == 12
    assert coco["categories"][0]["keypoints"] == P.JOINT_KEYS
    assert all(1 <= a <= 13 and 1 <= b <= 13 for a, b in coco["categories"][0]["skeleton"])


@pytest.mark.skipif(not (ROOT / "drawing_ref_test/images/7196182.jpg").exists(), reason="로컬 이미지 없음")
def test_dwpose_runs_on_illustrations():
    """실제 DWPose (처음 실행 시 모델 다운로드). 사전 시험에서 네 모델 모두 잡았던 쉬운 그림 + SD 체형 그림."""
    model = M.DWPoseModel()
    pred = model.predict(str(ROOT / "drawing_ref_test/images/7196182.jpg"))
    assert pred is not None and len(pred.keypoints) == 13 and 0 < pred.mean_conf <= 1
    assert pred.keypoints[P.J["head"]]["y"] < pred.keypoints[P.J["l_hip"]]["y"] < pred.keypoints[P.J["l_ankle"]]["y"]
    assert all("conf" in k for k in pred.keypoints)
    assert model.predict(str(ROOT / "drawing_ref_test/images/7194930.jpg")) is not None
