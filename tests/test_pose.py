"""core/pose.py 단위 테스트 (DB·브라우저 불필요)."""
import random

import pytest

from core import pose as P


def make_pose(seed: int) -> list[dict]:
    """표준 자세에서 팔·다리를 무작위로 크게 움직인 포즈."""
    rnd = random.Random(seed)
    kps = P.default_pose()
    for k in kps[3:]:  # 팔꿈치 아래 관절만 크게 흔든다
        k["x"] = min(max(k["x"] + rnd.uniform(-0.25, 0.25), 0), 1)
        k["y"] = min(max(k["y"] + rnd.uniform(-0.2, 0.2), 0), 1)
    return kps


def jitter(kps: list[dict], amount: float, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    return [{**k, "x": k["x"] + rnd.uniform(-amount, amount), "y": k["y"] + rnd.uniform(-amount, amount)} for k in kps]


def candidates(n: int = 12, aspect: float = 0.7) -> list[P.Candidate]:
    return [P.Candidate(image_id=f"img{i}", keypoints=make_pose(i), aspect=aspect, facing="front") for i in range(n)]


def test_validate_clamps_and_rounds():
    kps = P.default_pose()
    kps[0] = {"x": 1.234567, "y": -0.5, "state": "occluded"}
    out = P.validate_keypoints(kps)
    assert out[0] == {"x": 1.0, "y": 0.0, "state": "occluded"}


@pytest.mark.parametrize("bad", [[], [{"x": 0, "y": 0}] * 12, [{"x": "a", "y": 0}] * 13,
                                 [{"x": 0, "y": 0, "state": "hidden"}] * 13])
def test_validate_rejects(bad):
    with pytest.raises(P.PoseError):
        P.validate_keypoints(bad)


def test_identical_pose_scores_100():
    kps = make_pose(1)
    m = P.match(kps, 0.7, kps, 0.7)
    assert m.score == pytest.approx(100) and m.mean_angle == pytest.approx(0, abs=1e-6)


def test_same_pose_with_noise_is_top3():
    cands = candidates(12)
    for target in range(12):
        query = jitter(cands[target].keypoints, 0.015, seed=target)
        hits = P.search(query, 0.7, cands, top_k=3)
        assert cands[target].image_id in [h.candidate.image_id for h in hits], target


def test_scale_and_position_invariant():
    kps = make_pose(3)
    small = [{**k, "x": 0.2 + k["x"] * 0.5, "y": 0.1 + k["y"] * 0.5} for k in kps]
    assert P.match(kps, 0.7, small, 0.7).score == pytest.approx(100)


def test_aspect_ratio_is_corrected():
    """같은 실제 모양이면 이미지 비율이 달라도 같은 포즈로 본다."""
    kps = make_pose(4)
    # 가로가 2배 넓은 이미지에 같은 모양을 그리면 x 비율은 절반이 된다
    wide = [{**k, "x": 0.25 + (k["x"] - 0.5) * 0.5} for k in kps]
    assert P.match(kps, 0.7, wide, 1.4).score == pytest.approx(100, abs=1e-6)
    assert P.match(kps, 0.7, wide, 0.7).score < 100  # 보정하지 않으면 달라진다


def test_mirror_option():
    cands = candidates(12)
    query = P.mirror_keypoints(cands[5].keypoints)
    on = P.search(query, 0.7, cands, allow_mirror=True, top_k=1)
    assert on[0].candidate.image_id == "img5" and on[0].match.flip
    off = P.search(query, 0.7, cands, allow_mirror=False, top_k=12)
    assert "img5" not in [h.candidate.image_id for h in off[:1]]


def test_mirror_twice_is_identity():
    kps = make_pose(7)
    assert P.mirror_keypoints(P.mirror_keypoints(kps)) == [{**k, "x": round(k["x"], 4)} for k in kps]


def test_absent_joint_excludes_bones():
    kps = make_pose(2)
    other = [dict(k) for k in kps]
    other[P.J["l_wrist"]] = {"x": 0.0, "y": 0.0, "state": "absent"}  # 위치가 엉망이어도 없음이면 무시
    m = P.match(kps, 0.7, other, 0.7, allow_mirror=False)
    assert m.score == pytest.approx(100) and m.n_bones == len(P.BONES) - 1


def test_occluded_has_lower_weight():
    kps = make_pose(2)
    moved = [dict(k) for k in kps]
    moved[P.J["l_wrist"]] = {**moved[P.J["l_wrist"]], "x": moved[P.J["l_wrist"]]["x"] + 0.05}
    occluded = [dict(k) for k in moved]
    occluded[P.J["l_wrist"]]["state"] = "occluded"
    s_vis = P.match(kps, 0.7, moved, 0.7, allow_mirror=False).score
    s_occ = P.match(kps, 0.7, occluded, 0.7, allow_mirror=False).score
    assert s_occ > s_vis  # 같은 오차라도 가려진 관절은 점수에 덜 반영된다


def test_cutoff_rejects_large_bone_error():
    kps = P.default_pose()
    bent = [dict(k) for k in kps]
    # 왼쪽 아래팔을 반대 방향으로 꺾는다 (각도 오차 > 60도)
    e, w = bent[P.J["l_elbow"]], bent[P.J["l_wrist"]]
    bent[P.J["l_wrist"]] = {**w, "x": e["x"] - (w["x"] - e["x"]), "y": e["y"] - (w["y"] - e["y"])}
    assert P.match(kps, 0.7, bent, 0.7, allow_mirror=False) is None
    assert P.match(kps, 0.7, bent, 0.7, allow_mirror=False, max_angle=180) is not None


def test_part_selection_ignores_other_bones():
    kps = make_pose(8)
    legs_changed = [dict(k) for k in kps]
    for key in ("l_knee", "l_ankle", "r_knee", "r_ankle"):
        legs_changed[P.J[key]] = {**legs_changed[P.J[key]], "x": 0.5, "y": 0.6}
    assert P.match(kps, 0.7, legs_changed, 0.7, part="arms").score == pytest.approx(100)
    assert P.match(kps, 0.7, legs_changed, 0.7, part="full") is None or \
        P.match(kps, 0.7, legs_changed, 0.7, part="full").score < 100


def test_coverage_too_low_returns_none():
    kps = P.default_pose()
    mostly_absent = [{**k, "state": "absent"} for k in kps]
    for key in ("l_shoulder", "l_elbow"):
        mostly_absent[P.J[key]]["state"] = "visible"
    assert P.match(kps, 0.7, mostly_absent, 0.7) is None


def test_facing_filter_and_dedupe():
    base = make_pose(9)
    cands = [P.Candidate("a", base, 0.7, "front", "kim"), P.Candidate("a", jitter(base, 0.05), 0.7, "front", "lee"),
             P.Candidate("b", base, 0.7, "back", "kim")]
    hits = P.search(base, 0.7, cands)
    assert [h.candidate.image_id for h in hits] == ["a", "b"] and hits[0].candidate.annotator == "kim"
    assert [h.candidate.image_id for h in P.search(base, 0.7, cands, facings=["back"])] == ["b"]


def test_normalize_torso_length_one():
    import numpy as np
    n = P.normalize(P.default_pose(), 0.7)
    shoulder_mid = (n[P.J["l_shoulder"]] + n[P.J["r_shoulder"]]) / 2
    hip_mid = (n[P.J["l_hip"]] + n[P.J["r_hip"]]) / 2
    assert np.allclose(hip_mid, 0) and np.linalg.norm(shoulder_mid) == pytest.approx(1)
