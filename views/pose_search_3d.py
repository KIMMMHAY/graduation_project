import streamlit as st

from components.mannequin import mannequin
from core import pose as P
from core.dataset import load_metadata
from views.common import current_user
from views.pose_common import aspects_of, load_poses_or_stop, render_results, run_search, search_options

RESULT = "pose3d_result"
NONCE = "pose3d_nonce"

st.title("🧍 포즈 검색 (3D 마네킹)")
st.info("마네킹의 관절 공을 끌어 포즈를 만들고, 원하는 각도로 카메라를 돌린 뒤 **검색**을 누르세요. "
        "현재 화면에 보이는 모습(2D 투영) 그대로 라벨된 일러스트와 비교합니다. "
        "좌우는 **캐릭터 기준**이에요 (파랑 = 캐릭터의 왼쪽).")
df = load_metadata()
me = current_user()
aspects = aspects_of(df)
poses = load_poses_or_stop()
done = poses[(poses["status"] == P.DONE) & poses["source"].isin(P.SEARCHABLE_SOURCES) & poses["image_id"].isin(aspects)]
st.caption(f"검색 대상: 완료된 포즈 {done['image_id'].nunique()}장 (사람이 확정한 것만)")

nonce = st.session_state.get(NONCE, 0)
version = f"mannequin:{nonce}"
left, right = st.columns([3, 2])
with left:
    snap = mannequin(key=f"pose3d_{version}", version=version, height=560)
with right:
    if snap:
        st.caption(f"현재 마네킹 방향: **{P.FACINGS[snap['facing']]}**")
        same_facing = st.toggle("마네킹과 같은 방향의 그림만", value=False, key="pose3d_same_facing",
                                help="정면 마네킹이면 정면 그림만, 뒤에서 보면 뒷모습 그림만 찾습니다.")
    else:
        same_facing = False
        st.caption("마네킹을 불러오는 중...")
    options = search_options("pose3d")
    if same_facing and snap:
        options["facings"] = [snap["facing"]]
    c1, c2 = st.columns(2)
    do_search = c1.button("🔎 검색", type="primary", width="stretch", disabled=snap is None)
    if c2.button("↺ 새 마네킹", width="stretch", help="포즈와 카메라를 처음 상태로 되돌립니다."):
        st.session_state[NONCE] = nonce + 1
        st.session_state.pop(RESULT, None)
        st.rerun()

if do_search and snap:
    if (res := run_search(snap["keypoints"], snap["aspect"], options, aspects, mode="3d")) is not None:
        res["mannequin_state"] = snap["state"]
        st.session_state[RESULT] = res

res = st.session_state.get(RESULT)
if res:
    st.divider()
    if snap and res["query"]["keypoints"] != snap["keypoints"]:
        st.caption("⚠️ 마네킹이나 카메라를 움직였습니다. 바뀐 모습으로 보려면 **검색**을 다시 누르세요.")
    render_results(df, res, me)
