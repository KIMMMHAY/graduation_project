import streamlit as st

from components.pose_editor import pin_editor
from core import pose as P
from core.dataset import load_metadata
from views.common import current_user
from views.pose_common import (DEFAULT_ASPECT, GUIDE, aspects_of, load_poses_or_stop, render_results,
                               run_search, search_options)

RESULT = "pose_search_result"
NONCE = "pose_search_nonce"
START_DEFAULT = "__default__"

st.title("🔎 포즈 검색 (2D)")
st.info(GUIDE)
df = load_metadata()
me = current_user()
aspects = aspects_of(df)
poses = load_poses_or_stop()
done = poses[(poses["status"] == P.DONE) & poses["source"].isin(P.SEARCHABLE_SOURCES) & poses["image_id"].isin(aspects)]
st.caption(f"검색 대상: 완료된 포즈 {done['image_id'].nunique()}장 (사람이 확정한 것만)")

# ---------- 질의 포즈 ----------
starts = [START_DEFAULT] + sorted(done["image_id"].unique())
start = st.selectbox("시작 포즈", starts, key="pose_search_start",
                     format_func=lambda s: "표준 자세" if s == START_DEFAULT else f"라벨된 포즈에서 시작: {s}",
                     help="기존에 라벨된 포즈를 불러와 조금 바꿔 검색해 볼 수 있습니다.")
if start == START_DEFAULT:
    initial, q_aspect = P.default_pose(), DEFAULT_ASPECT
else:
    src = done[done["image_id"] == start].iloc[0]
    initial, q_aspect = P.validate_keypoints(src["keypoints"]), aspects[start]
nonce = st.session_state.get(NONCE, 0)
version = f"search:{start}:{nonce}"

left, right = st.columns([3, 2])
with left:
    query = pin_editor(key=f"pose_search_editor_{version}", version=version, keypoints=initial, aspect=q_aspect, height=560)
with right:
    options = search_options("pose_search")
    c1, c2 = st.columns(2)
    do_search = c1.button("🔎 검색", type="primary", width="stretch")
    if c2.button("↺ 초기화", width="stretch"):
        st.session_state[NONCE] = nonce + 1
        st.session_state.pop(RESULT, None)
        st.rerun()

if do_search:
    if (res := run_search(query, q_aspect, options, aspects)) is not None:
        st.session_state[RESULT] = res

# ---------- 결과 ----------
res = st.session_state.get(RESULT)
if res:
    st.divider()
    if res["query"]["keypoints"] != query:
        st.caption("⚠️ 핀을 옮겼습니다. 바뀐 포즈로 보려면 **검색**을 다시 누르세요.")
    render_results(df, res, me)
