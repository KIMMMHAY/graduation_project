import streamlit as st

from components.pose_editor import pin_editor
from core import pose as P
from core import poses as store
from core.dataset import load_metadata
from core.db import DBError
from views.common import current_user
from views.pose_common import DEFAULT_ASPECT, GUIDE, aspects_of, load_poses_or_stop, pose_thumbnail

RESULT = "pose_search_result"
NONCE = "pose_search_nonce"
COLS = 4
START_DEFAULT = "__default__"

st.title("🔎 포즈 검색 (2D)")
st.info(GUIDE)
df = load_metadata()
me = current_user()
aspects = aspects_of(df)
row_of = {i: n for n, i in enumerate(df["id"])}
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
    part = st.radio("비교할 부위", list(P.PARTS), format_func=lambda p: P.PARTS[p][0], horizontal=True, key="pose_search_part")
    allow_mirror = st.toggle("좌우 반전 포함", value=True, key="pose_search_mirror",
                             help="좌우가 뒤집힌 포즈도 찾습니다. 결과 썸네일은 뒤집어서 보여줍니다.")
    facings = st.multiselect("몸이 향한 방향", list(P.FACINGS), default=list(P.FACINGS), format_func=P.FACINGS.get,
                             key="pose_search_facing")
    max_angle = st.slider("허용 범위 (뼈 하나의 최대 각도 오차)", 20, 120, int(P.DEFAULT_MAX_ANGLE), 5, format="%d°",
                          key="pose_search_max_angle", help="이보다 크게 어긋난 뼈가 하나라도 있으면 결과에서 뺍니다.")
    top_k = st.number_input("결과 수", 4, 48, 12, 4, key="pose_search_topk")
    c1, c2 = st.columns(2)
    do_search = c1.button("🔎 검색", type="primary", width="stretch")
    if c2.button("↺ 초기화", width="stretch"):
        st.session_state[NONCE] = nonce + 1
        st.session_state.pop(RESULT, None)
        st.rerun()

if do_search:
    options = {"part": part, "mirror": allow_mirror, "facings": sorted(facings), "max_angle": max_angle}
    try:
        cands = store.searchable_candidates(aspects)
    except DBError as e:
        st.error(f"검색 대상을 불러오지 못했습니다. {e}")
        st.stop()
    hits = P.search(query, q_aspect, cands, part=part, allow_mirror=allow_mirror,
                    facings=facings or None, max_angle=max_angle, top_k=int(top_k))
    st.session_state[RESULT] = {
        "query": {"keypoints": query, "aspect": q_aspect}, "options": options, "n_candidates": len(cands),
        "hits": [{"image_id": h.candidate.image_id, "annotator": h.candidate.annotator, "facing": h.candidate.facing,
                  "keypoints": h.candidate.keypoints, "score": h.match.score, "mean_angle": h.match.mean_angle,
                  "flip": h.match.flip} for h in hits],
    }

# ---------- 결과 ----------
res = st.session_state.get(RESULT)
if res:
    st.divider()
    if res["query"]["keypoints"] != query:
        st.caption("⚠️ 핀을 옮겼습니다. 바뀐 포즈로 보려면 **검색**을 다시 누르세요.")
    hits = res["hits"]
    st.subheader(f"가까운 순 {len(hits)}장")
    st.caption(f"후보 {res['n_candidates']}개 포즈 중 허용 범위 안에 든 결과 · 점수는 100에 가까울수록 비슷함")
    if not hits:
        st.info("허용 범위 안에 드는 포즈가 없습니다. 허용 범위를 넓히거나 비교할 부위를 줄여 보세요.")
    qhash = store.query_hash(res["query"], res["options"])
    verdicts = {}
    if me:
        try:
            ev = store.load_pose_evals()
            ev = ev[(ev["evaluator"] == me) & (ev["query_hash"] == qhash)]
            verdicts = dict(zip(ev["result_image_id"], ev["verdict"]))
        except DBError as e:
            st.warning(f"평가 기록을 불러오지 못했습니다. {e}")
    elif hits:
        st.caption("사이드바에 **내 이름**을 입력하면 비슷함/다름 평가를 저장할 수 있어요.")

    def evaluate(rank: int, hit: dict, verdict: str) -> None:
        try:
            store.save_pose_eval(me, res["query"], res["options"], hit["image_id"], hit["annotator"],
                                 rank, hit["score"], hit["flip"], verdict)
        except DBError as e:
            st.error(f"평가를 저장하지 못했습니다. {e}")

    cols = st.columns(COLS)
    for rank, hit in enumerate(hits, start=1):
        with cols[(rank - 1) % COLS]:
            img_path = df["img_path"].iat[row_of[hit["image_id"]]]
            st.image(pose_thumbnail(img_path, hit["keypoints"], hit["flip"]), width="stretch")
            facing = P.FACINGS.get(hit["facing"], "?")
            st.markdown(f"#{rank} · **{hit['score']:.1f}점** · 평균 {hit['mean_angle']:.0f}°  \n"
                        f"`{hit['image_id']}` · {facing}" + (" · ↔ 반전" if hit["flip"] else ""))
            current = verdicts.get(hit["image_id"])
            b1, b2 = st.columns(2)
            for col, value, text in ((b1, "similar", "👍 비슷함"), (b2, "different", "👎 다름")):
                col.button(text, key=f"pose_ev_{qhash}_{hit['image_id']}_{value}", width="stretch",
                           type="primary" if current == value else "secondary", disabled=not me,
                           on_click=evaluate, args=(rank, hit, value))

    with st.expander("평가 현황 (팀 전체)"):
        try:
            ev = store.load_pose_evals()
        except DBError as e:
            st.warning(str(e))
            ev = None
        if ev is None or ev.empty:
            st.caption("아직 평가가 없습니다.")
        else:
            ev["is_similar"] = ev["verdict"] == "similar"
            ev["부위"] = ev["options"].map(lambda o: P.PARTS.get((o or {}).get("part", "full"), ("?",))[0])
            a, b, c = st.columns(3)
            a.metric("평가한 검색", ev["query_hash"].nunique())
            b.metric("평가 수", len(ev))
            c.metric("비슷함 비율", f"{ev['is_similar'].mean():.0%}")
            st.dataframe(ev.groupby("부위").agg(평가수=("verdict", "size"), 비슷함비율=("is_similar", "mean"))
                         .style.format({"비슷함비율": "{:.0%}"}))
