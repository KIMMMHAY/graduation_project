import random

import pandas as pd
import streamlit as st
from PIL import Image

from core.dataset import load_metadata, thumbnail
from core.evaluation import DIFFERENT, SIMILAR, VERDICT_LABEL, eval_path, load_evals, save_verdict
from methods import ImageSearch, methods_of
from views.common import PENDING_QUERY, image_card, pick_method

TOP_K = 12
COLS = 4
QUERY_KEY = "sim_query"

st.title("🔍 유사 검색")
df = load_metadata()
method = pick_method(methods_of(ImageSearch), "검색 방식", key="sim_method")
evaluator = st.session_state.get("evaluator", "").strip()

# 갤러리에서 "유사 이미지 찾기"로 넘어온 경우
if PENDING_QUERY in st.session_state:
    st.session_state[QUERY_KEY] = st.session_state.pop(PENDING_QUERY)
st.session_state.setdefault(QUERY_KEY, 0)


def pick_random():
    st.session_state[QUERY_KEY] = random.randrange(len(df))


def show_hits(hits, query_id: str | None) -> None:
    """결과 그리드. query_id가 있으면 평가 버튼을 붙인다."""
    done = {}
    if query_id and evaluator:
        ev = load_evals(method.key)
        ev = ev[(ev["evaluator"] == evaluator) & (ev["query_id"] == query_id)]
        done = dict(zip(ev["result_id"], ev["verdict"]))

    cols = st.columns(COLS)
    for rank, hit in enumerate(hits, start=1):
        rid = df["id"].iat[hit.idx]
        with cols[(rank - 1) % COLS]:
            image_card(df, hit.idx, caption=f"#{rank} · **{method.format_score(hit.score)}**")
            if not query_id:
                continue
            verdict = done.get(rid)
            b1, b2 = st.columns(2)
            for col, value, icon in ((b1, SIMILAR, "👍"), (b2, DIFFERENT, "👎")):
                col.button(
                    f"{icon} {VERDICT_LABEL[value]}", key=f"ev_{query_id}_{rid}_{value}",
                    type="primary" if verdict == value else "secondary",
                    disabled=not evaluator, width="stretch",
                    on_click=save_verdict,
                    args=(method.key, evaluator, query_id, rid, rank, hit.score, value),
                )
    if query_id and evaluator:
        n_sim = sum(v == SIMILAR for v in done.values())
        st.info(f"이 쿼리: {len(done)}/{len(hits)}장 평가 · 비슷함 {n_sim}장")


tab_data, tab_upload, tab_stats = st.tabs(["데이터셋 이미지로 검색", "내 이미지로 검색", "평가 현황"])

with tab_data:
    if not evaluator:
        st.warning("왼쪽 사이드바에 **평가자 이름**을 입력해야 비슷함/다름 평가를 저장할 수 있어요.")
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    c1.selectbox("검색할 이미지", range(len(df)), key=QUERY_KEY,
                 format_func=lambda i: f"{df['id'].iat[i]}  —  {' '.join(df['tag_list'].iat[i][:6])}")
    c2.button("🎲 랜덤", on_click=pick_random, width="stretch")

    q = st.session_state[QUERY_KEY]
    left, right = st.columns([1, 3])
    with left:
        st.subheader("검색 이미지")
        image_card(df, q)
    with right:
        st.subheader("태그")
        st.write(" ".join(f"`{t}`" for t in df["tag_list"].iat[q]))

    st.divider()
    st.subheader(f"가까운 순 {TOP_K}장")
    show_hits(method.search_by_image(q, TOP_K), query_id=df["id"].iat[q])

with tab_upload:
    st.caption("데이터셋에 없는 그림으로 검색해 봅니다. 이 탭의 결과는 평가 저장 대상이 아닙니다.")
    file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "webp"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=240)
        try:
            show_hits(method.search_by_upload(img, TOP_K), query_id=None)
        except NotImplementedError:
            st.info(f"{method.name}은(는) 업로드 검색을 지원하지 않습니다.")

with tab_stats:
    ev = load_evals(method.key)
    st.caption(f"저장 위치: `{eval_path(method.key)}`")
    if ev.empty:
        st.info("아직 평가 기록이 없습니다.")
    else:
        ev["is_similar"] = ev["verdict"] == SIMILAR
        m1, m2, m3 = st.columns(3)
        m1.metric("평가한 쿼리", ev["query_id"].nunique())
        m2.metric("평가 수", len(ev))
        m3.metric("비슷함 비율", f"{ev['is_similar'].mean():.0%}")

        st.markdown("**평가자별**")
        st.dataframe(ev.groupby("evaluator").agg(평가수=("verdict", "size"), 비슷함비율=("is_similar", "mean"))
                     .style.format({"비슷함비율": "{:.0%}"}))
        st.markdown(f"**{method.score_name} 구간별 비슷함 비율** — 어느 값까지 믿을 만한지 판단하는 근거")
        bins = pd.cut(ev["score"], bins=8)
        st.dataframe(ev.groupby(bins, observed=True).agg(평가수=("verdict", "size"), 비슷함비율=("is_similar", "mean"))
                     .style.format({"비슷함비율": "{:.0%}"}))
        st.download_button("CSV 다운로드", ev.drop(columns="is_similar").to_csv(index=False).encode("utf-8-sig"),
                           file_name=eval_path(method.key).name, mime="text/csv")
