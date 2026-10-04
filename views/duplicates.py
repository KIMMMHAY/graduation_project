import streamlit as st

from core.dataset import load_metadata
from methods import PairFinder, methods_of
from views.common import image_card, paginate, pick_method

PER_PAGE = 10

st.title("👯 중복 확인")
df = load_metadata()
method = pick_method(methods_of(PairFinder), "검색 방식", key="dup_method")

lo, hi, default = method.threshold_range
threshold = st.slider(f"{method.score_name} 임계값 (이하를 중복으로 판단)", lo, hi, default)
pairs = method.find_pairs(threshold)

st.metric("중복 의심 쌍", f"{len(pairs)}쌍")
if not pairs:
    st.info("임계값 이하인 쌍이 없습니다. 슬라이더를 올려 보세요.")
    st.stop()

start, end = paginate(len(pairs), PER_PAGE, key="dup_page")
for i, j, score in pairs[start:end]:
    st.markdown(f"#### {method.format_score(score)}")
    a, b, info = st.columns([2, 2, 3])
    with a:
        image_card(df, i)
    with b:
        image_card(df, j)
    with info:
        ti, tj = set(df["tag_list"].iat[i]), set(df["tag_list"].iat[j])
        st.caption(f"태그 겹침 {len(ti & tj)} / {len(ti | tj)}")
        st.write("공통: " + " ".join(f"`{t}`" for t in sorted(ti & tj)))
    st.divider()
