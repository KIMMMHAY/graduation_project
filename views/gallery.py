import streamlit as st

from core.dataset import load_metadata
from methods import TagSource, methods_of
from methods.predicted_tags import predicted_tags_of
from views.common import go_similar, image_card, paginate, pick_method

COLS = 4
PREVIEW_TAGS = 8

st.title("🖼️ 갤러리")
df = load_metadata()
source = pick_method(methods_of(TagSource), "태그 출처", key="gallery_source")
counts = source.tag_counts()

c1, c2, c3 = st.columns([4, 1, 2])
selected = c1.multiselect(
    "태그 필터", [t for t, _ in counts.most_common()],
    format_func=lambda t: f"{t} ({counts[t]})", placeholder="태그를 선택하세요 (많이 쓰인 순)",
)
match_all = c2.radio("조건", ["모두 포함", "하나라도"], horizontal=False) == "모두 포함"
keyword = c3.text_input("태그 부분 검색", placeholder="예: sword, sitting").strip().lower()

idxs = source.filter_by_tags(selected, match_all)
if keyword:
    idxs = [i for i in idxs if any(keyword in t for t in source.tags_of(i))]

st.caption(f"{len(idxs)} / {len(df)}장")
if not idxs:
    st.info("조건에 맞는 이미지가 없습니다.")
    st.stop()

start, end = paginate(len(idxs), 24, key="gallery_page")
cols = st.columns(COLS)
for n, idx in enumerate(idxs[start:end]):
    tags = df["tag_list"].iat[idx]  # 아래 표시는 항상 Safebooru 원본 태그
    with cols[n % COLS]:
        image_card(df, idx)
        st.caption(" · ".join(tags[:PREVIEW_TAGS]) + (" …" if len(tags) > PREVIEW_TAGS else ""))
        if predicted := predicted_tags_of(df["id"].iat[idx]):
            st.markdown("🤖 " + " ".join(f":blue-background[{name} {p:.2f}]" for name, p in predicted))
        with st.expander(f"태그 {len(tags)}개"):
            st.write(" ".join(f"`{t}`" for t in tags))
        if st.button("유사 이미지 찾기", key=f"sim_{idx}", on_click=go_similar, args=(idx,)):
            st.switch_page("views/similar.py")
