"""드로잉 레퍼런스 기술 검증 웹앱.  실행: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="드로잉 레퍼런스 검증", page_icon="🎨", layout="wide")

pages = {
    "검증": [
        st.Page("views/gallery.py", title="갤러리", icon="🖼️", default=True),
        st.Page("views/similar.py", title="유사 검색", icon="🔍"),
        st.Page("views/duplicates.py", title="중복 확인", icon="👯"),
    ],
    "자동 태깅": [
        st.Page("views/labeling.py", title="라벨링", icon="🏷️"),
        st.Page("views/train.py", title="학습 실행", icon="🧠"),
        st.Page("views/predictions.py", title="예측 확인", icon="🤖"),
        st.Page("views/tags_admin.py", title="태그 관리", icon="🗂️"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.text_input("내 이름", key="evaluator", placeholder="예: 김하영",
                  help="유사 검색 평가와 라벨링을 저장할 때 누가 했는지 기록합니다.")
    from core.db import is_configured
    st.caption("태그·라벨 저장소: **Supabase**" if is_configured() else ":red[Supabase 접속 정보 없음 — .env 확인]")

nav.run()
