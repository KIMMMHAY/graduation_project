"""드로잉 레퍼런스 기술 검증 웹앱.  실행: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="드로잉 레퍼런스 검증", page_icon="🎨", layout="wide")

pages = {
    "팀": [
        st.Page("views/team.py", title="팀 현황", icon="👥"),
    ],
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
    "포즈": [
        st.Page("views/pose_label.py", title="포즈 라벨링", icon="🦴"),
        st.Page("views/pose_search.py", title="포즈 검색 (2D)", icon="🔎"),
        st.Page("views/pose_search_3d.py", title="포즈 검색 (3D)", icon="🧍"),
        st.Page("views/pose_accuracy.py", title="포즈 정확도", icon="📏"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.text_input("내 이름", key="evaluator", placeholder="예: 김하영",
                  help="유사 검색 평가와 라벨링을 저장할 때 누가 했는지 기록합니다.")
    from core.db import is_configured
    st.caption("태그·라벨 저장소: **Supabase**" if is_configured() else ":red[Supabase 접속 정보 없음 — .env 확인]")
    from views.common import name_check
    name_check()

from core.dataset import META_CSV  # noqa: E402
if not is_configured() and not META_CSV.exists():  # 이미지 목록을 읽을 곳이 없으면 모든 페이지가 오류로 멈춘다
    st.error("**접속 정보 파일(`.env`)이 없습니다.** 처음 설치했다면 아래 순서대로 해 주세요.\n\n"
             "1. 프로젝트 폴더의 `.env.example`을 복사해 이름을 `.env`로 바꿉니다 (`.env.txt`가 되지 않게 주의)\n"
             "2. 팀장에게 받은 `SUPABASE_URL`, `SUPABASE_KEY` 값을 넣습니다\n"
             "3. 터미널에서 `Ctrl+C`로 앱을 끄고 `python -m streamlit run app.py`로 다시 켭니다\n\n"
             "자세한 안내는 `README.md`에 있습니다.")
    st.stop()

nav.run()
