import streamlit as st

from core.dataset import load_metadata
from views import labeling_by_image, labeling_by_tag
from views.common import current_user, load_tags_or_stop

VIEW_IMAGE, VIEW_TAG = "이미지별 (태그 한 번에)", "태그별 (하나씩)"
# Streamlit은 화면에 그리지 않은 위젯의 값을 지운다. 방식을 전환했다가 돌아와도 선택이 유지되도록 다시 넣어 둔다
KEEP_PREFIXES = ("label_tag", "label_mode", "label_order", "label_hide", "il_order", "il_hide_mine", "il_hide_team")
for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(KEEP_PREFIXES)]:
    st.session_state[k] = st.session_state[k]

st.title("🏷️ 라벨링")
df = load_metadata()
tags = load_tags_or_stop()
labeler = current_user()
if not labeler:
    st.warning("왼쪽 사이드바에 **내 이름**을 입력해야 라벨을 저장할 수 있어요.")

view = st.radio("라벨링 방식", [VIEW_IMAGE, VIEW_TAG], horizontal=True, key="label_view",
                help="이미지별: 한 장을 보고 해당하는 태그를 모두 체크합니다. "
                     "태그별: 태그 하나를 정하고 이미지마다 해당/아님을 누릅니다(새 태그만 빠르게 채울 때).")
if view == VIEW_IMAGE:
    labeling_by_image.render(df, tags, labeler)
else:
    labeling_by_tag.render(df, tags, labeler)
