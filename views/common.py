"""페이지들이 같이 쓰는 UI 조각."""
import streamlit as st

from core.dataset import thumbnail

PENDING_QUERY = "pending_query_idx"  # 갤러리 → 유사 검색으로 넘길 이미지


def pick_method(methods: list, label: str, key: str):
    if not methods:
        st.error("이 기능을 지원하는 검색 방식이 등록되어 있지 않습니다.")
        st.stop()
    if len(methods) == 1:
        method = methods[0]
    else:
        method = st.selectbox(label, methods, format_func=lambda m: m.name, key=key)
    st.caption(f"**{method.name}** — {method.description}")
    return method


def image_card(df, idx: int, caption: str | None = None) -> None:
    row = df.iloc[idx]
    st.image(thumbnail(row["img_path"]), width="stretch")
    st.markdown(f"[`{row['id']}`]({row['source_page']})" + (f" · {caption}" if caption else ""))


def paginate(n_items: int, per_page: int, key: str) -> tuple[int, int]:
    n_pages = max(1, -(-n_items // per_page))
    page = st.number_input(f"페이지 (전체 {n_pages})", 1, n_pages, 1, key=key) if n_pages > 1 else 1
    start = (page - 1) * per_page
    return start, min(start + per_page, n_items)


def go_similar(idx: int) -> None:
    st.session_state[PENDING_QUERY] = idx


def current_user() -> str:
    """입장한 팀원의 이름 (명단 기준, 유사 검색 평가자 = 라벨러). 방문자면 빈 문자열."""
    return st.session_state.get("evaluator", "").strip()


def is_visitor() -> bool:
    """방문자로 입장한 세션 (보기 전용). 입장 화면이 없는 실행(AppTest 등)에서는 False."""
    return st.session_state.get("auth_role") == "visitor"


def cannot_save_notice(what: str = "저장") -> None:
    """이름이 없어 저장할 수 없을 때 페이지 위쪽 안내. 방문자면 보기 전용 안내를 띄운다."""
    if is_visitor():
        st.info(f"👀 방문자 모드입니다. 화면은 모두 볼 수 있지만 {what}은(는) 팀원만 할 수 있어요.")
    else:
        st.warning(f"왼쪽 사이드바에서 팀원으로 입장해야 {what}할 수 있어요.")


def load_tags_or_stop(include_inactive: bool = False, require_any: bool = True):
    from core.db import DBError
    from core.tagging import load_tags
    try:
        tags = load_tags(include_inactive=include_inactive)
    except DBError as e:
        st.error(f"태그를 불러오지 못했습니다. {e}")
        st.stop()
    if require_any and not tags:
        st.warning("사용 중인 태그가 없습니다. **태그 관리** 페이지에서 태그를 추가하세요.")
        st.page_link("views/tags_admin.py", label="태그 관리로 이동", icon="🗂️")
        st.stop()
    return tags


def load_labels_or_stop():
    from core.db import DBError
    from core.tagging import load_labels
    try:
        return load_labels()
    except DBError as e:
        st.error(f"라벨을 불러오지 못했습니다. {e}")
        st.stop()


def save_label_safely(image_id: str, tag_key: str, value: int, labeler: str) -> bool:
    """버튼 on_click용. 저장 실패 시 앱을 멈추지 않고 화면 위에 안내를 띄운다. 성공 여부를 돌려준다."""
    from core.db import DBError
    from core.tagging import save_label
    try:
        save_label(image_id, tag_key, value, labeler)
        return True
    except DBError as e:
        st.error(f"라벨 저장에 실패했습니다. {e}")
        return False


def tag_label(t) -> str:
    return f"{t.name} ({t.group})" if t.group else t.name
