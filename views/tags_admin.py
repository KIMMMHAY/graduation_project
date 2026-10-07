import pandas as pd
import streamlit as st

from core.dataset import load_metadata
from core.db import DBError
from core.tagging import (NEGATIVE, POSITIVE, TagError, add_tag, consolidate, delete_tag, invalidate_tags_cache,
                          set_tag_active, update_tag)
from views.common import cannot_save_notice, is_visitor, load_labels_or_stop, load_tags_or_stop

MSG = "tags_admin_msg"
NEW_FIELDS = {"new_key": "", "new_name": "", "new_group": None, "new_definition": "", "new_hint": ""}

st.title("🗂️ 태그 관리")
st.caption("팀 태그는 Supabase에 저장되어 모든 팀원이 같은 목록을 씁니다. 다른 팀원이 바꾼 내용은 최대 30초 뒤에 보입니다.")
read_only = is_visitor()
if read_only:
    cannot_save_notice("태그 추가·수정·삭제")


def run(action, *args, success: str, on_success=None) -> None:
    """버튼 콜백 공통: 실행하고 결과 메시지를 다음 화면에 띄운다."""
    try:
        action(*args)
    except TagError as e:
        st.session_state[MSG] = ("error", str(e))
        return
    except DBError as e:
        st.session_state[MSG] = ("error", f"저장하지 못했습니다. {e}")
        return
    st.session_state[MSG] = ("success", success)
    if on_success:
        on_success()


if msg := st.session_state.pop(MSG, None):
    (st.success if msg[0] == "success" else st.error)(msg[1])

st.button("🔄 새로고침", on_click=invalidate_tags_cache, help="다른 팀원이 바꾼 내용을 바로 불러옵니다.")

df = load_metadata()
tags = load_tags_or_stop(include_inactive=True, require_any=False)
labels = load_labels_or_stop()
merged = consolidate(labels)
n_labels = labels.groupby("tag_key").size()
n_pos = merged[merged["value"] == POSITIVE].groupby("tag_key").size()
n_neg = merged[merged["value"] == NEGATIVE].groupby("tag_key").size()
hint_images = {t.key: int(df["tag_list"].map(lambda ts, h=t.booru_hint: h in ts).sum()) if t.booru_hint else None
               for t in tags}

# ---------- 분류 축별 목록 ----------
groups = list(dict.fromkeys(t.group or "(분류 없음)" for t in tags))
if not tags:
    st.info("아직 태그가 없습니다. 아래에서 첫 태그를 추가하세요.")
for group in groups:
    in_group = [t for t in tags if (t.group or "(분류 없음)") == group]
    st.subheader(f"{group}  ·  {sum(t.active for t in in_group)}개 사용 중")
    st.dataframe(
        pd.DataFrame({
            "상태": ["✅ 사용 중" if t.active else "⏸️ 사용 안 함" for t in in_group],
            "이름": [t.name for t in in_group],
            "key": [t.key for t in in_group],
            "정의": [t.definition for t in in_group],
            "Safebooru 태그": [f"{t.booru_hint} ({hint_images[t.key]}장)" if t.booru_hint else "" for t in in_group],
            "해당": [int(n_pos.get(t.key, 0)) for t in in_group],
            "해당 아님": [int(n_neg.get(t.key, 0)) for t in in_group],
            "라벨 건수": [int(n_labels.get(t.key, 0)) for t in in_group],
        }),
        hide_index=True, width="stretch",
        column_config={
            "정의": st.column_config.TextColumn(width="large"),
            "해당": st.column_config.NumberColumn(help="팀 다수결 기준"),
            "해당 아님": st.column_config.NumberColumn(help="팀 다수결 기준"),
            "라벨 건수": st.column_config.NumberColumn(help="모든 라벨러의 라벨 수 합계. 1건 이상이면 삭제할 수 없습니다"),
        },
    )

# ---------- 수정 ----------
if tags:
    st.divider()
    st.subheader("✏️ 태그 수정")
    by_key = {t.key: t for t in tags}
    key = st.selectbox("수정할 태그", list(by_key), key="edit_tag",
                       format_func=lambda k: f"{by_key[k].name} ({by_key[k].group})" + ("" if by_key[k].active else " · 사용 안 함"))
    tag = by_key[key]
    with st.form(f"edit_{key}"):
        st.text_input("key (변경 불가)", tag.key, disabled=True)
        st.text_input("표시 이름", tag.name, key=f"edit_name_{key}")
        st.text_area("정의 (라벨링 기준)", tag.definition, key=f"edit_def_{key}")
        st.form_submit_button(
            "저장", type="primary", disabled=read_only,
            on_click=lambda: run(update_tag, key, st.session_state[f"edit_name_{key}"],
                                 st.session_state[f"edit_def_{key}"], success=f"'{key}' 태그를 저장했습니다."))

    used = int(n_labels.get(key, 0))
    if not tag.active:
        st.button("▶️ 다시 사용", disabled=read_only, on_click=run, args=(set_tag_active, key, True),
                  kwargs={"success": f"'{tag.name}' 태그를 다시 사용합니다."})
    elif used:
        st.caption(f"라벨이 {used}건 있어 삭제할 수 없습니다. 사용 안 함으로 바꾸면 라벨링·학습·예측 화면에서 숨겨지고, 라벨은 보존됩니다.")
        st.button("⏸️ 사용 안 함", disabled=read_only, on_click=run, args=(set_tag_active, key, False),
                  kwargs={"success": f"'{tag.name}' 태그를 사용 안 함으로 바꿨습니다."})
    else:
        st.caption("아직 라벨이 없는 태그라 삭제할 수 있습니다.")
        sure = st.checkbox(f"'{tag.name}' 태그를 삭제합니다 (되돌릴 수 없음)", key=f"del_ok_{key}")
        st.button("🗑️ 삭제", disabled=not sure or read_only, on_click=run, args=(delete_tag, key),
                  kwargs={"success": f"'{tag.name}' 태그를 삭제했습니다."})

# ---------- 추가 ----------
st.divider()
st.subheader("➕ 새 태그 추가")
for k, v in NEW_FIELDS.items():
    st.session_state.setdefault(k, v)


def reset_new_fields() -> None:
    for k, v in NEW_FIELDS.items():
        st.session_state[k] = v


def submit_new() -> None:
    s = st.session_state
    run(add_tag, s["new_key"], s["new_name"], s["new_group"] or "", s["new_definition"], s["new_hint"],
        success=f"'{s['new_name'].strip()}' 태그를 추가했습니다.", on_success=reset_new_fields)


with st.form("add_tag"):
    c1, c2, c3 = st.columns(3)
    c1.text_input("영문 key", key="new_key", placeholder="예: low_angle", help="영문 소문자·숫자·밑줄. 만든 뒤에는 바꿀 수 없습니다.")
    c2.text_input("표시 이름", key="new_name", placeholder="예: 로우앵글")
    c3.selectbox("분류 축", [g for g in groups if g != "(분류 없음)"], key="new_group", accept_new_options=True,
                 placeholder="선택하거나 새로 입력", help="목록에 없으면 직접 입력하면 새 분류 축이 만들어집니다.")
    st.text_area("정의 (라벨링 기준)", key="new_definition",
                 placeholder="예: 카메라가 인물의 허리 아래에서 올려다보는 구도")
    st.text_input("대응되는 Safebooru 태그 (선택)", key="new_hint", placeholder="예: from_below",
                  help="입력하면 라벨링 페이지에서 이 태그가 붙은 이미지를 먼저 보여줄 수 있어 양성 예시를 빨리 모읍니다.")
    st.form_submit_button("추가", type="primary", disabled=read_only, on_click=submit_new)
