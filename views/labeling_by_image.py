"""라벨링 '이미지별 (태그 한 번에)' 화면.

이미지 한 장을 보고 사용 중인 팀 태그 전체를 체크박스로 지정해 한 번에 저장한다.
체크 = 해당(1), 해제 = 아님(0), 보류 = 저장 안 함. 오른쪽 체크박스 영역은 st.fragment라서
체크할 때마다 그 영역만 다시 그려진다 (DB를 다시 읽지 않음, 체크 수 안내가 바로 바뀜).
"""
import hashlib
import json
import random

import numpy as np
import pandas as pd
import streamlit as st

from core import image_labeling as IL
from core.db import DBError
from core.tagging import NEGATIVE, POSITIVE, PRED_THRESHOLD, TagError, consolidate, save_labels
from methods.predicted_tags import load_predictions
from views.common import load_labels_or_stop

IMAGE_WIDTH = 520
ORDER_RANDOM, ORDER_PARTIAL = "무작위", "새 태그만 남은 이미지 우선"
POS, REVIEW, LAST = "il_pos", "il_review", "il_last"
SKIPPED, HELD_DONE, NONCE, MSG = "il_skipped", "il_held_done", "il_nonce", "il_msg"


def _cb_key(image_id: str, tag_key: str, version: str) -> str:
    return f"il_cb_{image_id}_{tag_key}_{version}"


def render(df: pd.DataFrame, tags: list, labeler: str) -> None:
    active = [t.key for t in tags]
    labels = load_labels_or_stop()
    mine_done = IL.finished_images(labels, labeler, active) if labeler else set()
    partial = IL.partial_images(labels, labeler, active) if labeler else set()
    team_done = IL.team_finished_images(labels, active)

    # ---------- 진행 현황 ----------
    m1, m2, m3 = st.columns(3)
    m1.metric("내가 끝낸 이미지", f"{len(mine_done)} / {len(df)}", help="사용 중인 모든 태그에 내 라벨이 있는 이미지")
    m2.metric("팀에서 끝낸 이미지", f"{len(team_done)} / {len(df)}", help="한 명 이상이 모든 태그를 끝낸 이미지")
    m3.metric("사용 중 태그", len(tags))
    _tag_status_table(tags, labels)

    # ---------- 순서·필터 ----------
    c1, c2, c3 = st.columns([2, 2, 2])
    order = c1.radio("표시 순서", [ORDER_RANDOM, ORDER_PARTIAL], horizontal=True, key="il_order",
                     help="'새 태그만 남은 이미지'는 일부 태그만 라벨링한 이미지(태그가 나중에 추가된 경우)를 먼저 보여줍니다.")
    hide_mine = c2.toggle("내가 끝낸 이미지 숨기기", value=True, key="il_hide_mine")
    hide_team = c3.toggle("다른 팀원이 끝낸 이미지도 숨기기", value=False, key="il_hide_team")

    seed = st.session_state.setdefault("label_seed", random.randrange(1 << 30))
    ids = [df["id"].iat[i] for i in np.random.default_rng(seed).permutation(len(df))]  # 기존과 같은 고정 시드
    if order == ORDER_PARTIAL:
        ids = [i for i in ids if i in partial] + [i for i in ids if i not in partial]
    skipped = st.session_state.setdefault(SKIPPED, set())
    held_done = st.session_state.setdefault(HELD_DONE, set())  # 보류하고 저장한 이미지 (이번 세션에서는 다시 안 보임)
    if hide_mine:
        ids = [i for i in ids if i not in mine_done and i not in held_done]
    if hide_team:
        ids = [i for i in ids if i not in team_done]
    ids = [i for i in ids if i not in skipped]

    review = st.session_state.get(REVIEW)
    if skipped:
        st.button(f"건너뛴 {len(skipped)}장 다시 보기", on_click=skipped.clear)
    if msg := st.session_state.pop(MSG, None):
        (st.success if msg[0] == "ok" else st.error)(msg[1])
    if not ids and not review:
        st.success("모든 이미지를 끝냈습니다. 🎉 숨기기를 끄면 다시 볼 수 있어요.")
        return

    pos = min(st.session_state.get(POS, 0), max(len(ids) - 1, 0))
    st.session_state[POS] = pos
    image_id = review or ids[pos]
    row = df[df["id"] == image_id].iloc[0]

    # ---------- AI 추천 (표시만, 자동 체크 안 함) ----------
    preds = load_predictions()
    probs = dict(zip(preds.loc[preds["image_id"] == image_id, "tag_key"], preds.loc[preds["image_id"] == image_id, "prob"]))
    hints = {}
    for t in tags:
        badges = []
        if probs.get(t.key, 0) >= PRED_THRESHOLD:
            badges.append(f"🤖AI 추천 {probs[t.key]:.2f}")
        if t.booru_hint and t.booru_hint in row["tag_list"]:
            badges.append("🏷️원본 태그 일치")
        if badges:
            hints[t.key] = " · ".join(badges)

    saved = IL.my_values(labels, labeler, image_id) if labeler else {}
    nonce = st.session_state.setdefault(NONCE, 0)
    # 저장된 내 값이 바뀌면(저장 직후 다시 보기 등) 체크박스를 새로 만든다. 이미지가 바뀌면 키가 달라져 초기화된다
    version = hashlib.md5(json.dumps([saved, nonce], sort_keys=True).encode()).hexdigest()[:8]

    left, right = st.columns([11, 9])
    with left:
        st.image(row["img_path"], width=IMAGE_WIDTH)
        st.markdown(f"[`{image_id}`]({row['source_page']})")
    with right:
        if review:
            st.markdown("#### 다시 보는 중")
        else:
            st.markdown(f"#### 남은 {len(ids)}장" if hide_mine else f"#### {pos + 1} / {len(ids)}")
        if saved and set(saved) != set(active):
            st.caption(f"🆕 이 이미지는 {len(active) - len(set(saved) & set(active))}개 태그만 새로 확인하면 됩니다 "
                       "(이미 라벨링한 태그는 내 값으로 채워져 있음).")
        _checkbox_panel(tags, image_id, saved, hints, version, labeler, ids, pos, hide_mine, review)


def _tag_status_table(tags, labels) -> None:
    import training  # 학습 조건(양성·음성 각 20건)과 같은 기준을 쓴다
    merged = consolidate(labels)
    with st.expander("태그별 현황 (팀 전체, 다수결)"):
        rows = []
        for t in tags:
            v = merged.loc[merged["tag_key"] == t.key, "value"]
            pos, neg = int((v == POSITIVE).sum()), int((v == NEGATIVE).sum())
            ok = pos >= training.MIN_POS and neg >= training.MIN_NEG
            rows.append({"분류": t.group, "태그": t.name, "해당": pos, "아님": neg,
                         "학습 가능": "✅" if ok else f"❌ 양성 {max(training.MIN_POS - pos, 0)}·음성 {max(training.MIN_NEG - neg, 0)}건 부족"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


@st.fragment
def _checkbox_panel(tags, image_id, saved, hints, version, labeler, ids, pos, hide_mine, review) -> None:
    groups = list(dict.fromkeys(t.group or "(분류 없음)" for t in tags))
    keys = {t.key: _cb_key(image_id, t.key, version) for t in tags}
    names = {t.key: t.name for t in tags}

    for t in tags:  # 시작값: 내 라벨 그대로 (1이면 체크). 위젯 value= 대신 세션에 넣어야 'AI 추천 체크'와 충돌 경고가 없다
        st.session_state.setdefault(keys[t.key], saved.get(t.key) == POSITIVE)

    def check_recommended() -> None:
        for key in hints:  # 이미 직접 체크한 것은 그대로 두고 추천 태그만 추가로 체크
            st.session_state[keys[key]] = True

    with st.container(height=430, border=True):  # 태그가 많아도 아래 버튼 위치가 고정되도록 스크롤
        for group in groups:
            in_group = [t for t in tags if (t.group or "(분류 없음)") == group]
            st.markdown(f"**{group}** · {len(in_group)}개")
            cols = st.columns(3)
            for n, t in enumerate(in_group):
                label = t.name + (f"  :gray[{hints[t.key]}]" if t.key in hints else "")
                cols[n % 3].checkbox(label, key=keys[t.key], help=t.definition or "(정의 없음)")
    held = set(st.multiselect("🤔 잘 모르겠는 태그 (보류)", list(keys), format_func=names.get,
                              key=f"il_hold_{image_id}_{version}",
                              help="보류한 태그는 저장하지 않습니다. 이미 내 라벨이 있으면 그대로 둡니다."))
    checked = {key: bool(st.session_state.get(k, False)) for key, k in keys.items()}
    values = IL.values_to_save(checked, held)
    n_pos = sum(values.values())
    n_neg = len(values) - n_pos
    st.info(f"태그 {len(tags)}개 중 **{n_pos}개 체크**, 보류 {len(held)}개 → 나머지 **{n_neg}개는 '아님'으로 저장**됩니다."
            + (f"  \n체크했지만 보류한 태그({len([k for k in held if checked[k]])}개)는 저장하지 않습니다."
               if any(checked[k] for k in held) else ""))

    b1, b2 = st.columns([3, 2])
    if b1.button("💾 저장하고 다음", type="primary", width="stretch", disabled=not labeler, key=f"il_save_{image_id}_{version}"):
        try:
            save_labels(image_id, values, labeler)
        except (DBError, TagError) as e:
            st.error(f"저장하지 못했습니다. 이 이미지에 그대로 있습니다. {e}")  # 일부만 저장되지 않는다 (한 번의 요청)
            return
        st.session_state[LAST] = (image_id, n_pos, len(held))
        if held:
            st.session_state[HELD_DONE].add(image_id)
        if st.session_state.pop(REVIEW, None) is None and not hide_mine:
            st.session_state[POS] = pos + 1  # 숨기기를 켜면 저장한 이미지가 목록에서 빠지므로 같은 위치가 곧 다음
        st.session_state[MSG] = ("ok", f"`{image_id}` 저장: 해당 {n_pos} · 아님 {n_neg}" + (f" · 보류 {len(held)}" if held else ""))
        st.rerun()
    b2.button("🤖 AI 추천 모두 체크", width="stretch", disabled=not hints, on_click=check_recommended,
              key=f"il_ai_{image_id}_{version}", help="직접 체크한 것은 그대로 두고 추천 태그를 추가로 체크합니다. 저장은 직접 눌러야 합니다.")

    n1, n2, n3 = st.columns(3)
    if n1.button("⏭️ 건너뛰기", width="stretch", key=f"il_skip_{image_id}"):
        st.session_state[SKIPPED].add(image_id)  # 아무것도 저장하지 않는다
        st.session_state.pop(REVIEW, None)
        st.rerun()
    if n2.button("◀ 이전", width="stretch", disabled=not review and pos == 0, key=f"il_prev_{image_id}"):
        st.session_state.pop(REVIEW, None)
        st.session_state[POS] = max(pos - 1, 0)
        st.rerun()
    if n3.button("다음 ▶", width="stretch", disabled=not review and pos >= len(ids) - 1, key=f"il_next_{image_id}"):
        st.session_state.pop(REVIEW, None)
        st.session_state[POS] = pos + 1
        st.rerun()
    st.caption("건너뛰기·이전·다음은 저장하지 않습니다 (지금 체크한 내용은 사라집니다).")

    last = st.session_state.get(LAST)
    if last and last[0] != image_id:
        st.caption(f"방금 저장: `{last[0]}` → 해당 {last[1]}개" + (f", 보류 {last[2]}개" if last[2] else ""))
        if st.button("↩️ 방금 것 다시 보기", width="stretch", key="il_review_btn",
                     help="잘못 저장했을 때 그 이미지로 돌아가 고칠 수 있습니다."):
            st.session_state[REVIEW] = last[0]
            st.rerun()
