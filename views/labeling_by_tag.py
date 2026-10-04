"""라벨링 '태그별 (하나씩)' 화면 — 기존 화면을 함수로 감싼 것. 동작은 바꾸지 않는다."""
import random

import numpy as np
import streamlit as st

from core.tagging import NEGATIVE, POSITIVE, consolidate
from methods.predicted_tags import load_predictions
from views.common import image_card, load_labels_or_stop, paginate, save_label_safely, tag_label

PER_PAGE = 12
COLS = 4
SINGLE_WIDTH = 520  # 한 장씩 보기의 이미지 너비(px)
ORDER_RANDOM, ORDER_HINT, ORDER_UNSURE = "무작위", "Safebooru 힌트 태그 우선", "헷갈리는 것 우선 (확률 0.5 근처)"
MODE_SINGLE, MODE_GRID = "한 장씩", f"그리드 ({PER_PAGE}장)"
LAST = "label_last"  # 한 장씩 보기에서 방금 라벨링한 (tag_key, image_id, value)
VALUE_TEXT = {POSITIVE: "해당", NEGATIVE: "아님"}


def render(df, tags, labeler) -> None:
    """기존 라벨링 화면 본문 (st.stop()으로 끝나는 분기 포함)."""
    by_key = {t.key: t for t in tags}
    tag = by_key[st.selectbox("라벨링할 태그", list(by_key), format_func=lambda k: tag_label(by_key[k]), key="label_tag")]
    st.info(f"**{tag.name}** — {tag.definition or '(정의 없음: 태그 관리 페이지에서 정의를 적어 주세요)'}"
            + (f"\n\nSafebooru 대응 태그: `{tag.booru_hint}`" if tag.booru_hint else ""))

    labels = load_labels_or_stop()
    tag_labels = labels[labels["tag_key"] == tag.key]
    merged = consolidate(tag_labels)
    mine = dict(zip(tag_labels.loc[tag_labels["labeler"] == labeler, "image_id"],
                    tag_labels.loc[tag_labels["labeler"] == labeler, "value"]))

    m1, m2, m3 = st.columns(3)
    m1.metric("양성 (해당)", int((merged["value"] == POSITIVE).sum()), help="팀 전체, 다수결 기준")
    m2.metric("음성 (해당 아님)", int((merged["value"] == NEGATIVE).sum()), help="팀 전체, 다수결 기준")
    m3.metric("내가 라벨링한 수", len(mine))

    # ---------- 표시 순서 ----------
    preds = load_predictions()
    tag_preds = preds[preds["tag_key"] == tag.key]
    probs = dict(zip(tag_preds["image_id"], tag_preds["prob"]))  # 학습된 모델이 있을 때만 채워진다
    orders = [ORDER_RANDOM] + ([ORDER_HINT] if tag.booru_hint else []) + ([ORDER_UNSURE] if probs else [])
    c0, c1, c2 = st.columns([2, 3, 2])
    mode = c0.radio("보기 방식", [MODE_SINGLE, MODE_GRID], horizontal=True, key="label_mode")
    order = c1.radio("표시 순서", orders, horizontal=True, key="label_order")
    hide_done = c2.toggle("내가 라벨링한 이미지 숨기기", value=True, key="label_hide")

    seed = st.session_state.setdefault("label_seed", random.randrange(1 << 30))
    idxs = list(np.random.default_rng(seed).permutation(len(df)))  # 새로고침해도 순서가 유지되도록 고정 시드
    if order == ORDER_HINT:
        has_hint = {i for i in idxs if tag.booru_hint in df["tag_list"].iat[i]}
        idxs = [i for i in idxs if i in has_hint] + [i for i in idxs if i not in has_hint]
    elif order == ORDER_UNSURE:
        idxs.sort(key=lambda i: abs(probs.get(df["id"].iat[i], 1.0) - 0.5))

    skipped = st.session_state.setdefault("label_skipped", {}).setdefault(tag.key, set())
    if hide_done:
        idxs = [i for i in idxs if df["id"].iat[i] not in mine]
    idxs = [i for i in idxs if df["id"].iat[i] not in skipped]

    pos_key, review_key = f"label_pos_{tag.key}", f"label_review_{tag.key}"
    review = st.session_state.get(review_key)  # '방금 것 다시 보기'로 고르는 이미지 (목록에서 빠졌어도 보여준다)

    if skipped:
        st.button(f"건너뛴 {len(skipped)}장 다시 보기", on_click=skipped.clear)
    if not idxs and not (mode == MODE_SINGLE and review):
        st.success("이 태그는 더 라벨링할 이미지가 없습니다. 🎉")
        st.stop()


    def skip(image_id: str) -> None:
        skipped.add(image_id)


    def caption_of(idx: int) -> str:
        image_id, extra = df["id"].iat[idx], []
        if tag.booru_hint and tag.booru_hint in df["tag_list"].iat[idx]:
            extra.append(f"`{tag.booru_hint}`")
        if image_id in probs:
            extra.append(f"예측 {probs[image_id]:.2f}")
        return " · ".join(extra)


    # ---------- 한 장씩 보기: 누르면 바로 다음 이미지 ----------
    if mode == MODE_SINGLE:
        def act(image_id: str, value: int | None) -> None:
            """value=None은 건너뛰기."""
            if value is None:
                skipped.add(image_id)
            elif save_label_safely(image_id, tag.key, value, labeler):
                st.session_state[LAST] = (tag.key, image_id, value)
            else:
                return  # 저장 실패 시 그대로 머무른다
            reviewing = st.session_state.pop(review_key, None) is not None
            # 숨기기가 켜져 있거나 건너뛰면 그 이미지가 목록에서 빠지므로 같은 위치가 곧 다음 이미지다
            if not reviewing and not hide_done and value is not None:
                st.session_state[pos_key] = st.session_state.get(pos_key, 0) + 1

        def move(step: int) -> None:
            st.session_state.pop(review_key, None)
            st.session_state[pos_key] = max(0, st.session_state.get(pos_key, 0) + step)

        pos = min(st.session_state.get(pos_key, 0), max(len(idxs) - 1, 0))
        st.session_state[pos_key] = pos
        row = {i: n for n, i in enumerate(df["id"])}
        idx = row[review] if review else idxs[pos]
        image_id = df["id"].iat[idx]
        current = mine.get(image_id)

        left, right = st.columns([3, 2])
        with left:
            st.image(df["img_path"].iat[idx], width=SINGLE_WIDTH)
        with right:
            if review:
                st.markdown("#### 다시 보는 중")
            else:  # 숨기기를 켜면 라벨링한 이미지가 빠지므로 '남은 장수'가 진행 상황이다
                st.markdown(f"#### 남은 {len(idxs)}장" if hide_done else f"#### {pos + 1} / {len(idxs)}")
            st.markdown(f"[`{image_id}`]({df['source_page'].iat[idx]})" + (f" · {cap}" if (cap := caption_of(idx)) else ""))
            if current is not None:
                st.caption(f"내 라벨: **{VALUE_TEXT[current]}** (다시 누르면 바뀝니다)")
            for value, text in ((POSITIVE, "⭕ 해당"), (NEGATIVE, "❌ 아님")):
                st.button(text, key=f"one_{tag.key}_{image_id}_{value}", width="stretch",
                          type="primary" if current == value else "secondary", disabled=not labeler,
                          on_click=act, args=(image_id, value))
            st.button("⏭️ 건너뛰기", key=f"one_{tag.key}_{image_id}_skip", width="stretch", on_click=act, args=(image_id, None))

            n1, n2 = st.columns(2)
            n1.button("◀ 이전", width="stretch", disabled=not review and pos == 0, on_click=move, args=(-1,))
            n2.button("다음 ▶", width="stretch", disabled=not review and pos >= len(idxs) - 1, on_click=move, args=(1,))

            last = st.session_state.get(LAST)
            if last and last[0] == tag.key and last[1] != image_id:
                st.divider()
                st.caption(f"방금 라벨링: `{last[1]}` → **{VALUE_TEXT[last[2]]}**")
                st.button("↩️ 방금 것 다시 보기", width="stretch", on_click=lambda: st.session_state.update({review_key: last[1]}),
                          help="잘못 눌렀을 때 그 이미지로 돌아가 라벨을 고칠 수 있습니다.")
        st.stop()

    # ---------- 그리드 보기 ----------
    start, end = paginate(len(idxs), PER_PAGE, key="label_page")
    st.caption(f"남은 이미지 {len(idxs)}장")
    cols = st.columns(COLS)
    for n, idx in enumerate(idxs[start:end]):
        image_id = df["id"].iat[idx]
        current = mine.get(image_id)
        with cols[n % COLS]:
            image_card(df, idx, caption=caption_of(idx) or None)
            b1, b2, b3 = st.columns(3)
            for col, value, text in ((b1, POSITIVE, "해당"), (b2, NEGATIVE, "아님")):
                col.button(text, key=f"lb_{tag.key}_{image_id}_{value}", width="stretch",
                           type="primary" if current == value else "secondary", disabled=not labeler,
                           on_click=save_label_safely, args=(image_id, tag.key, value, labeler))
            b3.button("건너뛰기", key=f"lb_{tag.key}_{image_id}_skip", width="stretch", on_click=skip, args=(image_id,))
