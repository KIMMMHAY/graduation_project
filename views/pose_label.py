import hashlib
import json
import random

import numpy as np
import streamlit as st

from components.pose_editor import pin_editor
from core import pose as P
from core import poses as store
from core.dataset import ON_CLOUD, load_metadata
from core.db import DBError
from core.pose_models import DEFAULT_MODEL, MODELS
from core.pose_predict import predict_and_save
from views.common import current_user
from views.pose_common import GUIDE, aspects_of, load_poses_or_stop

POS = "pose_pos"            # 남은 목록에서 현재 위치
NONCE = "pose_nonce"        # 이미지별 편집기 초기화 횟수 (바뀌면 편집기를 새로 만든다)
START = "pose_start"        # 이미지별 시작 핀: auto(내 기록 → AI → 표준) / default(표준 자세) / model(AI 제안)
MSG = "pose_msg"
STATUS_TEXT = {P.DRAFT: "임시 저장", P.DONE: "완료", P.SKIPPED: "건너뜀"}
AI_PRE, AI_LIVE, AI_OFF = "pre", "live", "off"
AI_MODES = {AI_PRE: "미리 계산된 결과 사용", AI_LIVE: "열 때마다 바로 추정 (약 3초)", AI_OFF: "사용 안 함"}


@st.cache_resource(show_spinner="AI 포즈 모델 불러오는 중... (처음에는 내려받느라 1분 정도 걸릴 수 있어요)")
def get_model():
    return MODELS[DEFAULT_MODEL]()


def plain(kps: list[dict]) -> list[dict]:
    """편집기와 비교·저장할 때는 신뢰도(conf)를 뺀 {x, y, state}만 쓴다."""
    return [{"x": k["x"], "y": k["y"], "state": k["state"]} for k in kps]

st.title("🦴 포즈 라벨링")
st.info(GUIDE)
df = load_metadata()
me = current_user()
if not me:
    st.warning("왼쪽 사이드바에 **내 이름**을 입력해야 저장할 수 있어요.")

poses = load_poses_or_stop()
prog = store.progress(poses)
human = poses[poses["source"] != P.MODEL]
mine_df = human[human["annotator"] == me]

m1, m2, m3 = st.columns(3)
m1.metric("완료한 이미지", f"{prog['done_images']} / {len(df)}", help="팀 전체 기준, 한 명이라도 완료하면 1장")
m2.metric("건너뛴 이미지", prog["skipped_images"], help="인물 없음·판단 불가")
m3.metric("내가 완료한 수", int((mine_df["status"] == P.DONE).sum()))
if not prog["by_annotator"].empty:
    with st.expander("작성자별 완료 수"):
        st.dataframe(prog["by_annotator"].rename("완료 수").rename_axis("작성자"), width="content")

if msg := st.session_state.pop(MSG, None):
    (st.success if msg[0] == "ok" else st.error)(msg[1])

c1, c2 = st.columns([3, 2])
# 서버에서는 AI 모델을 돌리지 않는다 (메모리 부족으로 앱 전체가 멈출 수 있음). PC에서 미리 계산한 결과만 쓴다
ai_modes = [m for m in AI_MODES if not (ON_CLOUD and m == AI_LIVE)]
if st.session_state.get("pose_ai_mode") not in ai_modes:
    st.session_state.pop("pose_ai_mode", None)
ai_mode = c1.radio("🤖 AI 제안 핀", ai_modes, format_func=AI_MODES.get, horizontal=True, key="pose_ai_mode",
                   help="AI(DWPose)가 추정한 위치에 핀을 미리 놓습니다. 틀린 핀만 고치면 됩니다.\n\n"
                        "'바로 추정'으로 만든 결과도 저장되어 다음부터는 기다리지 않습니다.")
reopen = c2.toggle("완료·건너뛴 이미지도 보기 (다시 열어 수정)", key="pose_reopen")
closed = set(human.loc[human["status"].isin([P.DONE, P.SKIPPED]), "image_id"])
seed = st.session_state.setdefault("pose_seed", random.randrange(1 << 30))
order = [df["id"].iat[i] for i in np.random.default_rng(seed).permutation(len(df))]
ids = order if reopen else [i for i in order if i not in closed]

# 주소에 ?image=<id>가 있으면 그 이미지를 연다 (팀원에게 특정 이미지를 링크로 공유할 때)
linked = st.query_params.get("image")
if linked in set(order) and st.session_state.get("pose_linked") != linked:
    st.session_state["pose_linked"] = linked
    if linked not in ids:
        ids = [linked] + ids
    st.session_state[POS] = ids.index(linked)
elif st.session_state.get("pose_linked") in set(order) - set(ids):
    ids = [st.session_state["pose_linked"]] + ids  # 링크로 연 완료 이미지는 계속 목록에 둔다

if not ids:
    st.success("모든 이미지의 포즈 라벨링이 끝났습니다. 🎉 수정하려면 위 토글을 켜세요.")
    st.stop()

pos = min(st.session_state.get(POS, 0), len(ids) - 1)
st.session_state[POS] = pos
image_id = ids[pos]
row = df[df["id"] == image_id].iloc[0]
aspect = aspects_of(df)[image_id]
mine = mine_df[mine_df["image_id"] == image_id]
mine = mine.iloc[0] if not mine.empty else None
nonce = st.session_state.setdefault(NONCE, {}).get(image_id, 0)
start = st.session_state.setdefault(START, {}).get(image_id, "auto")
version = f"{image_id}:{me}:{nonce}"

# ---------- AI 제안 핀 ----------
model_kps, model_facing, model_note = None, None, None
if start == "model" or (start == "auto" and mine is None and ai_mode != AI_OFF):
    mrow = store.model_poses(DEFAULT_MODEL, poses)
    mrow = mrow[mrow["image_id"] == image_id]
    if not mrow.empty:
        r = mrow.iloc[0]
        if r["status"] == P.DONE:
            model_kps, model_facing = P.validate_keypoints(r["keypoints"]), r["facing"]
        else:
            model_note = "AI가 이 그림에서 인물을 찾지 못해 표준 자세로 시작합니다."
    elif not ON_CLOUD and (ai_mode == AI_LIVE or start == "model"):
        try:
            with st.spinner("AI가 포즈를 추정하는 중... (3초 정도)"):
                pred = predict_and_save(get_model(), image_id, row["img_path"])
            if pred is None:
                model_note = "AI가 이 그림에서 인물을 찾지 못해 표준 자세로 시작합니다."
            else:
                model_kps, model_facing = pred.keypoints, pred.facing
        except Exception as e:  # 모델·DB 문제가 있어도 라벨링은 계속할 수 있어야 한다
            model_note = f"AI 추정에 실패해 표준 자세로 시작합니다. ({type(e).__name__}: {e})"
    elif ON_CLOUD:
        model_note = ("이 이미지는 아직 AI 추정이 없어 표준 자세로 시작합니다. "
                      "PC에서 `python predict_poses.py`로 미리 계산하면 서버에도 AI 제안 핀이 나옵니다.")
    else:
        model_note = "이 이미지는 아직 AI 추정이 없습니다. **포즈 정확도** 페이지에서 미리 계산하거나 '바로 추정'을 고르세요."

if start == "auto" and mine is not None:
    initial, start_source = plain(P.validate_keypoints(mine["keypoints"])), mine["source"]
elif model_kps is not None and start in ("auto", "model"):
    initial, start_source = plain(model_kps), P.MODEL_CORRECTED
    conf = [k.get("conf", 0) for k in model_kps]
    model_note = f"🤖 AI 제안 핀에서 시작합니다 (DWPose · 평균 신뢰도 {sum(conf) / len(conf):.2f}). 틀린 핀만 고쳐 주세요."
else:
    initial, start_source = P.default_pose(), P.MANUAL

# 시작 핀이 바뀌면(AI 모드 전환, AI 추정 완료 등) 편집기를 새로 만든다. 같으면 사용자가 옮긴 핀을 유지한다.
version = f"{version}:{hashlib.md5(json.dumps(initial).encode()).hexdigest()[:8]}"

left, right = st.columns([3, 1])
with left:
    if model_note:
        st.caption(model_note)
    keypoints = pin_editor(key=f"pose_editor_{version}", version=version, keypoints=initial, aspect=aspect,
                           image_path=row["img_path"], height=640)
changed = keypoints != initial

with right:
    st.markdown(f"#### {pos + 1} / {len(ids)}")
    st.markdown(f"[`{image_id}`]({row['source_page']}) · [링크](?image={image_id})",
                help="'링크'를 복사해 팀원에게 보내면 이 이미지가 바로 열립니다.")
    if mine is not None:
        st.caption(f"내 기록: **{STATUS_TEXT[mine['status']]}**" + (f" · {P.FACINGS[mine['facing']]}" if mine["facing"] else ""))
    others = human[(human["image_id"] == image_id) & (human["annotator"] != me) & (human["status"] == P.DONE)]
    if not others.empty:
        st.caption(f"다른 팀원 완료: {', '.join(others['annotator'])}")

    if mine is not None and mine["facing"] in P.FACINGS:
        facing_default = mine["facing"]
    elif model_facing in P.FACINGS:
        facing_default = model_facing  # AI가 어깨 방향으로 추정한 값 (틀릴 수 있음)
    else:
        facing_default = "front"
    facing = st.radio("몸이 향한 방향", list(P.FACINGS), format_func=P.FACINGS.get, key=f"pose_facing_{image_id}",
                      index=list(P.FACINGS).index(facing_default), horizontal=True)

    def save(status: str) -> bool:
        try:
            store.save_pose(image_id, me, keypoints, facing if status != P.SKIPPED else None, status,
                            source=start_source)
            return True
        except (DBError, P.PoseError) as e:
            st.session_state[MSG] = ("error", f"저장하지 못했습니다. {e}")
            return False

    def advance(saved_closed: bool) -> None:
        if st.session_state.pop("pose_linked", None) is not None:  # 링크로 연 이미지는 끝났으니 평소 목록으로
            st.query_params.pop("image", None)
            if not reopen and image_id in closed:  # 원래 목록에 없던 이미지라 앞에 끼워 넣었었다
                st.session_state[POS] = 0
                return
        # 숨기기 상태에서 완료/건너뛰면 그 이미지가 목록에서 빠지므로 같은 위치가 곧 다음 이미지다
        if reopen or not saved_closed:
            st.session_state[POS] = pos + 1

    if st.button("✅ 완료 (저장 후 다음)", type="primary", width="stretch", disabled=not me):
        if save(P.DONE):
            st.session_state[MSG] = ("ok", f"`{image_id}` 저장했습니다.")
            advance(True)
        st.rerun()
    if st.button("⏭️ 건너뛰기 (인물 없음·판단 불가)", width="stretch", disabled=not me):
        if save(P.SKIPPED):
            advance(True)
        st.rerun()
    if st.button("↺ 초기화 (표준 자세로)", width="stretch"):
        st.session_state[NONCE][image_id] = nonce + 1
        st.session_state[START][image_id] = "default"
        st.rerun()
    if ai_mode != AI_OFF and st.button("🤖 AI 제안 핀으로 다시 시작", width="stretch",
                                       help="지금 핀을 버리고 AI가 추정한 위치로 되돌립니다."):
        st.session_state[NONCE][image_id] = nonce + 1
        st.session_state[START][image_id] = "model"
        st.rerun()

    n1, n2 = st.columns(2)
    nav = None
    if n1.button("◀ 이전", width="stretch", disabled=pos == 0):
        nav = -1
    if n2.button("다음 ▶", width="stretch", disabled=pos >= len(ids) - 1):
        nav = 1
    if nav:
        # 아직 완료하지 않은 이미지에서 핀을 옮겼다면 임시 저장해 둔다 (완료·건너뛴 기록은 덮어쓰지 않음)
        if me and changed and (mine is None or mine["status"] == P.DRAFT):
            save(P.DRAFT)
        st.session_state[POS] = max(pos + nav, 0)
        st.rerun()

    if changed and mine is not None and mine["status"] != P.DRAFT:
        st.caption("⚠️ 바꾼 내용은 **완료**를 눌러야 저장됩니다.")
    st.divider()
    st.caption("핀 드래그: 이동 · 빈 곳 드래그: 화면 이동 · 휠/＋－: 확대·축소\n\n"
               "핀 선택 후 보임/가려짐/없음 (키보드 1·2·3)")
