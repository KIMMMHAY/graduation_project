import pandas as pd
import streamlit as st

from core import pose as P
from core import pose_accuracy as A
from core.dataset import load_metadata
from core.pose_models import DEFAULT_MODEL, MODELS, OCCLUDED_MIN_CONF, VISIBLE_MIN_CONF
from core.pose_predict import run_batch
from views.pose_common import aspects_of, load_poses_or_stop

RUNNING = "pose_batch_running"

st.title("📏 포즈 정확도")
st.caption("AI 포즈 모델이 일러스트에서 관절을 얼마나 정확히 찾는지, 사람이 확정한 핀과 비교합니다. "
           "**'일러스트에서 포즈 추출이 되는가'에 대한 검증 결과**입니다.")
df = load_metadata()
poses = load_poses_or_stop()
models = poses[poses["source"] == P.MODEL]

# ---------- 미리 계산 ----------
st.subheader("🤖 AI 추정 미리 계산")
status = (models.groupby(["annotator", "status"]).size().unstack(fill_value=0)
          .reindex(columns=[P.DONE, P.SKIPPED], fill_value=0)) if not models.empty \
    else pd.DataFrame(columns=[P.DONE, P.SKIPPED], dtype=int)
if DEFAULT_MODEL not in status.index:  # 아직 한 장도 계산하지 않았어도 표에 보이게
    status.loc[DEFAULT_MODEL] = [0, 0]
status["남은 이미지"] = len(df) - status[P.DONE] - status[P.SKIPPED]
st.dataframe(status.rename(columns={P.DONE: "인물 검출", P.SKIPPED: "검출 실패"}).rename_axis("모델"), width="content")

c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
limit = c1.number_input("이번에 처리할 장수", 1, len(df), min(50, len(df)), 10, key="pose_batch_limit")
running = st.session_state.get(RUNNING, False)
c2.button(f"{MODELS[DEFAULT_MODEL].label}로 미리 계산 (아직 없는 이미지만)", type="primary", disabled=running,
          on_click=lambda: st.session_state.update({RUNNING: True}))
st.caption("1장에 약 3초. 페이지를 벗어나면 멈추지만, 다시 누르면 이어서 합니다. "
           "전체를 한 번에 하려면 터미널에서 `python predict_poses.py`를 실행하세요(약 25분).")
if running:
    try:
        bar = st.progress(0.0, text="AI 모델 불러오는 중...")

        def progress(n, total, image_id):
            bar.progress(n / total, text=f"{n}/{total} · {image_id}")

        with st.spinner("추정 중..."):
            r = run_batch(df, MODELS[DEFAULT_MODEL](), int(limit), progress=progress)
        st.session_state["pose_batch_msg"] = ("ok", f"완료: {r['processed']}장 처리, 인물 검출 {r['detected']}장")
    except Exception as e:
        st.session_state["pose_batch_msg"] = ("error", f"미리 계산에 실패했습니다: {type(e).__name__}: {e}")
    finally:
        st.session_state[RUNNING] = False
    st.rerun()
if msg := st.session_state.pop("pose_batch_msg", None):
    (st.success if msg[0] == "ok" else st.error)(msg[1])

# ---------- 정확도 리포트 ----------
st.divider()
st.subheader("📊 정확도 리포트")
human = poses[poses["source"].isin(P.SEARCHABLE_SOURCES) & (poses["status"] == P.DONE)]
if human.empty:
    st.info("아직 사람이 완료한 포즈가 없습니다. **포즈 라벨링**에서 포즈를 완료하면 여기에 결과가 나옵니다.")
    st.stop()
if models.empty:
    st.info("아직 AI 추정 결과가 없습니다. 위에서 미리 계산을 실행하세요.")
    st.stop()

sources = {"all": "전체", P.MANUAL: "표준 자세에서 직접 찍음", P.MODEL_CORRECTED: "AI 핀을 고쳐 완료"}
pick = st.radio("비교할 사람 라벨", list(sources), format_func=sources.get, horizontal=True, key="pose_acc_source",
                help="AI 핀에서 시작하면 사람이 AI 핀을 그대로 두는 경향이 있어 AI 점수가 실제보다 좋게 나옵니다. "
                     "공정한 비교는 '직접 찍음'입니다.")
if pick != "all":
    human = human[human["source"] == pick]
cmp = A.compare(human, models, aspects_of(df))
summ = A.summary(cmp)
if summ.empty:
    st.info("비교할 수 있는 이미지가 없습니다 (같은 이미지에 사람 포즈와 AI 추정이 모두 있어야 합니다).")
    st.stop()

st.markdown(f"**PCK@{P.PCK_THRESHOLD}**: 관절 위치 오차가 몸통 길이의 {P.PCK_THRESHOLD:.0%} 이내면 정답. "
            "**평균 오차**: 몸통 길이 대비 거리 (0.1 = 몸통의 10%).")
st.dataframe(summ.rename(columns={"model": "모델", "images": "비교 이미지", "detection_rate": "검출률",
                                  "pck": "PCK (전체 관절)", "mean_error": "평균 오차"})
             .style.format({"검출률": "{:.0%}", "PCK (전체 관절)": "{:.0%}", "평균 오차": "{:.3f}"}, na_rep="-"),
             hide_index=True, width="stretch")

pj = A.per_joint(cmp)
if not pj.empty:
    st.markdown("**관절별**")
    table = pj.pivot(index="joint_name", columns="model", values=["pck", "mean_error", "n"])
    table = table.reindex([name for _, name in P.JOINTS]).dropna(how="all")
    table.columns = [f"{m} · {dict(pck='PCK', mean_error='평균 오차', n='비교 수')[k]}" for k, m in table.columns]
    fmt = {c: ("{:.0%}" if "PCK" in c else "{:.3f}" if "오차" in c else "{:.0f}") for c in table.columns}
    def pck_color(v):  # 낮을수록 빨강, 높을수록 초록 (matplotlib 없이)
        if pd.isna(v):
            return ""
        return f"background-color: rgba({int(220 * (1 - v))}, {int(170 * v)}, 60, 0.25)"

    st.dataframe(table.rename_axis("관절").style.format(fmt, na_rep="-")
                 .map(pck_color, subset=[c for c in table.columns if "PCK" in c]), width="stretch")

with st.expander("신뢰도 구간별 PCK (보임/가려짐/없음 기준값 조정용)"):
    st.caption(f"현재 기준: 신뢰도 {VISIBLE_MIN_CONF} 이상 보임, {OCCLUDED_MIN_CONF}~{VISIBLE_MIN_CONF} 가려짐, "
               f"{OCCLUDED_MIN_CONF} 미만 없음. 낮은 구간의 PCK가 높으면 기준을 낮춰도 됩니다.")
    bc = A.by_confidence(cmp)
    if bc.empty:
        st.caption("신뢰도 정보가 있는 비교 결과가 없습니다.")
    else:
        st.dataframe(bc.rename(columns={"model": "모델", "conf_bin": "신뢰도", "pck": "PCK", "n": "관절 수"})
                     .style.format({"PCK": "{:.0%}"}), hide_index=True, width="content")
