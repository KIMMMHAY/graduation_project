import pandas as pd
import streamlit as st

import training
from core.tagging import REPORT_CSV, consolidate
from views.common import load_labels_or_stop, load_tags_or_stop

RUNNING = "train_running"
RESULT = "train_result"
ERROR = "train_error"

# 모델은 한 번 불러오면 앱이 켜져 있는 동안 재사용
get_embedder = st.cache_resource(show_spinner="CLIP 모델 불러오는 중... (처음엔 다운로드로 몇 분 걸릴 수 있어요)")(
    training.load_embedder)

st.title("🧠 학습 실행")
st.caption(f"임베딩 모델: `{training.EMBED_ID}` · 학습 조건: 양성 {training.MIN_POS}건 이상, 음성 {training.MIN_NEG}건 이상")

tags = load_tags_or_stop()
status = training.label_status(tags, consolidate(load_labels_or_stop()))
cached_ids, _ = training.load_embedding_cache()

st.subheader("태그별 라벨 현황")
st.dataframe(
    pd.DataFrame({
        "태그": status["name"], "분류": status["group"], "양성": status["n_pos"], "음성": status["n_neg"],
        "학습 가능": status["trainable"].map({True: "✅ 가능", False: "❌ 불가"}), "사유": status["reason"],
    }),
    hide_index=True, width="stretch",
)
st.caption(f"임베딩 캐시: {len(cached_ids)}장")


def start() -> None:
    st.session_state[RUNNING] = True
    st.session_state.pop(RESULT, None)
    st.session_state.pop(ERROR, None)


running = st.session_state.get(RUNNING, False)
st.button("학습 실행", type="primary", disabled=running, on_click=start)

if running:
    try:
        with st.status("학습 실행 중...", expanded=True) as box:
            bar = st.progress(0.0)

            def progress(stage: str, message: str, frac: float | None) -> None:
                box.update(label=f"{stage}: {message}")
                st.write(f"**{stage}** · {message}")
                if frac is not None:
                    bar.progress(min(max(frac, 0.0), 1.0), text=stage)

            st.session_state[RESULT] = training.train_all(get_embedder=get_embedder, progress=progress)
    except training.TrainingError as e:
        st.session_state[ERROR] = str(e)
    except Exception as e:  # 예상 못 한 오류도 화면에 한국어로 남긴다
        st.session_state[ERROR] = f"알 수 없는 오류로 실패했습니다: {type(e).__name__}: {e}"
    finally:
        st.session_state[RUNNING] = False
    st.rerun()  # 버튼을 다시 활성화하고 라벨 현황을 새로 그린다

# ---------- 결과 ----------
if err := st.session_state.get(ERROR):
    st.error(err)
result: training.TrainResult | None = st.session_state.get(RESULT)
if result is not None:
    if result.trained:
        st.success(f"학습 완료 · {len(result.report)}개 태그 · 새로 추출한 임베딩 {result.n_new_embeddings}장 · "
                   f"{result.seconds:.1f}초")
    else:
        st.warning("라벨 부족: 학습 가능한 태그가 없습니다. 위 표의 사유를 확인하고 라벨링 페이지에서 라벨을 더 모아 주세요.")
    with st.expander("실행 기록"):
        st.code("\n".join(result.log), language=None)

if REPORT_CSV.exists():
    report = pd.read_csv(REPORT_CSV, encoding="utf-8-sig")
    st.subheader("태그별 평가 지표 (평가용 20%)")
    st.caption(f"마지막 학습: {report['trained_at'].iat[0]} · F1 {training.F1_WARN} 미만은 빨간색으로 표시")
    table = pd.DataFrame({
        "태그": report["name"], "양성": report["n_pos"], "음성": report["n_neg"], "평가 건수": report["n_test"],
        "Precision": report["precision"], "Recall": report["recall"], "F1": report["f1"],
    })
    low = report["f1"] < training.F1_WARN
    st.dataframe(
        table.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"})
        .apply(lambda r: ["background-color: rgba(255, 75, 75, 0.25)" if low.iat[r.name] else ""] * len(r), axis=1),
        hide_index=True, width="stretch",
    )
