"""포즈 라벨링·검색 페이지 공통."""
import json

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from core import pose as P
from core import poses as store
from core.dataset import local_image
from core.db import DBError

GUIDE = ("좌우는 **캐릭터 기준**입니다. 캐릭터의 왼팔이 `왼쪽`(파랑)이에요. "
         "정면을 보는 그림이면 캐릭터의 왼쪽이 화면 **오른쪽**에 옵니다.")
DEFAULT_ASPECT = 0.75
SIDE_RGB = {"l": (37, 99, 235), "r": (234, 88, 12), "c": (22, 163, 74)}


def aspects_of(df: pd.DataFrame) -> dict[str, float]:
    """image_id → 가로/세로. metadata의 원본 크기 기준 (저장된 축소 이미지와 비율이 같다)."""
    w = pd.to_numeric(df["width"], errors="coerce")
    h = pd.to_numeric(df["height"], errors="coerce")
    ratio = (w / h).where((w > 0) & (h > 0), DEFAULT_ASPECT)
    return dict(zip(df["id"], ratio.astype(float)))


def load_poses_or_stop() -> pd.DataFrame:
    try:
        return store.load_poses()
    except DBError as e:
        st.error(f"포즈 데이터를 불러오지 못했습니다. {e}")
        st.stop()


@st.cache_data(show_spinner=False, max_entries=256)
def _overlay(img_path: str, keypoints_json: str, flip: bool, size: int) -> Image.Image:
    kps = json.loads(keypoints_json)
    with Image.open(local_image(img_path)) as src:
        img = src.convert("RGB")
    img.thumbnail((size, size))
    w, h = img.size
    draw = ImageDraw.Draw(img)

    def point(i):
        if i in (P.SHOULDER_MID, P.HIP_MID):
            a, b = (P.J["l_shoulder"], P.J["r_shoulder"]) if i == P.SHOULDER_MID else (P.J["l_hip"], P.J["r_hip"])
            ok = kps[a]["state"] != P.ABSENT and kps[b]["state"] != P.ABSENT
            return ((kps[a]["x"] + kps[b]["x"]) / 2 * w, (kps[a]["y"] + kps[b]["y"]) / 2 * h), ok
        return (kps[i]["x"] * w, kps[i]["y"] * h), kps[i]["state"] != P.ABSENT

    for _, a, b, _, side in P.BONES:
        (pa, oka), (pb, okb) = point(a), point(b)
        if oka and okb:
            draw.line([pa, pb], fill=(255, 255, 255), width=5)
            draw.line([pa, pb], fill=SIDE_RGB[side], width=3)
    for i, (key, _) in enumerate(P.JOINTS):
        (x, y), ok = point(i)
        if ok:
            side = "c" if key == "head" else key[0]
            r = 4
            fill = SIDE_RGB[side] if kps[i]["state"] == P.VISIBLE else (255, 255, 255)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=SIDE_RGB[side], width=2)
    return ImageOps.mirror(img) if flip else img


def pose_thumbnail(img_path: str, keypoints: list[dict], flip: bool = False, size: int = 384) -> Image.Image:
    """핀을 그려 넣은 썸네일. flip이면 좌우를 뒤집어 질의 포즈와 같은 방향으로 보여준다."""
    return _overlay(img_path, json.dumps(keypoints), flip, size)


# ---------- 포즈 검색 (2D·3D 페이지 공통) ----------

def search_options(prefix: str) -> dict:
    """검색 옵션 위젯. 위젯 키는 prefix로 페이지마다 구분한다."""
    part = st.radio("비교할 부위", list(P.PARTS), format_func=lambda p: P.PARTS[p][0], horizontal=True,
                    key=f"{prefix}_part")
    allow_mirror = st.toggle("좌우 반전 포함", value=True, key=f"{prefix}_mirror",
                             help="좌우가 뒤집힌 포즈도 찾습니다. 결과 썸네일은 뒤집어서 보여줍니다.")
    facings = st.multiselect("몸이 향한 방향", list(P.FACINGS), default=list(P.FACINGS), format_func=P.FACINGS.get,
                             key=f"{prefix}_facing")
    max_angle = st.slider("허용 범위 (뼈 하나의 최대 각도 오차)", 20, 180, int(P.DEFAULT_MAX_ANGLE), 5, format="%d°",
                          key=f"{prefix}_max_angle", help="이보다 크게 어긋난 뼈가 하나라도 있으면 결과에서 뺍니다.")
    top_k = st.number_input("결과 수", 4, 48, 12, 4, key=f"{prefix}_topk")
    return {"part": part, "mirror": allow_mirror, "facings": sorted(facings), "max_angle": max_angle,
            "top_k": int(top_k)}


def run_search(query: list[dict], aspect: float, options: dict, aspects: dict[str, float],
               mode: str = "2d") -> dict | None:
    """검색해서 결과를 세션에 넣을 수 있는 dict로. 실패하면 화면에 안내하고 None."""
    try:
        cands = store.searchable_candidates(aspects)
    except DBError as e:
        st.error(f"검색 대상을 불러오지 못했습니다. {e}")
        return None
    hits = P.search(query, aspect, cands, part=options["part"], allow_mirror=options["mirror"],
                    facings=options["facings"] or None, max_angle=options["max_angle"], top_k=options["top_k"])
    saved_options = {k: v for k, v in options.items() if k != "top_k"} | ({"mode": mode} if mode != "2d" else {})
    return {
        "query": {"keypoints": query, "aspect": aspect}, "options": saved_options, "n_candidates": len(cands),
        "hits": [{"image_id": h.candidate.image_id, "annotator": h.candidate.annotator, "facing": h.candidate.facing,
                  "keypoints": h.candidate.keypoints, "score": h.match.score, "mean_angle": h.match.mean_angle,
                  "flip": h.match.flip} for h in hits],
    }


def render_results(df: pd.DataFrame, res: dict, me: str, cols: int = 4) -> None:
    """검색 결과 그리드 + 비슷함/다름 평가 + 팀 평가 현황."""
    row_of = {i: n for n, i in enumerate(df["id"])}
    hits = res["hits"]
    st.subheader(f"가까운 순 {len(hits)}장")
    st.caption(f"후보 {res['n_candidates']}개 포즈 중 허용 범위 안에 든 결과 · 점수는 100에 가까울수록 비슷함")
    if not hits:
        st.info("허용 범위 안에 드는 포즈가 없습니다. 허용 범위를 넓히거나 비교할 부위를 줄여 보세요.")
    qhash = store.query_hash(res["query"], res["options"])
    verdicts = {}
    if me:
        try:
            ev = store.load_pose_evals()
            ev = ev[(ev["evaluator"] == me) & (ev["query_hash"] == qhash)]
            verdicts = dict(zip(ev["result_image_id"], ev["verdict"]))
        except DBError as e:
            st.warning(f"평가 기록을 불러오지 못했습니다. {e}")
    elif hits:
        st.caption("사이드바에 **내 이름**을 입력하면 비슷함/다름 평가를 저장할 수 있어요.")

    def evaluate(rank: int, hit: dict, verdict: str) -> None:
        try:
            store.save_pose_eval(me, res["query"], res["options"], hit["image_id"], hit["annotator"],
                                 rank, hit["score"], hit["flip"], verdict)
        except DBError as e:
            st.error(f"평가를 저장하지 못했습니다. {e}")

    grid = st.columns(cols)
    for rank, hit in enumerate(hits, start=1):
        with grid[(rank - 1) % cols]:
            img_path = df["img_path"].iat[row_of[hit["image_id"]]]
            st.image(pose_thumbnail(img_path, hit["keypoints"], hit["flip"]), width="stretch")
            facing = P.FACINGS.get(hit["facing"], "?")
            st.markdown(f"#{rank} · **{hit['score']:.1f}점** · 평균 {hit['mean_angle']:.0f}°  \n"
                        f"`{hit['image_id']}` · {facing}" + (" · ↔ 반전" if hit["flip"] else ""))
            current = verdicts.get(hit["image_id"])
            b1, b2 = st.columns(2)
            for col, value, text in ((b1, "similar", "👍 비슷함"), (b2, "different", "👎 다름")):
                col.button(text, key=f"pose_ev_{qhash}_{hit['image_id']}_{value}", width="stretch",
                           type="primary" if current == value else "secondary", disabled=not me,
                           on_click=evaluate, args=(rank, hit, value))

    with st.expander("평가 현황 (팀 전체)"):
        try:
            ev = store.load_pose_evals()
        except DBError as e:
            st.warning(str(e))
            ev = None
        if ev is None or ev.empty:
            st.caption("아직 평가가 없습니다.")
        else:
            ev["is_similar"] = ev["verdict"] == "similar"
            ev["검색 방식"] = ev["options"].map(lambda o: "3D 마네킹" if (o or {}).get("mode") == "3d" else "2D 핀")
            ev["부위"] = ev["options"].map(lambda o: P.PARTS.get((o or {}).get("part", "full"), ("?",))[0])
            a, b, c = st.columns(3)
            a.metric("평가한 검색", ev["query_hash"].nunique())
            b.metric("평가 수", len(ev))
            c.metric("비슷함 비율", f"{ev['is_similar'].mean():.0%}")
            st.dataframe(ev.groupby(["검색 방식", "부위"]).agg(평가수=("verdict", "size"), 비슷함비율=("is_similar", "mean"))
                         .style.format({"비슷함비율": "{:.0%}"}))
