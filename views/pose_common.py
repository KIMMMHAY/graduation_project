"""포즈 라벨링·검색 페이지 공통."""
import json

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from core import pose as P
from core import poses as store
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
    with Image.open(img_path) as src:
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
