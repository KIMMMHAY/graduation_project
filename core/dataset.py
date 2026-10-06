"""데이터셋(metadata.csv + images/) 로딩과 썸네일 관리."""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "drawing_ref_test"
IMG_DIR = DATA_DIR / "images"
META_CSV = DATA_DIR / "metadata.csv"
CACHE_DIR = DATA_DIR / "cache"
THUMB_DIR = CACHE_DIR / "thumbs"
THUMB_SIZE = 384


def _local_path(image_id: str, csv_path) -> Path:
    # csv의 path는 만든 사람 PC의 절대경로라서, 파일명만 따서 images/ 기준으로 다시 잡는다
    if isinstance(csv_path, str) and csv_path:
        return IMG_DIR / csv_path.replace("\\", "/").rsplit("/", 1)[-1]
    return IMG_DIR / f"{image_id}.jpg"


def read_metadata() -> pd.DataFrame:
    """행 번호(0..n-1)가 모든 검색 모듈에서 공통 인덱스로 쓰인다. 웹앱에서는 load_metadata()를 쓴다."""
    if not META_CSV.exists():  # 배포 서버: 로컬 데이터가 없으므로 Supabase에서 읽는다
        return _read_metadata_from_db()
    df = pd.read_csv(META_CSV, dtype={"id": str})
    df = df.drop_duplicates(subset="id")
    df["tags"] = df["tags"].fillna("")
    df["img_path"] = [str(_local_path(i, p)) for i, p in zip(df["id"], df.get("path", [None] * len(df)))]
    df = df[df["img_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    df["tag_list"] = df["tags"].str.split()
    return df


def _read_metadata_from_db() -> pd.DataFrame:
    from core import db  # db.py가 dataset.ROOT를 import하므로 순환 import를 피하려고 함수 안에서 import

    rows = db.call(db.fetch_all, "images", "id,url,source_page,tags,width,height")
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    df["tag_list"] = df["tags"].map(lambda t: list(t) if isinstance(t, list) else str(t or "").split())
    df["tags"] = df["tag_list"].map(" ".join)
    df["img_path"] = df["url"]  # 서버에서는 파일 경로 대신 이미지 URL을 쓴다
    return df


load_metadata = st.cache_data(show_spinner="메타데이터 불러오는 중...")(read_metadata)


def thumbnail(img_path: str) -> str:
    """갤러리용 축소 이미지 경로. 없으면 만들어서 cache/thumbs/에 저장한다."""
    if img_path.startswith(("http://", "https://")):  # 서버: URL은 그대로 쓴다
        return img_path
    src = Path(img_path)
    dst = THUMB_DIR / src.name
    if not dst.exists():
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.save(dst, quality=85)
    return str(dst)