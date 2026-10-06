"""데이터셋(metadata.csv + images/) 로딩과 썸네일 관리."""
import hashlib
import urllib.request
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
REMOTE_DIR = CACHE_DIR / "remote"  # 서버: URL 이미지를 한 번 내려받아 두는 곳
THUMB_SIZE = 384
DOWNLOAD_TIMEOUT = 30


def is_url(img_path: str) -> bool:
    return img_path.startswith(("http://", "https://"))


def local_image(img_path: str) -> str:
    """파일로 열 수 있는 이미지 경로. 서버처럼 img_path가 URL이면 내려받아 캐시하고 그 경로를 돌려준다.

    PIL·OpenCV·CLIP 등 파일 경로가 필요한 곳은 img_path를 바로 쓰지 말고 이 함수를 거친다.
    """
    if not is_url(img_path):
        return img_path
    suffix = Path(img_path.split("?", 1)[0]).suffix.lower()
    dst = REMOTE_DIR / (hashlib.md5(img_path.encode()).hexdigest() + (suffix if len(suffix) <= 5 else ""))
    if not dst.exists():
        REMOTE_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(img_path, headers={"User-Agent": "drawing-ref-validation/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r:
                data = r.read()
        except OSError as e:
            raise FileNotFoundError(f"이미지를 내려받지 못했습니다 ({img_path}): {e}") from e
        tmp = dst.with_name(dst.name + ".part")  # 받다가 끊겨도 깨진 파일이 캐시로 남지 않게
        tmp.write_bytes(data)
        tmp.replace(dst)
    return str(dst)


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

    rows = db.call(db.fetch_all, "images", "id,url,source_page,tags,width,height,phash")
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
    if is_url(img_path):  # 서버: URL은 브라우저가 직접 불러오므로 그대로 쓴다
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