"""Phash(지각 해시) 기반 유사 검색 / 중복 탐지."""
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from core.dataset import CACHE_DIR
from methods.base import Hit, ImageSearch, PairFinder, SearchMethod

CACHE_CSV = CACHE_DIR / "phash.csv"


def _bits(h: imagehash.ImageHash) -> np.ndarray:
    return h.hash.flatten().astype(bool)


class PhashMethod(SearchMethod, ImageSearch, PairFinder):
    key = "phash"
    name = "Phash"
    description = "이미지를 64비트 해시로 바꿔 해밍 거리(0=동일, 64=완전히 다름)로 비교"
    score_name = "거리"
    higher_is_better = False
    threshold_range = (0, 20, 6)

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        hashes = self._load_hashes()
        self.bits = np.stack([_bits(hashes[i]) for i in df["id"]])
        # 모든 쌍의 해밍 거리. 수천 장까지는 메모리 문제 없음
        self.dist = (self.bits[:, None, :] != self.bits[None, :, :]).sum(axis=2).astype(np.uint8)

    def _load_hashes(self) -> dict[str, imagehash.ImageHash]:
        cached = {}
        if CACHE_CSV.exists():
            c = pd.read_csv(CACHE_CSV, dtype=str)
            cached = {i: imagehash.hex_to_hash(h) for i, h in zip(c["id"], c["phash"])}
        elif "phash" in self.df.columns:  # 서버: 로컬 캐시 파일이 없으면 Supabase에 저장된 값을 쓴다
            cached = {i: imagehash.hex_to_hash(h) for i, h in zip(self.df["id"], self.df["phash"]) if isinstance(h, str) and h}
        missing = [(i, p) for i, p in zip(self.df["id"], self.df["img_path"]) if i not in cached]
        if missing:
            for i, p in missing:
                with Image.open(p) as img:
                    cached[i] = imagehash.phash(img)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": list(cached), "phash": [str(h) for h in cached.values()]}).to_csv(CACHE_CSV, index=False)
        return cached

    def format_score(self, score: float) -> str:
        return f"거리 {int(score)}"

    def _top_k(self, d: np.ndarray, k: int, exclude: int | None = None) -> list[Hit]:
        order = np.argsort(d, kind="stable")
        return [Hit(int(j), float(d[j])) for j in order if j != exclude][:k]

    def search_by_image(self, idx: int, k: int) -> list[Hit]:
        return self._top_k(self.dist[idx], k, exclude=idx)

    def search_by_upload(self, img: Image.Image, k: int) -> list[Hit]:
        q = _bits(imagehash.phash(img))
        return self._top_k((self.bits != q).sum(axis=1), k)

    def find_pairs(self, threshold: float) -> list[tuple[int, int, float]]:
        i, j = np.where(np.triu(self.dist <= threshold, k=1))
        d = self.dist[i, j]
        order = np.argsort(d, kind="stable")
        return [(int(i[o]), int(j[o]), float(d[o])) for o in order]