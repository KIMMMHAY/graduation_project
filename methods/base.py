"""검색 방식 공통 인터페이스.

검색 방식 하나 = SearchMethod를 상속한 클래스 하나. 지원하는 기능의 믹스인을 같이 상속하면
해당 기능을 쓰는 페이지에 자동으로 나타난다.

    ImageSearch  이미지 → 비슷한 이미지        (유사 검색 페이지)   예: Phash, CLIP 임베딩
    PairFinder   임계값 이하인 이미지 쌍       (중복 확인 페이지)   예: Phash
    TagSource    이미지별 태그 + 태그 필터     (갤러리 페이지)      예: Safebooru, WD Tagger, CLIP 태깅
    (예정) PoseSearch  관절 좌표 → 비슷한 포즈  예: OpenPose 결과

새 방식 추가 방법:
    1. methods/새방식.py 에 클래스 작성 (key, name 지정 + 믹스인 메서드 구현)
    2. methods/__init__.py 의 METHOD_CLASSES 에 추가
모든 메서드의 이미지 인덱스(idx)는 load_metadata() 결과의 행 번호다.
무거운 모델 결과는 __init__에서 drawing_ref_test/cache/ 에 저장해 두고 재사용하는 것을 권장한다.
"""
from abc import ABC
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class Hit:
    idx: int      # 데이터셋 행 번호
    score: float  # 방식별 점수 (Phash는 해밍 거리, CLIP이면 코사인 유사도 등)


class SearchMethod(ABC):
    key: str = ""          # 파일명에 쓰는 짧은 영문 id (평가 파일: eval_{key}.csv)
    name: str = ""         # 화면 표시 이름
    description: str = ""
    score_name: str = "점수"
    higher_is_better: bool = True

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def format_score(self, score: float) -> str:
        return f"{self.score_name} {score:.3g}"


class ImageSearch:
    def search_by_image(self, idx: int, k: int) -> list[Hit]:
        """데이터셋 안의 이미지(idx)와 비슷한 k장. 자기 자신은 제외."""
        raise NotImplementedError

    def search_by_upload(self, img: Image.Image, k: int) -> list[Hit]:
        """외부 이미지로 검색. 지원하지 않으면 구현하지 않아도 된다."""
        raise NotImplementedError


class PairFinder:
    # 임계값 슬라이더 (최소, 최대, 기본값)
    threshold_range: tuple[float, float, float] = (0.0, 1.0, 0.5)

    def find_pairs(self, threshold: float) -> list[tuple[int, int, float]]:
        """임계값 기준으로 '같은 그림'으로 보이는 (i, j, score) 목록, 가까운 순."""
        raise NotImplementedError


class TagSource:
    def tags_of(self, idx: int) -> list[str]:
        raise NotImplementedError

    def tag_counts(self) -> Counter:
        counts = Counter()
        for i in range(len(self.df)):
            counts.update(self.tags_of(i))
        return counts

    def filter_by_tags(self, tags: list[str], match_all: bool = True) -> list[int]:
        if not tags:
            return list(range(len(self.df)))
        wanted = set(tags)
        test = wanted.issubset if match_all else (lambda s: not wanted.isdisjoint(s))
        return [i for i in range(len(self.df)) if test(set(self.tags_of(i)))]
