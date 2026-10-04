"""Safebooru에서 수집한 원본 태그 (사람이 붙인 정답 태그 역할)."""
from methods.base import SearchMethod, TagSource


class SafebooruTags(SearchMethod, TagSource):
    key = "safebooru"
    name = "Safebooru 태그"
    description = "Safebooru 사용자들이 직접 붙인 태그"

    def tags_of(self, idx: int) -> list[str]:
        return self.df["tag_list"].iat[idx]
