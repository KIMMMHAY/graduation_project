"""Supabase 연결 테스트: 접속 → 테이블 확인 → 라벨 쓰기/읽기/덮어쓰기 → 테스트 행 정리.

실행: python check_supabase.py
"""
import sys

from core.cli import quiet_streamlit, utf8_console

quiet_streamlit()

from core import db  # noqa: E402
from core.tagging import load_labels, load_tags, save_label  # noqa: E402

TEST_LABELER = "__connection_test__"
TABLES = ["images", "tags", "labels", "predictions"]


def step(text: str) -> None:
    print(f"- {text} ... ", end="", flush=True)


def main() -> int:
    try:
        step("접속 정보 확인")
        if not db.is_configured():
            print("실패\n  .env에 SUPABASE_URL, SUPABASE_KEY가 없습니다.")
            return 1
        print("OK")

        counts = {}
        for t in TABLES:
            step(f"{t} 테이블 읽기")
            counts[t] = db.get_client().table(t).select("*", count="exact").limit(1).execute().count
            print(f"OK ({counts[t]}행)")
        if not counts["images"]:
            print("  images가 비어 있어 쓰기 테스트를 할 수 없습니다. python migrate_to_supabase.py를 먼저 실행하세요.")
            return 1

        image_id = db.get_client().table("images").select("id").limit(1).execute().data[0]["id"]
        tag_key = load_tags()[0].key

        step("라벨 쓰기")
        save_label(image_id, tag_key, 1, TEST_LABELER)
        print("OK")
        step("라벨 읽기")
        mine = load_labels().query("labeler == @TEST_LABELER")
        assert mine["value"].tolist() == [1], mine
        print("OK")
        step("같은 라벨 덮어쓰기")
        save_label(image_id, tag_key, 0, TEST_LABELER)
        mine = load_labels().query("labeler == @TEST_LABELER")
        assert len(mine) == 1 and mine["value"].iat[0] == 0, mine
        print("OK (1행 유지, 값 1→0)")

        step("테스트 행 정리")
        deleted = db.get_client().table("labels").delete().eq("labeler", TEST_LABELER).execute().data
        print("OK" if deleted else "건너뜀 (삭제 권한 없음 — publishable 키는 정상)")
    except AssertionError as e:
        print(f"실패\n  결과가 예상과 다릅니다:\n{e}")
        return 1
    except Exception as e:
        print(f"실패\n  {db.friendly_error(e)}")
        return 1
    print("\n연결 테스트 통과: 읽기/쓰기 모두 정상입니다.")
    return 0


if __name__ == "__main__":
    utf8_console()
    sys.exit(main())
