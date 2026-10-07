"""Supabase 연결. 접속 정보는 프로젝트 폴더의 .env (또는 SUPABASE.env)에서 읽는다."""
import os
from functools import lru_cache

from dotenv import load_dotenv

from core.dataset import ROOT

ENV_FILES = [ROOT / ".env", ROOT / "SUPABASE.env"]
PAGE_SIZE = 1000  # Supabase(PostgREST)가 한 번에 돌려주는 최대 행 수


class DBError(RuntimeError):
    pass


# 쓰기 허용 여부 확인 함수. 웹앱(app.py)이 매 실행마다 "방문자면 DBError"를 거는 함수를 넣는다.
# 함수는 호출될 때 그 사용자 세션의 상태를 읽으므로 서버 하나에 여러 사람이 접속해도 각자 판단된다.
# 웹앱 밖(training.py 등 스크립트, 테스트)에서는 None → 제한 없음.
_write_guard = None


def set_write_guard(fn) -> None:
    global _write_guard
    _write_guard = fn


def check_write() -> None:
    """DB에 쓰기 전에 부른다. 허용되지 않으면 DBError."""
    if _write_guard is not None:
        _write_guard()


def _load_env() -> None:
    for f in ENV_FILES:
        if f.exists():
            load_dotenv(f, override=False)


def is_configured() -> bool:
    _load_env()
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))


@lru_cache(maxsize=1)
def get_client():
    from supabase import create_client
    if not is_configured():
        raise DBError("Supabase 접속 정보가 없습니다. 프로젝트 폴더의 .env에 SUPABASE_URL, SUPABASE_KEY를 넣어 주세요.")
    url = os.environ["SUPABASE_URL"].strip()
    if "supabase.com/dashboard" in url:
        raise DBError("SUPABASE_URL에 대시보드 주소가 들어 있습니다. https://<프로젝트ID>.supabase.co 형식이어야 합니다.")
    return create_client(url, os.environ["SUPABASE_KEY"].strip())


def friendly_error(e: Exception) -> str:
    """Supabase/PostgREST 오류를 팀원이 이해할 수 있는 한국어로."""
    code = getattr(e, "code", None) or ""
    msg = getattr(e, "message", None) or str(e)
    if code == "PGRST202" or "Could not find the function" in msg:
        return "DB에 필요한 함수가 없습니다. supabase/schema.sql을 Supabase SQL Editor에서 먼저 실행해 주세요."
    if code == "PGRST205" or "Could not find the table" in msg:
        return "DB에 테이블이 없습니다. supabase/schema.sql을 Supabase SQL Editor에서 먼저 실행해 주세요."
    if code == "23503":
        return "DB의 images 테이블에 없는 이미지입니다. python migrate_to_supabase.py를 실행해 이미지를 등록해 주세요."
    if code == "42501" and "permission denied for table" in msg:
        return "DB 테이블 사용 권한(GRANT)이 없습니다. supabase/schema.sql의 'GRANT' 부분을 SQL Editor에서 실행해 주세요."
    if code == "42501":
        return "DB 접근 규칙(RLS)에 막혔습니다. 허용되지 않는 작업이거나, schema.sql의 RLS 정책 부분이 실행되지 않았습니다."
    if "401" in msg or "Invalid API key" in msg:
        return "SUPABASE_KEY가 올바르지 않습니다."
    return f"DB 오류: {msg}"


def call(fn, *args, **kwargs):
    """DB 작업을 실행하고, 실패하면 한국어 메시지의 DBError로 바꿔 던진다."""
    try:
        return fn(*args, **kwargs)
    except DBError:
        raise
    except Exception as e:
        raise DBError(friendly_error(e)) from e


def fetch_all(table: str, columns: str = "*") -> list[dict]:
    """페이지를 넘겨 가며 테이블 전체를 읽는다."""
    rows, start = [], 0
    while True:
        page = get_client().table(table).select(columns).range(start, start + PAGE_SIZE - 1).execute().data
        rows += page
        if len(page) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def upsert(table: str, rows: list[dict], on_conflict: str, chunk: int = 500, ignore_duplicates: bool = False) -> None:
    """ignore_duplicates=True면 이미 있는 행은 건드리지 않고 새 행만 넣는다."""
    check_write()
    for i in range(0, len(rows), chunk):
        get_client().table(table).upsert(rows[i:i + chunk], on_conflict=on_conflict, returning="minimal",
                                         ignore_duplicates=ignore_duplicates).execute()
