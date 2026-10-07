"""팀원 확인 (입장 화면). 명단은 DB members 테이블에 있고, 앱은 member_name 함수로 이메일 하나만 물어본다.

저장소가 공개라 이메일을 코드·파일에 두지 않는다. 팀원 키로는 명단 전체를 읽을 수 없다(schema.sql 참고).
이메일 확인은 비밀번호가 아니다: 팀원 이메일을 아는 사람은 들어올 수 있다. 편집 권한을 나누는 용도다.
"""
import re

from core import db

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(normalize_email(email)))


def member_name(email: str) -> str | None:
    """등록된(사용 중) 팀원이면 이름, 아니면 None. DB 문제는 db.DBError(한국어 메시지)."""
    email = normalize_email(email)
    if not is_email(email):
        return None
    name = db.call(lambda: db.get_client().rpc("member_name", {"p_email": email}).execute().data)
    return name.strip() if isinstance(name, str) and name.strip() else None
