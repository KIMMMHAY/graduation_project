"""웹앱 없이 실행하는 스크립트(training.py 등)용 설정."""
import logging
import sys


def quiet_streamlit() -> None:
    """core 모듈이 st.cache_data를 쓰기 때문에 단독 실행 시 뜨는 "No runtime found" 경고를 숨긴다.

    core.dataset을 import하기 전에 호출해야 한다. streamlit이 로거를 만들 때 레벨을 덮어쓰므로
    해당 모듈을 먼저 불러온 뒤 레벨을 바꾼다.
    """
    import streamlit.runtime.caching.cache_data_api  # noqa: F401
    logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)


def utf8_console() -> None:
    """Windows 콘솔에서 한글 깨짐 방지. __main__에서만 호출."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
