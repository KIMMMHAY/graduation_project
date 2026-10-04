"""핀 편집기(web/pose_editor/pin_editor.js)를 Streamlit에 붙이는 어댑터 (st.components.v2)."""
import base64
import hashlib
import io
from pathlib import Path

import streamlit as st
from PIL import Image

from core import pose as P
from core.dataset import ROOT

_JS = (ROOT / "web" / "pose_editor" / "pin_editor.js").read_text(encoding="utf-8") + "\n" + \
      (Path(__file__).parent / "pose_editor_streamlit.js").read_text(encoding="utf-8")
_NAME = "pose_pin_editor"
_PIN_EDITOR = st.components.v2.component(_NAME, js=_JS)


def _renderer():
    """등록이 사라졌으면 다시 등록한다.

    컴포넌트 등록은 Streamlit 런타임에 저장되는데, 파이썬 모듈은 캐시된 채로 런타임만 새로 생기는 경우가 있다
    (테스트 실행기, 서버 재시작 등). 그때 'is not registered' 오류가 나지 않도록 매번 확인한다.
    """
    global _PIN_EDITOR
    try:
        from streamlit.runtime import Runtime
        if Runtime.exists() and Runtime.instance().bidi_component_registry.get(_NAME) is None:
            _PIN_EDITOR = st.components.v2.component(_NAME, js=_JS)
    except Exception:
        pass  # 내부 API가 바뀌어도 기존 등록으로 동작은 시도한다
    return _PIN_EDITOR

JOINTS_META = [{"key": k, "name": name, "side": "c" if k == "head" else k[0]} for k, name in P.JOINTS]
BONES_META = [[a, b, side] for _, a, b, _, side in P.BONES]
MAX_IMAGE_SIDE = 1200


@st.cache_data(show_spinner=False, max_entries=64)
def image_data_url(path: str) -> str:
    """브라우저로 보낼 이미지 (data URL). 너무 크면 줄인다."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pin_editor(*, key: str, version: str, keypoints: list[dict], aspect: float,
               image_path: str | None = None, height: int = 620) -> list[dict]:
    """핀 편집기를 그리고 현재 핀 위치를 돌려준다.

    version이 바뀌면 편집기를 새로 만든다(다른 이미지, 초기화). 같은 version이면 사용자가 옮긴 핀을 유지한다.
    """
    data = {"version": version, "keypoints": keypoints, "aspect": aspect, "height": height,
            "image": image_data_url(image_path) if image_path else None,
            "joints": JOINTS_META, "bones": BONES_META}
    # 키에는 작성자 이름 등 임의의 문자가 들어올 수 있는데, v2 컴포넌트 키는 '__'를 쓸 수 없다 → 해시로 바꾼다
    safe_key = "pin_editor_" + hashlib.md5(key.encode("utf-8")).hexdigest()
    result = _renderer()(key=safe_key, data=data, on_pose_change=lambda: None)
    state = result.get("pose") if result is not None else None
    if isinstance(state, dict) and state.get("version") == version:
        try:
            return P.validate_keypoints(state["keypoints"])
        except P.PoseError:
            pass
    return keypoints
