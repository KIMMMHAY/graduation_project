"""3D 마네킹(web/mannequin/mannequin.js)을 Streamlit에 붙이는 어댑터 (st.components.v2).

three.js는 .streamlit/config.toml의 enableStaticServing으로 static/vendor/three/ 를 제공해 불러온다.
"""
import hashlib
from pathlib import Path

import streamlit as st

from core import pose as P
from core.dataset import ROOT

_JS = (ROOT / "web" / "mannequin" / "mannequin.js").read_text(encoding="utf-8") + "\n" + \
      (Path(__file__).parent / "mannequin_streamlit.js").read_text(encoding="utf-8")
_NAME = "pose_mannequin"
_MANNEQUIN = st.components.v2.component(_NAME, js=_JS)
THREE_URL = "app/static/vendor/three/three.module.js"  # 페이지 주소 기준 상대 경로


def _renderer():
    """등록이 사라졌으면 다시 등록한다 (components/pose_editor.py와 같은 이유)."""
    global _MANNEQUIN
    try:
        from streamlit.runtime import Runtime
        if Runtime.exists() and Runtime.instance().bidi_component_registry.get(_NAME) is None:
            _MANNEQUIN = st.components.v2.component(_NAME, js=_JS)
    except Exception:
        pass
    return _MANNEQUIN


def mannequin(*, key: str, version: str, height: int = 560, state: dict | None = None) -> dict | None:
    """마네킹을 그리고 현재 투영 결과를 돌려준다: {keypoints, aspect, facing, state}. 아직 준비 전이면 None."""
    data = {"version": version, "height": height, "state": state, "three_url": THREE_URL}
    safe_key = "mannequin_" + hashlib.md5(key.encode("utf-8")).hexdigest()  # v2 키에는 '__'를 쓸 수 없다
    result = _renderer()(key=safe_key, data=data, on_pose_change=lambda: None)
    snap = result.get("pose") if result is not None else None
    if not isinstance(snap, dict) or snap.get("version") != version:
        return None
    try:
        return {**snap, "keypoints": P.validate_keypoints(snap["keypoints"])}
    except (P.PoseError, KeyError):
        return None
