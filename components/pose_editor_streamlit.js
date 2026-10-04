// Streamlit components.v2 어댑터. 실행 시 web/pose_editor/pin_editor.js 뒤에 이어 붙여 하나의 모듈로 쓴다.
// 같은 version이면 편집 중인 상태를 유지하고, version이 바뀌면(다른 이미지·초기화) 새로 만든다.
const PIN_EDITOR_CACHE = (window.__pinEditorCache = window.__pinEditorCache || {});

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let host = parentElement.querySelector('.pe-host');
  if (!host) {
    const style = document.createElement('style');
    style.textContent = PIN_EDITOR_CSS;
    parentElement.appendChild(style);
    host = document.createElement('div');
    host.className = 'pe-host';
    parentElement.appendChild(host);
  }
  if (host.__editor && host.__version === data.version) return;
  if (host.__editor) host.__editor.destroy();

  // Streamlit이 컴포넌트를 다시 그려도 방금 옮긴 핀과 확대 상태가 사라지지 않도록 브라우저 메모리에 보관
  const cached = PIN_EDITOR_CACHE[data.version] || {};
  host.__version = data.version;
  host.__editor = new PinEditor(host, {
    ...data,
    keypoints: cached.keypoints || data.keypoints,
    initialView: cached.view,
    onChange(keypoints) {
      PIN_EDITOR_CACHE[data.version] = { ...PIN_EDITOR_CACHE[data.version], keypoints };
      setStateValue('pose', { version: data.version, keypoints });
    },
    onViewChange(view) {
      PIN_EDITOR_CACHE[data.version] = { ...PIN_EDITOR_CACHE[data.version], view };
    },
  });
}
