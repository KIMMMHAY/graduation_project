// Streamlit components.v2 어댑터. 실행 시 web/mannequin/mannequin.js 뒤에 이어 붙여 하나의 모듈로 쓴다.
// three.js는 Streamlit 정적 파일(/app/static/...)에서 불러온다 → 인터넷 없이 동작.
// 같은 version이면 조작 중인 마네킹을 유지하고, version이 바뀌면(초기화 등) 새로 만든다.
const MANNEQUIN_CACHE = (window.__mannequinCache = window.__mannequinCache || {});

export default async function (component) {
  const { data, parentElement, setStateValue } = component;
  let host = parentElement.querySelector('.mq-host');
  if (!host) {
    const style = document.createElement('style');
    style.textContent = MANNEQUIN_CSS;
    parentElement.appendChild(style);
    host = document.createElement('div');
    host.className = 'mq-host';
    parentElement.appendChild(host);
  }
  if (host.__version === data.version) return;  // 이미 만들었거나 만드는 중
  host.__version = data.version;
  host.__mq?.destroy();
  host.__mq = null;

  const THREE = await import(new URL(data.three_url, document.baseURI).href);
  if (host.__version !== data.version) return;  // 불러오는 사이에 다른 버전이 요청됨

  const send = (snap) => {
    MANNEQUIN_CACHE[data.version] = snap.state;
    setStateValue('pose', { version: data.version, ...snap });
  };
  const cached = MANNEQUIN_CACHE[data.version];
  host.__mq = new Mannequin(host, { THREE, height: data.height, state: cached || data.state, onChange: send });
  window.__mannequin = host.__mq;  // 테스트·디버깅용
  if (!cached) send(host.__mq.snapshot());  // 처음 상태도 알려서 바로 검색할 수 있게
}
