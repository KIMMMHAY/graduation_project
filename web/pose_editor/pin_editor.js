/**
 * 포즈 핀 편집기 — 프레임워크에 의존하지 않는 순수 ES 모듈 (Streamlit 없이도 동작).
 *
 *   const editor = new PinEditor(container, {
 *     image: 'data:image/jpeg;base64,...' | null,   // null이면 빈 캔버스 (포즈 검색용)
 *     aspect: 0.75,                                   // 가로 / 세로
 *     keypoints: [{x, y, state}, ...],                // 13개, 좌표는 0~1 비율
 *     joints: [{key, name, side}], bones: [[a, b, side]],  // a/b의 13·14는 어깨·골반 중점
 *     height: 620, readonly: false,
 *     onChange(keypoints) {}, onViewChange(view) {}, initialView: {x, y, w, h},
 *   });
 *   editor.getKeypoints(); editor.destroy();
 *
 * 조작: 핀 드래그 = 이동, 빈 곳 드래그 = 화면 이동, 휠/버튼 = 확대·축소,
 *       핀 선택 후 상태 버튼(또는 키보드 1·2·3) = 보임/가려짐/없음.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
export const SIDE_COLORS = { l: '#2563eb', r: '#ea580c', c: '#16a34a' };
export const STATE_LABELS = { visible: '보임', occluded: '가려짐', absent: '없음' };
const STATE_KEYS = ['visible', 'occluded', 'absent'];
const WORLD_H = 1000;      // 내부 좌표계 높이 (가로는 aspect × 1000)
const PIN_RADIUS_PX = 7;   // 화면에서 보이는 핀 반지름 (확대해도 같은 크기)
const MAX_ZOOM = 12;

function el(tag, attrs = {}, parent = null) {
  const node = tag.startsWith('svg:') ? document.createElementNS(SVG_NS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

export class PinEditor {
  constructor(container, opts) {
    this.opts = opts;
    this.W = WORLD_H * opts.aspect;
    this.H = WORLD_H;
    this.kps = opts.keypoints.map((k) => ({ ...k }));
    this.selected = null;
    this.drag = null;
    this.view = opts.initialView ? { ...opts.initialView } : { x: 0, y: 0, w: this.W, h: this.H };
    this.root = el('div', { class: 'pe-root', tabindex: '0' }, container);
    this._buildToolbar();
    this._buildCanvas();
    this._bind();
    this.render();
  }

  // ---------- 공개 API ----------
  getKeypoints() {
    return this.kps.map((k) => ({ x: +k.x.toFixed(4), y: +k.y.toFixed(4), state: k.state }));
  }

  setKeypoints(keypoints) {
    this.kps = keypoints.map((k) => ({ ...k }));
    this.render();
  }

  destroy() {
    this.resizeObserver?.disconnect();
    this.root.remove();
  }

  // ---------- 화면 구성 ----------
  _buildToolbar() {
    const bar = el('div', { class: 'pe-toolbar' }, this.root);
    const zoom = el('div', { class: 'pe-group' }, bar);
    this.btnZoomOut = el('button', { type: 'button', title: '축소 (-)', text: '－' }, zoom);
    this.btnZoomIn = el('button', { type: 'button', title: '확대 (+)', text: '＋' }, zoom);
    this.btnFit = el('button', { type: 'button', title: '전체 보기 (0)', text: '전체' }, zoom);

    const sel = el('div', { class: 'pe-group' }, bar);
    this.selLabel = el('span', { class: 'pe-selected' }, sel);
    this.stateButtons = STATE_KEYS.map((s, i) =>
      el('button', { type: 'button', 'data-state': s, title: `${STATE_LABELS[s]} (${i + 1})`, text: STATE_LABELS[s] }, sel));

    const legend = el('div', { class: 'pe-legend' }, bar);
    el('span', { class: 'pe-dot', style: `background:${SIDE_COLORS.l}` }, legend);
    el('span', { text: '캐릭터의 왼쪽' }, legend);
    el('span', { class: 'pe-dot', style: `background:${SIDE_COLORS.r}` }, legend);
    el('span', { text: '캐릭터의 오른쪽' }, legend);

    el('div', {
      class: 'pe-guide',
      text: '좌우는 캐릭터 기준입니다. 정면을 보는 그림이면 캐릭터의 왼쪽(파랑)이 화면 오른쪽에 옵니다.',
    }, this.root);
  }

  _buildCanvas() {
    const wrap = el('div', { class: 'pe-canvas', style: `height:${this.opts.height || 620}px` }, this.root);
    this.svg = el('svg:svg', { preserveAspectRatio: 'xMidYMid meet' }, wrap);
    if (this.opts.image) {
      el('svg:image', { href: this.opts.image, x: 0, y: 0, width: this.W, height: this.H, preserveAspectRatio: 'none' }, this.svg);
    } else {
      el('svg:rect', { x: 0, y: 0, width: this.W, height: this.H, class: 'pe-blank' }, this.svg);
      for (let i = 1; i < 10; i++) {  // 빈 캔버스용 보조선
        el('svg:line', { x1: 0, y1: this.H * i / 10, x2: this.W, y2: this.H * i / 10, class: 'pe-grid' }, this.svg);
        el('svg:line', { x1: this.W * i / 10, y1: 0, x2: this.W * i / 10, y2: this.H, class: 'pe-grid' }, this.svg);
      }
    }
    this.layer = el('svg:g', {}, this.svg);
  }

  _bind() {
    this.btnZoomIn.onclick = () => this._zoomAt(1.4);
    this.btnZoomOut.onclick = () => this._zoomAt(1 / 1.4);
    this.btnFit.onclick = () => { this.view = { x: 0, y: 0, w: this.W, h: this.H }; this._viewChanged(); };
    this.stateButtons.forEach((b) => { b.onclick = () => this._setState(b.dataset.state); });

    this.svg.addEventListener('pointerdown', (e) => this._onDown(e));
    this.svg.addEventListener('pointermove', (e) => this._onMove(e));
    this.svg.addEventListener('pointerup', (e) => this._onUp(e));
    this.svg.addEventListener('pointercancel', (e) => this._onUp(e));
    this.svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      this._zoomAt(Math.exp(-e.deltaY * 0.0015), this._toWorld(e));
    }, { passive: false });
    this.root.addEventListener('keydown', (e) => {
      if (e.key === '+' || e.key === '=') this._zoomAt(1.4);
      else if (e.key === '-') this._zoomAt(1 / 1.4);
      else if (e.key === '0') this.btnFit.onclick();
      else if (['1', '2', '3'].includes(e.key)) this._setState(STATE_KEYS[+e.key - 1]);
      else return;
      e.preventDefault();
    });
    this.resizeObserver = new ResizeObserver(() => this.render());  // 창 크기가 바뀌면 핀 크기만 다시 계산
    this.resizeObserver.observe(this.svg);
  }

  // ---------- 좌표 ----------
  _toWorld(e) {
    const pt = this.svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(this.svg.getScreenCTM().inverse());
    return { x: p.x, y: p.y };
  }

  _unitsPerPx() {
    const ctm = this.svg.getScreenCTM();
    return ctm && ctm.a ? 1 / ctm.a : this.view.w / 600;
  }

  _point(i) {  // 13 = 어깨 중점, 14 = 골반 중점
    if (i === 13 || i === 14) {
      const [a, b] = i === 13 ? [1, 2] : [7, 8];
      const ka = this.kps[a], kb = this.kps[b];
      const state = [ka.state, kb.state].includes('absent') ? 'absent'
        : [ka.state, kb.state].includes('occluded') ? 'occluded' : 'visible';
      return { x: (ka.x + kb.x) / 2 * this.W, y: (ka.y + kb.y) / 2 * this.H, state };
    }
    const k = this.kps[i];
    return { x: k.x * this.W, y: k.y * this.H, state: k.state };
  }

  // ---------- 조작 ----------
  _onDown(e) {
    if (this.opts.readonly) return;
    this.root.focus({ preventScroll: true });
    const idx = e.target.dataset ? e.target.dataset.idx : undefined;
    this.svg.setPointerCapture(e.pointerId);
    if (idx !== undefined) {
      this.selected = +idx;
      this.drag = { type: 'pin', moved: false };
    } else {
      this.drag = { type: 'pan', sx: e.clientX, sy: e.clientY, view: { ...this.view } };
    }
    this.render();
  }

  _onMove(e) {
    if (!this.drag) return;
    if (this.drag.type === 'pin') {
      const p = this._toWorld(e);
      const k = this.kps[this.selected];
      k.x = clamp(p.x / this.W, 0, 1);
      k.y = clamp(p.y / this.H, 0, 1);
      this.drag.moved = true;
      this.render();
    } else {
      const u = this._unitsPerPx();
      this.view.x = this.drag.view.x - (e.clientX - this.drag.sx) * u;
      this.view.y = this.drag.view.y - (e.clientY - this.drag.sy) * u;
      this._clampView();
      this._applyView();
    }
  }

  _onUp(e) {
    if (!this.drag) return;
    const d = this.drag;
    this.drag = null;
    if (this.svg.hasPointerCapture(e.pointerId)) this.svg.releasePointerCapture(e.pointerId);
    if (d.type === 'pin' && d.moved) this._changed();
    if (d.type === 'pan') this._viewChanged();
  }

  _setState(state) {
    if (this.opts.readonly || this.selected === null) return;
    this.kps[this.selected].state = state;
    this.render();
    this._changed();
  }

  _zoomAt(factor, center) {
    const w = clamp(this.view.w / factor, this.W / MAX_ZOOM, this.W);
    const h = w * this.H / this.W;
    const c = center || { x: this.view.x + this.view.w / 2, y: this.view.y + this.view.h / 2 };
    this.view = { x: c.x - (c.x - this.view.x) * (w / this.view.w), y: c.y - (c.y - this.view.y) * (h / this.view.h), w, h };
    this._clampView();
    this._viewChanged();
  }

  _clampView() {  // 화면이 이미지 밖으로 너무 벗어나지 않게
    this.view.x = clamp(this.view.x, -this.view.w * 0.5, this.W - this.view.w * 0.5);
    this.view.y = clamp(this.view.y, -this.view.h * 0.5, this.H - this.view.h * 0.5);
  }

  _changed() { this.opts.onChange?.(this.getKeypoints()); }

  _viewChanged() {
    this.render();
    this.opts.onViewChange?.({ ...this.view });
  }

  _applyView() {
    const v = this.view;
    this.svg.setAttribute('viewBox', `${v.x} ${v.y} ${v.w} ${v.h}`);
  }

  // ---------- 그리기 ----------
  render() {
    this._applyView();
    const u = this._unitsPerPx();
    const r = PIN_RADIUS_PX * u;
    this.layer.replaceChildren();

    for (const [a, b, side] of this.opts.bones) {
      const pa = this._point(a), pb = this._point(b);
      const absent = pa.state === 'absent' || pb.state === 'absent';
      const occluded = pa.state === 'occluded' || pb.state === 'occluded';
      const common = { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, 'stroke-linecap': 'round' };
      el('svg:line', { ...common, stroke: 'white', 'stroke-width': 5 * u, opacity: absent ? 0.2 : 0.7 }, this.layer);
      el('svg:line', {
        ...common, stroke: absent ? '#9ca3af' : SIDE_COLORS[side], 'stroke-width': 3 * u, opacity: absent ? 0.35 : 1,
        'stroke-dasharray': absent || occluded ? `${6 * u} ${4 * u}` : 'none',
      }, this.layer);
    }

    this.opts.joints.forEach((j, i) => {
      const p = this._point(i);
      const color = SIDE_COLORS[j.side];
      const isSel = i === this.selected;
      if (isSel) el('svg:circle', { cx: p.x, cy: p.y, r: r * 1.9, fill: 'none', stroke: '#111827', 'stroke-width': 2 * u }, this.layer);
      const pin = el('svg:circle', {
        cx: p.x, cy: p.y, r: isSel ? r * 1.25 : r, 'data-idx': i, class: 'pe-pin',
        fill: p.state === 'visible' ? color : p.state === 'occluded' ? 'white' : '#9ca3af',
        stroke: p.state === 'absent' ? '#6b7280' : p.state === 'occluded' ? color : 'white',
        'stroke-width': (p.state === 'occluded' ? 3 : 2) * u,
        'stroke-dasharray': p.state === 'occluded' ? `${3 * u} ${2 * u}` : 'none',
        opacity: p.state === 'absent' ? 0.6 : 1,
      }, this.layer);
      el('svg:title', { text: `${j.name} · ${STATE_LABELS[p.state]}` }, pin);
      if (isSel) {
        el('svg:text', {
          x: p.x + r * 2.2, y: p.y - r * 1.6, 'font-size': 13 * u, class: 'pe-label',
          'stroke-width': 3 * u, text: `${j.name} · ${STATE_LABELS[p.state]}`,
        }, this.layer);
      }
    });

    const sel = this.selected === null ? null : this.opts.joints[this.selected];
    this.selLabel.textContent = sel ? `선택: ${sel.name}` : '핀을 클릭해 선택';
    this.stateButtons.forEach((b) => {
      b.disabled = !sel || this.opts.readonly;
      b.classList.toggle('pe-active', !!sel && this.kps[this.selected].state === b.dataset.state);
    });
  }
}

export const PIN_EDITOR_CSS = `
.pe-root { font-family: inherit; outline: none; }
.pe-toolbar { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin-bottom: 6px; }
.pe-group { display: flex; gap: 4px; align-items: center; }
.pe-toolbar button { border: 1px solid #d1d5db; background: #fff; color: #111827; border-radius: 6px;
  padding: 3px 10px; font-size: 13px; cursor: pointer; }
.pe-toolbar button:disabled { opacity: .4; cursor: default; }
.pe-toolbar button.pe-active { background: #111827; color: #fff; border-color: #111827; }
.pe-selected { font-size: 13px; min-width: 120px; color: #374151; }
.pe-legend { display: flex; gap: 6px; align-items: center; font-size: 12px; color: #4b5563; }
.pe-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.pe-guide { font-size: 12px; color: #92400e; background: #fef3c7; border-radius: 6px; padding: 4px 8px; margin-bottom: 6px; }
.pe-canvas { width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; background: #f3f4f6; overflow: hidden; }
.pe-canvas svg { width: 100%; height: 100%; display: block; touch-action: none; cursor: grab; user-select: none; }
.pe-pin { cursor: move; }
.pe-blank { fill: #fff; }
.pe-grid { stroke: #e5e7eb; stroke-width: 1; vector-effect: non-scaling-stroke; }
.pe-label { fill: #111827; stroke: #fff; paint-order: stroke; font-weight: 600; pointer-events: none; }
`;
