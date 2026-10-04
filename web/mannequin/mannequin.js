/**
 * 3D 마네킹 — 프레임워크에 의존하지 않는 순수 ES 모듈 (Streamlit 없이도 동작).
 * three.js는 직접 import하지 않고 생성자로 받는다 → 번들러·로컬 파일·CDN 어디서든 같은 코드로 쓴다.
 *
 *   import * as THREE from '.../three.module.js';
 *   const mq = new Mannequin(container, { THREE, height: 560, state, onChange(snapshot) {} });
 *   mq.snapshot()      // { keypoints: 13개 {x, y, state}, aspect: 1, facing, state }
 *   mq.getState() / mq.setState(state)   // 관절 회전(쿼터니언) + 카메라
 *   mq.resetPose(); mq.setView('front' | 'side' | 'back' | 'high' | 'low'); mq.destroy();
 *
 * 조작: 관절 공 드래그 = 그 뼈를 회전 (끝점이 마우스를 따라감), Shift + 가슴 공 드래그 = 몸 전체 기울이기,
 *       빈 곳 드래그 = 카메라 회전, 휠 = 확대·축소.
 * 좌우는 캐릭터 기준: 캐릭터는 +z(카메라 쪽)를 보고 서 있고, 캐릭터의 왼쪽(l_)이 +x다.
 * 관절 조작 방식(구 위 드래그로 뼈 회전)은 x6ud/pose-search(MIT)의 아이디어를 참고했다. 코드는 새로 작성.
 */

export const JOINT_KEYS = ['head', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
  'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle'];
const PIVOTS = ['root', 'waist', 'neck', 'l_shoulder', 'l_elbow', 'r_shoulder', 'r_elbow', 'l_hip', 'l_knee', 'r_hip', 'r_knee'];
// 끌 수 있는 관절 공 → 회전시킬 관절(pivot)과 그 뼈의 끝점
const HANDLES = {
  head: ['neck', 'head'], chest: ['waist', 'chest'],
  l_elbow: ['l_shoulder', 'l_elbow'], l_wrist: ['l_elbow', 'l_wrist'],
  r_elbow: ['r_shoulder', 'r_elbow'], r_wrist: ['r_elbow', 'r_wrist'],
  l_knee: ['l_hip', 'l_knee'], l_ankle: ['l_knee', 'l_ankle'],
  r_knee: ['r_hip', 'r_knee'], r_ankle: ['r_knee', 'r_ankle'],
};
const COLORS = { l: 0x2563eb, r: 0xea580c, c: 0x16a34a, body: 0xd6d3d1, hover: 0xfacc15 };
const VIEWS = {  // [방위각, 고도] (라디안). 측면은 캐릭터의 왼쪽에서 본 모습
  front: [0, 0.05], side: [Math.PI / 2, 0.05], back: [Math.PI, 0.05], high: [0, 0.85], low: [0, -0.55],
};
const VIEW_LABELS = { front: '정면', side: '측면', back: '뒤', high: '하이앵글', low: '로우앵글' };
const TARGET_Y = 0.95;

export class Mannequin {
  constructor(container, opts) {
    this.T = opts.THREE;
    this.opts = opts;
    this.nodes = {};
    this.handles = [];
    this.cam = { azimuth: VIEWS.front[0], elevation: VIEWS.front[1], distance: 3.4 };
    this.root = document.createElement('div');
    this.root.className = 'mq-root';
    container.appendChild(this.root);
    this._buildToolbar();
    this._buildScene();
    this._buildBody();
    this.rest = this.getState().pose;
    if (opts.state) this.setState(opts.state, false);
    this._bind();
    this._placeCamera();
    this.render();
  }

  // ---------- 공개 API ----------
  snapshot() {
    return { keypoints: this.getKeypoints(), aspect: 1, facing: this.getFacing(), state: this.getState() };
  }

  getState() {
    const pose = {};
    for (const name of PIVOTS) pose[name] = this.nodes[name].quaternion.toArray().map((v) => +v.toFixed(5));
    return { pose, camera: { ...this.cam } };
  }

  setState(state, emit = true) {
    for (const [name, q] of Object.entries(state.pose || {})) if (this.nodes[name]) this.nodes[name].quaternion.fromArray(q);
    if (state.camera) this.cam = { ...this.cam, ...state.camera };
    this._placeCamera();
    this.render();
    if (emit) this._changed();
  }

  resetPose() { this.setState({ pose: this.rest }); }

  setView(name) {
    [this.cam.azimuth, this.cam.elevation] = VIEWS[name];
    this._placeCamera();
    this.render();
    this._changed();
  }

  /** 13관절을 현재 카메라에서 2D로 투영. 크기·위치와 무관하게 비교하므로, 정사각형 틀(0.1~0.9)에 맞춰 넣는다. */
  getKeypoints() {
    const T = this.T;
    this.scene.updateMatrixWorld(true);
    const { w, h } = this._size();
    const px = JOINT_KEYS.map((k) => {
      const v = this.nodes[k].getWorldPosition(new T.Vector3()).project(this.camera);
      return [(v.x + 1) / 2 * w, (1 - v.y) / 2 * h];
    });
    const xs = px.map((p) => p[0]), ys = px.map((p) => p[1]);
    const minX = Math.min(...xs), minY = Math.min(...ys);
    const span = Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY) || 1;
    const offX = (span - (Math.max(...xs) - minX)) / 2, offY = (span - (Math.max(...ys) - minY)) / 2;
    return px.map(([x, y]) => ({
      x: +(0.1 + 0.8 * (x - minX + offX) / span).toFixed(4),
      y: +(0.1 + 0.8 * (y - minY + offY) / span).toFixed(4),
      state: 'visible',
    }));
  }

  /** 몸통이 카메라를 향한 정도로 정면/측면/뒷모습 판단 */
  getFacing() {
    const T = this.T;
    this.scene.updateMatrixWorld(true);
    const chest = this.nodes.chest.getWorldPosition(new T.Vector3());
    const forward = new T.Vector3(0, 0, 1).applyQuaternion(this.nodes.waist.getWorldQuaternion(new T.Quaternion()));
    const toCam = this.camera.position.clone().sub(chest).normalize();
    const c = forward.dot(toCam);
    return c > Math.SQRT1_2 ? 'front' : c < -Math.SQRT1_2 ? 'back' : 'side';
  }

  /** 관절의 화면 좌표 (테스트·안내용) */
  screenPositionOf(key) {
    const T = this.T;
    this.scene.updateMatrixWorld(true);
    const v = this.nodes[key].getWorldPosition(new T.Vector3()).project(this.camera);
    const r = this.canvas.getBoundingClientRect();
    return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height };
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    this.resizeObserver?.disconnect();
    this.renderer.dispose();
    this.root.remove();
  }

  // ---------- 구성 ----------
  _buildToolbar() {
    const bar = document.createElement('div');
    bar.className = 'mq-toolbar';
    for (const [name, label] of Object.entries(VIEW_LABELS)) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      b.onclick = () => this.setView(name);
      bar.appendChild(b);
    }
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.textContent = '포즈 초기화';
    reset.className = 'mq-reset';
    reset.onclick = () => this.resetPose();
    bar.appendChild(reset);
    const hint = document.createElement('div');
    hint.className = 'mq-hint';
    hint.textContent = '관절 공 드래그: 뼈 회전 · Shift+가슴 공: 몸 전체 기울이기 · 빈 곳 드래그: 카메라 회전 · 휠: 확대. '
      + '파랑 = 캐릭터의 왼쪽, 주황 = 오른쪽';
    this.root.append(bar, hint);
  }

  _buildScene() {
    const T = this.T;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'mq-canvas';
    this.canvas.style.height = `${this.opts.height || 560}px`;
    this.root.appendChild(this.canvas);
    this.renderer = new T.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.scene = new T.Scene();
    this.scene.background = new T.Color(0xf3f4f6);
    this.camera = new T.PerspectiveCamera(35, 1, 0.05, 50);
    this.scene.add(new T.HemisphereLight(0xffffff, 0x8a8a8a, 2.2));
    const sun = new T.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 4, 3);
    this.scene.add(sun);
    this.grid = new T.GridHelper(4, 16, 0xbdbdbd, 0xdedede);
    this.grid.position.y = 0.06;
    this.scene.add(this.grid);
    this.raycaster = new T.Raycaster();
  }

  _buildBody() {
    const T = this.T;
    const mat = (color) => new T.MeshStandardMaterial({ color, roughness: 0.65 });
    const node = (name, parent, pos) => {
      const o = new T.Object3D();
      o.name = name;
      o.position.set(...pos);
      parent.add(o);
      this.nodes[name] = o;
      return o;
    };
    const segment = (parent, offset, radius, color = COLORS.body) => {  // parent 원점 → offset 까지의 캡슐
      const v = new T.Vector3(...offset);
      const mesh = new T.Mesh(new T.CapsuleGeometry(radius, v.length(), 4, 12), mat(color));
      mesh.position.copy(v.clone().multiplyScalar(0.5));
      mesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), v.clone().normalize());
      parent.add(mesh);
    };
    const ball = (parent, radius, color, handle) => {
      const m = new T.Mesh(new T.SphereGeometry(radius, 20, 14), mat(color));
      parent.add(m);
      if (handle) {
        m.userData = { handle, color };
        this.handles.push(m);
      }
      return m;
    };

    const root = node('root', this.scene, [0, 1.0, 0]);          // 골반 중점
    const waist = node('waist', root, [0, 0, 0]);
    const chest = node('chest', waist, [0, 0.5, 0]);             // 어깨 중점
    const neck = node('neck', chest, [0, 0, 0]);
    const head = node('head', neck, [0, 0.2, 0]);
    segment(waist, [0, 0.5, 0], 0.11);
    segment(neck, [0, 0.1, 0], 0.04);
    ball(head, 0.11, COLORS.c, 'head');
    ball(chest, 0.05, COLORS.c, 'chest');
    ball(root, 0.04, COLORS.c);

    for (const side of ['l', 'r']) {
      const s = side === 'l' ? 1 : -1;
      const color = COLORS[side];
      const arm = [0.8 * s, -0.6, 0];  // 팔은 비스듬히 아래로 (2D 표준 자세와 비슷하게)
      const sh = node(`${side}_shoulder`, chest, [0.19 * s, 0, 0]);
      const el = node(`${side}_elbow`, sh, arm.map((v) => v * 0.28));
      const wr = node(`${side}_wrist`, el, arm.map((v) => v * 0.25));
      const hip = node(`${side}_hip`, root, [0.09 * s, 0, 0]);
      const kn = node(`${side}_knee`, hip, [0.02 * s, -0.42, 0]);
      const an = node(`${side}_ankle`, kn, [0.01 * s, -0.4, 0]);
      segment(chest, [0.19 * s, 0, 0], 0.05);
      segment(root, [0.09 * s, 0, 0], 0.07);
      segment(sh, el.position.toArray(), 0.045);
      segment(el, wr.position.toArray(), 0.04);
      segment(hip, kn.position.toArray(), 0.06);
      segment(kn, an.position.toArray(), 0.05);
      ball(sh, 0.045, color);
      ball(hip, 0.045, color);
      ball(el, 0.05, color, `${side}_elbow`);
      ball(wr, 0.05, color, `${side}_wrist`);
      ball(kn, 0.055, color, `${side}_knee`);
      ball(an, 0.055, color, `${side}_ankle`);
    }
  }

  // ---------- 카메라 ----------
  _size() {
    const r = this.canvas.getBoundingClientRect();
    return { w: Math.max(r.width, 1), h: Math.max(r.height, 1) };
  }

  _placeCamera() {
    const { azimuth: az, elevation: el, distance: d } = this.cam;
    this.camera.position.set(d * Math.cos(el) * Math.sin(az), TARGET_Y + d * Math.sin(el), d * Math.cos(el) * Math.cos(az));
    this.camera.lookAt(0, TARGET_Y, 0);
    this.camera.updateMatrixWorld();
    if (this.grid) this.grid.visible = this.camera.position.y > this.grid.position.y;  // 바닥 아래서 볼 때 격자가 가리지 않게
  }

  render() {
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => this._draw());
    this._draw();  // 바로 한 번 그려서 투영 좌표가 항상 최신이 되게
  }

  _draw() {
    const { w, h } = this._size();
    const canvasW = Math.round(w * this.renderer.getPixelRatio());
    if (this.renderer.domElement.width !== canvasW) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
    this.renderer.render(this.scene, this.camera);
  }

  // ---------- 조작 ----------
  _bind() {
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => this._onDown(e));
    c.addEventListener('pointermove', (e) => this._onMove(e));
    c.addEventListener('pointerup', (e) => this._onUp(e));
    c.addEventListener('pointercancel', (e) => this._onUp(e));
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.cam.distance = Math.min(Math.max(this.cam.distance * Math.exp(e.deltaY * 0.001), 1.5), 8);
      this._placeCamera();
      this.render();
      clearTimeout(this._wheelTimer);
      this._wheelTimer = setTimeout(() => this._changed(), 300);
    }, { passive: false });
    this.resizeObserver = new ResizeObserver(() => this.render());
    this.resizeObserver.observe(this.canvas);
  }

  _ray(e) {
    const r = this.canvas.getBoundingClientRect();
    const ndc = new this.T.Vector2((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    return this.raycaster.ray;
  }

  _pick(e) {
    this._ray(e);
    const hit = this.raycaster.intersectObjects(this.handles, false)[0];
    return hit ? hit.object : null;
  }

  _onDown(e) {
    const T = this.T;
    this.canvas.setPointerCapture(e.pointerId);
    const handle = this._pick(e);
    if (!handle) {
      this.drag = { type: 'orbit', x: e.clientX, y: e.clientY, cam: { ...this.cam } };
      return;
    }
    let [pivotName, endName] = HANDLES[handle.userData.handle];
    if (handle.userData.handle === 'chest' && e.shiftKey) pivotName = 'root';  // 몸 전체 기울이기
    const pivot = this.nodes[pivotName];
    this.scene.updateMatrixWorld(true);
    const P = pivot.getWorldPosition(new T.Vector3());
    const E = this.nodes[endName].getWorldPosition(new T.Vector3());
    this.drag = {
      type: 'bone', pivot, P, radius: E.distanceTo(P), startDir: E.clone().sub(P).normalize(),
      startWorld: pivot.getWorldQuaternion(new T.Quaternion()),
      parentInv: pivot.parent.getWorldQuaternion(new T.Quaternion()).invert(),
      // 끝점이 카메라 쪽(앞)에 있으면 구의 앞면, 뒤에 있으면 뒷면을 따라 움직인다 (드래그 시작 시 튀지 않게)
      front: E.clone().sub(P).dot(this.camera.position.clone().sub(P)) >= 0,
      moved: false,
    };
  }

  _onMove(e) {
    const T = this.T;
    if (!this.drag) {
      const h = this._pick(e);
      this.canvas.style.cursor = h ? 'grab' : 'default';
      for (const m of this.handles) m.material.color.setHex(m === h ? COLORS.hover : m.userData.color);
      this.render();
      return;
    }
    if (this.drag.type === 'orbit') {
      const d = this.drag;
      this.cam.azimuth = d.cam.azimuth - (e.clientX - d.x) * 0.01;
      this.cam.elevation = Math.min(Math.max(d.cam.elevation + (e.clientY - d.y) * 0.01, -1.45), 1.45);
      this._placeCamera();
      this.render();
      return;
    }
    const d = this.drag;
    const ray = this._ray(e);
    // 피벗을 중심으로 한 구와 마우스 광선의 교점 → 뼈의 새 방향
    const oc = ray.origin.clone().sub(d.P);
    const b = oc.dot(ray.direction), c = oc.lengthSq() - d.radius ** 2, disc = b * b - c;
    let hit;
    if (disc >= 0) {
      const t = d.front ? -b - Math.sqrt(disc) : -b + Math.sqrt(disc);
      hit = ray.origin.clone().add(ray.direction.clone().multiplyScalar(t));
    } else {  // 구 밖을 가리키면 광선에서 가장 가까운 방향으로
      hit = ray.closestPointToPoint(d.P, new T.Vector3());
    }
    const dir = hit.sub(d.P).normalize();
    const delta = new T.Quaternion().setFromUnitVectors(d.startDir, dir);
    d.pivot.quaternion.copy(d.parentInv.clone().multiply(delta.multiply(d.startWorld.clone())));
    d.moved = true;
    this.render();
  }

  _onUp(e) {
    if (!this.drag) return;
    const d = this.drag;
    this.drag = null;
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    if (d.type === 'orbit' || d.moved) this._changed();
  }

  _changed() { this.opts.onChange?.(this.snapshot()); }
}

export const MANNEQUIN_CSS = `
.mq-root { font-family: inherit; }
.mq-toolbar { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.mq-toolbar button { border: 1px solid #d1d5db; background: #fff; color: #111827; border-radius: 6px;
  padding: 3px 10px; font-size: 13px; cursor: pointer; }
.mq-toolbar .mq-reset { margin-left: 12px; }
.mq-hint { font-size: 12px; color: #4b5563; margin-bottom: 6px; }
.mq-canvas { width: 100%; display: block; border: 1px solid #e5e7eb; border-radius: 8px; touch-action: none; }
`;
