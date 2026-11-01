// In-browser vision with MediaPipe Tasks: hands (ISL signs, thumbs) and face
// (expression from blendshapes, head nod/shake from the transform matrix,
// lip-activity syllable count from jawOpen). Video never leaves the device.
import {
  FilesetResolver, HandLandmarker, FaceLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export class Vision {
  constructor(video, overlay) {
    this.video = video; this.overlay = overlay; this.octx = overlay.getContext('2d');
    this.listeners = { frame: [] };
    this.headBuf = [];            // {t, pitch, yaw}
    this.exprBuf = [];            // recent expression labels
    this.lastHead = { label: null, t: 0 };
    this.utter = null;            // per-utterance accumulator
    this.jawHigh = false;
    this.state = { hands: [], features: null, thumbs: null, expression: 'neutral', head: null, jaw: 0 };
    this.frameNo = 0;
  }

  on(evt, fn) { this.listeners[evt].push(fn); }

  async init() {
    const fileset = await FilesetResolver.forVisionTasks(WASM);
    const make = async (Cls, modelAssetPath, extra) => {
      try {
        return await Cls.createFromOptions(fileset, { baseOptions: { modelAssetPath, delegate: 'GPU' }, runningMode: 'VIDEO', ...extra });
      } catch {
        return await Cls.createFromOptions(fileset, { baseOptions: { modelAssetPath, delegate: 'CPU' }, runningMode: 'VIDEO', ...extra });
      }
    };
    [this.hands, this.face] = await Promise.all([
      make(HandLandmarker, HAND_MODEL, { numHands: 2 }),
      make(FaceLandmarker, FACE_MODEL, { numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true }),
    ]);
    this.loop();
  }

  loop() {
    const tick = () => {
      if (this.video.readyState >= 2) this.process(performance.now());
      requestAnimationFrame(tick);
    };
    tick();
  }

  process(now) {
    const v = this.video;
    if (this.overlay.width !== v.videoWidth) { this.overlay.width = v.videoWidth; this.overlay.height = v.videoHeight; }
    this.octx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    const hr = this.hands.detectForVideo(v, now);
    const handed = hr.handedness || hr.handednesses || [];
    this.state.hands = hr.landmarks.map((lm, i) => ({ lm, side: handed[i]?.[0]?.categoryName || 'Right' }));
    this.state.features = handFeatures(this.state.hands);
    this.state.thumbs = this.state.hands.length === 1 ? thumbsDirection(this.state.hands[0].lm) : null;
    this.state.openPalm = this.state.hands.length === 1 && isOpenPalm(this.state.hands[0].lm);
    for (const h of this.state.hands) this.drawPoints(h.lm, '#34c38f', 3);

    if (this.frameNo++ % 2 === 0) {            // face at half rate is plenty
      const fr = this.face.detectForVideo(v, now);
      if (fr.faceBlendshapes?.length) {
        const bs = Object.fromEntries(fr.faceBlendshapes[0].categories.map((c) => [c.categoryName, c.score]));
        this.updateExpression(bs);
        this.updateJaw(bs.jawOpen || 0);
        this.updateBrows(bs.browInnerUp || 0, now);
      }
      if (fr.facialTransformationMatrixes?.length) this.updateHead(fr.facialTransformationMatrixes[0].data, now);
      if (fr.faceLandmarks?.length) {
        const lips = [13, 14, 78, 308, 0, 17, 61, 291].map((i) => fr.faceLandmarks[0][i]);
        this.drawPoints(lips, '#f5b301', 2);
      }
    }
    if (this.lastHead.label && now - this.lastHead.t > 1500) this.lastHead.label = null;
    this.state.head = this.lastHead.label;
    for (const fn of this.listeners.frame) fn(this.state);
  }

  drawPoints(points, color, r) {
    const { width: W, height: H } = this.overlay;
    this.octx.fillStyle = color;
    for (const p of points) { this.octx.beginPath(); this.octx.arc(p.x * W, p.y * H, r, 0, 2 * Math.PI); this.octx.fill(); }
  }

  updateExpression(bs) {
    const avg = (...k) => k.reduce((a, n) => a + (bs[n] || 0), 0) / k.length;
    const smile = avg('mouthSmileLeft', 'mouthSmileRight');
    const browDown = avg('browDownLeft', 'browDownRight');
    const squint = avg('eyeSquintLeft', 'eyeSquintRight');
    const sneer = avg('noseSneerLeft', 'noseSneerRight');
    const frown = avg('mouthFrownLeft', 'mouthFrownRight');
    let label = 'neutral';
    if (smile > 0.45) label = 'smile';
    else if (browDown > 0.35 && (squint > 0.35 || sneer > 0.25)) label = 'pain';
    else if (frown > 0.3 || browDown > 0.5) label = 'frown';
    else if ((bs.browInnerUp || 0) > 0.5 && (bs.jawOpen || 0) > 0.3) label = 'surprise';
    this.exprBuf.push(label); if (this.exprBuf.length > 12) this.exprBuf.shift();
    this.state.expression = mode(this.exprBuf);
    if (this.utter) this.utter.expr.push(label);
  }

  updateJaw(jaw) {
    this.state.jaw = jaw;
    if (!this.utter) return;
    if (!this.jawHigh && jaw > 0.18) { this.jawHigh = true; this.utter.syllables++; }
    else if (this.jawHigh && jaw < 0.08) this.jawHigh = false;
  }

  /** Two quick eyebrow raises (within 1.5 s) set state.browDouble to the time they happened. */
  updateBrows(v, now) {
    if (!this.browHigh && v > 0.55) {
      this.browHigh = true;
      this.browRaises = (this.browRaises || []).filter((t) => now - t < 1500).concat(now);
      if (this.browRaises.length >= 2) { this.state.browDouble = now; this.browRaises = []; }
    } else if (this.browHigh && v < 0.3) this.browHigh = false;
  }

  updateHead(m, now) {
    // column-major 4x4; pitch (nod) about x, yaw (shake) about y, in degrees
    const pitch = Math.atan2(m[6], m[10]) * 180 / Math.PI;
    const yaw = Math.asin(Math.max(-1, Math.min(1, -m[2]))) * 180 / Math.PI;
    this.headBuf.push({ t: now, pitch, yaw });
    while (this.headBuf.length && now - this.headBuf[0].t > 1400) this.headBuf.shift();
    if (this.headBuf.length < 8) return;
    const osc = (key) => {
      const v = this.headBuf.map((h) => h[key]);
      const range = Math.max(...v) - Math.min(...v);
      let turns = 0, dir = 0;
      for (let i = 2; i < v.length; i++) {
        const d = v[i] - v[i - 2];
        if (Math.abs(d) < 1.2) continue;
        const s = Math.sign(d);
        if (dir && s !== dir) turns++;
        dir = s;
      }
      return { range, turns };
    };
    const p = osc('pitch'), y = osc('yaw');
    let label = null;
    if (p.range > 8 && p.turns >= 2 && y.range < p.range * 0.7) label = 'nod';
    else if (y.range > 10 && y.turns >= 2 && p.range < y.range * 0.7) label = 'shake';
    if (label) {
      this.lastHead = { label, t: now };
      if (this.utter) this.utter.head = label;
      this.headBuf = [];
    }
  }

  beginUtterance() { this.utter = { syllables: 0, expr: [], head: null, start: performance.now() }; this.jawHigh = false; }

  /** Cues gathered during the utterance (plus a head gesture made just before it ended). */
  endUtterance() {
    const u = this.utter || { syllables: 0, expr: [], head: null };
    this.utter = null;
    return {
      syllables: u.syllables || null,
      expression: u.expr.length ? mode(u.expr) : this.state.expression,
      head: u.head || this.lastHead.label,
      thumbs: this.state.thumbs,
    };
  }

  /** Cues right now (for sign/gesture-only turns). */
  snapshot() {
    return { syllables: null, expression: this.state.expression, head: this.lastHead.label, thumbs: this.state.thumbs };
  }
}

function mode(arr) {
  const c = {}; let best = arr[0], n = 0;
  for (const a of arr) { c[a] = (c[a] || 0) + 1; if (c[a] > n) { n = c[a]; best = a; } }
  return best;
}

// ---------------------------------------------------------------- hand features
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, (a.z - b.z) || 0);

/**
 * Feature vector for ISL signs (works for one- and two-handed letters):
 *   [presentLeft, presentRight,
 *    left hand  in its own wrist frame (63), right hand in its own frame (63),
 *    left hand  in the shared two-hand frame (63), right hand in the shared frame (63)]
 * The shared frame keeps the relative position of the two hands, which many ISL letters depend on.
 */
export function handFeatures(hands) {
  if (!hands.length) return null;
  const slots = { Left: null, Right: null };
  for (const h of hands) {
    const side = slots[h.side] ? (h.side === 'Left' ? 'Right' : 'Left') : h.side;
    slots[side] = h.lm;
  }
  const all = hands.flatMap((h) => h.lm);
  const c = all.reduce((a, p) => ({ x: a.x + p.x / all.length, y: a.y + p.y / all.length, z: a.z + p.z / all.length }), { x: 0, y: 0, z: 0 });
  const gScale = Math.max(...all.map((p) => Math.hypot(p.x - c.x, p.y - c.y))) || 1;
  const local = (lm) => {
    if (!lm) return new Array(63).fill(0);
    const w = lm[0];
    const s = Math.max(...lm.map((p) => Math.hypot(p.x - w.x, p.y - w.y))) || 1;
    return lm.flatMap((p) => [(p.x - w.x) / s, (p.y - w.y) / s, (p.z - w.z) / s]);
  };
  const global = (lm) => (lm ? lm.flatMap((p) => [(p.x - c.x) / gScale, (p.y - c.y) / gScale, (p.z - c.z) / gScale]) : new Array(63).fill(0));
  const pres = [slots.Left ? 2 : 0, slots.Right ? 2 : 0];
  return [...pres, ...local(slots.Left), ...local(slots.Right), ...global(slots.Left), ...global(slots.Right)].map((v) => Math.round(v * 1e4) / 1e4);
}

/** All five fingers stretched out and pointing up, as in a raised "stop / help" hand. */
export function isOpenPalm(lm) {
  const wrist = lm[0], size = dist(wrist, lm[9]) || 1e-3;
  const extended = [[8, 6], [12, 10], [16, 14], [20, 18]].every(([tip, pip]) => dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.2);
  const upright = [8, 12, 16, 20].every((i) => lm[i].y < wrist.y - size * 0.8);
  const thumbOut = dist(lm[4], lm[5]) > size * 0.45;
  return extended && upright && thumbOut;
}

/** 'up' | 'down' | null for a single hand making a thumbs gesture. */
export function thumbsDirection(lm) {
  const wrist = lm[0], size = dist(wrist, lm[9]) || 1e-3;
  const curled = [[8, 6], [12, 10], [16, 14], [20, 18]].every(([tip, pip]) => dist(lm[tip], wrist) < dist(lm[pip], wrist) * 1.15);
  if (!curled) return null;
  const tip = lm[4], mcp = lm[2];
  const others = [8, 12, 16, 20].map((i) => lm[i].y);
  if (tip.y < mcp.y - size * 0.45 && tip.y < Math.min(...others) - size * 0.2) return 'up';
  if (tip.y > mcp.y + size * 0.45 && tip.y > Math.max(...others) + size * 0.2) return 'down';
  return null;
}
