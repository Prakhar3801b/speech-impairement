// k-nearest-neighbour classifier for ISL signs + a stabiliser that turns a
// noisy per-frame prediction into discrete "sign entered" events.

export class SignClassifier {
  constructor() { this.samples = []; this.threshold = Infinity; }

  load(rows) {
    this.samples = rows.map((r) => ({ label: r.label, kind: r.kind, phrase_id: r.phrase_id, f: Float32Array.from(r.features) }));
    this.threshold = this.computeThreshold();
  }

  get size() { return this.samples.length; }

  // Reject frames far from every training sample (no sign / unknown sign):
  // 3x the median nearest-neighbour distance between samples of the same label.
  computeThreshold() {
    const d = [];
    const sample = this.samples.length > 400 ? this.samples.filter((_, i) => i % 3 === 0) : this.samples;
    for (const a of sample) {
      let best = Infinity;
      for (const b of this.samples) if (a !== b && a.label === b.label) best = Math.min(best, l2(a.f, b.f));
      if (best < Infinity) d.push(best);
    }
    if (!d.length) return 3.0;
    d.sort((x, y) => x - y);
    return Math.max(0.6, d[Math.floor(d.length / 2)] * 3);
  }

  predict(features, k = 5) {
    if (!features || !this.samples.length) return null;
    const f = Float32Array.from(features);
    const near = this.samples.map((s) => ({ s, d: l2(f, s.f) })).sort((a, b) => a.d - b.d).slice(0, k);
    const votes = {};
    for (const { s, d } of near) votes[s.label] = (votes[s.label] || 0) + 1 / (d + 1e-3);
    const [label, score] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
    const total = Object.values(votes).reduce((a, b) => a + b, 0);
    const best = near.find((n) => n.s.label === label);
    return { label, kind: best.s.kind, phrase_id: best.s.phrase_id, distance: best.d,
             confidence: score / total, known: best.d < this.threshold };
  }
}

function l2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

/**
 * Emits a sign once it has been held for `holdFrames` consecutive frames.
 * The same sign repeats only after the hand changes/leaves, or after `repeatMs` of holding
 * (so double letters such as "OO" are possible).
 */
export class SignStabilizer {
  constructor(onSign, { holdFrames = 12, minConf = 0.7, repeatMs = 1600 } = {}) {
    Object.assign(this, { onSign, holdFrames, minConf, repeatMs });
    this.current = null; this.count = 0; this.emitted = null; this.emittedAt = 0;
  }

  push(pred) {
    const ok = pred && pred.known && pred.confidence >= this.minConf;
    const label = ok ? pred.label : null;
    if (label !== this.current) {
      this.current = label; this.count = 0;
      if (!label) this.emitted = null;
      return;
    }
    if (!label) return;
    this.count++;
    const now = performance.now();
    if (this.count >= this.holdFrames && (this.emitted !== label || now - this.emittedAt > this.repeatMs)) {
      this.emitted = label; this.emittedAt = now;
      this.onSign(pred);
    }
  }
}

export const LETTERS = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ', ...'0123456789'];
export const CONTROLS = ['SPACE', 'DEL', 'SEND'];
