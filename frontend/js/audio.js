// Microphone capture -> 16 kHz mono WAV, plus live FFT spectrogram drawing.

export class Recorder {
  constructor() {
    this.ctx = null; this.analyser = null; this.chunks = []; this.recording = false;
  }

  async init(stream) {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = this.ctx.createMediaStreamSource(stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.2;
    // ScriptProcessor is deprecated but universally supported and fine for a prototype.
    const proc = this.ctx.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (e) => {
      if (this.recording) this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    src.connect(this.analyser);
    src.connect(proc);
    const mute = this.ctx.createGain(); mute.gain.value = 0;
    proc.connect(mute); mute.connect(this.ctx.destination);
  }

  start() {
    if (this.ctx?.state === 'suspended') this.ctx.resume();
    this.chunks = []; this.recording = true; this.startedAt = performance.now();
  }

  /** Stop and return {blob, seconds} as a 16 kHz 16-bit mono WAV. */
  stop() {
    this.recording = false;
    const n = this.chunks.reduce((a, c) => a + c.length, 0);
    const pcm = new Float32Array(n);
    let o = 0;
    for (const c of this.chunks) { pcm.set(c, o); o += c.length; }
    const down = downsample(pcm, this.ctx.sampleRate, 16000);
    return { blob: encodeWav(down, 16000), seconds: down.length / 16000 };
  }
}

function downsample(x, from, to) {
  if (from === to) return x;
  const ratio = from / to, n = Math.floor(x.length / ratio), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {           // box-filter average to avoid aliasing
    const a = Math.floor(i * ratio), b = Math.min(x.length, Math.floor((i + 1) * ratio));
    let s = 0; for (let j = a; j < b; j++) s += x[j];
    out[i] = s / Math.max(1, b - a);
  }
  return out;
}

function encodeWav(x, sr) {
  const buf = new ArrayBuffer(44 + x.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  w(0, 'RIFF'); v.setUint32(4, 36 + x.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) {
    const s = Math.max(-1, Math.min(1, x[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

// Colour map (dark -> green -> yellow) for spectrogram intensities in [0,1].
export function heat(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0, 2 * t - 0.6)));
  const g = Math.round(255 * Math.min(1, t * 1.4));
  const b = Math.round(255 * Math.max(0, 0.5 - t) * 0.9);
  return `rgb(${r},${g},${b})`;
}

/** Scrolling waterfall spectrogram of the analyser (log-frequency axis). */
export function startLiveSpectrogram(canvas, analyser) {
  const ctx = canvas.getContext('2d');
  const bins = new Uint8Array(analyser.frequencyBinCount);
  const resize = () => { canvas.width = canvas.clientWidth; };
  resize(); window.addEventListener('resize', resize);
  const draw = () => {
    if (!canvas.width && canvas.clientWidth) canvas.width = canvas.clientWidth;
    const { width: W, height: H } = canvas;
    if (!W || !H) { requestAnimationFrame(draw); return; }   // tab hidden
    analyser.getByteFrequencyData(bins);
    ctx.drawImage(canvas, -2, 0);
    const maxBin = Math.floor(bins.length * 0.5);       // up to ~half Nyquist (speech band)
    for (let y = 0; y < H; y++) {
      const frac = 1 - y / H;
      const bin = Math.min(maxBin - 1, Math.floor(Math.pow(maxBin, frac)));
      ctx.fillStyle = heat(bins[bin] / 255);
      ctx.fillRect(W - 2, y, 2, 1);
    }
    requestAnimationFrame(draw);
  };
  draw();
}

/** Draw a [frames][mels] matrix (values 0..1) as a heatmap; low frequencies at the bottom. */
export function drawMatrix(canvas, matrix) {
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth || 300;
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
  if (!matrix?.length) return;
  const T = matrix.length, M = matrix[0].length, cw = W / T, ch = H / M;
  for (let t = 0; t < T; t++) for (let m = 0; m < M; m++) {
    ctx.fillStyle = heat(matrix[t][m]);
    ctx.fillRect(t * cw, H - (m + 1) * ch, Math.ceil(cw), Math.ceil(ch));
  }
}
