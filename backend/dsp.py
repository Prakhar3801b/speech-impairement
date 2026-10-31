"""Spectrogram features and template matching (speaker-dependent).

Pipeline:  WAV -> 16 kHz mono -> pre-emphasis -> 25 ms Hamming frames (10 ms hop)
           -> |FFT|^2 -> 40-band mel filterbank -> log  (= log-mel spectrogram)
           -> DCT -> 13 MFCC + deltas -> per-utterance mean/variance normalisation
Matching:  dynamic time warping (DTW) against the patient's own enrolled recordings.
           DTW absorbs the slow / irregular timing typical of dysarthric speech.

Pure numpy, so it runs anywhere and costs no API credits.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

SR = 16000
N_FFT = 512
WIN = 400   # 25 ms
HOP = 160   # 10 ms
N_MELS = 40
N_MFCC = 13


# ---------------------------------------------------------------- audio I/O
def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV (8/16/32-bit int) into mono float32 in [-1, 1]."""
    with wave.open(io.BytesIO(data), "rb") as w:
        n_ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr


def write_wav(x: np.ndarray, sr: int = SR) -> bytes:
    x16 = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x16.tobytes())
    return buf.getvalue()


def resample(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * target / sr))
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)


def duration_s(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------------------------------------------------------------- features
def _mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                    fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    mel_to_hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)  # noqa: E731
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        fb[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        fb[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


_FB = _mel_filterbank()
_WINDOW = np.hamming(WIN).astype(np.float32)


def _dct_matrix(n_in: int = N_MELS, n_out: int = N_MFCC) -> np.ndarray:
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    m = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)
    m[0] /= np.sqrt(2.0)
    return m.astype(np.float32)


_DCT = _dct_matrix()


def frames(x: np.ndarray) -> np.ndarray:
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * _WINDOW


def trim_silence(x: np.ndarray, rel_db: float = -35.0, pad_frames: int = 5) -> np.ndarray:
    """Drop leading/trailing frames whose energy is far below the loudest frame."""
    if len(x) < WIN:
        return x
    fr = frames(x)
    energy_db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)
    voiced = np.where(energy_db > energy_db.max() + rel_db)[0]
    if len(voiced) == 0:
        return x
    start = max(0, voiced[0] - pad_frames) * HOP
    end = min(len(x), (voiced[-1] + pad_frames) * HOP + WIN)
    return x[start:end]


def logmel(x: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram, shape [frames, N_MELS]."""
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]) if len(x) > 1 else x  # pre-emphasis
    spec = np.abs(np.fft.rfft(frames(x), n=N_FFT)) ** 2
    return np.log(spec @ _FB.T + 1e-8).astype(np.float32)


def _deltas(f: np.ndarray, width: int = 2) -> np.ndarray:
    padded = np.pad(f, ((width, width), (0, 0)), mode="edge")
    num = sum(k * (padded[width + k: len(f) + width + k] - padded[width - k: len(f) + width - k])
              for k in range(1, width + 1))
    return num / (2 * sum(k * k for k in range(1, width + 1)))


def mfcc_from_logmel(lm: np.ndarray) -> np.ndarray:
    c = lm @ _DCT.T
    feats = np.hstack([c, _deltas(c)])
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)  # CMVN
    return feats.astype(np.float32)


@dataclass
class Features:
    mfcc: np.ndarray     # [frames, 2*N_MFCC]
    logmel: np.ndarray   # [frames, N_MELS]


def extract(wav_bytes: bytes) -> Features:
    x, sr = read_wav(wav_bytes)
    x = trim_silence(resample(x, sr))
    lm = logmel(x)
    return Features(mfcc=mfcc_from_logmel(lm), logmel=lm)


def spectrogram_preview(lm: np.ndarray, max_frames: int = 120) -> list[list[float]]:
    """Down-sample a log-mel matrix to a small 0..1 grid for drawing in the browser."""
    if len(lm) == 0:
        return []
    step = max(1, int(np.ceil(len(lm) / max_frames)))
    small = lm[::step]
    lo, hi = np.percentile(small, 5), np.percentile(small, 99)
    norm = np.clip((small - lo) / (hi - lo + 1e-6), 0, 1)
    return np.round(norm, 3).tolist()


# ---------------------------------------------------------------- DTW matching
def _pool(x: np.ndarray, k: int = 2) -> np.ndarray:
    """Average-pool frames (10 ms -> 20 ms) to halve DTW cost."""
    n = len(x) // k
    return x[: n * k].reshape(n, k, -1).mean(axis=1) if n >= 4 else x


def dtw_distance(query: np.ndarray, template: np.ndarray) -> float:
    """DTW cost normalised by query length.

    Step pattern: each query frame advances the template by 0, 1 or 2 frames
    (local slope 0..2). With no horizontal steps every row depends only on the
    previous row, so the recursion is vectorised with numpy. Slope 0 lets a slow
    (dysarthric) query stretch over a template; slope 2 allows up to 2x faster.
    """
    a, b = _pool(query), _pool(template)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    d = np.sqrt(np.maximum(
        (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T, 0.0))
    inf = np.inf
    prev = np.full(m, inf)
    prev[0] = d[0, 0]
    if m > 1:
        prev[1] = d[0, 1] * 1.5  # allow starting one frame in, with a small penalty
    for i in range(1, n):
        stay = prev
        step1 = np.concatenate(([inf], prev[:-1]))
        step2 = np.concatenate(([inf, inf], prev[:-2])) if m > 2 else np.full(m, inf)
        prev = d[i] + np.minimum(np.minimum(stay, step1), step2)
    end = prev[-1] if m == 1 else min(prev[-1], prev[-2])
    # inf when the query is too short to cover the template (>2x faster): no match
    return float(end / n)


def match(query: np.ndarray, templates: list[tuple[int, np.ndarray]], top_k: int = 5,
          temperature: float = 0.35) -> list[dict]:
    """Return the closest phrases. `templates` is a list of (phrase_id, mfcc).

    The best (minimum) distance per phrase is kept, then converted to a
    confidence with a softmax over negative distances.
    """
    best: dict[int, float] = {}
    for pid, tmpl in templates:
        dist = dtw_distance(query, tmpl)
        if dist < best.get(pid, float("inf")):
            best[pid] = dist
    best = {k: v for k, v in best.items() if np.isfinite(v)}
    if not best:
        return []
    ids = list(best)
    dists = np.array([best[i] for i in ids])
    logits = -(dists - dists.min()) / (temperature * max(dists.min(), 1e-6))
    probs = np.exp(logits) / np.exp(logits).sum()
    order = np.argsort(dists)[:top_k]
    return [{"phrase_id": ids[i], "distance": round(float(dists[i]), 4),
             "confidence": round(float(probs[i]), 4)} for i in order]
