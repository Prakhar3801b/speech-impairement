"""Self-supervised speech embeddings (WavLM) for template matching, run locally with ONNX Runtime.

A model trained on tens of thousands of hours of speech turns every 20 ms of audio into a vector
that describes the speech sound being made. Unlike MFCCs, that vector stays fairly stable across
microphones, rooms and background noise, so DTW against the patient's own recordings has less
to absorb. No API credits; the model runs on the CPU.

The model file is built (or downloaded) once with:  python -m scripts.build_speech_model
Without it, or without onnxruntime installed, `Embedder.available` is False and Clio uses the
spectral features in dsp.py instead.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._session = None
        self._lock = threading.Lock()
        self._failed = False
        meta_path = self.path.with_suffix(".json") if self.path else None
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path and meta_path.exists() else {}

    @property
    def name(self) -> str:
        """Identifies the model + layer, so stored embeddings from another model are recomputed."""
        return self.path.stem if self.path else ""

    @property
    def available(self) -> bool:
        if self._failed or not self.path or not self.path.exists():
            return False
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import onnxruntime as ort
                    opts = ort.SessionOptions()
                    opts.log_severity_level = 3
                    self._session = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
                    self._input = self._session.get_inputs()[0].name
        return self._session

    def __call__(self, x: np.ndarray) -> np.ndarray | None:
        """16 kHz mono float signal -> [frames, dim] embeddings, one per 20 ms."""
        if not self.available or len(x) < 1600:   # under 0.1 s: nothing useful to embed
            return None
        try:
            sess = self._load()
            x = x.astype(np.float32)
            if self.meta.get("normalize_input"):
                x = (x - x.mean()) / (x.std() + 1e-7)
            h = sess.run(None, {self._input: x[None, :]})[0][0]
        except Exception as e:  # a broken model file must not take the patient's turn down
            log.warning("speech embedding failed, falling back to spectral features: %s", e)
            self._failed = True
            return None
        return h.astype(np.float32)


def for_dtw(emb: np.ndarray, center: bool = True) -> np.ndarray:
    """Frames for DTW: optionally mean-removed, then L2-normalised (Euclidean DTW = cosine).

    Removing the clip's mean vector works like cepstral mean normalisation: it cancels what is
    constant over the clip (voice quality, microphone) and keeps what changes (the sounds)."""
    h = emb - emb.mean(axis=0) if center else emb
    return (h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def clip_vector(emb: np.ndarray) -> np.ndarray:
    """One vector per clip (mean + std pooling), L2-normalised: the input to prototype matching."""
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    v = np.concatenate([e.mean(axis=0), e.std(axis=0)])
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
