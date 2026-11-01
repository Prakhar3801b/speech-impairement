"""Optional personal speech recogniser: a Whisper model fine-tuned on this patient's own speech.

Train it with train/finetune_whisper.py on the clips the patient agreed to keep, then set
LOCAL_ASR_MODEL to the output folder. When set, it runs before Scribe: free, offline, and,
once trained on enough of the patient's speech, better at it than a general recogniser.
Needs torch + transformers (requirements-train.txt); without them this module stays switched off.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
WHISPER_LANG = {"eng": "en", "en": "en", "hin": "hi", "hi": "hi"}


class LocalASR:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self._model = self._proc = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def available(self) -> bool:
        if self._failed or not self.path or not (self.path / "config.json").exists():
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        with self._lock:
            if self._model is None:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                self._proc = WhisperProcessor.from_pretrained(self.path)
                self._model = WhisperForConditionalGeneration.from_pretrained(self.path).eval()
        return self._model, self._proc

    def transcribe(self, x: np.ndarray, language: str = "auto") -> str:
        """16 kHz mono float signal -> text ('' when unavailable or on error)."""
        if not self.available or len(x) < 1600:
            return ""
        try:
            import torch
            model, proc = self._load()
            inputs = proc(x, sampling_rate=16000, return_tensors="pt")
            kw = {"task": "transcribe"}
            if WHISPER_LANG.get(language):
                kw["language"] = WHISPER_LANG[language]
            with torch.no_grad():
                ids = model.generate(inputs.input_features, max_new_tokens=64, **kw)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            log.warning("local ASR failed, using Scribe instead: %s", e)
            self._failed = True
            return ""
