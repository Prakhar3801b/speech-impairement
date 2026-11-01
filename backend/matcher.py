"""Which phrase did the patient just say? Matching against the patient's own recordings.

Three layers, best available first:
* speech-model embeddings (WavLM, backend/embed.py) + DTW: most robust to noise and microphone;
* PNCC-style cepstra + prosody (dsp.py) + DTW: noise-robust, no model needed;
* plain MFCC + DTW: only for old templates saved before their audio was kept.

When a phrase has enough recordings (PROTOTYPE_MIN_TEMPLATES, reached through "learn as you talk"),
a prototype score is blended in: the average of that phrase's clip vectors compared with the new
clip. Prototypes are cheap and steady; DTW keeps the order of sounds and works from one recording.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import dsp
from .config import settings
from .db import Database
from .embed import Embedder, clip_vector, for_dtw


@dataclass
class Template:
    id: int
    phrase_id: int
    source: str
    mfcc: np.ndarray
    robust: np.ndarray | None = None   # PNCC-style cepstra + weighted prosody
    emb: np.ndarray | None = None      # embeddings prepared for DTW
    vec: np.ndarray | None = None      # clip vector for prototypes


def robust_track(feats: dsp.Features, prosody_weight: float | None = None) -> np.ndarray:
    w = settings.prosody_weight if prosody_weight is None else prosody_weight
    return np.hstack([feats.robust, w * feats.prosody]).astype(np.float32) if w else feats.robust


class Matcher:
    def __init__(self, db: Database, embedder: Embedder | None = None):
        self.db = db
        self.embedder = embedder or Embedder(settings.ssl_model_path)
        self.items: list[Template] = []
        self.reload()

    # ------------------------------------------------------------ templates
    def save(self, phrase_id: int, feats: dsp.Features, source: str = "enrolled") -> None:
        """Store a template and add it to the in-memory set (no reload: that re-reads every template)."""
        emb = self.embed(feats)
        res = self.db.execute(
            "INSERT INTO audio_templates(phrase_id, mfcc, logmel, n_frames, n_mfcc, n_mels, source, audio, "
            "embedding, emb_frames, emb_dim, emb_model, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (phrase_id, feats.mfcc.tobytes(), feats.logmel.tobytes(), len(feats.mfcc), feats.mfcc.shape[1],
             feats.logmel.shape[1], source, dsp.pcm16(feats.audio) if feats.audio is not None else None,
             emb.tobytes() if emb is not None else None, len(emb) if emb is not None else None,
             emb.shape[1] if emb is not None else None, self.embedder.name if emb is not None else None,
             time.time()))
        t = Template(res.lastrowid or 0, phrase_id, source, feats.mfcc)
        if feats.audio is not None and feats.robust is not None:
            t.robust = robust_track(feats)
        if emb is not None:
            t.emb, t.vec = for_dtw(emb, settings.ssl_center), clip_vector(emb)
        self.items.append(t)

    def trim_learned(self, phrase_id: int, keep: int) -> None:
        """Keep only the newest `keep` learned templates of a phrase (recorded ones are never dropped)."""
        learned = sorted((t for t in self.items if t.phrase_id == phrase_id and t.source == "learned"),
                         key=lambda t: t.id)
        drop = learned[:-keep] if keep else learned
        if drop:
            ids = [t.id for t in drop]
            self.db.execute(f"DELETE FROM audio_templates WHERE id IN ({','.join('?' * len(ids))})", ids)
            gone = set(ids)
            self.items = [t for t in self.items if t.id not in gone]

    def remove(self, template_id: int | None = None, phrase_id: int | None = None) -> None:
        self.items = [t for t in self.items if t.id != template_id and t.phrase_id != phrase_id]

    def reload(self) -> None:
        rows = self.db.query("SELECT id, phrase_id, source, mfcc, n_frames, n_mfcc, audio, embedding, "
                             "emb_frames, emb_dim, emb_model FROM audio_templates ORDER BY id")
        use_emb = self.embedder.available
        items = []
        for r in rows:
            t = Template(r["id"], r["phrase_id"], r["source"],
                         np.frombuffer(r["mfcc"], dtype=np.float32).reshape(r["n_frames"], r["n_mfcc"]))
            if r["audio"]:
                feats = dsp.extract_signal(dsp.from_pcm16(r["audio"]))
                t.robust = robust_track(feats)
                emb = None
                if use_emb and r["embedding"] and r["emb_model"] == self.embedder.name:
                    emb = np.frombuffer(r["embedding"], dtype=np.float32).reshape(r["emb_frames"], r["emb_dim"])
                elif use_emb:   # new or changed model: compute once and store
                    emb = self.embedder(feats.audio)
                    if emb is not None:
                        self.db.execute("UPDATE audio_templates SET embedding=?, emb_frames=?, emb_dim=?, "
                                        "emb_model=? WHERE id=?",
                                        (emb.tobytes(), len(emb), emb.shape[1], self.embedder.name, r["id"]))
                if emb is not None:
                    t.emb, t.vec = for_dtw(emb, settings.ssl_center), clip_vector(emb)
            items.append(t)
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------ matching
    @property
    def kind(self) -> str:
        """Feature type used for this template set. Every template must have it, so distances compare."""
        want = settings.matcher
        has_emb = bool(self.items) and all(t.emb is not None for t in self.items)
        has_robust = bool(self.items) and all(t.robust is not None for t in self.items)
        if want in ("auto", "ssl") and has_emb:
            return "ssl"
        if want in ("auto", "ssl", "robust") and has_robust:
            return "robust"
        return "mfcc"

    def embed(self, feats: dsp.Features) -> np.ndarray | None:
        if feats.audio is None or settings.matcher not in ("auto", "ssl") or not self.embedder.available:
            return None
        return self.embedder(feats.audio)

    def match(self, feats: dsp.Features, emb: np.ndarray | None = None, top_k: int = 5) -> tuple[list[dict], dict]:
        """Return (matches, info). `emb` is the query's raw embedding (from `embed`), when available."""
        kind = self.kind
        if kind == "ssl" and emb is None:
            kind = "robust" if all(t.robust is not None for t in self.items) else "mfcc"
        if kind == "ssl":
            query, pool = for_dtw(emb, settings.ssl_center), 1          # already 20 ms frames
            tracks = [(t.phrase_id, t.emb) for t in self.items]
        elif kind == "robust":
            query, pool = robust_track(feats), 2
            tracks = [(t.phrase_id, t.robust) for t in self.items]
        else:
            query, pool = feats.mfcc, 2
            tracks = [(t.phrase_id, t.mfcc) for t in self.items]
        best = dsp.best_distances(query, tracks, pool)
        temperature = {"ssl": settings.match_temperature_ssl, "robust": settings.match_temperature_robust,
                       "mfcc": settings.match_temperature_mfcc}[kind]
        matches = dsp.to_confidence(best, top_k=len(best), temperature=temperature)
        info = {"features": kind, "prototype": False}
        if kind == "ssl" and settings.prototype_weight > 0:
            matches, used = self._blend_prototypes(matches, clip_vector(emb))
            info["prototype"] = used
        return matches[:top_k], info

    def _blend_prototypes(self, matches: list[dict], qvec: np.ndarray) -> tuple[list[dict], bool]:
        groups: dict[int, list[np.ndarray]] = {}
        for t in self.items:
            if t.vec is not None:
                groups.setdefault(t.phrase_id, []).append(t.vec)
        return blend_prototypes(matches, qvec, groups)


def blend_prototypes(matches: list[dict], qvec: np.ndarray, groups: dict[int, list[np.ndarray]],
                     min_templates: int | None = None, weight: float | None = None,
                     temperature: float | None = None) -> tuple[list[dict], bool]:
    """Mix a prototype score into the DTW confidences of phrases with enough recordings."""
    min_templates = settings.prototype_min_templates if min_templates is None else min_templates
    w = settings.prototype_weight if weight is None else weight
    temperature = settings.prototype_temperature if temperature is None else temperature
    protos = {pid: np.mean(v, axis=0) for pid, v in groups.items() if len(v) >= min_templates}
    protos = {pid: v / (np.linalg.norm(v) + 1e-8) for pid, v in protos.items()}
    in_both = [m for m in matches if m["phrase_id"] in protos]
    if len(in_both) < 2:
        return matches, False
    sims = np.array([float(qvec @ protos[m["phrase_id"]]) for m in in_both])
    logits = (sims - sims.max()) / temperature
    probs = np.exp(logits) / np.exp(logits).sum()
    mass = sum(m["confidence"] for m in in_both)   # keep the total confidence unchanged
    for m, p, s in zip(in_both, probs, sims):
        m["confidence"] = round((1 - w) * m["confidence"] + w * mass * float(p), 4)
        m["prototype_similarity"] = round(float(s), 4)
    return sorted(matches, key=lambda m: -m["confidence"]), True
