"""ElevenLabs client: Scribe speech-to-text, text-to-speech, Voice Design and
Instant Voice Cloning, built to stay inside the free tier:

* every TTS result and every transcript is cached in SQLite (never pay twice);
* a clip that sounds almost the same as one already transcribed re-uses that
  transcript (DTW over MFCCs, the same matcher as the spectrogram templates),
  so a patient repeating "paani chahiye" is paid for once;
* silence is trimmed and clips are capped before upload, so fewer seconds are billed;
* several API keys can be configured; when one runs out the next takes over
  (see budget.py), and when all are out the app falls back to free paths;
* a semaphore keeps us within the plan's concurrent-request limit (2 on Free);
* credit usage is logged as events so the Insights tab can show the burn rate;
* without an API key the client runs in MOCK mode and the browser falls back to
  its own speech synthesis.

Endpoint shapes follow the public REST API; verify model ids against your plan.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from typing import Any

import httpx
import numpy as np

from . import dsp
from .budget import Key, KeyPool
from .config import settings
from .db import Database

API = "https://api.elevenlabs.io"

FINGERPRINT_SCHEMA = """
CREATE TABLE IF NOT EXISTS stt_fingerprints (
    audio_hash  TEXT PRIMARY KEY,
    role        TEXT NOT NULL,             -- patient | partner | consent
    language    TEXT NOT NULL,
    mfcc        BLOB NOT NULL,             -- float32 [frames, coeffs]
    n_frames    INTEGER NOT NULL,
    n_mfcc      INTEGER NOT NULL,
    text        TEXT NOT NULL,
    words_json  TEXT NOT NULL,
    language_detected TEXT,
    created_at  REAL NOT NULL
);
"""
MAX_FINGERPRINTS = 300      # per role; oldest dropped from the in-memory index


class ElevenLabsError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"ElevenLabs error {status}: {detail}")
        self.status = status
        self.detail = detail


class BudgetExhausted(ElevenLabsError):
    def __init__(self):
        super().__init__(402, "all configured ElevenLabs keys are out of credits for this month")


def _quota_error(e: ElevenLabsError) -> bool:
    text = json.dumps(e.detail).lower() if not isinstance(e.detail, str) else e.detail.lower()
    return e.status == 402 or (e.status in (401, 429) and ("quota" in text or "credits" in text))


class ElevenLabs:
    def __init__(self, db: Database, api_key: str | None = None, pool: KeyPool | None = None):
        """`db` holds this account's caches; `pool` (the API keys and their budgets) can be shared."""
        keys = settings.api_keys if api_key is None else [k for k in api_key.split(",") if k.strip()]
        self.pool = pool or KeyPool(db, keys, settings.credits_per_key, settings.credit_reserve)
        self.sem = asyncio.Semaphore(max(1, settings.elevenlabs_concurrency))
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._open_caches(db)

    def for_account(self, db: Database) -> "ElevenLabs":
        """A client for one account: its own transcript/speech caches, shared keys, budget and HTTP clients."""
        el = object.__new__(ElevenLabs)
        el.pool, el.sem, el._clients = self.pool, self.sem, self._clients
        el._open_caches(db)
        return el

    def _open_caches(self, db: Database) -> None:
        self.db = db
        with db.lock:
            db.script(FINGERPRINT_SCHEMA)
        self._prints: dict[str, list[dict]] = {}
        for r in db.query("SELECT * FROM stt_fingerprints ORDER BY created_at DESC LIMIT ?", (MAX_FINGERPRINTS * 3,)):
            r["mfcc"] = np.frombuffer(r["mfcc"], dtype=np.float32).reshape(r["n_frames"], r["n_mfcc"])
            self._prints.setdefault(r["role"], []).insert(0, r)

    @property
    def enabled(self) -> bool:
        return bool(self.pool)

    def _http(self, key: Key) -> httpx.AsyncClient:
        if key.id not in self._clients:
            self._clients[key.id] = httpx.AsyncClient(
                base_url=API, headers={"xi-api-key": key.secret},
                timeout=httpx.Timeout(30.0, connect=5.0))
        return self._clients[key.id]

    async def aclose(self) -> None:
        for c in self._clients.values():
            await c.aclose()

    async def _send(self, key: Key, method: str, path: str, **kw) -> httpx.Response:
        async with self.sem:
            r = await self._http(key).request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json()
            except ValueError:
                detail = r.text[:300]
            raise ElevenLabsError(r.status_code, detail)
        return r

    async def _sync(self, key: Key) -> None:
        """Best effort: read the account's own credit counter for this key."""
        if not self.pool.needs_sync(key.id):
            return
        try:
            body = (await self._send(key, "GET", "/v1/user/subscription")).json()
            self.pool.sync(key.id, body.get("character_count"), body.get("character_limit"))
        except (ElevenLabsError, httpx.HTTPError, ValueError):
            self.pool.mark_synced(key.id)    # key may lack the permission; keep the local estimate

    async def _request(self, method: str, path: str, cost: float = 0.0,
                       only_key: str | None = None, **kw) -> tuple[httpx.Response, str]:
        """Send with the first key that can afford `cost`; on a quota error move to the next one."""
        last: ElevenLabsError | None = None
        for key in self.pool.candidates(cost, only_key):
            await self._sync(key)
            if self.pool.remaining(key.id) < cost:
                continue
            try:
                r = await self._send(key, method, path, **kw)
            except ElevenLabsError as e:
                if _quota_error(e):
                    self.pool.mark_exhausted(key.id)
                    last = e
                    continue
                raise
            if cost:
                self.pool.charge(key.id, cost)
            return r, key.id
        if last is not None or self.pool:
            raise BudgetExhausted()
        raise ElevenLabsError(503, "ElevenLabs API key not configured")

    def budget(self) -> dict:
        return {"keys": self.pool.summary(), "fraction_left": self.pool.fraction_left(),
                "saver_mode": self.saver_mode}

    @property
    def saver_mode(self) -> bool:
        return bool(self.pool) and self.pool.fraction_left() < settings.saver_mode_below

    # ------------------------------------------------------------------ STT
    def _nearest_transcript(self, mfcc: np.ndarray, role: str, lang: str) -> dict | None:
        """Re-use the transcript of an acoustically near-identical earlier clip.

        Accepted only when the best match is close in absolute DTW terms, has a
        similar length, and clearly beats the best clip with a *different* text."""
        cands = [p for p in self._prints.get(role, []) if p["language"] == lang]
        if not cands or len(mfcc) < 8:
            return None
        scored = []
        for p in cands:
            ratio = len(mfcc) / max(1, p["n_frames"])
            if 0.6 <= ratio <= 1.7:
                scored.append((dsp.dtw_distance(mfcc, p["mfcc"]), p))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0])
        best_d, best = scored[0]
        rival = next((d for d, p in scored[1:] if p["text"] != best["text"]), math.inf)
        if best_d <= settings.stt_reuse_max_dist and rival >= best_d * 1.15:
            return {**best, "distance": best_d}
        return None

    def _remember(self, audio_hash: str, role: str, lang: str, mfcc: np.ndarray, text: str,
                  words: list, detected: str | None) -> None:
        if not text or len(mfcc) < 8:
            return
        row = {"audio_hash": audio_hash, "role": role, "language": lang, "mfcc": mfcc,
               "n_frames": len(mfcc), "n_mfcc": mfcc.shape[1], "text": text,
               "words_json": json.dumps(words, ensure_ascii=False), "language_detected": detected,
               "created_at": time.time()}
        self.db.execute(
            "INSERT OR REPLACE INTO stt_fingerprints(audio_hash, role, language, mfcc, n_frames, n_mfcc, "
            "text, words_json, language_detected, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (audio_hash, role, lang, mfcc.astype(np.float32).tobytes(), row["n_frames"], row["n_mfcc"],
             text, row["words_json"], detected, row["created_at"]))
        prints = self._prints.setdefault(role, [])
        prints.append(row)
        del prints[:-MAX_FINGERPRINTS]

    async def transcribe(self, wav: bytes, language: str = "auto", session_id: str | None = None,
                         role: str = "patient", features: dsp.Features | None = None,
                         reuse_similar: bool = True) -> dict:
        """Scribe speech-to-text, cheapest source first:
        exact cache -> near-identical earlier clip -> Scribe (silence trimmed, length capped).
        `reuse_similar=False` skips the near-identical step (evaluation must measure Scribe itself)."""
        audio_hash = hashlib.sha256(wav).hexdigest()
        lang = language or "auto"
        cached = self.db.one(
            "SELECT * FROM stt_cache WHERE audio_hash=? AND model_id=? AND language=?",
            (audio_hash, settings.stt_model, lang))
        if cached:
            self.db.execute(
                "UPDATE stt_cache SET hits=hits+1 WHERE audio_hash=? AND model_id=? AND language=?",
                (audio_hash, settings.stt_model, lang))
            return {"text": cached["text"], "words": json.loads(cached["words_json"]),
                    "language": cached["language_detected"], "cached": True, "mock": False}

        x, sr = dsp.read_wav(wav)
        x = dsp.trim_silence(dsp.resample(x, sr))[: int(settings.max_stt_seconds * dsp.SR)]
        feats = features or await asyncio.to_thread(lambda: dsp.Features(
            mfcc=dsp.mfcc_from_logmel(dsp.logmel(x)), logmel=np.zeros((0, dsp.N_MELS), np.float32)))
        near = self._nearest_transcript(feats.mfcc, role, lang) if reuse_similar else None
        if near:
            self.db.log_event("credit", {"service": "stt", "seconds": 0, "saved_seconds": round(len(x) / dsp.SR, 2),
                                         "source": "fuzzy-cache"}, session_id)
            return {"text": near["text"], "words": json.loads(near["words_json"]),
                    "language": near["language_detected"], "cached": "fuzzy", "mock": False,
                    "reuse_distance": round(near["distance"], 3)}
        if not self.enabled:
            return {"text": "", "words": [], "language": None, "cached": False, "mock": True}

        secs = len(x) / dsp.SR
        upload = dsp.write_wav(x)
        data = {"model_id": settings.stt_model, "tag_audio_events": "false"}
        if lang in ("eng", "hin"):
            data["language_code"] = lang
        try:
            r, _ = await self._request(
                "POST", "/v1/speech-to-text", cost=secs / 60 * settings.stt_credits_per_minute, data=data,
                files={"file": ("utterance.wav", upload, "audio/wav")})
        except BudgetExhausted:
            return {"text": "", "words": [], "language": None, "cached": False, "mock": True, "budget": "exhausted"}
        body = r.json()
        words = []
        for w in body.get("words", []):
            if w.get("type", "word") != "word":
                continue
            lp = w.get("logprob")
            words.append({"text": w.get("text", "").strip(),
                          "start": w.get("start"), "end": w.get("end"),
                          "confidence": round(math.exp(lp), 3) if isinstance(lp, (int, float)) else None})
        text = (body.get("text") or "").strip()
        self.db.execute(
            "INSERT OR REPLACE INTO stt_cache(audio_hash, model_id, language, text, words_json, "
            "language_detected, seconds, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (audio_hash, settings.stt_model, lang, text, json.dumps(words, ensure_ascii=False),
             body.get("language_code"), secs, time.time()))
        self._remember(audio_hash, role, lang, feats.mfcc, text, words, body.get("language_code"))
        self.db.log_event("credit", {"service": "stt", "seconds": round(secs, 2),
                                     "trimmed_seconds": round(max(0.0, dsp.duration_s(wav) - secs), 2)}, session_id)
        return {"text": text, "words": words, "language": body.get("language_code"),
                "cached": False, "mock": False}

    # ------------------------------------------------------------------ TTS
    @staticmethod
    def tts_key(text: str, voice_id: str, model_id: str) -> str:
        return hashlib.sha256(f"{text.strip()}|{voice_id}|{model_id}".encode()).hexdigest()

    def cached_tts(self, text: str, voice_id: str) -> dict | None:
        key = self.tts_key(text, voice_id, settings.tts_model)
        row = self.db.one("SELECT audio, mime FROM tts_cache WHERE key=?", (key,))
        if row:
            self.db.execute("UPDATE tts_cache SET hits=hits+1 WHERE key=?", (key,))
        return row

    async def speak(self, text: str, voice_id: str, session_id: str | None = None,
                    owner_key: str | None = None) -> dict:
        """Return {'audio', 'mime', 'cached'} or {'mock': True}. A designed or cloned voice lives in
        one ElevenLabs account, so `owner_key` pins the request to that account's key."""
        row = self.cached_tts(text, voice_id)
        if row:
            return {"audio": row["audio"], "mime": row["mime"], "cached": True}
        if not self.enabled:
            return {"mock": True}
        try:
            r, _ = await self._request(
                "POST", f"/v1/text-to-speech/{voice_id}", cost=len(text) * settings.tts_credits_per_char,
                only_key=owner_key, params={"output_format": "mp3_44100_128"},
                json={"text": text, "model_id": settings.tts_model})
        except BudgetExhausted:
            return {"mock": True, "budget": "exhausted"}
        audio = r.content
        self.db.execute(
            "INSERT OR REPLACE INTO tts_cache(key, text, voice_id, model_id, audio, mime, chars, created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (self.tts_key(text, voice_id, settings.tts_model), text.strip(), voice_id,
             settings.tts_model, audio, "audio/mpeg", len(text), time.time()))
        self.db.log_event("credit", {"service": "tts", "chars": len(text),
                                     "model": settings.tts_model}, session_id)
        return {"audio": audio, "mime": "audio/mpeg", "cached": False}

    # ------------------------------------------------------------- voices
    async def design_previews(self, description: str, sample_text: str | None = None) -> list[dict]:
        """Voice Design: synthetic voice from a text description (no recordings, no consent needed)."""
        if not self.enabled:
            raise ElevenLabsError(503, "ElevenLabs API key not configured")
        payload: dict[str, Any] = {"voice_description": description}
        if sample_text and len(sample_text) >= 100:
            payload["text"] = sample_text
        else:
            payload["auto_generate_text"] = True
        r, kid = await self._request("POST", "/v1/text-to-voice/design", json=payload)
        self.db.log_event("credit", {"service": "voice_design", "key_id": kid})
        return [{"generated_voice_id": p.get("generated_voice_id"),
                 "audio_base64": p.get("audio_base_64"),
                 "media_type": p.get("media_type", "audio/mpeg"), "key_id": kid}
                for p in r.json().get("previews", [])]

    async def design_save(self, generated_voice_id: str, name: str, description: str,
                          key_id: str | None = None) -> tuple[str, str]:
        r, kid = await self._request("POST", "/v1/text-to-voice", only_key=key_id, json={
            "voice_name": name, "voice_description": description,
            "generated_voice_id": generated_voice_id})
        return r.json()["voice_id"], kid

    async def clone(self, name: str, files: list[tuple[str, bytes, str]], description: str) -> tuple[str, str]:
        """Instant Voice Cloning (needs a paid plan). Caller MUST enforce consent first."""
        if not self.enabled:
            raise ElevenLabsError(503, "ElevenLabs API key not configured")
        r, kid = await self._request(
            "POST", "/v1/voices/add",
            data={"name": name, "description": description},
            files=[("files", f) for f in files])
        return r.json()["voice_id"], kid

    async def delete_voice(self, voice_id: str, key_id: str | None = None) -> None:
        if self.enabled:
            await self._request("DELETE", f"/v1/voices/{voice_id}", only_key=key_id)
