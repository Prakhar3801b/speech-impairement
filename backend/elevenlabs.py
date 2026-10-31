"""ElevenLabs client: Scribe speech-to-text, text-to-speech, Voice Design and
Instant Voice Cloning, built for the free tier:

* every TTS result and every transcript is cached in SQLite (never pay twice);
* a semaphore keeps us within the plan's concurrent-request limit (2 on Free);
* credit usage is logged as events so the Metrics tab can show the burn rate;
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

from .config import settings
from .db import Database
from .dsp import duration_s

API = "https://api.elevenlabs.io"


class ElevenLabsError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"ElevenLabs error {status}: {detail}")
        self.status = status
        self.detail = detail


class ElevenLabs:
    def __init__(self, db: Database, api_key: str | None = None):
        self.db = db
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self.sem = asyncio.Semaphore(max(1, settings.elevenlabs_concurrency))
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=API, headers={"xi-api-key": self.api_key},
                timeout=httpx.Timeout(30.0, connect=5.0))
        return self._client

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    async def _request(self, method: str, path: str, **kw) -> httpx.Response:
        async with self.sem:
            r = await self._http().request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json()
            except ValueError:
                detail = r.text[:300]
            raise ElevenLabsError(r.status_code, detail)
        return r

    # ------------------------------------------------------------------ STT
    async def transcribe(self, wav: bytes, language: str = "auto",
                         session_id: str | None = None) -> dict:
        """Scribe speech-to-text with a transcript cache keyed by audio hash."""
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
        if not self.enabled:
            return {"text": "", "words": [], "language": None, "cached": False, "mock": True}

        data = {"model_id": settings.stt_model, "tag_audio_events": "false"}
        if lang in ("eng", "hin"):
            data["language_code"] = lang
        r = await self._request(
            "POST", "/v1/speech-to-text", data=data,
            files={"file": ("utterance.wav", wav, "audio/wav")})
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
        secs = duration_s(wav)
        self.db.execute(
            "INSERT OR REPLACE INTO stt_cache(audio_hash, model_id, language, text, words_json, "
            "language_detected, seconds, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (audio_hash, settings.stt_model, lang, text, json.dumps(words, ensure_ascii=False),
             body.get("language_code"), secs, time.time()))
        self.db.log_event("credit", {"service": "stt", "seconds": round(secs, 2)}, session_id)
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

    async def speak(self, text: str, voice_id: str, session_id: str | None = None) -> dict:
        """Return {'audio': bytes, 'mime': str, 'cached': bool} or {'mock': True}."""
        row = self.cached_tts(text, voice_id)
        if row:
            return {"audio": row["audio"], "mime": row["mime"], "cached": True}
        if not self.enabled:
            return {"mock": True}
        r = await self._request(
            "POST", f"/v1/text-to-speech/{voice_id}",
            params={"output_format": "mp3_44100_128"},
            json={"text": text, "model_id": settings.tts_model})
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
        r = await self._request("POST", "/v1/text-to-voice/design", json=payload)
        self.db.log_event("credit", {"service": "voice_design"})
        return [{"generated_voice_id": p.get("generated_voice_id"),
                 "audio_base64": p.get("audio_base_64"),
                 "media_type": p.get("media_type", "audio/mpeg")}
                for p in r.json().get("previews", [])]

    async def design_save(self, generated_voice_id: str, name: str, description: str) -> str:
        r = await self._request("POST", "/v1/text-to-voice", json={
            "voice_name": name, "voice_description": description,
            "generated_voice_id": generated_voice_id})
        return r.json()["voice_id"]

    async def clone(self, name: str, files: list[tuple[str, bytes, str]], description: str) -> str:
        """Instant Voice Cloning (needs a paid plan). Caller MUST enforce consent first."""
        if not self.enabled:
            raise ElevenLabsError(503, "ElevenLabs API key not configured")
        r = await self._request(
            "POST", "/v1/voices/add",
            data={"name": name, "description": description},
            files=[("files", f) for f in files])
        return r.json()["voice_id"]

    async def delete_voice(self, voice_id: str) -> None:
        if self.enabled:
            await self._request("DELETE", f"/v1/voices/{voice_id}")
