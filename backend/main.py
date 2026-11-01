"""FastAPI app: REST API + static frontend.

Run:  uvicorn backend.main:app --reload    (from the project folder)

Every /api route except /api/auth/* needs a signed-in account (cookie), or for the partner's
phone a signed partner link. Route code uses `db`, `el`, `templates`, ...: stand-ins that resolve
to the signed-in account's own store and matcher (see AccountMiddleware).
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import consent as consent_mod
from . import dsp, fusion, intro, llm
from .auth import COOKIE, SESSION_DAYS, Accounts, AuthError
from .config import ROOT, settings
from .db import Database
from .elevenlabs import ElevenLabs, ElevenLabsError
from .local_asr import LocalASR
from .matcher import Matcher
from .seed import seed_phrases


# ---------------------------------------------------------------- request models
class PhraseIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    language: str = "en"
    category: str = "general"
    aliases: list[str] = []
    keywords: list[str] = []
    is_emergency: bool = False


class SignIn(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    kind: str = "letter"          # letter | phrase | control
    phrase_id: int | None = None
    samples: list[list[float]]
    learned: bool = False         # captured during a confirmed turn rather than a training recording


class ConfirmIn(BaseModel):
    session_id: str
    text: str
    phrase_id: int | None = None
    rank: int = 0                     # 0 = top suggestion; -1 = typed / custom
    actions: int = 1                  # number of user inputs needed to confirm
    time_to_confirm_ms: float = 0
    sources: list[str] = []
    raw_text: str = ""
    ranker: str = ""
    language: str = ""


class TrainingConsentIn(BaseModel):
    signer: str = Field(min_length=1, max_length=120)
    relationship: str = "self"
    agreed: bool = False


TRAINING_STATEMENT = (
    "Keep my confirmed spoken phrases on this device, with the text I confirmed, to train a speech "
    "recogniser for my voice. They are not uploaded. I can stop this and delete them at any time.")


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    session_id: str | None = None


class SelectVoice(BaseModel):
    voice_id: str


class DesignIn(BaseModel):
    description: str = Field(min_length=20, max_length=1000)
    sample_text: str | None = None


class DesignSave(BaseModel):
    generated_voice_id: str
    name: str
    description: str
    key_id: str | None = None          # account the preview was generated in



class SignupIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=120)


class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)


MAX_LEARNED_TEMPLATES = 6      # per phrase, on top of the ones the patient recorded
MAX_LEARNED_SIGN_SAMPLES = 40  # per sign


# ---------------------------------------------------------------- accounts
@dataclass
class Account:
    user: dict
    db: Database
    el: ElevenLabs
    templates: Matcher
    recent_audio: dict = field(default_factory=dict)     # session -> (time, features) of the last utterance
    partner_latest: dict = field(default_factory=dict)   # session -> last partner turn + suggestions


_current: contextvars.ContextVar[Account] = contextvars.ContextVar("clio_account")


class _Current:
    """Stands in for one attribute of the signed-in account, so route code reads `db.query(...)`."""

    def __init__(self, attr: str):
        self._attr = attr

    def _get(self):
        return getattr(_current.get(), self._attr)

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __len__(self):
        return len(self._get())

    def __getitem__(self, k):
        return self._get()[k]

    def __setitem__(self, k, v):
        self._get()[k] = v


PARTNER_PATHS = ("/api/partner", "/api/history/")   # what a partner link may use (one conversation only)


class AccountMiddleware:
    """Resolves the account for every /api request (login cookie, or a partner link) before the route runs."""

    def __init__(self, app, resolve):
        self.app, self.resolve = app, resolve

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith("/api/") or path.startswith("/api/auth/"):
            return await self.app(scope, receive, send)
        acct = await run_in_threadpool(self.resolve, Request(scope))
        if acct is None:
            return await JSONResponse({"error": "auth", "detail": "Please sign in."}, status_code=401)(scope, receive, send)
        token = _current.set(acct)
        try:
            await self.app(scope, receive, send)
        finally:
            _current.reset(token)


# ---------------------------------------------------------------- state
def create_app(db_path: str | None = None, elevenlabs_key: str | None = None) -> FastAPI:
    core_url = db_path or settings.database_url or settings.db_path   # an explicit path (tests) wins
    accounts = Accounts(core_url)
    user_dir = Path(db_path or settings.db_path).parent / "users"
    base_el = ElevenLabs(accounts.core, api_key=elevenlabs_key)   # shares keys, budget, HTTP clients
    local_asr = LocalASR(settings.local_asr_model)
    open_accounts: dict[int, Account] = {}
    open_lock = threading.Lock()

    def account(uid: int) -> Account | None:
        with open_lock:
            if uid not in open_accounts:
                user = accounts.user(uid)
                if not user:
                    return None
                udb = accounts.open_user_db(uid, user_dir)
                seed_phrases(udb)
                open_accounts[uid] = Account(user, udb, base_el.for_account(udb), Matcher(udb))
            return open_accounts[uid]

    def resolve(req: Request) -> Account | None:
        user = accounts.user_for_token(req.cookies.get(COOKIE))
        if user:
            return account(user["id"])
        if req.url.path.startswith(PARTNER_PATHS):
            uid = accounts.check_partner_key(req.query_params.get("k"), req.query_params.get("s", ""))
            if uid:
                req.scope.setdefault("state", {})["partner_session"] = req.query_params.get("s")
                return account(uid)
        return None

    db, el, templates = _Current("db"), _Current("el"), _Current("templates")
    recent_audio, partner_latest = _Current("recent_audio"), _Current("partner_latest")

    def partner_scope_ok(request: Request, session_id: str) -> None:
        """A partner link only covers the conversation it was made for."""
        allowed = request.scope.get("state", {}).get("partner_session")
        if allowed is not None and allowed != session_id:
            raise HTTPException(403, "this partner link is for another conversation")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await base_el.aclose()

    app = FastAPI(title="Clio: PS16 conversation assistant", lifespan=lifespan)
    app.add_middleware(AccountMiddleware, resolve=resolve)
    app.state.accounts, app.state.el, app.state.account = accounts, base_el, account

    @app.exception_handler(AuthError)
    async def _auth_error(_req: Request, exc: AuthError):
        return JSONResponse(status_code=400, content={"error": "auth", "detail": str(exc)})

    # ------------------------------------------------------------ sign up / sign in
    def _signed_in(request: Request, user: dict) -> JSONResponse:
        resp = JSONResponse({"user": user})
        resp.set_cookie(COOKIE, accounts.start_session(user["id"]), max_age=SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=request.url.scheme == "https")
        return resp

    @app.post("/api/auth/signup")
    def signup(body: SignupIn, request: Request):
        return _signed_in(request, accounts.signup(body.email, body.password, body.name))

    @app.post("/api/auth/login")
    def login(body: LoginIn, request: Request):
        return _signed_in(request, accounts.login(body.email, body.password))

    @app.post("/api/auth/logout")
    def logout(request: Request):
        accounts.end_session(request.cookies.get(COOKIE))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    @app.get("/api/auth/me")
    def me(request: Request):
        user = accounts.user_for_token(request.cookies.get(COOKIE))
        if not user:
            return JSONResponse({"error": "auth", "detail": "Please sign in."}, status_code=401)
        return {"user": user, "database": "postgres" if accounts.core.pg else "sqlite"}

    @app.get("/api/partner/link")
    def partner_link(session_id: str):
        """Signed key for the partner's phone: lets it use this one conversation without signing in."""
        return {"key": accounts.partner_key(_current.get().user["id"], session_id)}

    @app.exception_handler(ElevenLabsError)
    async def _el_error(_req: Request, exc: ElevenLabsError):
        return JSONResponse(status_code=502 if exc.status >= 500 else exc.status,
                            content={"error": "elevenlabs", "status": exc.status, "detail": exc.detail})

    @app.exception_handler(consent_mod.ConsentError)
    async def _consent_error(_req: Request, exc: consent_mod.ConsentError):
        return JSONResponse(status_code=403, content={"error": "consent", "detail": str(exc)})

    def current_voice() -> str:
        return db.get_setting("voice_id", settings.default_voice_id) or settings.default_voice_id

    # ------------------------------------------------------------ status
    @app.get("/api/status")
    def status():
        return {
            "elevenlabs": el.enabled, "llm": settings.llm_enabled,
            "stt_model": settings.stt_model, "tts_model": settings.tts_model,
            "llm_model": settings.llm_model if settings.llm_enabled else None,
            "voice_id": current_voice(), "concurrency": settings.elevenlabs_concurrency,
            "cache": db.cache_stats(), "credits": credit_usage(), "budget": el.budget(),
            "templates": len(templates), "matcher": templates.kind if len(templates) else None,
            "speech_model": templates.embedder.name if templates.embedder.available else None,
            "local_asr": local_asr.available, "training": training_summary(),
            "user": _current.get().user, "database": "postgres" if accounts.core.pg else "sqlite",
        }

    def credit_usage() -> dict:
        since = time.time() - 30 * 86400
        rows = db.query("SELECT data FROM events WHERE kind='credit' AND created_at>?", (since,))
        tts_chars = stt_secs = saved_secs = designs = skipped = 0.0
        for r in rows:
            d = json.loads(r["data"])
            tts_chars += d.get("chars", 0)
            stt_secs += d.get("seconds", 0)
            saved_secs += d.get("saved_seconds", 0) + d.get("trimmed_seconds", 0)
            designs += 1 if d.get("service") == "voice_design" else 0
            skipped += 1 if d.get("source") == "skipped" else 0
        return {"window_days": 30, "tts_chars": int(tts_chars), "stt_minutes": round(stt_secs / 60, 2),
                "stt_minutes_saved": round(saved_secs / 60, 2), "stt_calls_skipped": int(skipped),
                "voice_designs": int(designs)}

    # ------------------------------------------------------------ phrases

    @app.get("/api/phrases")
    def list_phrases():
        return db.phrases()

    @app.post("/api/phrases")
    def add_phrase(p: PhraseIn):
        return {"id": db.add_phrase(p.text.strip(), p.language, p.category, p.aliases, p.keywords,
                                    p.is_emergency)}

    @app.delete("/api/phrases/{phrase_id}")
    def delete_phrase(phrase_id: int):
        db.execute("DELETE FROM phrases WHERE id=?", (phrase_id,))
        templates.remove(phrase_id=phrase_id)
        return {"ok": True}

    @app.post("/api/phrases/precache")
    async def precache_phrases():
        """Generate TTS once for every phrase in the current voice (spends credits once)."""
        voice = current_voice()
        done = cached = 0
        for p in db.phrases():
            if el.cached_tts(p["text"], voice):
                cached += 1
                continue
            res = await el.speak(p["text"], voice)
            if res.get("mock"):
                return {"mock": True, "generated": 0, "already_cached": cached}
            done += 1
        return {"generated": done, "already_cached": cached}

    def phrases_now() -> list[dict]:
        """The phrasebook, annotated with how often each phrase is said at this time of day."""
        prior = db.phrase_time_prior()
        phrases = db.phrases()
        for p in phrases:
            p["time_prior"] = prior.get(p["id"], 0.0)
        return phrases

    # ------------------------------------------------------------ spectrogram templates
    @app.post("/api/templates")
    async def enroll_template(phrase_id: int = Form(...), audio: UploadFile = File(...)):
        if not db.one("SELECT id FROM phrases WHERE id=?", (phrase_id,)):
            raise HTTPException(404, "phrase not found")
        wav = await audio.read()
        try:
            feats = await asyncio.to_thread(dsp.extract, wav)
        except Exception as e:  # malformed upload
            raise HTTPException(400, f"could not read WAV audio: {e}") from e
        if len(feats.mfcc) < 10:
            raise HTTPException(400, "recording too short or silent")
        await asyncio.to_thread(templates.save, phrase_id, feats)
        return {"ok": True, "frames": len(feats.mfcc), "spectrogram": dsp.spectrogram_preview(feats.logmel)}

    @app.get("/api/templates/{phrase_id}")
    def template_previews(phrase_id: int):
        rows = db.query("SELECT id, logmel, n_frames, n_mels FROM audio_templates WHERE phrase_id=?",
                        (phrase_id,))
        return [{"id": r["id"], "spectrogram": dsp.spectrogram_preview(
            np.frombuffer(r["logmel"], dtype=np.float32).reshape(r["n_frames"], r["n_mels"]))}
            for r in rows]

    @app.delete("/api/templates/{template_id}")
    def delete_template(template_id: int):
        db.execute("DELETE FROM audio_templates WHERE id=?", (template_id,))
        templates.remove(template_id=template_id)
        return {"ok": True}

    # ------------------------------------------------------------ ISL signs

    @app.get("/api/signs")
    def list_signs():
        rows = db.query("SELECT label, kind, phrase_id, features FROM sign_samples")
        for r in rows:
            r["features"] = json.loads(r["features"])
        return rows

    @app.get("/api/signs/summary")
    def sign_summary():
        return db.query("SELECT label, kind, phrase_id, COUNT(*) AS n FROM sign_samples "
                        "GROUP BY label, kind, phrase_id ORDER BY kind, label")

    @app.post("/api/signs")
    def add_signs(s: SignIn):
        if s.kind not in ("letter", "phrase", "control"):
            raise HTTPException(400, "kind must be letter, phrase or control")
        label = s.label.strip().upper() if s.kind != "phrase" else s.label.strip()
        now = time.time()
        source = "learned" if s.learned else "recorded"
        for f in s.samples:
            db.execute("INSERT INTO sign_samples(label, kind, phrase_id, features, source, created_at) "
                       "VALUES(?,?,?,?,?,?)", (label, s.kind, s.phrase_id, json.dumps(f), source, now))
        if s.learned:   # keep only the newest learned samples per sign; recorded ones are never dropped
            db.execute("DELETE FROM sign_samples WHERE label=? AND source='learned' AND id NOT IN "
                       "(SELECT id FROM sign_samples WHERE label=? AND source='learned' ORDER BY id DESC LIMIT ?)",
                       (label, label, MAX_LEARNED_SIGN_SAMPLES))
        return {"ok": True, "label": label, "added": len(s.samples)}

    @app.delete("/api/signs/{label}")
    def delete_sign(label: str):
        db.execute("DELETE FROM sign_samples WHERE label=?", (label,))
        return {"ok": True}

    # ------------------------------------------------------------ conversation
    def log_latency(session_id: str, kind: str, stages: dict) -> None:
        db.log_event("latency", {"kind": kind, **{k: round(v, 1) for k, v in stages.items()}}, session_id)

    @app.post("/api/partner")
    async def partner_turn(request: Request, session_id: str = Form(...), language: str = Form("auto"),
                           text: str = Form(""), use_llm: bool = Form(True),
                           audio: UploadFile | None = File(None)):
        partner_scope_ok(request, session_id)
        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        stt = {"text": text.strip(), "mock": False, "cached": False}
        if audio is not None and not stt["text"]:
            wav = await audio.read()
            stt = await el.transcribe(wav, language, session_id, role="partner")
            stages["stt_ms"] = (time.perf_counter() - t0) * 1000
        partner_text = stt["text"]
        if not partner_text:
            return {"transcript": "", "suggestions": [], "stt": stt,
                    "note": "No transcript (mock mode or silence). Type the partner's words instead."}
        turn_id = db.add_turn(session_id, "partner", partner_text, source="typed" if text else "scribe")
        phrases = phrases_now()
        t1 = time.perf_counter()
        suggestions = fusion.suggest_replies(partner_text, phrases)
        stages["fusion_ms"] = (time.perf_counter() - t1) * 1000
        if use_llm and settings.llm_enabled:
            t2 = time.perf_counter()
            llm_out = await llm.rank({"partner_last": partner_text, "history": db.history(session_id),
                                      "shortlist": [{"text": c["text"], "score": c["score"]} for c in suggestions],
                                      "phrasebook": [p["text"] for p in phrases]}, task="reply")
            stages["llm_ms"] = (time.perf_counter() - t2) * 1000
            if llm_out:   # the question's own answers stay first; the LLM orders the rest
                shaped = [c for c in suggestions if any(s.startswith("question:") for s in c["sources"])]
                suggestions = fusion._dedupe(shaped + _attach_phrase_ids(llm_out, phrases))
        stages["total_ms"] = (time.perf_counter() - t0) * 1000
        log_latency(session_id, "partner", stages)
        out = {"transcript": partner_text, "suggestions": suggestions, "stt": stt, "latency": stages,
               "question": fusion.question_shape(partner_text), "turn_id": turn_id}
        partner_latest[session_id] = out
        return out

    @app.get("/api/partner/latest")
    def partner_latest_turn(session_id: str, after: int = 0):
        """Latest partner turn for a session (lets the laptop pick up turns spoken on the partner's phone)."""
        last = partner_latest.get(session_id)
        return last if last and last["turn_id"] > after else {}

    @app.get("/api/lan")
    def lan_addresses():
        """This machine's LAN addresses, so the partner's phone can open the partner page."""
        import socket
        addrs = set()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("10.255.255.255", 1))   # no packet is sent; picks the outbound interface
                addrs.add(sock.getsockname()[0])
        except OSError:
            pass
        return {"addresses": sorted(a for a in addrs if not a.startswith("127."))}

    def _attach_phrase_ids(cands: list[dict], phrases: list[dict]) -> list[dict]:
        idx = {fusion.normalize(p["text"]): p for p in phrases}
        for c in cands:
            p = idx.get(fusion.normalize(c["text"]))
            if p:
                c["phrase_id"] = p["id"]
        return cands

    @app.post("/api/patient")
    async def patient_turn(session_id: str = Form(...), language: str = Form("auto"),
                           cues: str = Form("{}"), spelled: str = Form(""),
                           sign_phrase_ids: str = Form("[]"), use_llm: bool = Form(True),
                           use_scribe: bool = Form(True), use_spectrogram: bool = Form(True),
                           audio: UploadFile | None = File(None)):
        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        wav = await audio.read() if audio is not None else b""
        try:
            cue_obj = fusion.Cues.from_dict(json.loads(cues or "{}"))
            sign_ids = [int(i) for i in json.loads(sign_phrase_ids or "[]")]
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"bad cues/sign_phrase_ids JSON: {e}") from e

        # Local spectrogram matching runs first (a few ms, no credits). When it is already
        # decisive, Scribe is skipped, which keeps the free-tier STT minutes for unclear speech.
        feats, matches, spec_preview, match_info = None, [], [], {}
        if wav:
            feats = await asyncio.to_thread(dsp.extract, wav)
            spec_preview = dsp.spectrogram_preview(feats.logmel)
            recent_audio[session_id] = (time.time(), feats)
            if use_spectrogram and len(templates):
                t = time.perf_counter()
                emb = await asyncio.to_thread(templates.embed, feats) if templates.kind == "ssl" else None
                matches, match_info = await asyncio.to_thread(templates.match, feats, emb)
                stages["spectrogram_ms"] = (time.perf_counter() - t) * 1000
        top = matches[0]["confidence"] if matches else 0.0
        skip_at = 0.6 if el.saver_mode else settings.skip_stt_confidence
        skip_stt = bool(matches) and top >= skip_at

        stt = {"text": "", "words": [], "mock": not el.enabled, "cached": False}
        if wav and use_scribe and skip_stt:
            stt["skipped"] = f"spectrogram match {top:.2f}"
            db.log_event("credit", {"service": "stt", "seconds": 0, "source": "skipped",
                                    "saved_seconds": round(dsp.duration_s(wav), 2)}, session_id)
        elif wav and use_scribe:
            t = time.perf_counter()
            local_text = (await asyncio.to_thread(local_asr.transcribe, feats.audio, language)
                          if local_asr.available else "")
            if local_text:   # the patient's own fine-tuned recogniser: free, so Scribe is not called
                stt = {"text": local_text, "words": [], "mock": False, "cached": False, "local": True}
                stages["local_asr_ms"] = (time.perf_counter() - t) * 1000
                db.log_event("credit", {"service": "stt", "seconds": 0, "source": "local",
                                        "saved_seconds": round(dsp.duration_s(wav), 2)}, session_id)
            else:
                stt = await el.transcribe(wav, language, session_id, role="patient", features=feats)
                if not stt.get("mock"):
                    stages["stt_ms"] = (time.perf_counter() - t) * 1000
        phrases = phrases_now()
        by_id = {p["id"]: p for p in phrases}
        for m in matches:
            m["text"] = by_id.get(m["phrase_id"], {}).get("text")
        history = db.history(session_id)
        partner_last = next((h["text"] for h in reversed(history) if h["speaker"] == "partner"), "")
        inp = fusion.FusionInput(asr_text=stt.get("text", ""), template_matches=matches, spelled=spelled,
                                 sign_phrase_ids=sign_ids, cues=cue_obj, partner_last=partner_last,
                                 language=language)
        t1 = time.perf_counter()
        fast = fusion.fast_path(inp, phrases)
        heuristic = fast or fusion.heuristic_rank(inp, phrases)
        stages["fusion_ms"] = (time.perf_counter() - t1) * 1000
        candidates, ranker = heuristic, ("fast-path" if fast else "heuristic")
        if not fast and use_llm and settings.llm_enabled and (inp.asr_text or matches or spelled or sign_ids):
            t2 = time.perf_counter()
            llm_out = await llm.rank({
                "partner_last": partner_last, "history": history, "asr_text": inp.asr_text,
                "asr_words": stt.get("words", []), "language_hint": language,
                "spectrogram_matches": [{"text": m["text"], "confidence": m["confidence"]} for m in matches],
                "isl_spelled": spelled, "sign_phrases": [by_id[i]["text"] for i in sign_ids if i in by_id],
                "cues": cue_obj.__dict__,
                "shortlist": [{"text": c["text"], "score": c["score"], "sources": c["sources"]} for c in heuristic],
            })
            stages["llm_ms"] = (time.perf_counter() - t2) * 1000
            if llm_out:
                candidates, ranker = _attach_phrase_ids(llm_out, phrases), "llm"
        stages["total_ms"] = (time.perf_counter() - t0) * 1000
        log_latency(session_id, "patient", stages)
        return {"candidates": candidates, "heuristic": heuristic, "ranker": ranker,
                "transcript": stt.get("text", ""),
                "stt": {k: stt.get(k) for k in ("mock", "cached", "language", "skipped", "budget", "local")},
                "spectrogram_matches": matches, "matcher": match_info, "spectrogram": spec_preview, "latency": stages,
                "partner_last": partner_last}


    @app.post("/api/confirm")
    def confirm(c: ConfirmIn):
        db.add_turn(c.session_id, "patient", c.text, raw_text=c.raw_text, source=",".join(c.sources))
        learned = kept = False
        when, feats = recent_audio.pop(c.session_id, (0.0, None))
        spoken = (feats is not None and feats.audio is not None and time.time() - when < 120
                  and len(feats.mfcc) >= 10 and any(s in ("speech", "spectrogram") for s in c.sources))
        if c.phrase_id:
            db.execute("UPDATE phrases SET use_count=use_count+1 WHERE id=?", (c.phrase_id,))
            # Learn as you talk: when the confirmed phrase came from the patient's own speech, that
            # utterance becomes one more spectrogram template for it (no separate recording session).
            if spoken:
                templates.save(c.phrase_id, feats, source="learned")
                templates.trim_learned(c.phrase_id, MAX_LEARNED_TEMPLATES)
                learned = True
        # Opt-in: keep the utterance with the confirmed text, to train a personal recogniser later.
        if spoken and db.get_setting("training_clips") == "on":
            db.execute("INSERT INTO training_clips(phrase_id, text, language, audio, seconds, created_at) "
                       "VALUES(?,?,?,?,?,?)", (c.phrase_id, c.text, c.language, dsp.pcm16(feats.audio),
                                               len(feats.audio) / dsp.SR, time.time()))
            kept = True
        db.log_event("effort", {"rank": c.rank, "actions": c.actions, "time_ms": round(c.time_to_confirm_ms),
                                "ranker": c.ranker, "sources": c.sources, "phrase_id": c.phrase_id}, c.session_id)
        return {"ok": True, "learned_template": learned, "kept_for_training": kept}

    # ------------------------------------------------------------ guided voice intro (setup)
    @app.get("/api/intro/prompts")
    def intro_prompts(language: str = "en"):
        return {"prompts": intro.prompts(db.phrases(), language),
                "profile": json.loads(db.get_setting("voice_profile") or "null")}

    @app.post("/api/intro")
    async def intro_submit(words: UploadFile = File(...), phrase_ids: str = Form(...),
                           about: UploadFile | None = File(None), language: str = Form("auto")):
        """Part 2 (`words`, said with pauses) becomes one template per phrase; part 1 (`about`,
        free speech) gives the voice profile and name suggestions."""
        try:
            ids = [int(i) for i in json.loads(phrase_ids)]
            x, sr = dsp.read_wav(await words.read())
        except Exception as e:
            raise HTTPException(400, f"could not read the recording: {e}") from e
        known = {p["id"]: p["text"] for p in db.phrases()}
        if not ids or any(i not in known for i in ids):
            raise HTTPException(400, "unknown phrase id")
        pieces = await asyncio.to_thread(intro.segment, dsp.resample(x, sr), len(ids))
        if not pieces:
            raise HTTPException(422, f"We could not hear {len(ids)} separate words. Try again with a clear "
                                     "pause after each word.")
        saved = []
        for pid, piece in zip(ids, pieces):
            feats = await asyncio.to_thread(dsp.extract_signal, piece)
            if len(feats.mfcc) >= 10:
                await asyncio.to_thread(templates.save, pid, feats, "enrolled")
                saved.append({"phrase_id": pid, "text": known[pid], "seconds": round(len(piece) / dsp.SR, 2)})

        profile, transcript, ideas = None, "", []
        if about is not None:
            wav = await about.read()
            ax, asr = dsp.read_wav(wav)
            profile = await asyncio.to_thread(intro.voice_profile, dsp.resample(ax, asr))
            db.set_setting("voice_profile", json.dumps(profile))
            stt = await el.transcribe(wav, language, None, role="intro", reuse_similar=False)
            transcript = stt.get("text", "")
            ideas = intro.suggestions(transcript, list(known.values()))
        db.log_event("intro", {"saved": len(saved), "profile": profile})
        return {"saved": saved, "profile": profile, "transcript": transcript, "suggestions": ideas}

    # ------------------------------------------------------------ personal model training data (opt-in)
    def training_summary() -> dict:
        row = db.one("SELECT COUNT(*) AS n, COALESCE(SUM(seconds), 0) AS s, COUNT(DISTINCT text) AS t "
                     "FROM training_clips")
        return {"enabled": db.get_setting("training_clips") == "on", "clips": row["n"],
                "minutes": round(row["s"] / 60, 1), "distinct_texts": row["t"],
                "signer": db.get_setting("training_signer"), "statement": TRAINING_STATEMENT}

    @app.get("/api/training")
    def training():
        return training_summary()

    @app.post("/api/training/consent")
    def training_consent(t: TrainingConsentIn):
        if not t.agreed:
            raise HTTPException(400, "the patient (or their guardian) must agree first")
        db.set_setting("training_clips", "on")
        db.set_setting("training_signer", t.signer.strip())
        db.log_event("consent", {"action": "training_clips_on", "signer": t.signer.strip(),
                                 "relationship": t.relationship, "statement": TRAINING_STATEMENT})
        return training_summary()

    @app.post("/api/training/off")
    def training_off(delete: bool = Form(False)):
        db.set_setting("training_clips", "off")
        if delete:
            db.execute("DELETE FROM training_clips")
        db.log_event("consent", {"action": "training_clips_off", "deleted": delete})
        return training_summary()

    @app.post("/api/emergency")
    def emergency(session_id: str = Form(""), trigger: str = Form("button")):
        db.log_event("emergency", {"trigger": trigger}, session_id or None)
        p = next((p for p in db.phrases() if p["is_emergency"]), None)
        return {"text": p["text"] if p else "I need help now"}


    @app.post("/api/speak")
    async def speak(s: SpeakIn):
        t0 = time.perf_counter()
        voice = current_voice()
        row = db.one("SELECT key_id FROM voices WHERE voice_id=?", (voice,))
        res = await el.speak(s.text, voice, s.session_id, owner_key=row["key_id"] if row else None)
        if res.get("mock"):
            return JSONResponse({"mock": True})
        log_latency(s.session_id or "", "tts", {"tts_ms": (time.perf_counter() - t0) * 1000,
                                                 "cached": 1.0 if res["cached"] else 0.0})
        return Response(content=res["audio"], media_type=res["mime"],
                        headers={"X-Cache": "hit" if res["cached"] else "miss"})

    @app.get("/api/history/{session_id}")
    def history(session_id: str, request: Request):
        partner_scope_ok(request, session_id)
        return db.history(session_id, limit=50)

    # ------------------------------------------------------------ voices & consent
    @app.get("/api/voices")
    def voices():
        rows = db.query("SELECT v.*, c.voice_owner, c.revoked_at FROM voices v "
                        "LEFT JOIN consent_records c ON c.id=v.consent_id WHERE v.deleted_at IS NULL")
        return {"selected": current_voice(), "default": settings.default_voice_id, "voices": rows}


    @app.post("/api/voices/select")
    def select_voice(v: SelectVoice):
        if v.voice_id != settings.default_voice_id:
            row = db.one("SELECT * FROM voices WHERE voice_id=? AND deleted_at IS NULL", (v.voice_id,))
            if not row:
                raise HTTPException(404, "unknown or deleted voice")
            if row["kind"] == "cloned":
                consent_mod.require_active(db, row["consent_id"])
        db.set_setting("voice_id", v.voice_id)
        return {"ok": True, "voice_id": v.voice_id}


    @app.post("/api/voices/design")
    async def design(d: DesignIn):
        return {"previews": await el.design_previews(d.description, d.sample_text)}


    @app.post("/api/voices/design/save")
    async def design_save(d: DesignSave):
        vid, kid = await el.design_save(d.generated_voice_id, d.name, d.description, d.key_id)
        db.execute("INSERT OR REPLACE INTO voices(voice_id, name, kind, description, key_id, created_at) "
                   "VALUES(?,?,?,?,?,?)", (vid, d.name, "designed", d.description, kid, time.time()))
        db.set_setting("voice_id", vid)
        return {"voice_id": vid}

    @app.post("/api/voices/clone")
    async def clone(consent_id: int = Form(...), name: str = Form(...),
                    files: list[UploadFile] = File(...)):
        record = consent_mod.require_active(db, consent_id)   # hard consent gate
        payload = [(f.filename or "sample.wav", await f.read(), f.content_type or "audio/wav") for f in files]
        vid, kid = await el.clone(name, payload, f"Consented clone of {record['voice_owner']} (consent #{consent_id})")
        db.execute("INSERT OR REPLACE INTO voices(voice_id, name, kind, description, consent_id, key_id, created_at) "
                   "VALUES(?,?,?,?,?,?,?)", (vid, name, "cloned", record["purpose"], consent_id, kid, time.time()))
        db.log_event("consent", {"action": "voice_cloned", "consent_id": consent_id, "voice_id": vid})
        db.set_setting("voice_id", vid)
        return {"voice_id": vid}

    @app.get("/api/consent")
    def list_consent():
        return db.query("SELECT * FROM consent_records ORDER BY id DESC")

    @app.get("/api/consent/statement")
    def consent_statement(owner: str, signer: str, relationship: str, purpose: str):
        return {"statement": consent_mod.build_statement(owner, signer, relationship, purpose)}

    @app.post("/api/consent")
    async def grant_consent(voice_owner: str = Form(...), signer: str = Form(...),
                            relationship: str = Form("self"), purpose: str = Form(...),
                            typed_signature: str = Form(...), agreed: bool = Form(False),
                            statement_audio: UploadFile | None = File(None)):
        audio = await statement_audio.read() if statement_audio else None
        return consent_mod.create(db, voice_owner, signer, relationship, purpose,
                                  typed_signature, agreed, audio)

    @app.post("/api/consent/{consent_id}/revoke")
    async def revoke_consent(consent_id: int):
        owners = {r["voice_id"]: r["key_id"] for r in db.query(
            "SELECT voice_id, key_id FROM voices WHERE consent_id=?", (consent_id,))}
        voices_to_delete = consent_mod.revoke(db, consent_id)
        errors = []
        for vid in voices_to_delete:
            try:
                await el.delete_voice(vid, owners.get(vid))
            except ElevenLabsError as e:
                errors.append({"voice_id": vid, "error": str(e)})
        if current_voice() in voices_to_delete:
            db.set_setting("voice_id", settings.default_voice_id)
        return {"revoked": consent_id, "deleted_voices": voices_to_delete, "errors": errors}

    @app.get("/api/audit")
    def audit():
        return db.events("consent", limit=200)

    # ------------------------------------------------------------ metrics
    @app.get("/api/metrics")
    def metrics():
        lat = db.events("latency", limit=1000)
        stages: dict[str, dict[str, list[float]]] = {}
        for e in lat:
            d = e["data"]
            for k, v in d.items():
                if k.endswith("_ms"):
                    stages.setdefault(d.get("kind", "?"), {}).setdefault(k, []).append(v)
        latency = {kind: {k: {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1),
                              "p95": round(float(np.percentile(v, 95)), 1)} for k, v in ks.items()}
                   for kind, ks in stages.items()}
        eff = [e["data"] for e in db.events("effort", limit=1000)]
        effort = {}
        if eff:
            effort = {"n": len(eff),
                      "mean_actions": round(float(np.mean([e["actions"] for e in eff])), 2),
                      "median_time_ms": round(float(np.median([e["time_ms"] for e in eff]))),
                      "top1_rate": round(sum(1 for e in eff if e["rank"] == 0) / len(eff), 3),
                      "custom_rate": round(sum(1 for e in eff if e["rank"] < 0) / len(eff), 3)}
        return {"latency": latency, "effort": effort, "cache": db.cache_stats(), "credits": credit_usage(),
                "budget": el.budget()}

    app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
    return app


app = create_app()
