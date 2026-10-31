"""FastAPI app: REST API + static frontend.

Run:  uvicorn backend.main:app --reload    (from the ps16-voice-assistant folder)
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import consent as consent_mod
from . import dsp, fusion, llm
from .config import ROOT, settings
from .db import Database
from .elevenlabs import ElevenLabs, ElevenLabsError
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



# ---------------------------------------------------------------- state
class TemplateIndex:
    """In-memory copy of enrolled spectrogram templates (reloaded on change)."""

    def __init__(self, db: Database):
        self.db = db
        self.items: list[tuple[int, np.ndarray]] = []
        self.reload()

    def reload(self) -> None:
        rows = self.db.query("SELECT phrase_id, mfcc, n_frames, n_mfcc FROM audio_templates")
        self.items = [(r["phrase_id"], np.frombuffer(r["mfcc"], dtype=np.float32)
                       .reshape(r["n_frames"], r["n_mfcc"])) for r in rows]


def create_app(db_path: str | None = None, elevenlabs_key: str | None = None) -> FastAPI:
    db = Database(db_path or settings.db_path)
    seed_phrases(db)
    el = ElevenLabs(db, api_key=elevenlabs_key)
    templates = TemplateIndex(db)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await el.aclose()

    app = FastAPI(title="PS16 Voice Restoration Assistant", lifespan=lifespan)
    app.state.db, app.state.el, app.state.templates = db, el, templates

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
            "cache": db.cache_stats(), "credits": credit_usage(),
            "templates": len(templates.items),
        }

    def credit_usage() -> dict:
        since = time.time() - 30 * 86400
        rows = db.query("SELECT data FROM events WHERE kind='credit' AND created_at>?", (since,))
        tts_chars = stt_secs = designs = 0.0
        for r in rows:
            d = json.loads(r["data"])
            tts_chars += d.get("chars", 0)
            stt_secs += d.get("seconds", 0)
            designs += 1 if d.get("service") == "voice_design" else 0
        return {"window_days": 30, "tts_chars": int(tts_chars), "stt_minutes": round(stt_secs / 60, 2),
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
        templates.reload()
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
        db.execute(
            "INSERT INTO audio_templates(phrase_id, mfcc, logmel, n_frames, n_mfcc, n_mels, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (phrase_id, feats.mfcc.tobytes(), feats.logmel.tobytes(), len(feats.mfcc),
             feats.mfcc.shape[1], feats.logmel.shape[1], time.time()))
        templates.reload()
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
        templates.reload()
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
        for f in s.samples:
            db.execute("INSERT INTO sign_samples(label, kind, phrase_id, features, created_at) "
                       "VALUES(?,?,?,?,?)", (label, s.kind, s.phrase_id, json.dumps(f), now))
        return {"ok": True, "label": label, "added": len(s.samples)}

    @app.delete("/api/signs/{label}")
    def delete_sign(label: str):
        db.execute("DELETE FROM sign_samples WHERE label=?", (label,))
        return {"ok": True}

    # ------------------------------------------------------------ conversation
    def log_latency(session_id: str, kind: str, stages: dict) -> None:
        db.log_event("latency", {"kind": kind, **{k: round(v, 1) for k, v in stages.items()}}, session_id)

    @app.post("/api/partner")
    async def partner_turn(session_id: str = Form(...), language: str = Form("auto"),
                           text: str = Form(""), use_llm: bool = Form(True),
                           audio: UploadFile | None = File(None)):
        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        stt = {"text": text.strip(), "mock": False, "cached": False}
        if audio is not None and not stt["text"]:
            wav = await audio.read()
            stt = await el.transcribe(wav, language, session_id)
            stages["stt_ms"] = (time.perf_counter() - t0) * 1000
        partner_text = stt["text"]
        if not partner_text:
            return {"transcript": "", "suggestions": [], "stt": stt,
                    "note": "No transcript (mock mode or silence). Type the partner's words instead."}
        db.add_turn(session_id, "partner", partner_text, source="typed" if text else "scribe")
        phrases = db.phrases()
        t1 = time.perf_counter()
        suggestions = fusion.suggest_replies(partner_text, phrases)
        stages["fusion_ms"] = (time.perf_counter() - t1) * 1000
        if use_llm and settings.llm_enabled:
            t2 = time.perf_counter()
            llm_out = await llm.rank({"partner_last": partner_text, "history": db.history(session_id),
                                      "shortlist": [{"text": c["text"], "score": c["score"]} for c in suggestions],
                                      "phrasebook": [p["text"] for p in phrases]}, task="reply")
            stages["llm_ms"] = (time.perf_counter() - t2) * 1000
            if llm_out:
                suggestions = _attach_phrase_ids(llm_out, phrases)
        stages["total_ms"] = (time.perf_counter() - t0) * 1000
        log_latency(session_id, "partner", stages)
        return {"transcript": partner_text, "suggestions": suggestions, "stt": stt, "latency": stages}

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

        async def run_stt():
            if not (wav and use_scribe):
                return {"text": "", "words": [], "mock": not el.enabled, "cached": False}
            t = time.perf_counter()
            res = await el.transcribe(wav, language, session_id)
            if not res.get("mock"):
                stages["stt_ms"] = (time.perf_counter() - t) * 1000
            return res

        async def run_templates():
            if not (wav and use_spectrogram and templates.items):
                return [], []
            t = time.perf_counter()
            feats = await asyncio.to_thread(dsp.extract, wav)
            matches = await asyncio.to_thread(dsp.match, feats.mfcc, templates.items)
            stages["spectrogram_ms"] = (time.perf_counter() - t) * 1000
            return matches, dsp.spectrogram_preview(feats.logmel)

        stt, (matches, spec_preview) = await asyncio.gather(run_stt(), run_templates())
        phrases = db.phrases()
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
                "transcript": stt.get("text", ""), "stt": {k: stt.get(k) for k in ("mock", "cached", "language")},
                "spectrogram_matches": matches, "spectrogram": spec_preview, "latency": stages,
                "partner_last": partner_last}


    @app.post("/api/confirm")
    def confirm(c: ConfirmIn):
        db.add_turn(c.session_id, "patient", c.text, raw_text=c.raw_text, source=",".join(c.sources))
        if c.phrase_id:
            db.execute("UPDATE phrases SET use_count=use_count+1 WHERE id=?", (c.phrase_id,))
        db.log_event("effort", {"rank": c.rank, "actions": c.actions, "time_ms": round(c.time_to_confirm_ms),
                                "ranker": c.ranker, "sources": c.sources}, c.session_id)
        return {"ok": True}


    @app.post("/api/speak")
    async def speak(s: SpeakIn):
        t0 = time.perf_counter()
        res = await el.speak(s.text, current_voice(), s.session_id)
        if res.get("mock"):
            return JSONResponse({"mock": True})
        log_latency(s.session_id or "", "tts", {"tts_ms": (time.perf_counter() - t0) * 1000,
                                                 "cached": 1.0 if res["cached"] else 0.0})
        return Response(content=res["audio"], media_type=res["mime"],
                        headers={"X-Cache": "hit" if res["cached"] else "miss"})

    @app.get("/api/history/{session_id}")
    def history(session_id: str):
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
        vid = await el.design_save(d.generated_voice_id, d.name, d.description)
        db.execute("INSERT OR REPLACE INTO voices(voice_id, name, kind, description, created_at) "
                   "VALUES(?,?,?,?,?)", (vid, d.name, "designed", d.description, time.time()))
        db.set_setting("voice_id", vid)
        return {"voice_id": vid}

    @app.post("/api/voices/clone")
    async def clone(consent_id: int = Form(...), name: str = Form(...),
                    files: list[UploadFile] = File(...)):
        record = consent_mod.require_active(db, consent_id)   # hard consent gate
        payload = [(f.filename or "sample.wav", await f.read(), f.content_type or "audio/wav") for f in files]
        vid = await el.clone(name, payload, f"Consented clone of {record['voice_owner']} (consent #{consent_id})")
        db.execute("INSERT OR REPLACE INTO voices(voice_id, name, kind, description, consent_id, created_at) "
                   "VALUES(?,?,?,?,?,?)", (vid, name, "cloned", record["purpose"], consent_id, time.time()))
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
        voices_to_delete = consent_mod.revoke(db, consent_id)
        errors = []
        for vid in voices_to_delete:
            try:
                await el.delete_voice(vid)
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
        return {"latency": latency, "effort": effort, "cache": db.cache_stats(), "credits": credit_usage()}

    app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
    return app


app = create_app()
