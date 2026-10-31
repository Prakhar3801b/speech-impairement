import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import consent, dsp, fusion
from backend.db import Database
from backend.main import create_app
from backend.metrics import cer, wer


# ---------------------------------------------------------------- synthetic "words"
def word(freqs, seg=0.12, sr=16000, stretch=1.0, noise=0.01, seed=0):
    """A pseudo-word: a sequence of harmonic tone segments (formant-like)."""
    rng = np.random.default_rng(seed)
    parts = []
    for f in freqs:
        n = int(seg * stretch * sr)
        t = np.arange(n) / sr
        tone = sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3))
        parts.append(tone * np.hanning(n))
    x = np.concatenate([np.zeros(1600), *parts, np.zeros(1600)])
    x = 0.3 * x / np.abs(x).max() + noise * rng.standard_normal(len(x))
    return dsp.write_wav(x.astype(np.float32), sr)


WORDS = {
    1: [300, 600, 900, 400],
    2: [800, 300, 300, 700],
    3: [500, 500, 1200, 1200],
    4: [1000, 700, 400, 250],
}


def test_dtw_matches_own_template_even_when_slower():
    templates = [(pid, dsp.extract(word(f, seed=pid)).mfcc) for pid, f in WORDS.items()]
    for pid, f in WORDS.items():
        # slower (dysarthric-like) and noisier re-recording of the same word
        q = dsp.extract(word(f, stretch=1.6, noise=0.03, seed=100 + pid)).mfcc
        best = dsp.match(q, templates)[0]
        assert best["phrase_id"] == pid, (pid, dsp.match(q, templates))


def test_wav_roundtrip_and_resample():
    x = np.sin(np.linspace(0, 100, 44100)).astype(np.float32) * 0.5
    y, sr = dsp.read_wav(dsp.write_wav(x, 44100))
    assert sr == 44100 and abs(len(dsp.resample(y, sr)) - 16000) <= 1


def test_wer_cer():
    assert wer("I want water", "I want water") == 0
    assert wer("I want water", "I want") == pytest.approx(1 / 3)
    assert cer("paani", "pani") == pytest.approx(1 / 5)


def test_phonetic_similarity_handles_blurred_speech():
    assert fusion.similarity("bani", "paani") > 0.7
    assert fusion.similarity("i wan wader", "I want water") > 0.7


def test_fast_path_nod_answers_yes_no_question():
    phrases = [{"id": 1, "text": "Yes", "is_emergency": False, "language": "en"}]
    inp = fusion.FusionInput(partner_last="Do you want tea?", cues=fusion.Cues(head="nod"))
    assert fusion.fast_path(inp, phrases)[0]["text"] == "Yes"
    inp = fusion.FusionInput(partner_last="Kya chai chahiye?", cues=fusion.Cues(head="shake"))
    assert fusion.fast_path(inp, phrases)[0]["text"] == "Nahi"


def test_consent_rules(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    with pytest.raises(consent.ConsentError):
        consent.create(db, "Ravi", "Ravi", "self", "aac", "someone else", True)
    with pytest.raises(consent.ConsentError):
        consent.create(db, "Ravi", "Ravi", "self", "aac", "Ravi", False)
    rec = consent.create(db, "Ravi", "Ravi", "self", "aac", "ravi", True)
    assert consent.require_active(db, rec["id"])["voice_owner"] == "Ravi"
    consent.revoke(db, rec["id"])
    with pytest.raises(consent.ConsentError):
        consent.require_active(db, rec["id"])


# ---------------------------------------------------------------- API (mock mode: no keys)
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.Settings.llm_enabled", property(lambda self: False))
    app = create_app(db_path=str(tmp_path / "api.db"), elevenlabs_key="")
    with TestClient(app) as c:
        yield c


def test_status_and_seeded_phrases(client):
    s = client.get("/api/status").json()
    assert s["elevenlabs"] is False
    assert len(client.get("/api/phrases").json()) > 20


def test_conversation_flow_with_templates_and_signs(client):
    phrases = client.get("/api/phrases").json()
    water = next(p for p in phrases if p["text"] == "I want water")
    pain = next(p for p in phrases if p["text"] == "I am in pain")
    # enroll two spectrogram templates
    for p, f in ((water, WORDS[1]), (pain, WORDS[2])):
        r = client.post("/api/templates", data={"phrase_id": p["id"]},
                        files={"audio": ("a.wav", word(f), "audio/wav")})
        assert r.status_code == 200 and r.json()["frames"] > 10
    # partner speaks (typed fallback)
    r = client.post("/api/partner", data={"session_id": "s1", "text": "What would you like to drink?"}).json()
    assert any("water" in c["text"].lower() for c in r["suggestions"])
    # patient: slurred audio matching the "water" template, plus ISL spelling "WA"
    r = client.post("/api/patient", data={"session_id": "s1", "spelled": "WA", "cues": json.dumps({"expression": "neutral"})},
                    files={"audio": ("p.wav", word(WORDS[1], stretch=1.4, noise=0.03, seed=7), "audio/wav")}).json()
    assert r["candidates"][0]["text"] == "I want water", r
    assert r["spectrogram_matches"][0]["phrase_id"] == water["id"]
    assert "spectrogram_ms" in r["latency"] and r["spectrogram"]
    # confirm + speak (mock -> browser TTS)
    assert client.post("/api/confirm", json={"session_id": "s1", "text": "I want water", "phrase_id": water["id"],
                                              "rank": 0, "actions": 1, "time_to_confirm_ms": 900}).json()["ok"]
    assert client.post("/api/speak", json={"text": "I want water"}).json() == {"mock": True}
    m = client.get("/api/metrics").json()
    assert m["effort"]["top1_rate"] == 1.0 and "patient" in m["latency"]
    assert [t["speaker"] for t in client.get("/api/history/s1").json()] == ["partner", "patient"]


def test_sign_samples_crud(client):
    r = client.post("/api/signs", json={"label": "a", "kind": "letter", "samples": [[0.1] * 254, [0.2] * 254]})
    assert r.json()["label"] == "A"
    assert client.get("/api/signs/summary").json()[0]["n"] == 2
    client.delete("/api/signs/A")
    assert client.get("/api/signs").json() == []


def test_clone_requires_active_consent(client):
    files = {"files": ("v.wav", word(WORDS[3]), "audio/wav")}
    r = client.post("/api/voices/clone", data={"consent_id": 999, "name": "x"}, files=files)
    assert r.status_code == 403
    c = client.post("/api/consent", data={"voice_owner": "Asha", "signer": "Meera", "relationship": "guardian",
                                          "purpose": "aac", "typed_signature": "Meera", "agreed": "true"}).json()
    client.post(f"/api/consent/{c['id']}/revoke")
    r = client.post("/api/voices/clone", data={"consent_id": c["id"], "name": "x"}, files=files)
    assert r.status_code == 403 and "revoked" in r.json()["detail"]
    assert [e["data"]["action"] for e in client.get("/api/audit").json()] == ["revoked", "granted"]


def test_emergency_sign_fast_path(client):
    phrases = client.get("/api/phrases").json()
    help_ = next(p for p in phrases if p["is_emergency"])
    r = client.post("/api/patient", data={"session_id": "s2", "sign_phrase_ids": json.dumps([help_["id"]])}).json()
    assert r["ranker"] == "fast-path" and r["candidates"][0]["text"] == help_["text"]
