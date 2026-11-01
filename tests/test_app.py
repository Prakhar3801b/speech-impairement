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
        sign_up(c)
        yield c


def sign_up(c, email="carer@example.com"):
    r = c.post("/api/auth/signup", json={"email": email, "password": "long-enough-1", "name": "Ravi"})
    assert r.status_code == 200, r.text
    return r.json()["user"]


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


# ---------------------------------------------------------------- question-shaped replies
def test_question_shape_and_shaped_replies(tmp_path):
    assert fusion.question_shape("Paani thanda ya normal?") == {"type": "choice", "options": ["Thanda", "Normal"]}
    assert fusion.question_shape("Would you like hot water or cold water?")["options"] == ["Hot water", "Cold water"]
    assert fusion.question_shape("Kya dawai li?") == {"type": "yes_no", "options": ["Haan", "Nahi"]}
    assert fusion.question_shape("Are you cold?")["options"] == ["Yes", "No"]
    assert fusion.question_shape("What would you like to drink?")["type"] == "open"
    db = Database(str(tmp_path / "q.db"))
    from backend.seed import seed_phrases
    seed_phrases(db)
    r = fusion.suggest_replies("Aap chai lenge ya coffee?", db.phrases())
    assert [c["text"] for c in r[:2]] == ["Chai", "Coffee"] and all(c["confidence"] > 0 for c in r)


# ---------------------------------------------------------------- free-tier budget (fake ElevenLabs)
import asyncio

import httpx

from backend.elevenlabs import API, ElevenLabs


def fake_elevenlabs(el, handlers):
    """Route each key's requests to handler(key_index, request); records the calls."""
    calls = []
    for i, key in enumerate(el.pool.keys):
        def handle(req, i=i):
            calls.append((i, req.url.path))
            if req.url.path == "/v1/user/subscription":
                return httpx.Response(404)
            return handlers(i, req)
        el._clients[key.id] = httpx.AsyncClient(base_url=API, transport=httpx.MockTransport(handle))
    return calls


def test_tts_falls_back_to_next_key_when_quota_is_used_up(tmp_path):
    el = ElevenLabs(Database(str(tmp_path / "b.db")), api_key="key-one,key-two")
    calls = fake_elevenlabs(el, lambda i, req: httpx.Response(401, json={"detail": {"status": "quota_exceeded"}})
                            if i == 0 else httpx.Response(200, content=b"mp3"))
    res = asyncio.run(el.speak("Hello", "voice"))
    assert res["audio"] == b"mp3" and not res["cached"]
    first, second = el.pool.summary()
    assert first["exhausted"] and not second["exhausted"] and second["used"] > 0
    asyncio.run(el.speak("Hello again", "voice"))
    assert (0, "/v1/text-to-speech/voice") not in calls[3:]          # the used-up key is not tried again
    assert asyncio.run(el.speak("Hello", "voice"))["cached"] is True   # repeats are free


def test_all_keys_used_up_falls_back_to_browser_voice(tmp_path):
    el = ElevenLabs(Database(str(tmp_path / "b.db")), api_key="key-one")
    fake_elevenlabs(el, lambda i, req: httpx.Response(402, json={"detail": "quota"}))
    assert asyncio.run(el.speak("Hi", "voice")) == {"mock": True, "budget": "exhausted"}
    assert el.budget()["fraction_left"] == 0.0


def test_near_identical_clip_reuses_transcript(tmp_path):
    el = ElevenLabs(Database(str(tmp_path / "b.db")), api_key="key-one")
    calls = fake_elevenlabs(el, lambda i, req: httpx.Response(200, json={"text": "paani chahiye", "words": []}))
    stt = lambda wav: asyncio.run(el.transcribe(wav, role="patient"))  # noqa: E731
    assert stt(word(WORDS[1], seed=1))["text"] == "paani chahiye"
    again = stt(word(WORDS[1], stretch=1.1, noise=0.02, seed=2))        # same phrase, new recording
    assert again["cached"] == "fuzzy" and again["text"] == "paani chahiye"
    stt(word(WORDS[3], seed=3))                                          # a different phrase is sent
    assert [p for _, p in calls].count("/v1/speech-to-text") == 2


def test_patient_turn_skips_scribe_when_template_match_is_clear(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.Settings.llm_enabled", property(lambda self: False))
    app = create_app(db_path=str(tmp_path / "api.db"), elevenlabs_key="key-one")
    calls = fake_elevenlabs(app.state.el, lambda i, req: httpx.Response(200, json={"text": "x", "words": []}))
    with TestClient(app) as c:
        sign_up(c)
        water = next(p for p in c.get("/api/phrases").json() if p["text"] == "I want water")
        c.post("/api/templates", data={"phrase_id": water["id"]}, files={"audio": ("a.wav", word(WORDS[1]), "audio/wav")})
        r = c.post("/api/patient", data={"session_id": "s3"},
                   files={"audio": ("p.wav", word(WORDS[1], stretch=1.3, seed=5), "audio/wav")}).json()
        assert r["stt"]["skipped"] and r["candidates"][0]["text"] == "I want water"
        assert not any(p == "/v1/speech-to-text" for _, p in calls)
        assert c.get("/api/metrics").json()["credits"]["stt_calls_skipped"] == 1


# ---------------------------------------------------------------- learning as you talk
def test_confirmed_spoken_phrase_becomes_a_template(client):
    water = next(p for p in client.get("/api/phrases").json() if p["text"] == "I want water")
    pain = next(p for p in client.get("/api/phrases").json() if p["text"] == "I am in pain")
    for p, f in ((water, WORDS[1]), (pain, WORDS[2])):
        client.post("/api/templates", data={"phrase_id": p["id"]}, files={"audio": ("a.wav", word(f), "audio/wav")})
    client.post("/api/patient", data={"session_id": "s4"},
                files={"audio": ("p.wav", word(WORDS[1], stretch=1.3, seed=8), "audio/wav")})
    r = client.post("/api/confirm", json={"session_id": "s4", "text": "I want water", "phrase_id": water["id"],
                                          "sources": ["spectrogram"]}).json()
    assert r["learned_template"] is True
    assert next(p for p in client.get("/api/phrases").json() if p["id"] == water["id"])["n_templates"] == 2
    # a typed message (no recent audio for the session) teaches nothing
    r = client.post("/api/confirm", json={"session_id": "s4", "text": "I want water", "phrase_id": water["id"],
                                          "sources": ["spectrogram"]}).json()
    assert r["learned_template"] is False


def test_learned_sign_samples_are_capped_and_recorded_ones_kept(client):
    client.post("/api/signs", json={"label": "B", "samples": [[0.5] * 254] * 3})
    client.post("/api/signs", json={"label": "B", "samples": [[0.1] * 254] * 45, "learned": True})
    assert client.get("/api/signs/summary").json()[0]["n"] == 3 + 40


def test_time_of_day_prior_favours_phrases_said_at_this_hour(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    from backend.seed import seed_phrases
    seed_phrases(db)
    water = next(p for p in db.phrases() if p["text"] == "I want water")
    for _ in range(3):
        db.add_turn("s", "patient", "I want water")
    db.add_turn("s", "patient", "I am in pain")          # said once: too few to count
    assert db.phrase_time_prior() == {water["id"]: 1.0}


def test_partner_phone_feed_and_emergency(client):
    r = client.post("/api/partner", data={"session_id": "s5", "text": "Are you cold?"}).json()
    latest = client.get("/api/partner/latest", params={"session_id": "s5", "after": 0}).json()
    assert latest["turn_id"] == r["turn_id"] and latest["question"]["type"] == "yes_no"
    assert client.get("/api/partner/latest", params={"session_id": "s5", "after": r["turn_id"]}).json() == {}
    e = client.post("/api/emergency", data={"session_id": "s5", "trigger": "open palm"}).json()
    assert e["text"] == next(p for p in client.get("/api/phrases").json() if p["is_emergency"])["text"]


# ---------------------------------------------------------------- real-data WER script
def test_run_wer_reads_torgo_layout_and_measures_scribe(tmp_path):
    from eval import run_wer
    for spk, ref in (("F01", "the boy ran"), ("FC01", "hello")):
        d = tmp_path / spk / "Session1"
        (d / "wav_headMic").mkdir(parents=True)
        (d / "prompts").mkdir()
        (d / "wav_headMic" / "0001.wav").write_bytes(word(WORDS[1], seed=len(spk)))
        (d / "prompts" / "0001.txt").write_text(ref)
        (d / "wav_headMic" / "0002.wav").write_bytes(word(WORDS[1], seed=9))
        (d / "prompts" / "0002.txt").write_text("input/images/picture.jpg")   # not read speech: skipped
    items = run_wer.collect(tmp_path, limit=10)
    assert sorted((i["speaker"], i["group"]) for i in items) == [("F01", "dysarthric"), ("FC01", "control")]
    el = ElevenLabs(Database(str(tmp_path / "w.db")), api_key="key-one")
    calls = fake_elevenlabs(el, lambda i, req: httpx.Response(200, json={"text": "the boy", "words": []}))
    rows = asyncio.run(run_wer.evaluate(items, el))
    assert [p for _, p in calls].count("/v1/speech-to-text") == 2        # near-identical clips are not re-used
    assert next(r for r in rows if r["speaker"] == "F01")["wer"] == pytest.approx(1 / 3, abs=1e-4)
    assert "REAL data" in run_wer.report(rows)


# ---------------------------------------------------------------- accounts
def test_api_needs_login_and_accounts_are_separate(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.Settings.llm_enabled", property(lambda self: False))
    app = create_app(db_path=str(tmp_path / "core.db"), elevenlabs_key="")
    with TestClient(app) as a, TestClient(app) as b:
        assert a.get("/api/phrases").status_code == 401
        assert a.post("/api/auth/login", json={"email": "x@example.com", "password": "nope-nope"}).status_code == 400
        sign_up(a, "a@example.com")
        sign_up(b, "b@example.com")
        assert b.post("/api/auth/signup", json={"email": "a@example.com", "password": "long-enough-1",
                                                 "name": "Dup"}).status_code == 400
        a.post("/api/phrases", json={"text": "Call Amit"})
        assert any(p["text"] == "Call Amit" for p in a.get("/api/phrases").json())
        assert not any(p["text"] == "Call Amit" for p in b.get("/api/phrases").json())
        a.post("/api/auth/logout")
        assert a.get("/api/phrases").status_code == 401
        assert a.post("/api/auth/login", json={"email": "A@example.com", "password": "long-enough-1"}).status_code == 200


def test_partner_link_covers_one_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.Settings.llm_enabled", property(lambda self: False))
    app = create_app(db_path=str(tmp_path / "core.db"), elevenlabs_key="")
    with TestClient(app) as laptop, TestClient(app) as phone:
        sign_up(laptop)
        key = laptop.get("/api/partner/link", params={"session_id": "s1"}).json()["key"]
        assert phone.get("/api/history/s1").status_code == 401
        r = phone.post("/api/partner", params={"s": "s1", "k": key}, data={"session_id": "s1", "text": "Paani chahiye?"})
        assert r.status_code == 200 and r.json()["question"]["type"] == "yes_no"
        assert phone.get("/api/history/s1", params={"s": "s1", "k": key}).json()[0]["text"] == "Paani chahiye?"
        assert phone.post("/api/partner", params={"s": "s1", "k": key},
                          data={"session_id": "other", "text": "hi"}).status_code == 403
        assert phone.get("/api/history/s2", params={"s": "s2", "k": key}).status_code == 401
        assert phone.get("/api/phrases", params={"s": "s1", "k": key}).status_code == 401


def test_sqlite_statements_translate_to_postgres():
    from backend.db import _schema_to_pg, _to_pg
    sql, rid = _to_pg("INSERT INTO turns(session_id, speaker) VALUES(?,?)")
    assert sql.endswith("VALUES(%s,%s) RETURNING id") and rid
    sql, rid = _to_pg("INSERT OR REPLACE INTO voices(voice_id, name, kind) VALUES(?,?,?)")
    assert "ON CONFLICT (voice_id) DO UPDATE SET name=EXCLUDED.name, kind=EXCLUDED.kind" in sql and not rid
    assert _to_pg("INSERT OR IGNORE INTO key_usage(key_id, month) VALUES(?, ?)")[0].endswith("ON CONFLICT DO NOTHING")
    assert "LIKE '%%a'" in _to_pg("SELECT 1 WHERE x LIKE '%a'")[0]
    ddl = _schema_to_pg("id INTEGER PRIMARY KEY AUTOINCREMENT, a BLOB, b REAL")
    assert ddl == "id BIGSERIAL PRIMARY KEY, a BYTEA, b DOUBLE PRECISION"


# ---------------------------------------------------------------- guided intro
def words_with_pauses(freq_lists, gap=0.5, sr=16000):
    parts = []
    for i, f in enumerate(freq_lists):
        x, _ = dsp.read_wav(word(f, seed=40 + i, noise=0.002))
        parts += [x, np.zeros(int(gap * sr), np.float32)]
    return dsp.write_wav(np.concatenate(parts), sr)


def test_intro_splits_words_into_templates_and_profiles_voice(client):
    prompts = client.get("/api/intro/prompts").json()["prompts"]
    assert [p["say"] for p in prompts][:3] == ["Yes", "No", "Water"]
    use = prompts[:4]
    about = word([220, 240, 230, 250, 210], seg=0.3)   # a voiced, pitched "sentence"
    r = client.post("/api/intro", data={"phrase_ids": json.dumps([p["phrase_id"] for p in use])},
                    files={"words": ("w.wav", words_with_pauses(list(WORDS.values())), "audio/wav"),
                           "about": ("a.wav", about, "audio/wav")})
    assert r.status_code == 200, r.text
    out = r.json()
    assert [s["phrase_id"] for s in out["saved"]] == [p["phrase_id"] for p in use]
    assert out["profile"]["pitch_hz"] and 150 < out["profile"]["pitch_hz"] < 300
    assert client.get("/api/status").json()["matcher"] in ("robust", "ssl")
    # a later, slower "Water" is recognised from the intro alone
    q = client.post("/api/patient", data={"session_id": "i1", "use_llm": "false"},
                    files={"audio": ("q.wav", word(WORDS[3], stretch=1.4, seed=77), "audio/wav")}).json()
    assert q["spectrogram_matches"][0]["phrase_id"] == use[2]["phrase_id"]
    # too few words heard -> clear error, nothing saved
    bad = client.post("/api/intro", data={"phrase_ids": json.dumps([p["phrase_id"] for p in prompts])},
                      files={"words": ("w.wav", words_with_pauses(list(WORDS.values())[:2]), "audio/wav")})
    assert bad.status_code == 422


def test_intro_name_suggestions():
    from backend import intro
    assert intro.suggestions("My name is Ravi. My son Amit lives in Pune.", ["Yes"]) == \
        ["Please call Ravi", "Please call Amit", "Please call Pune"]
