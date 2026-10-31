# Saathi: multimodal conversation assistant for people with speech impairments (PS16)

Prototype for **AI Manthan 2.0, Problem Statement 16**. The app listens to the conversation partner and to the patient,
and watches the patient's hands and face. It combines everything into 3 suggested messages, and speaks the one the
patient confirms in a personalised ElevenLabs voice.

| Signal | How | Where it runs |
|---|---|---|
| Patient & partner speech | ElevenLabs **Scribe** speech-to-text (English, Hindi, code-mixed) | ElevenLabs API (cached) |
| **Spectrogram matching** | log-mel / MFCC features + dynamic time warping against the patient's own enrolled recordings | Backend, numpy, no credits |
| **ISL static letters & signs** | MediaPipe hand landmarks (both hands), then a k-NN classifier trained in the app | Browser |
| Face & head | MediaPipe face blendshapes: expression (smile/frown/pain); head nod/shake; lip-activity syllable count | Browser |
| Context | Last partner utterance + history + personal phrasebook | Backend |
| Fusion | Fast path, then weighted heuristic ranker, then optional LLM re-ranker (Claude, structured JSON) | Backend |
| Voice | ElevenLabs **Flash v2.5** TTS; **Voice Design** (free tier) or consent-gated **Instant Voice Clone** | ElevenLabs API (cached) |

Video never leaves the browser; only small feature vectors and audio clips are sent to the backend.

## Quick start

```bash
cd ps16-voice-assistant
pip install -r requirements.txt
cp .env.example .env          # add ELEVENLABS_API_KEY (and optionally ANTHROPIC_API_KEY)
uvicorn backend.main:app --reload --port 8000
# open http://localhost:8000  (Chrome/Edge recommended; camera + mic permission needed)
```

Without keys the app runs in **mock mode**. Spectrogram matching, ISL signs, cues and the heuristic ranker all still work,
and speech output falls back to the browser's speech synthesis. Without Scribe, type the partner's words.

The browser loads MediaPipe from `cdn.jsdelivr.net` and the models from `storage.googleapis.com`.

### Demo flow
1. **Converse → Start camera & mic.**
2. **Spectrogram phrases:** pick a phrase and record it 3 times in the patient's own voice.
3. **ISL signs:** record letters (A–Z, 0–9), a few phrase signs (e.g. an emergency "help" sign) and the controls
   SPACE / DEL / SEND. Use 2–3 recordings per sign.
4. **Converse:** press **O** (or 🗣) while the partner talks; suggested replies appear immediately. Press **P** (or 🎙)
   while the patient speaks, signs or nods. Choose with **1–3**, a click, **Enter**, or by holding 👍 (👎 moves to the next suggestion).
5. **Metrics:** latency per stage (p50/p95), confirmation effort, cache hits and ElevenLabs usage.

## Architecture

```
Browser (frontend/)                                Backend (backend/, FastAPI)
 ├─ audio.js   mic → 16 kHz WAV, live FFT view  ──►  /api/partner  Scribe STT → context → reply suggestions
 ├─ vision.js  MediaPipe hands + face           ──►  /api/patient  ┌ Scribe STT ───────────┐ (parallel)
 │             → ISL features, thumbs,                             └ spectrogram DTW match ┘
 │               expression, nod/shake, syllables                    → fusion.fast_path / heuristic_rank
 ├─ signs.js   k-NN + stabiliser → letters/signs                     → llm.rank (optional, falls back)
 └─ app.js     UI, confirmation, effort logging ──►  /api/confirm, /api/speak (TTS cache → ElevenLabs)
                                                     /api/consent, /api/voices/*  (consent gate, audit)
                                                     SQLite: data/app.db
```

### Database (`data/app.db`, SQLite)
| Table | Contents |
|---|---|
| `tts_cache` | Generated speech audio keyed by sha256(text, voice, model), with hit counts |
| `stt_cache` | Scribe transcripts keyed by sha256(audio), with word timings and confidences |
| `phrases` | Personal phrasebook (En/Hi/Hinglish), with aliases, context keywords, emergency flag and usage count |
| `audio_templates` | Enrolled spectrogram templates (MFCC + log-mel as float32 blobs) |
| `sign_samples` | ISL sign training samples (254-dimension two-hand landmark features) |
| `consent_records`, `voices` | Voice-cloning consent and the voices created under it |
| `turns`, `events` | Conversation history; telemetry (latency, effort, credit usage, consent audit) |
| `settings` | Selected voice etc. |

### ElevenLabs free tier
The free tier is roughly **10k credits a month** with **2 concurrent requests**, no Instant Voice Cloning, and
non-commercial use with attribution. Check the current pricing page. The app is built around these limits:
* every TTS clip and transcript is cached, so repeats are free; **Phrasebook → Pre-generate** spends credits once per phrase;
* a semaphore limits ElevenLabs calls to `ELEVENLABS_CONCURRENCY` (default 2);
* Flash v2.5 is the live TTS model, because it costs the fewest credits and has the lowest latency;
* **Voice Design** works on Free. Cloning is built in, but it needs a paid plan **and** an active consent record;
* the Metrics tab tracks characters and minutes sent.

### Consent (cloning)
`POST /api/voices/clone` refuses (403) unless it references an active consent record. To create one, the signer must
type their name to match, tick agreement, and give a relationship (self / guardian / legal representative / caregiver).
Revoking consent deletes the voice at ElevenLabs and its cached audio. Every action goes to the audit log.

## Evaluation
```bash
python -m eval.run_eval      # writes eval/reports/results.{md,json}
python -m pytest -q          # unit + API tests (mock mode)
```
`eval/run_eval.py` runs an ablation of the fusion ranker on a **SIMULATED** dataset. The dysarthric ASR errors, cues,
ISL spelling and template scores are all synthetic, generated at 3 severities. Current output
(see `eval/reports/results.md`):

| Condition (SIMULATED) | WER | Intent top-1 | Intent top-3 |
|---|---|---|---|
| ASR only (no fusion) | 0.580 | 15.2% | 15.2% |
| + conversation context | 0.403 | 64.8% | 90.0% |
| + face/head/lip cues | 0.276 | 74.4% | 94.1% |
| + spectrogram templates | 0.020 | 97.4% | 99.6% |
| + ISL spelling (full) | 0.014 | 98.1% | 99.6% |

These numbers show that the fusion logic works. They are **not** real-world accuracy: the simulated template scores
are optimistic, and context alone *raises* WER on mild speech, which needs tuning. Before final submission, run TORGO / UA-Speech
clips through Scribe and the spectrogram matcher, and run live sessions with team members (labelled as acted).

## Known limitations / next steps
* MediaPipe in the browser has been checked for syntax, and the feature/classifier logic is unit-tested in Node. Live camera
  tracking still needs a run on a real machine, and the expression/nod thresholds will need tuning there.
* The ISL letter classifier is speaker-specific: you train it in the app, because no pretrained ISL model ships with it. Dynamic
  ISL letters (with movement) are not supported.
* The ElevenLabs endpoint shapes (Scribe, Voice Design, IVC) follow the public REST API, but have not been exercised with a
  live key in this environment. Check them on day 0.
* The LLM re-ranker is optional. Measure its latency against the 2 s budget, and set `LLM_TIMEOUT_S` to match.
