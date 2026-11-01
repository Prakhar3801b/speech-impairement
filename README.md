# Clio: multimodal conversation assistant for people with speech impairments (PS16)

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
pip install -r requirements.txt
cp .env.example .env          # add ELEVENLABS_API_KEY, optionally ANTHROPIC_API_KEY and DATABASE_URL (Neon)
python -m scripts.build_speech_model --download   # optional, ~95 MB: WavLM speech model for matching
uvicorn backend.main:app --reload --port 8000
# open http://localhost:8000, create an account (Chrome/Edge recommended; camera + mic permission needed)
```

**Accounts.** The site opens on a sign-in page. Each account (a patient, or a caregiver setting Clio up for one) has its
own phrasebook, voice recordings, voices, consent records and history. Passwords are hashed with scrypt; the login is an
HttpOnly cookie. The partner's phone gets a signed link (in the QR code) that covers one conversation only.

**Database.** Without `DATABASE_URL`, data is in local SQLite files (`data/app.db` for accounts, `data/users/u_<id>.db`
per account). With a Neon `DATABASE_URL` everything goes to Postgres: accounts in `public`, each account's data in its
own schema `u_<id>`, so one patient's rows can never appear in another's queries. Use Neon's **direct** connection
string (host without `-pooler`), because each account's connection sets its schema once.

**Speech model.** `--download` fetches a prebuilt int8 WavLM Base+ (12 layers). With torch and transformers installed
(`pip install -r requirements-train.txt`), `python -m scripts.build_speech_model` instead exports only the first 6 layers
(52 MB, about half the run time). Without a model, matching uses the noise-robust spectral features.

Without keys the app runs in **mock mode**. Spectrogram matching, ISL signs, cues and the heuristic ranker all still work,
and speech output falls back to the browser's speech synthesis. Without Scribe, type the partner's words.

The browser loads MediaPipe from `cdn.jsdelivr.net` and the models from `storage.googleapis.com`.

### Demo flow
1. **Setup** (opens on first visit, 4 steps): name and language; how the patient says "yes" (nod, thumbs up,
   tap/key, or auto-scan); **"Let Clio learn your voice"**; which voice to use. "Start talking" turns on the camera and microphone.
   The voice step takes about a minute: (1) optional free speech about themselves, which gives a voice profile (pitch,
   pace, room noise) and, through one Scribe call, names to add as phrases ("Please call Amit"); (2) six words said in one
   recording with a pause after each (Yes, No, Water, Pain, Toilet, Help, or the Hindi forms). The server splits that
   recording at the pauses and saves each word as a template, so matching works from the first conversation.
2. **Talk:** press **O** while the partner talks, or let the partner use their phone (below). Suggested replies appear
   at once. For yes/no and either/or questions ("Paani thanda ya normal?") the two answers are big cards side by side.
   Press **P** while the patient speaks, signs or nods. Choose with the "yes" movement from setup, **1–3**, a click or **Enter**.
3. **Teach phrases / Teach signs** (optional): record a phrase 3 times, or a sign 2–3 times. You can also skip this,
   because Clio learns from use: every confirmed spoken phrase becomes one more template for that phrase (up to 4 learned
   per phrase), and the hand shapes of confirmed signs become extra sign samples (up to 40 learned per sign).
4. **Help:** the red Help button, an open palm held to the camera for 2.5 s, or two quick eyebrow raises show a
   full-screen alert and speak the emergency phrase. The gestures can be switched off under Talk → Options.
5. **Insights:** latency per stage (p50/p95), confirmation effort, cache hits, credits per API key and minutes saved.

### Partner's phone
Talk → Conversation → **Partner phone** shows a QR code. The partner's phone gets a hold-to-talk button and large
captions of what the patient says; turns spoken there show up on the laptop with reply suggestions.
* Start the server with `--host 0.0.0.0` so the phone can reach it, and keep both devices on the same Wi-Fi.
* Phones only allow the microphone on **https**. Over plain http the phone still shows captions and accepts typed words.
  For the microphone, serve over https (for example `uvicorn ... --ssl-keyfile key.pem --ssl-certfile cert.pem`, or a tunnel).
* The API has no login. Only expose it on a network you trust, and stop the server after the session.

### Suggestions that fit the moment
* **Question-shaped:** yes/no questions answer Yes/No (Haan/Nahi for Hindi); either/or questions offer the two options
  in the partner's own words.
* **Time of day:** phrases the patient says at this time of day (night, morning, afternoon, evening) rank higher,
  once they have been said there at least twice.

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

### Phrase matching (`backend/matcher.py`)
The patient's clip is matched against their own recordings with dynamic time warping, using the best features available:
1. **WavLM speech-model embeddings** (`backend/embed.py`, ONNX Runtime on the CPU) when the model file is present;
2. otherwise **noise-robust cepstra** (`dsp.robust_cepstra`): spectral noise subtraction, PNCC-style power-law
   compression, plus energy/voicing/pitch tracks;
3. plain MFCC only for templates saved before their audio was kept.

Template audio is stored, so features are recomputed when the front-end or model changes. Once a phrase has 5+
recordings, a per-phrase prototype (average clip vector) is blended into the score. Confidence temperatures were
calibrated with `eval/run_matching.py` (the old setting made confidences so flat that Scribe was almost never skipped).

```bash
python -m eval.make_synthetic                                   # Windows: TTS voices + simulated distortion and noise
python -m eval.run_matching eval/data/synthetic --model models/wavlm-base-plus-l6.int8.onnx
python -m eval.run_matching path/to/UA-Speech --k 2             # real dysarthric speech (registration required)
```
On the synthetic set, noise-robust cepstra reached 88.9% top-1 against 81.2% for MFCC (48 words, 144 turns).
That is a comparison of front-ends on synthetic audio, not a prediction for patients.

### Database (one store per account: SQLite files or Neon Postgres schemas)
| Table | Contents |
|---|---|
| `tts_cache` | Generated speech audio keyed by sha256(text, voice, model), with hit counts |
| `stt_cache` | Scribe transcripts keyed by sha256(audio), with word timings and confidences |
| `phrases` | Personal phrasebook (En/Hi/Hinglish), with aliases, context keywords, emergency flag and usage count |
| `audio_templates` | Enrolled templates: trimmed audio, MFCC + log-mel, cached speech-model embeddings |
| `training_clips` | Opt-in only: confirmed utterances with their text, for training a personal recogniser |
| `sign_samples` | ISL sign training samples (254-dimension two-hand landmark features) |
| `consent_records`, `voices` | Voice-cloning consent and the voices created under it |
| `turns`, `events` | Conversation history; telemetry (latency, effort, credit usage, consent audit) |
| `settings` | Selected voice, voice profile from the intro, etc. |
| `users`, `auth_sessions` (core) | Accounts (scrypt password hashes) and login sessions (token hashes only) |

### ElevenLabs free tier
The free tier is roughly **10k credits a month** with **2 concurrent requests**, no Instant Voice Cloning, and
non-commercial use with attribution. Check the current pricing page. The app is built around these limits:
* every TTS clip and transcript is cached, so repeats are free; **Phrasebook → Pre-generate** spends credits once per phrase;
* a semaphore limits ElevenLabs calls to `ELEVENLABS_CONCURRENCY` (default 2);
* Flash v2.5 is the live TTS model, because it costs the fewest credits and has the lowest latency;
* **Voice Design** works on Free. Cloning is built in, but it needs a paid plan **and** an active consent record;
* the Insights tab tracks characters and minutes sent, and minutes saved.

**Keeping Scribe inside the free tier.** Before audio goes to Scribe it passes four cheaper checks:
1. *Exact cache*: the same audio bytes are never transcribed twice.
2. *Skip on a clear local match*: the patient's speech is first matched against their recorded phrases
   (spectrogram + DTW, free). If one phrase matches with confidence of at least `SKIP_STT_CONFIDENCE`, Scribe is not called.
   In saver mode (under `SAVER_MODE_BELOW` credits left) the bar drops to 0.6.
3. *Near-identical clip*: each transcribed clip keeps an MFCC fingerprint. A new clip within `STT_REUSE_MAX_DIST`
   (DTW) of an earlier one, and clearly closer to it than to any clip with a different text, re-uses that transcript.
4. *Trim and cap*: silence is cut from both ends and clips are capped at `MAX_STT_SECONDS`, so fewer seconds are billed.

**Several keys.** Put extra keys in `ELEVENLABS_API_KEYS` (comma separated). Keys are used in order; a key is skipped
when its estimated use reaches its limit or when ElevenLabs reports its quota as used up. Use is estimated locally and
corrected from `GET /v1/user/subscription` when the key can read it. A designed or cloned voice belongs to one account,
so speech in that voice always uses that account's key. When every key is out, speech falls back to the browser's
voice and the partner's words can be typed. Keys are never stored; the database keeps only a short hash of each.

### Consent (cloning)
`POST /api/voices/clone` refuses (403) unless it references an active consent record. To create one, the signer must
type their name to match, tick agreement, and give a relationship (self / guardian / legal representative / caregiver).
Revoking consent deletes the voice at ElevenLabs and its cached audio. Every action goes to the audit log.

## Evaluation
```bash
python -m eval.run_eval      # writes eval/reports/results.{md,json}
python -m pytest -q          # unit + API tests (mock mode)
```
**Real recordings (TORGO, UA-Speech, your own clips):**
```bash
python -m eval.run_wer path/to/TORGO --max 40 --dry-run   # counts clips and audio minutes, sends nothing
python -m eval.run_wer path/to/TORGO --max 40              # writes eval/reports/real_wer.{md,json}
```
It reads TORGO's layout (`Speaker/Session/wav_*/0001.wav` with `prompts/0001.txt`) or a transcript next to each clip,
skips prompts that are not read speech, reports dysarthric and control speakers separately, and lists the clips
Scribe gets most wrong. Requests go through the same budget and caches as the app, so re-runs are free; transcript
re-use between similar clips is switched off here so the numbers measure Scribe itself.

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
* Open-palm and double-eyebrow emergency gestures, nod-to-confirm and auto-scan timing (2 s) need tuning with a real
  camera and the real user. Any ISL sign that is an open palm held still will also trigger the alert after 2.5 s.
* Scribe credit cost per minute (`STT_CREDITS_PER_MINUTE`) is an estimate, and the transcript re-use distance
  (`STT_REUSE_MAX_DIST`) was tuned on synthetic audio. Check both against real speech and your plan.
* Live (streaming) Scribe is not used: streamed audio cannot be cached or re-used, so it would spend the free-tier
  minutes much faster. Recognition runs when a turn ends.
* Neon/Postgres support is unit-tested for its SQL translation only; no Postgres server was available here. Run the
  app once against your Neon `DATABASE_URL` and check the tables appear (`public.users`, `u_1.phrases`, ...).
* Accounts have no password reset, email verification or login rate limiting yet. Add these before a public launch.
* The WavLM matcher has not been measured against the spectral features yet (run `eval/run_matching.py` with `--model`),
  and the intro's word splitting relies on clear pauses; very breathy or continuous speech may need re-recording.
* Not built yet: seeding the sign classifier from `data/ISL_Dataset` (needs MediaPipe for Python, which is not installed
  in this environment), expressive v3 speech from facial affect, spoken consent, and several patient profiles per device.
