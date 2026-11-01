# Fusion ablation (SIMULATED data)

270 simulated turns (30 dialogues x 3 severities x 3 seeds). Patient ASR text, cues, spelling and template scores are synthetic; see eval/run_eval.py.

| Condition | WER | CER | Intent top-1 | Intent top-3 | WER mild | WER moderate | WER severe | ranker ms |
|---|---|---|---|---|---|---|---|---|
| asr_only (no fusion) | 0.580 | 0.293 | 15.2% | 15.2% | 0.316 | 0.579 | 0.846 | 0.00 |
| + conversation context | 0.403 | 0.358 | 64.8% | 90.0% | 0.396 | 0.343 | 0.469 | 43.37 |
| + face/head/lip cues | 0.276 | 0.251 | 74.4% | 94.1% | 0.242 | 0.271 | 0.315 | 43.67 |
| + spectrogram templates | 0.020 | 0.017 | 97.4% | 99.6% | 0.000 | 0.017 | 0.044 | 47.35 |
| + ISL spelling (full) | 0.014 | 0.012 | 98.1% | 99.6% | 0.000 | 0.008 | 0.034 | 48.36 |
