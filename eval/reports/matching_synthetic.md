# Phrase matching: front-end comparison (SYNTHETIC)

Data `eval\data\synthetic` (generic), 2 speakers, 48 words, K=2 enrolled recordings per word, 144 live turns. 2026-11-01.

> Synthetic: Windows TTS voices with simulated dysarthria-like distortion and noise.
> Use it to compare front-ends, not to predict accuracy for patients.

| Front-end | Top-1 | Top-3 | Match time | Best temperature | Scribe skipped (conf >= 0.85) | Correct when skipped |
|---|---|---|---|---|---|---|
| mfcc | 81.2% | 90.3% | 96 ms | 0.015 | 66% | 96% |
| robust | 88.9% | 96.5% | 83 ms | 0.01 | 77% | 96% |
| robust+prosody | 88.9% | 96.5% | 105 ms | 0.01 | 75% | 96% |

Best temperature: the softmax temperature with the most Scribe skips while skipped turns stay >= 95% correct (set MATCH_TEMPERATURE_* in .env).
