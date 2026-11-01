"""Guided voice intro: one short session at setup that replaces most of "Teach phrases".

Part 1, "tell us about yourself" (free speech): gives a voice profile (pitch, loudness, pace,
room noise) and, through one Scribe call, names and places for the phrasebook.
Part 2, "say each word with a pause": one recording, split at the pauses; each piece becomes
a template for its phrase, so the matcher works from the first conversation.
"""
from __future__ import annotations

import re

import numpy as np

from . import dsp

# What the patient is asked to say (short forms are easier), and the phrase each one stands for.
PROMPTS = [
    {"say": "Yes", "hi": "Haan", "phrase": "Yes"},
    {"say": "No", "hi": "Nahi", "phrase": "No"},
    {"say": "Water", "hi": "Paani", "phrase": "I want water"},
    {"say": "Pain", "hi": "Dard", "phrase": "I am in pain"},
    {"say": "Toilet", "hi": "Bathroom", "phrase": "I need to use the toilet"},
    {"say": "Help", "hi": "Madad", "phrase": "Help! Please call the nurse now"},
]
MIN_GAP_S = 0.25      # silences shorter than this are inside a word
MIN_WORD_S = 0.12
PAD_S = 0.08


def prompts(phrases: list[dict], language: str = "en") -> list[dict]:
    by_text = {p["text"]: p for p in phrases}
    out = []
    for p in PROMPTS:
        if p["phrase"] in by_text:
            out.append({"say": p["hi"] if language.startswith("hi") else p["say"],
                        "phrase_id": by_text[p["phrase"]]["id"], "phrase": p["phrase"]})
    return out


def _frame_db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.mean(dsp.frames(x) ** 2, axis=1) + 1e-10)


def segment(x: np.ndarray, expected: int) -> list[np.ndarray]:
    """Split a recording of words said with pauses into `expected` pieces.

    Speech frames are those well above the room's noise floor. Short gaps are bridged, specks
    dropped; if there are still too many pieces, the shortest pauses are joined first (a
    dysarthric speaker may pause inside a word). Returns [] if fewer words than expected were found."""
    db = _frame_db(x)
    floor, peak = np.percentile(db, 10), db.max()
    voiced = db > max(floor + 10, peak - 35)
    runs, start = [], None
    for i, v in enumerate(np.append(voiced, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    hop_s = dsp.HOP / dsp.SR
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * hop_s < MIN_GAP_S:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    merged = [r for r in merged if (r[1] - r[0]) * hop_s >= MIN_WORD_S]
    while len(merged) > expected:
        gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
        i = int(np.argmin(gaps))
        merged[i:i + 2] = [[merged[i][0], merged[i + 1][1]]]
    if len(merged) < expected:
        return []
    pad = int(PAD_S / hop_s)
    return [x[max(0, a - pad) * dsp.HOP:min(len(db), b + pad) * dsp.HOP + dsp.WIN] for a, b in merged]


def voice_profile(x: np.ndarray) -> dict:
    """Pitch, loudness, pace and room noise of free speech (the intro's first part)."""
    db = _frame_db(x)
    floor, peak = float(np.percentile(db, 10)), float(db.max())
    speech = db > max(floor + 10, peak - 35)
    f0, voicing = dsp.pitch_track(x)
    voiced_f0 = f0[(voicing > 0.45) & speech]
    # pace: bursts of energy (roughly syllables) per second of speech
    smooth = np.convolve(db, np.ones(5) / 5, mode="same")
    peaks = np.sum((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]) & speech[1:-1]
                   & (smooth[1:-1] > floor + 15))
    speech_s = float(speech.sum() * dsp.HOP / dsp.SR)
    return {
        "seconds": round(len(x) / dsp.SR, 1),
        "speech_seconds": round(speech_s, 1),
        "pitch_hz": round(float(np.median(voiced_f0)), 0) if len(voiced_f0) else None,
        "pitch_range_hz": [round(float(np.percentile(voiced_f0, 10))), round(float(np.percentile(voiced_f0, 90)))]
        if len(voiced_f0) else None,
        "syllables_per_s": round(float(peaks) / speech_s, 1) if speech_s > 0.5 else None,
        "noise_db": round(floor, 1),
        "snr_db": round(float(np.median(db[speech])) - floor, 1) if speech.any() else None,
    }


STOP = {"I", "I'm", "My", "Me", "The", "A", "An", "And", "Hello", "Hi", "Namaste", "Is", "Am", "We", "He", "She",
        "This", "That", "Mera", "Meri", "Main", "Hai", "Hoon", "Naam", "Ji", "Yes", "No", "Okay", "OK"}


def suggestions(transcript: str, existing: list[str]) -> list[str]:
    """Names and places in the intro (capitalised words mid-sentence) as phrase ideas."""
    have = {e.lower() for e in existing}
    names = []
    for sent in re.split(r"[.!?।]+", transcript):
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", sent)
        for w in words[1:]:
            if w[0].isupper() and w not in STOP and w not in names:
                names.append(w)
    out = [f"Please call {n}" for n in names[:6]]
    return [o for o in out if o.lower() not in have]
