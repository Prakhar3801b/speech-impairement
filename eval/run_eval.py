"""Offline ablation of the fusion ranker on a SIMULATED conversational dataset.

Everything in this dataset is simulated and labelled as such:
  * patient "ASR output" = the intended message corrupted with dysarthria-like
    errors (consonant blurring/dropping, merged and deleted words) at three severities;
  * face/head cues, ISL fingerspelling prefixes and spectrogram-template confidences
    are sampled from simple noise models.
It measures how much each extra signal helps the ranker. It is NOT a substitute
for testing on real atypical speech (TORGO / UA-Speech through Scribe) and real users.

Usage:  python -m eval.run_eval  [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

from backend.fusion import Cues, FusionInput, heuristic_rank, normalize
from backend.metrics import cer, wer
from backend.seed import PHRASES

OUT = Path(__file__).resolve().parent / "reports"

DIALOGUES = [  # (partner says, patient intends)
    ("Do you want some water?", "Yes"),
    ("Are you feeling cold?", "No"),
    ("Kya aapko paani chahiye?", "Haan"),
    ("Kya aap so rahe ho?", "Nahi"),
    ("What would you like to drink?", "I want water"),
    ("Kya peena hai, paani ya chai?", "Mujhe paani chahiye"),
    ("Should I make some tea?", "I would like tea, please"),
    ("Chai peeni hai?", "Chai chahiye"),
    ("It is almost lunch time, are you hungry?", "I am hungry"),
    ("Khana kha lo, bhookh lagi hai?", "Mujhe bhookh lagi hai"),
    ("Do you need anything?", "I need to use the toilet"),
    ("Kuch chahiye aapko?", "Mujhe bathroom jaana hai"),
    ("You look tired. Want to rest?", "I am tired, I want to rest"),
    ("Is it too warm in here?", "Please turn on the fan"),
    ("Are you comfortable lying like that?", "Please change my position"),
    ("Do you want me to call someone?", "Can you call my family?"),
    ("How are you feeling today?", "I am in pain"),
    ("Aap kaisa feel kar rahe ho? Tabiyat kaisi hai?", "Mujhe dard ho raha hai"),
    ("Where does it hurt?", "The pain is in my chest"),
    ("Where is the pain?", "The pain is in my head"),
    ("Did you take your medicine?", "I need my medicine"),
    ("How are you feeling now?", "I feel better now"),
    ("How are you? Kaise ho?", "Main theek hoon"),
    ("Hello! Good morning!", "Hello, how are you?"),
    ("Here is your blanket.", "Thank you"),
    ("How are you doing?", "I am fine"),
    ("What happened? Are you okay?", "I feel dizzy"),
    ("", "Help! Please call the nurse now"),
    ("", "I can't breathe"),
    ("Did you say something?", "Please repeat that"),
]

SEVERITY = {"mild": 0.15, "moderate": 0.35, "severe": 0.6}
_BLUR = {"p": "b", "b": "p", "t": "d", "d": "t", "k": "g", "g": "k", "s": "", "f": "p", "v": "b",
         "ch": "s", "sh": "s", "th": "d", "r": "", "l": "", "n": "", "m": "b"}


def degrade(text: str, p: float, rng: random.Random) -> str:
    """Simulated dysarthric ASR output (SIMULATED)."""
    words = []
    for w in normalize(text).split():
        if rng.random() < p * 0.35:          # word dropped entirely
            continue
        chars, i, out = w, 0, ""
        while i < len(chars):
            two = chars[i:i + 2]
            if two in _BLUR and rng.random() < p:
                out += _BLUR[two]; i += 2; continue
            c = chars[i]
            if c in _BLUR and rng.random() < p:
                out += _BLUR[c]
            elif c in "aeiou" and rng.random() < p * 0.3:
                out += rng.choice("aeiou")
            else:
                out += c
            i += 1
        if out:
            words.append(out)
    if len(words) > 1 and rng.random() < p * 0.5:  # merged words
        k = rng.randrange(len(words) - 1)
        words[k:k + 2] = [words[k] + words[k + 1]]
    return " ".join(words)


def simulate_cues(intended: dict, severity: float, rng: random.Random) -> Cues:
    text = intended["text"].lower()
    head = thumbs = None
    if text in ("yes", "haan") and rng.random() < 0.7:
        head = "nod"
    if text in ("no", "nahi") and rng.random() < 0.7:
        head = "shake"
    expression = "neutral"
    if intended["category"] in ("medical", "emergency") and rng.random() < 0.6:
        expression = "pain"
    if intended["category"] == "social" and rng.random() < 0.5:
        expression = "smile"
    syl = max(1, len(re.findall(r"[aeiouy]+", text)) + rng.choice([-1, 0, 0, 1]))
    return Cues(head=head, expression=expression, thumbs=thumbs, syllables=syl)


def simulate_templates(intended_id: int, phrase_ids: list[int], severity: float,
                       rng: random.Random) -> list[dict]:
    """Simulated spectrogram-template confidences: correct phrase usually, not always, on top."""
    true_conf = max(0.05, min(0.95, rng.gauss(0.65 - 0.3 * severity, 0.18)))
    distractors = rng.sample([i for i in phrase_ids if i != intended_id], 3)
    rest = 1 - true_conf
    weights = [rng.random() for _ in distractors]
    s = sum(weights)
    out = [{"phrase_id": intended_id, "confidence": true_conf}]
    out += [{"phrase_id": d, "confidence": rest * w / s} for d, w in zip(distractors, weights)]
    return sorted(out, key=lambda m: -m["confidence"])


def simulate_spelling(intended: dict, rng: random.Random) -> str:
    """Simulated ISL fingerspelling: first 2-4 letters of the most distinctive word, 50% of the time."""
    if rng.random() > 0.5:
        return ""
    words = [w for w in normalize(intended["text"]).split() if len(w) > 3] or normalize(intended["text"]).split()
    word = max(words, key=len)
    return word[: rng.randint(2, min(4, len(word)))]


CONDITIONS = {
    "asr_only (no fusion)": dict(context=False, cues=False, templates=False, signs=False),
    "+ conversation context": dict(context=True, cues=False, templates=False, signs=False),
    "+ face/head/lip cues": dict(context=True, cues=True, templates=False, signs=False),
    "+ spectrogram templates": dict(context=True, cues=True, templates=True, signs=False),
    "+ ISL spelling (full)": dict(context=True, cues=True, templates=True, signs=True),
}


def main(seeds: int = 3) -> dict:
    phrases = []
    for i, p in enumerate(PHRASES, 1):
        phrases.append({**p, "id": i, "use_count": 0, "aliases": p.get("aliases", []),
                        "keywords": p.get("keywords", []), "is_emergency": p.get("is_emergency", False)})
    by_text = {normalize(p["text"]): p for p in phrases}
    ids = [p["id"] for p in phrases]

    samples = []
    for seed in range(seeds):
        rng = random.Random(seed)
        for sev_name, sev in SEVERITY.items():
            for partner, intended_text in DIALOGUES:
                intended = by_text[normalize(intended_text)]
                samples.append(dict(
                    severity=sev_name, partner=partner, intended=intended,
                    asr=degrade(intended["text"], sev, rng),
                    cues=simulate_cues(intended, sev, rng),
                    templates=simulate_templates(intended["id"], ids, sev, rng),
                    spelled=simulate_spelling(intended, rng)))

    results = {}
    for name, c in CONDITIONS.items():
        rows = []
        for s in samples:
            inp = FusionInput(
                asr_text=s["asr"], partner_last=s["partner"] if c["context"] else "",
                cues=s["cues"] if c["cues"] else Cues(),
                template_matches=s["templates"] if c["templates"] else [],
                spelled=s["spelled"] if c["signs"] else "")
            t0 = time.perf_counter()
            if name.startswith("asr_only"):
                top = [s["asr"]]
            else:
                top = [x["text"] for x in heuristic_rank(inp, phrases)]
            ms = (time.perf_counter() - t0) * 1000
            ref = s["intended"]["text"]
            rows.append(dict(severity=s["severity"], wer=wer(ref, top[0] if top else ""),
                             cer=cer(ref, top[0] if top else ""),
                             top1=bool(top) and normalize(top[0]) == normalize(ref),
                             top3=any(normalize(t) == normalize(ref) for t in top[:3]), ms=ms))
        summary = {"all": _agg(rows)}
        for sev in SEVERITY:
            summary[sev] = _agg([r for r in rows if r["severity"] == sev])
        results[name] = summary

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"dataset": "SIMULATED", "n_samples": len(samples), "results": results}, indent=2))
    md = ["# Fusion ablation (SIMULATED data)", "",
          f"{len(samples)} simulated turns ({len(DIALOGUES)} dialogues x 3 severities x {seeds} seeds). "
          "Patient ASR text, cues, spelling and template scores are synthetic; see eval/run_eval.py.", "",
          "| Condition | WER | CER | Intent top-1 | Intent top-3 | WER mild | WER moderate | WER severe | ranker ms |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, s in results.items():
        a = s["all"]
        md.append(f"| {name} | {a['wer']:.3f} | {a['cer']:.3f} | {a['top1']:.1%} | {a['top3']:.1%} | "
                  f"{s['mild']['wer']:.3f} | {s['moderate']['wer']:.3f} | {s['severe']['wer']:.3f} | {a['ms']:.2f} |")
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return results


def _agg(rows: list[dict]) -> dict:
    return {"n": len(rows), "wer": statistics.mean(r["wer"] for r in rows),
            "cer": statistics.mean(r["cer"] for r in rows),
            "top1": statistics.mean(r["top1"] for r in rows), "top3": statistics.mean(r["top3"] for r in rows),
            "ms": statistics.mean(r["ms"] for r in rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    main(ap.parse_args().seeds)
