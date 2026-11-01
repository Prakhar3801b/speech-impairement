"""Compare phrase-matching front-ends on repeated recordings of the same words.

The protocol copies how Clio is used: for each speaker, the first K recordings of a word are the
enrolled templates, every later recording is a live turn to recognise among that speaker's words.

Front-ends compared:
  mfcc            log-mel MFCC + DTW (the original matcher)
  robust          PNCC-style cepstra with spectral noise subtraction + DTW
  robust+prosody  the same plus energy / voicing / pitch tracks
  ssl:<model>     speech-model embeddings + DTW (one row per --model given)
  ssl:<model>+proto   embeddings + DTW blended with per-word prototypes (needs K >= 2)

Layouts (auto-detected, or --layout):
  generic   ROOT/<speaker>/<word>/*.wav                      (eval.make_synthetic writes this)
  uaspeech  ROOT/**/<SPK>_B<n>_<WORD>_M<mic>.wav               (blocks B1..B3 are the repeats)
  torgo     ROOT/<SPK>/<Session>/wav_*/<n>.wav + prompts/<n>.txt  (single-word prompts only)

Usage:  python -m eval.run_matching ROOT [--k 2] [--model models/x.onnx ...] [--max-words 50]
Writes: eval/reports/matching.{md,json}
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend import dsp
from backend.embed import Embedder, clip_vector, for_dtw
from backend.matcher import blend_prototypes, robust_track

OUT = Path(__file__).resolve().parent / "reports"
SKIP_AT = 0.85   # SKIP_STT_CONFIDENCE default: how often would Scribe be skipped, and how safely?


# ---------------------------------------------------------------- loaders
def load_generic(root: Path) -> list[dict]:
    return [{"speaker": p.relative_to(root).parts[0], "label": p.parent.name, "path": p}
            for p in sorted(root.rglob("*.wav")) if len(p.relative_to(root).parts) >= 3]


def load_uaspeech(root: Path, mic: str | None) -> list[dict]:
    items = []
    for p in sorted(root.rglob("*.wav")):
        m = re.match(r"^([A-Z]+\d+)_B(\d)_([A-Z]+\d+)_M(\d)$", p.stem)
        if not m or (mic and m.group(4) != mic):
            continue
        spk, block, word = m.group(1), m.group(2), m.group(3)
        # uncommon words (UW*) differ per block, so the block is part of their label
        items.append({"speaker": spk, "label": word if not word.startswith("UW") else f"B{block}_{word}",
                      "path": p, "order": block})
    if not mic:   # one microphone per recording, so repeats are real repeats and not the same take
        first = {}
        for it in items:
            key = (it["speaker"], it["label"], it["order"])
            first.setdefault(key, it)
        items = list(first.values())
    return items


def load_torgo(root: Path) -> list[dict]:
    from eval.run_wer import transcript_for
    items = []
    for p in sorted(root.rglob("*.wav")):
        text = transcript_for(p)
        if text and len(text.split()) == 1:
            items.append({"speaker": p.relative_to(root).parts[0], "label": text.lower().strip(".,!?"), "path": p})
    return items


def detect(root: Path) -> str:
    sample = next(root.rglob("*.wav"), None)
    if sample is None:
        raise SystemExit(f"no .wav files under {root}")
    if re.match(r"^[A-Z]+\d+_B\d_[A-Z]+\d+_M\d$", sample.stem):
        return "uaspeech"
    if (sample.parent.parent / "prompts").is_dir():
        return "torgo"
    return "generic"


# ---------------------------------------------------------------- evaluation
def split(items: list[dict], k: int, max_words: int) -> dict[str, tuple[list, list]]:
    """speaker -> (templates, queries). Words with no recording left over after K are dropped."""
    by = defaultdict(lambda: defaultdict(list))
    for it in items:
        by[it["speaker"]][it["label"]].append(it)
    out = {}
    for spk, words in by.items():
        usable = sorted(w for w, clips in words.items() if len(clips) > k)[:max_words]
        tmpl, qry = [], []
        for w in usable:
            clips = sorted(words[w], key=lambda c: (c.get("order", ""), str(c["path"])))
            tmpl += clips[:k]
            qry += clips[k:]
        if len(usable) >= 2:
            out[spk] = (tmpl, qry)
    return out


def featurize(items: list[dict], embedders: dict[str, Embedder]) -> None:
    for it in items:
        x, sr = dsp.read_wav(it["path"].read_bytes())
        f = dsp.extract_signal(dsp.trim_silence(dsp.resample(x, sr)))
        it["tracks"] = {"mfcc": f.mfcc, "robust": robust_track(f, 0.0), "robust+prosody": robust_track(f, 0.5)}
        it["emb"] = {}
        for name, e in embedders.items():
            t = time.perf_counter()
            emb = e(f.audio)
            it.setdefault("embed_ms", {})[name] = (time.perf_counter() - t) * 1000
            if emb is not None:
                it["emb"][name] = emb


TEMPERATURES = [0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005]
MIN_PRECISION = 0.95   # a skipped Scribe call must almost never put wrong words in the patient's mouth


def evaluate(splits: dict, methods: list[str], k: int) -> dict[str, dict]:
    """Accuracy per front-end, plus the softmax temperature that lets the most turns skip Scribe
    while the skipped ones stay >= MIN_PRECISION correct (confidences are only as good as that)."""
    res = {}
    for meth in methods:
        runs, ms = [], []
        for tmpl, qry in splits.values():
            wid = {w: i for i, w in enumerate(sorted({t["label"] for t in tmpl}))}
            for q in qry:
                t0 = time.perf_counter()
                scored = distances(meth, q, tmpl, wid)
                ms.append((time.perf_counter() - t0) * 1000)
                if scored:
                    runs.append((scored, wid[q["label"]]))
        n = max(len(runs), 1)
        top = [[m["phrase_id"] for m in rank(sc, 0.35, meth, k)] for sc, _ in runs]
        r = {"queries": len(runs), "top1": sum(t[0] == y for t, (_, y) in zip(top, runs)) / n,
             "top3": sum(y in t[:3] for t, (_, y) in zip(top, runs)) / n,
             "match_ms": statistics.median(ms) if ms else 0.0, "calibration": []}
        for temp in TEMPERATURES:
            firsts = [(rank(sc, temp, meth, k)[0], y) for sc, y in runs]
            skipped = [(m, y) for m, y in firsts if m["confidence"] >= SKIP_AT]
            ok = sum(m["phrase_id"] == y for m, y in skipped)
            r["calibration"].append({"temperature": temp, "skip_rate": len(skipped) / n,
                                     "skip_precision": ok / len(skipped) if skipped else None})
        safe = [c for c in r["calibration"] if c["skip_precision"] is not None
                and c["skip_precision"] >= MIN_PRECISION and c["skip_rate"] * n >= 5]
        r["best"] = max(safe, key=lambda c: c["skip_rate"]) if safe else None
        res[meth] = r
    return res


def distances(meth: str, q: dict, tmpl: list[dict], wid: dict) -> dict | None:
    """The expensive part: best DTW distance per word (+ clip vectors for prototypes)."""
    if meth.startswith("ssl:"):
        name = meth[4:].removesuffix("+proto").removesuffix("+nocenter")
        center = not meth.endswith("+nocenter")
        if name not in q["emb"]:
            return None
        tracks = [(wid[t["label"]], for_dtw(t["emb"][name], center)) for t in tmpl if name in t["emb"]]
        out = {"best": dsp.best_distances(for_dtw(q["emb"][name], center), tracks, pool=1)}
        if meth.endswith("+proto"):
            groups = defaultdict(list)
            for t in tmpl:
                groups[wid[t["label"]]].append(clip_vector(t["emb"][name]))
            out["proto"] = (clip_vector(q["emb"][name]), groups)
        return out
    tracks = [(wid[t["label"]], t["tracks"][meth]) for t in tmpl]
    return {"best": dsp.best_distances(q["tracks"][meth], tracks, pool=2)}


def rank(scored: dict, temperature: float, meth: str, k: int) -> list[dict]:
    matches = dsp.to_confidence(scored["best"], top_k=len(scored["best"]), temperature=temperature)
    if "proto" in scored:
        qvec, groups = scored["proto"]
        matches, _ = blend_prototypes(matches, qvec, groups, min_templates=k)
    return matches


def report(res: dict, meta: dict) -> str:
    lines = [f"# Phrase matching: front-end comparison ({meta['data_kind']})", "",
             f"Data `{meta['root']}` ({meta['layout']}), {meta['speakers']} speakers, {meta['words']} words, "
             f"K={meta['k']} enrolled recordings per word, {meta['queries']} live turns. {meta['date']}.", ""]
    if meta["data_kind"] == "SYNTHETIC":
        lines += ["> Synthetic: Windows TTS voices with simulated dysarthria-like distortion and noise.",
                  "> Use it to compare front-ends, not to predict accuracy for patients.", ""]
    lines += [f"| Front-end | Top-1 | Top-3 | Match time | Best temperature | Scribe skipped (conf >= {SKIP_AT}) "
              "| Correct when skipped |", "|---|---|---|---|---|---|---|"]
    for meth, r in res.items():
        b = r["best"]
        cal = (f"{b['temperature']} | {b['skip_rate']:.0%} | {b['skip_precision']:.0%}" if b
               else f"none reaches {MIN_PRECISION:.0%} | 0% | –")
        lines.append(f"| {meth} | {r['top1']:.1%} | {r['top3']:.1%} | {r['match_ms']:.0f} ms | {cal} |")
    lines += ["", f"Best temperature: the softmax temperature with the most Scribe skips while skipped turns stay "
              f">= {MIN_PRECISION:.0%} correct (set MATCH_TEMPERATURE_* in .env)."]
    if meta.get("embed_ms"):
        lines += ["", "Embedding time per clip (median): " +
                  ", ".join(f"{k} {v:.0f} ms" for k, v in meta["embed_ms"].items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--layout", choices=["auto", "generic", "uaspeech", "torgo"], default="auto")
    ap.add_argument("--k", type=int, default=2, help="enrolled recordings per word")
    ap.add_argument("--model", action="append", default=[], help="speech-model ONNX file (repeatable)")
    ap.add_argument("--max-words", type=int, default=60, help="words per speaker (UA-Speech has 455)")
    ap.add_argument("--mic", default=None, help="UA-Speech microphone number (default: first per take)")
    ap.add_argument("--synthetic", action="store_true", help="label the report as synthetic data")
    a = ap.parse_args()

    layout = detect(a.root) if a.layout == "auto" else a.layout
    items = {"generic": lambda: load_generic(a.root), "uaspeech": lambda: load_uaspeech(a.root, a.mic),
             "torgo": lambda: load_torgo(a.root)}[layout]()
    splits = split(items, a.k, a.max_words)
    if not splits:
        raise SystemExit("no speaker has 2+ words with more than K recordings each")
    used = [it for t, q in splits.values() for it in t + q]
    embedders = {Path(m).name.split(".")[0]: Embedder(m) for m in a.model}
    for name, e in embedders.items():
        if not e.available:
            raise SystemExit(f"model {name} not usable (missing file or onnxruntime)")
    featurize(used, embedders)

    methods = ["mfcc", "robust", "robust+prosody"]
    for name in embedders:
        methods += [f"ssl:{name}", f"ssl:{name}+nocenter"] + ([f"ssl:{name}+proto"] if a.k >= 2 else [])
    res = evaluate(splits, methods, a.k)
    meta = {"root": str(a.root), "layout": layout, "k": a.k, "speakers": len(splits),
            "words": sum(len({t["label"] for t in s[0]}) for s in splits.values()),
            "queries": sum(len(s[1]) for s in splits.values()), "date": time.strftime("%Y-%m-%d"),
            "data_kind": "SYNTHETIC" if a.synthetic or "synthetic" in str(a.root).lower() else "REAL",
            "embed_ms": {n: statistics.median(it["embed_ms"][n] for it in used) for n in embedders}}
    OUT.mkdir(exist_ok=True)
    stem = "matching_synthetic" if meta["data_kind"] == "SYNTHETIC" else "matching"
    (OUT / f"{stem}.json").write_text(json.dumps({"meta": meta, "results": res}, indent=2), encoding="utf-8")
    (OUT / f"{stem}.md").write_text(report(res, meta), encoding="utf-8")
    print(report(res, meta))


if __name__ == "__main__":
    main()
