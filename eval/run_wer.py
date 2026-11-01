"""Scribe word error rate on REAL atypical speech (TORGO, UA-Speech, or your own labelled clips).

Point it at a folder of WAV files with transcripts. A transcript is found next to the audio
(`clip.txt` / `clip.lab`) or, as in TORGO, in a sibling `prompts/` folder (`../prompts/clip.txt`).
TORGO prompts that are not read speech (picture descriptions, "[say Ah-P-Eee...]") are skipped.

The first folder level under the root is the speaker. TORGO control speakers (FC*, MC*) are
reported separately from dysarthric speakers (F*, M*).

Every request goes through the app's ElevenLabs client, so the free-tier budget, key fallback and
transcript caches apply: re-running the script costs nothing for clips already transcribed.
Use --dry-run first to see how many audio minutes a run would send.

Usage:  python -m eval.run_wer DATA_DIR [--max 40] [--language eng] [--dry-run]
Writes: eval/reports/real_wer.{md,json}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

from backend import dsp
from backend.config import settings
from backend.db import Database
from backend.elevenlabs import ElevenLabs
from backend.metrics import cer, wer

OUT = Path(__file__).resolve().parent / "reports"


def transcript_for(wav: Path) -> str | None:
    for cand in (wav.with_suffix(".txt"), wav.with_suffix(".lab"), wav.parent.parent / "prompts" / (wav.stem + ".txt")):
        if cand.exists():
            text = cand.read_text(encoding="utf-8", errors="ignore").strip()
            if not text or text.startswith("[") or "/" in text or "xxx" in text.lower():
                return None   # not read speech
            return re.sub(r"\s+", " ", text)
    return None


def collect(root: Path, limit: int) -> list[dict]:
    items = []
    for wav in sorted(root.rglob("*.wav")):
        ref = transcript_for(wav)
        if ref is None:
            continue
        speaker = wav.relative_to(root).parts[0]
        items.append({"path": wav, "speaker": speaker, "reference": ref,
                      "group": "control" if re.match(r"^[FM]C", speaker) else "dysarthric"})
    # spread a limited run across speakers instead of taking one speaker's first N clips
    by_speaker: dict[str, list[dict]] = {}
    for it in items:
        by_speaker.setdefault(it["speaker"], []).append(it)
    picked, i = [], 0
    while len(picked) < limit and any(i < len(v) for v in by_speaker.values()):
        picked += [v[i] for v in by_speaker.values() if i < len(v)][: limit - len(picked)]
        i += 1
    return picked


async def evaluate(items: list[dict], el: ElevenLabs, language: str = "eng") -> list[dict]:
    rows = []
    for it in items:
        wav = it["path"].read_bytes()
        res = await el.transcribe(wav, language, "eval-real", role="eval", reuse_similar=False)
        if res.get("mock"):
            raise SystemExit("No ElevenLabs credits or key available: " + json.dumps({k: res.get(k) for k in ("budget",)}))
        rows.append({"speaker": it["speaker"], "group": it["group"], "file": it["path"].name,
                     "reference": it["reference"], "hypothesis": res["text"],
                     "wer": round(wer(it["reference"], res["text"]), 4), "cer": round(cer(it["reference"], res["text"]), 4),
                     "cached": bool(res.get("cached"))})
    return rows


def report(rows: list[dict]) -> str:
    def stats(rs):
        return (f"{len(rs)} | {statistics.mean(r['wer'] for r in rs):.3f} | {statistics.mean(r['cer'] for r in rs):.3f}"
                if rs else "0 | – | –")
    lines = ["# Scribe WER on real atypical speech (REAL data)", "",
             f"Model `{settings.stt_model}`, {len(rows)} clips, {time.strftime('%Y-%m-%d')}. Not simulated.", "",
             "| Group | Clips | WER | CER |", "|---|---|---|---|"]
    for g in ("dysarthric", "control"):
        lines.append(f"| {g} | {stats([r for r in rows if r['group'] == g])} |")
    lines += ["", "| Speaker | Group | Clips | WER | CER |", "|---|---|---|---|---|"]
    for spk in sorted({r["speaker"] for r in rows}):
        rs = [r for r in rows if r["speaker"] == spk]
        lines.append(f"| {spk} | {rs[0]['group']} | {stats(rs)} |")
    worst = sorted(rows, key=lambda r: -r["wer"])[:10]
    lines += ["", "## Worst clips (where Scribe fails)", "", "| Speaker | Reference | Scribe heard | WER |", "|---|---|---|---|"]
    lines += [f"| {r['speaker']} | {r['reference']} | {r['hypothesis'] or '(nothing)'} | {r['wer']:.2f} |" for r in worst]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--max", type=int, default=40, help="clips to send (keep it small on the free tier)")
    ap.add_argument("--language", default="eng")
    ap.add_argument("--dry-run", action="store_true", help="only count clips and audio minutes")
    a = ap.parse_args()
    items = collect(a.data_dir, a.max)
    minutes = sum(dsp.duration_s(it["path"].read_bytes()) for it in items) / 60
    print(f"{len(items)} clips from {len({i['speaker'] for i in items})} speakers, {minutes:.1f} audio minutes "
          f"(before silence trimming and caching)")
    if a.dry_run or not items:
        return
    el = ElevenLabs(Database(settings.db_path))
    rows = asyncio.run(evaluate(items, el, a.language))
    OUT.mkdir(exist_ok=True)
    (OUT / "real_wer.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "real_wer.md").write_text(report(rows), encoding="utf-8")
    print(report(rows))


if __name__ == "__main__":
    main()
