"""Multimodal fusion: combine the noisy Scribe transcript, spectrogram template
matches, ISL sign input, face/head cues and conversation context into a short
ranked list of candidate messages.

Three layers, cheapest first:
  1. fast path  - unambiguous sign/gesture answers without any model call
  2. heuristic  - weighted scoring over the phrasebook (always available, offline)
  3. LLM ranker - optional re-ranking / rewriting on top of (2) (see llm.py)

The heuristic is also the ablation baseline used by eval/run_eval.py.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# ------------------------------------------------------------------ text utils
_PUNCT = re.compile(r"[^\w\sऀ-ॿ]", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", text)).strip()


_PHONETIC_RULES = [
    ("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
    ("ch", "c"), ("jh", "j"), ("ck", "k"), ("aa", "a"), ("ee", "i"), ("oo", "u"), ("w", "v"),
    ("z", "s"), ("q", "k"), ("x", "ks"), ("y", "i"),
]
# dysarthria commonly blurs voicing; collapse voiced/unvoiced pairs
_VOICING = str.maketrans({"b": "p", "d": "t", "g": "k", "v": "f", "j": "c"})


def phonetic_key(text: str) -> str:
    """Crude phonetic skeleton for romanised English/Hindi: merges digraphs,
    voicing pairs and drops non-initial vowels, so 'paani', 'bani' and 'pani' collide."""
    out = []
    for word in normalize(text).split():
        for a, b in _PHONETIC_RULES:
            word = word.replace(a, b)
        word = word.translate(_VOICING)
        if not word:
            continue
        skeleton = word[0] + re.sub(r"[aeiou]", "", word[1:])
        out.append(re.sub(r"(.)\1+", r"\1", skeleton))
    return " ".join(out)


def similarity(a: str, b: str) -> float:
    """Blend of character, phonetic and token-overlap similarity in [0, 1]."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    char = SequenceMatcher(None, na, nb).ratio()
    pa, pb = phonetic_key(na), phonetic_key(nb)
    phon = SequenceMatcher(None, pa, pb).ratio() if pa and pb else 0.0
    ta, tb = set(na.split()), set(nb.split())
    token = len(ta & tb) / max(1, min(len(ta), len(tb)))
    # partial: does the (short) ASR text fuzzily appear inside the phrase?
    partial = max((SequenceMatcher(None, w, v).ratio() for w in na.split() for v in nb.split()), default=0)
    return max(char, phon, 0.6 * token + 0.4 * partial, 0.85 * partial)


def estimate_syllables(text: str) -> int:
    norm = normalize(text)
    if re.search(r"[ऀ-ॿ]", norm):
        return max(1, len(re.findall(r"[अ-औक-ह]", norm)) // 2)
    return max(1, len(re.findall(r"[aeiouy]+", norm)))


YES_TEXTS = {"yes", "haan"}
NO_TEXTS = {"no", "nahi"}
_AUX_START = re.compile(r"^(do|does|did|are|is|was|were|can|could|should|shall|will|would|have|has|may)\b")
_WH = re.compile(r"\b(what|which|where|when|who|whom|why|how|kaun|kaunsa|kahan|kab|kaise|kaisa|kaisi|"
                 r"kitna|kitni|kyun|kyon|kis|kisko)\b")
_CHOICE = re.compile(r"\b(or|ya)\b")


_HINDI_WORDS = re.compile(r"\b(kya|hai|hain|aap|aapko|tum|chahiye|ho|kaise|kaisa|mein|nahi|haan|"
                          r"lagi|raha|rahe|rahi|karo|kar|ka|ki|ke|ko|se)\b")


def looks_hindi(text: str) -> bool:
    t = normalize(text)
    return bool(re.search(r"[\u0900-\u097F]", t)) or len(_HINDI_WORDS.findall(t)) >= 2


def keyword_hit(keyword: str, normalized_text: str) -> bool:
    """Whole-word / whole-phrase match of a context keyword."""
    k = normalize(keyword)
    return bool(k) and re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", normalized_text) is not None


def is_yes_no_question(text: str) -> bool:
    """English: auxiliary-verb questions ('Do you…?'). Hindi: 'Kya …?', '… chahiye?', '… hai na?'.
    Choice questions ('water or tea?', 'paani ya chai?') and wh-questions are not yes/no."""
    t = normalize(text)
    if not t or _CHOICE.search(t) or _WH.search(t):
        return False
    if _AUX_START.search(t):
        return True
    return bool(re.search(r"^kya\b|\bchahiye\b|\bhai na\b|\bthik hai\b|\btheek hai\b", t)) and "?" in text


# ------------------------------------------------------------------ inputs
@dataclass
class Cues:
    head: str | None = None          # nod | shake | None
    expression: str | None = None    # neutral | smile | frown | pain | surprise
    thumbs: str | None = None        # up | down | None
    syllables: int | None = None     # lip-activity syllable estimate during the utterance

    @classmethod
    def from_dict(cls, d: dict | None) -> "Cues":
        d = d or {}
        syl = d.get("syllables")
        return cls(head=d.get("head"), expression=d.get("expression"), thumbs=d.get("thumbs"),
                   syllables=int(syl) if isinstance(syl, (int, float)) and syl > 0 else None)


@dataclass
class FusionInput:
    asr_text: str = ""
    template_matches: list[dict] = field(default_factory=list)   # [{phrase_id, confidence}]
    spelled: str = ""                                            # ISL fingerspelled letters
    sign_phrase_ids: list[int] = field(default_factory=list)     # phrase signs seen
    cues: Cues = field(default_factory=Cues)
    partner_last: str = ""
    language: str = "auto"                                       # auto | eng | hin


@dataclass
class Weights:
    asr: float = 1.0
    template: float = 1.2
    sign: float = 1.5
    context: float = 0.6
    cue: float = 0.8
    lip: float = 0.15
    frequency: float = 0.1
    language: float = 0.1
    time: float = 0.25        # said at this time of day before (see Database.phrase_time_prior)


# ------------------------------------------------------------------ fast path
def fast_path(inp: FusionInput, phrases: list[dict]) -> list[dict] | None:
    by_id = {p["id"]: p for p in phrases}
    # 1. an emergency phrase sign -> speak immediately (still shown for confirmation)
    for pid in inp.sign_phrase_ids:
        p = by_id.get(pid)
        if p and p["is_emergency"]:
            return [_cand(p["text"], 0.99, ["sign:emergency"], p, emotion="urgent")]
    # 2. yes/no question answered by nod/shake/thumbs with no speech
    if is_yes_no_question(inp.partner_last) and not inp.asr_text and not inp.template_matches:
        hindi = inp.language == "hin" or bool(re.search(r"\b(kya|chahiye|hai)\b", inp.partner_last.lower()))
        if inp.cues.head == "nod" or inp.cues.thumbs == "up":
            return [_cand("Haan" if hindi else "Yes", 0.95, ["gesture:" + (inp.cues.head or "thumbs_up")])]
        if inp.cues.head == "shake" or inp.cues.thumbs == "down":
            return [_cand("Nahi" if hindi else "No", 0.95, ["gesture:" + (inp.cues.head or "thumbs_down")])]
    return None


def _cand(text: str, conf: float, sources: list[str], phrase: dict | None = None,
          emotion: str = "neutral", score: float | None = None) -> dict:
    return {"text": text, "confidence": round(conf, 3), "sources": sources,
            "phrase_id": phrase["id"] if phrase else None,
            "language": phrase["language"] if phrase else None,
            "emotion": emotion, "score": round(score if score is not None else conf, 4)}


# ------------------------------------------------------------------ heuristic
def _emotion_for(cues: Cues, phrase: dict | None) -> str:
    if phrase and phrase.get("is_emergency"):
        return "urgent"
    return {"pain": "pained", "frown": "sad", "smile": "happy"}.get(cues.expression or "", "neutral")


def heuristic_rank(inp: FusionInput, phrases: list[dict], top_k: int = 3,
                   weights: Weights | None = None) -> list[dict]:
    w = weights or Weights()
    tmpl = {m["phrase_id"]: m["confidence"] for m in inp.template_matches}
    signs = set(inp.sign_phrase_ids)
    partner = normalize(inp.partner_last)
    yes_no = is_yes_no_question(inp.partner_last)
    max_use = max((p.get("use_count", 0) for p in phrases), default=0) or 1
    spelled = normalize(inp.spelled)

    scored: list[tuple[float, dict, list[str]]] = []
    for p in phrases:
        forms = [p["text"], *p.get("aliases", [])]
        s, src = 0.0, []
        if inp.asr_text:
            s_asr = max(similarity(inp.asr_text, f) for f in forms)
            s += w.asr * s_asr
            if s_asr > 0.5:
                src.append("speech")
        if p["id"] in tmpl:
            s += w.template * tmpl[p["id"]]
            if tmpl[p["id"]] > 0.2:
                src.append("spectrogram")
        if p["id"] in signs:
            s += w.sign
            src.append("sign")
        if spelled:
            words = [v for f in forms for v in normalize(f).split()]
            best = 0.0
            for v in words:
                if v.startswith(spelled) and len(spelled) >= 2:
                    best = max(best, 0.6 + 0.4 * len(spelled) / len(v))
                else:
                    best = max(best, 0.8 * SequenceMatcher(None, spelled, v).ratio() if len(spelled) >= 3 else 0)
            s += w.sign * best
            if best > 0.5:
                src.append("isl-spelling")
        if partner:
            kw = p.get("keywords", [])
            s_ctx = 1.0 if any(keyword_hit(k, partner) for k in kw) else 0.0
            if yes_no and p["category"] == "answer":
                s_ctx = max(s_ctx, 0.7)
            s += w.context * s_ctx
            if s_ctx:
                src.append("context")
        cue = inp.cues
        low = p["text"].lower()
        s_cue = 0.0
        if (cue.head == "nod" or cue.thumbs == "up") and low in YES_TEXTS:
            s_cue = 1.0
        elif (cue.head == "shake" or cue.thumbs == "down") and low in NO_TEXTS:
            s_cue = 1.0
        elif cue.expression == "pain" and p["category"] in ("medical", "emergency"):
            s_cue = 0.6
        elif cue.expression == "smile" and p["category"] == "social":
            s_cue = 0.3
        s += w.cue * s_cue
        if s_cue:
            src.append("face/head")
        if cue.syllables and (inp.asr_text or tmpl):
            s += w.lip * math.exp(-abs(cue.syllables - estimate_syllables(p["text"])) / 2.0)
        s += w.frequency * p.get("use_count", 0) / max_use
        s += w.time * p.get("time_prior", 0.0)
        lang = {"eng": "en", "hin": "hi"}.get(inp.language)
        if lang and p["language"] in (lang, "mixed"):
            s += w.language
        scored.append((s, p, src))

    scored.sort(key=lambda t: -t[0])
    out = [(_s, _cand(p["text"], 0, src, p, _emotion_for(inp.cues, p), _s)) for _s, p, src in scored[: top_k + 3]]

    # free-text candidates: the raw transcript and the spelled word, if nothing in
    # the phrasebook explains them well
    top_score = scored[0][0] if scored else 0
    if inp.asr_text and (not scored or similarity(inp.asr_text, scored[0][1]["text"]) < 0.6):
        out.append((top_score * 0.8, _cand(inp.asr_text.strip(), 0, ["speech:raw"], None,
                                           _emotion_for(inp.cues, None), top_score * 0.8)))
    if len(spelled) >= 2 and not any("isl-spelling" in c["sources"] for _, c in out[:top_k]):
        out.append((top_score * 0.75 + 0.2, _cand(inp.spelled.upper(), 0, ["isl-spelling:raw"], None,
                                                  _emotion_for(inp.cues, None), top_score * 0.75 + 0.2)))

    out.sort(key=lambda t: -t[0])
    out = _dedupe([c for _, c in out])[:top_k]
    return _softmax_confidence(out)   # softmax-ish confidences over the shortlist


def _dedupe(cands: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cands:
        key = normalize(c["text"])
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


_FILLERS = {"chahiye", "chaiye", "hai", "hain", "lenge", "loge", "logi", "lena", "please", "pasand",
             "karoge", "karogi", "karenge", "doon", "du", "de", "do", "na", "now", "today", "then"}


def choice_options(text: str) -> list[str]:
    """Pull the two options out of an either/or question.

    'Paani thanda ya normal?' -> ['Thanda', 'Normal']; 'Hot water or cold water?' -> ['Hot water',
    'Cold water']. The right-hand option sets how many words to take from the left."""
    parts = re.split(r"\s+(?:or|ya)\s+", normalize(text))
    if len(parts) != 2:
        return []
    strip = lambda ws: ws[: next((i for i in range(len(ws), 0, -1) if ws[i - 1] not in _FILLERS), 0)]  # noqa: E731
    left, right = strip(parts[0].split()), strip(parts[1].split())
    if not left or not right or len(right) > 3:
        return []
    left = left[-len(right):]
    if _AUX_START.match(" ".join(left)) or _WH.search(" ".join(left)):
        return []
    return [" ".join(left).capitalize(), " ".join(right).capitalize()]


def question_shape(text: str) -> dict:
    """What kind of answer the partner's words call for: yes_no, choice (with its options) or open."""
    options = choice_options(text) if _CHOICE.search(normalize(text)) else []
    if options:
        return {"type": "choice", "options": options}
    if is_yes_no_question(text):
        hindi = looks_hindi(text) or bool(re.search(r"\b(kya|chahiye|hai)\b", normalize(text)))
        return {"type": "yes_no", "options": ["Haan", "Nahi"] if hindi else ["Yes", "No"]}
    return {"type": "open", "options": []}


def _softmax_confidence(cands: list[dict], sharpness: float = 3.0) -> list[dict]:
    if cands:
        mx = max(c["score"] for c in cands)
        exps = [math.exp(sharpness * (c["score"] - mx)) for c in cands]
        for c, e in zip(cands, exps):
            c["confidence"] = round(e / sum(exps), 3)
    return cands


def suggest_replies(partner_text: str, phrases: list[dict], top_k: int = 4) -> list[dict]:
    """Precompute likely patient replies as soon as the partner finishes speaking.

    Yes/no and either/or questions put their own answers first (two big cards in the UI,
    so the patient answers with one action); open questions rank the phrasebook."""
    shape = question_shape(partner_text)
    shaped = []
    for opt in shape["options"]:   # speak the option as worded; link a phrase only when the text is the same
        p = next((q for q in phrases if normalize(q["text"]) == normalize(opt)), None)
        shaped.append(_cand(opt, 0, ["question:" + shape["type"]], p))
    partner = normalize(partner_text)
    max_use = max((q.get("use_count", 0) for q in phrases), default=0) or 1
    hindi = looks_hindi(partner_text)
    ranked = []
    for p in phrases:
        s = 0.0
        if any(keyword_hit(k, partner) for k in p.get("keywords", [])):
            s += 1.0
        forms = " ".join([p["text"], *p.get("aliases", [])]).lower()
        overlap = [t for t in partner.split() if len(t) > 3 and t in forms]
        s += 0.4 * min(len(overlap), 2)
        if is_yes_no_question(partner_text) and p["category"] == "answer":
            s += 0.8
        if hindi and p["language"] in ("hi", "mixed"):
            s += 0.3
        s += 0.2 * p.get("use_count", 0) / max_use
        s += 0.3 * p.get("time_prior", 0.0)
        ranked.append((s, p))
    ranked.sort(key=lambda t: -t[0])
    rest = [_cand(p["text"], 0, ["context"], p, score=s) for s, p in ranked[: top_k + 2] if s > 0]
    lead = (rest[0]["score"] if rest else 0.0) + 0.5        # the question's own answers always rank first
    for c in shaped:
        c["score"] = round(lead, 4)
    out = _dedupe(shaped + rest)[: max(top_k, len(shaped) + 2)]
    return _softmax_confidence(out)
