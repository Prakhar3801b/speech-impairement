"""Optional LLM re-ranker (Claude). Takes the heuristic shortlist plus all raw
signals and returns up to 3 candidate messages as schema-validated JSON.

If no Anthropic credentials are configured, or the call fails / is slow, callers
fall back to the heuristic ranking, so the app never blocks on the LLM.
"""
from __future__ import annotations

import json
import logging

import anthropic

from .config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You help a person with a speech impairment (dysarthria from ALS, cerebral palsy, \
Parkinson's or stroke, or after a laryngectomy) take part in a live conversation. Their speech \
recogniser output is often wrong. Combine every signal you are given to infer what they most likely \
mean, and propose up to 3 short candidate messages for them to confirm. They will choose one, \
and it will be spoken aloud in their voice.

Signals you may receive:
- partner_last / history: what the other person said. This is usually the strongest clue.
- asr_text: noisy transcript of the patient (English, Hindi or code-mixed Hinglish).
- spectrogram_matches: phrases whose enrolled recordings acoustically match this utterance; \
these are speaker-specific and often more reliable than asr_text for severe dysarthria.
- isl_spelled: letters fingerspelled in Indian Sign Language (may be a partial word).
- sign_phrases: phrases the patient signed directly.
- cues: head nod/shake, facial expression, thumbs up/down, lip-syllable count.
- shortlist: candidates from a heuristic ranker, with scores.

Rules:
- Write messages in first person, as the patient would say them; keep them short.
- Match the language of the conversation (English, Hindi in Roman script, or Hinglish).
- Prefer phrasebook wording when it fits; otherwise write a natural sentence.
- Never invent medical facts or urgency that no signal supports.
- Order candidates from most to least likely; confidences should roughly sum to 1."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "language": {"type": "string", "enum": ["en", "hi", "mixed"]},
                    "confidence": {"type": "number"},
                    "emotion": {"type": "string",
                                "enum": ["neutral", "happy", "sad", "urgent", "pained"]},
                    "rationale": {"type": "string"},
                },
                "required": ["text", "language", "confidence", "emotion", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(timeout=settings.llm_timeout_s, max_retries=0)
    return _client


async def rank(signals: dict, task: str = "intent") -> list[dict] | None:
    """Return LLM candidates, or None to signal 'use the heuristic result'."""
    if not settings.llm_enabled:
        return None
    instruction = (
        "Infer what the patient is trying to say right now."
        if task == "intent" else
        "The partner just spoke. Suggest the replies the patient is most likely to want to give.")
    try:
        response = await _get_client().beta.messages.create(
            model=settings.llm_model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                       f"{instruction}\n\n<signals>\n{json.dumps(signals, ensure_ascii=False)}\n</signals>"}],
            output_config={"effort": settings.llm_effort, "format": {
                "type": "json_schema", "schema": OUTPUT_SCHEMA}},
            # Opt-in server-side fallback if a request is declined by a safety classifier.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APITimeoutError:
        log.warning("LLM ranker timed out; using heuristic ranking")
        return None
    except anthropic.RateLimitError:
        log.warning("LLM ranker rate limited; using heuristic ranking")
        return None
    except anthropic.APIStatusError as e:
        log.warning("LLM ranker API error %s: %s", e.status_code, e.message)
        return None
    except anthropic.APIConnectionError:
        log.warning("LLM ranker unreachable; using heuristic ranking")
        return None

    if response.stop_reason == "refusal":
        log.warning("LLM ranker refused: %s", response.stop_details)
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("LLM ranker returned invalid JSON (stop_reason=%s)", response.stop_reason)
        return None
    out = []
    for c in data.get("candidates", [])[:3]:
        out.append({"text": c["text"].strip(), "confidence": round(float(c["confidence"]), 3),
                    "language": c["language"], "emotion": c["emotion"],
                    "rationale": c["rationale"], "sources": ["llm"], "phrase_id": None})
    return out or None
