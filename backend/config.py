"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so we avoid an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    elevenlabs_api_key: str = os.environ.get("ELEVENLABS_API_KEY", "")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    stt_model: str = os.environ.get("STT_MODEL", "scribe_v1")
    tts_model: str = os.environ.get("TTS_MODEL", "eleven_flash_v2_5")
    default_voice_id: str = os.environ.get("DEFAULT_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    elevenlabs_concurrency: int = int(os.environ.get("ELEVENLABS_CONCURRENCY", "2"))
    llm_model: str = os.environ.get("LLM_MODEL", "claude-opus-5-5")
    llm_effort: str = os.environ.get("LLM_EFFORT", "low")
    llm_timeout_s: float = float(os.environ.get("LLM_TIMEOUT_S", "4"))
    db_path: str = os.environ.get("DB_PATH", str(ROOT / "data" / "app.db"))
    # Postgres (e.g. Neon, direct connection string). When set, it replaces the SQLite files.
    database_url: str = os.environ.get("DATABASE_URL", "")
    # Free-tier budget. Extra keys (comma separated) take over when one runs out.
    elevenlabs_api_keys: str = os.environ.get("ELEVENLABS_API_KEYS", "")
    credits_per_key: float = float(os.environ.get("ELEVENLABS_CREDITS_PER_KEY", "10000"))
    credit_reserve: float = float(os.environ.get("ELEVENLABS_CREDIT_RESERVE", "0.05"))
    # Local estimates only; replaced by the account's own counter when it can be read.
    tts_credits_per_char: float = float(os.environ.get("TTS_CREDITS_PER_CHAR", "0.5"))
    stt_credits_per_minute: float = float(os.environ.get("STT_CREDITS_PER_MINUTE", "400"))
    # Below this share of credits left, the patient's speech skips Scribe when a local match is clear.
    saver_mode_below: float = float(os.environ.get("SAVER_MODE_BELOW", "0.25"))
    # Re-use the transcript of an acoustically near-identical earlier clip (DTW distance).
    stt_reuse_max_dist: float = float(os.environ.get("STT_REUSE_MAX_DIST", "3.0"))
    # A spectrogram template match this confident answers the patient turn without Scribe.
    skip_stt_confidence: float = float(os.environ.get("SKIP_STT_CONFIDENCE", "0.85"))
    max_stt_seconds: float = float(os.environ.get("MAX_STT_SECONDS", "15"))
    # Phrase matching (backend/matcher.py): auto | ssl | robust | mfcc. auto = best available.
    matcher: str = os.environ.get("MATCHER", "auto")
    ssl_model_path: str = os.environ.get("SSL_MODEL_PATH", str(ROOT / "models" / "wavlm-base-plus-l6.int8.onnx"))
    ssl_center: bool = os.environ.get("SSL_CENTER", "1") == "1"
    prosody_weight: float = float(os.environ.get("PROSODY_WEIGHT", "0.5"))
    # Softmax temperature per feature type, calibrated with eval/run_matching.py so that a
    # confidence >= SKIP_STT_CONFIDENCE is right >= 95% of the time (0.35 was far too flat).
    match_temperature_mfcc: float = float(os.environ.get("MATCH_TEMPERATURE_MFCC", "0.02"))
    match_temperature_robust: float = float(os.environ.get("MATCH_TEMPERATURE_ROBUST", "0.02"))
    match_temperature_ssl: float = float(os.environ.get("MATCH_TEMPERATURE_SSL", "0.02"))
    prototype_min_templates: int = int(os.environ.get("PROTOTYPE_MIN_TEMPLATES", "5"))
    prototype_weight: float = float(os.environ.get("PROTOTYPE_WEIGHT", "0.3"))
    prototype_temperature: float = float(os.environ.get("PROTOTYPE_TEMPERATURE", "0.02"))
    # Optional personal speech recogniser (fine-tuned Whisper folder); tried before Scribe.
    local_asr_model: str = os.environ.get("LOCAL_ASR_MODEL", "")

    @property
    def api_keys(self) -> list[str]:
        keys = [k.strip() for k in self.elevenlabs_api_keys.split(",")] + [self.elevenlabs_api_key]
        return [k for k in dict.fromkeys(keys) if k]

    @property
    def elevenlabs_enabled(self) -> bool:
        return bool(self.api_keys)

    @property
    def llm_enabled(self) -> bool:
        # The Anthropic SDK can also resolve credentials from ANTHROPIC_AUTH_TOKEN / profiles.
        return bool(self.anthropic_api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


settings = Settings()
