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

    @property
    def elevenlabs_enabled(self) -> bool:
        return bool(self.elevenlabs_api_key)

    @property
    def llm_enabled(self) -> bool:
        # The Anthropic SDK can also resolve credentials from ANTHROPIC_AUTH_TOKEN / profiles.
        return bool(self.anthropic_api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


settings = Settings()
