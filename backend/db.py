"""SQLite storage: caches (TTS audio, STT transcripts), phrasebook, spectrogram
templates, ISL sign samples, consent records, voices, conversation turns and
telemetry events.

One file, standard library only. A single connection guarded by a lock is plenty
for a single-user prototype; WAL mode keeps reads cheap.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS tts_cache (
    key         TEXT PRIMARY KEY,          -- sha256(text|voice|model)
    text        TEXT NOT NULL,
    voice_id    TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    audio       BLOB NOT NULL,
    mime        TEXT NOT NULL DEFAULT 'audio/mpeg',
    chars       INTEGER NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stt_cache (
    audio_hash  TEXT NOT NULL,             -- sha256(wav bytes)
    model_id    TEXT NOT NULL,
    language    TEXT NOT NULL,
    text        TEXT NOT NULL,
    words_json  TEXT NOT NULL,
    language_detected TEXT,
    seconds     REAL NOT NULL DEFAULT 0,
    hits        INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    PRIMARY KEY (audio_hash, model_id, language)
);

CREATE TABLE IF NOT EXISTS phrases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    language    TEXT NOT NULL DEFAULT 'en',  -- en | hi | mixed
    category    TEXT NOT NULL DEFAULT 'general',
    aliases     TEXT NOT NULL DEFAULT '[]',  -- JSON list: romanised / Devanagari / short forms
    keywords    TEXT NOT NULL DEFAULT '[]',  -- JSON list: context words that make this phrase likely
    is_emergency INTEGER NOT NULL DEFAULT 0,
    use_count   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_templates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase_id   INTEGER NOT NULL REFERENCES phrases(id) ON DELETE CASCADE,
    mfcc        BLOB NOT NULL,              -- float32 [frames, coeffs]
    logmel      BLOB NOT NULL,              -- float32 [frames, mels] (for visualisation)
    n_frames    INTEGER NOT NULL,
    n_mfcc      INTEGER NOT NULL,
    n_mels      INTEGER NOT NULL,
    source      TEXT NOT NULL DEFAULT 'enrolled',  -- enrolled | learned (from a confirmed turn)
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sign_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT NOT NULL,              -- 'A'..'Z', '0'..'9', or a phrase sign name
    kind        TEXT NOT NULL DEFAULT 'letter',  -- letter | phrase | control
    phrase_id   INTEGER REFERENCES phrases(id) ON DELETE SET NULL,
    features    TEXT NOT NULL,              -- JSON float list (hand-landmark feature vector)
    source      TEXT NOT NULL DEFAULT 'recorded',  -- recorded | learned (from a confirmed turn)
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS consent_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    voice_owner TEXT NOT NULL,
    signer      TEXT NOT NULL,
    relationship TEXT NOT NULL,             -- self | guardian | ...
    purpose     TEXT NOT NULL,
    statement   TEXT NOT NULL,
    statement_audio_hash TEXT,
    created_at  REAL NOT NULL,
    revoked_at  REAL
);

CREATE TABLE IF NOT EXISTS voices (
    voice_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,              -- default | designed | cloned
    description TEXT NOT NULL DEFAULT '',
    consent_id  INTEGER REFERENCES consent_records(id),
    key_id      TEXT,                       -- ElevenLabs account (key hash) that owns the voice
    created_at  REAL NOT NULL,
    deleted_at  REAL
);

CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    speaker     TEXT NOT NULL,              -- partner | patient
    text        TEXT NOT NULL,
    raw_text    TEXT,                       -- raw ASR before fusion (patient)
    source      TEXT,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    kind        TEXT NOT NULL,              -- latency | effort | credit | consent | ...
    data        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, created_at);
CREATE INDEX IF NOT EXISTS idx_templates_phrase ON audio_templates(phrase_id);
CREATE INDEX IF NOT EXISTS idx_signs_label ON sign_samples(label);
"""


class Database:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(voices)")}
            if "key_id" not in cols:   # added for multi-key fallback: the account that owns the voice
                self.conn.execute("ALTER TABLE voices ADD COLUMN key_id TEXT")
            for table, default in (("audio_templates", "enrolled"), ("sign_samples", "recorded")):
                if "source" not in {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN source TEXT NOT NULL DEFAULT '{default}'")
            self.conn.commit()

    # -- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, tuple(params))
            self.conn.commit()
            return cur

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- settings ----------------------------------------------------------
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # -- events ------------------------------------------------------------
    def log_event(self, kind: str, data: dict, session_id: str | None = None) -> None:
        self.execute(
            "INSERT INTO events(session_id, kind, data, created_at) VALUES(?,?,?,?)",
            (session_id, kind, json.dumps(data, ensure_ascii=False), time.time()),
        )

    def events(self, kind: str, limit: int = 1000) -> list[dict]:
        rows = self.query(
            "SELECT * FROM events WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)
        )
        for r in rows:
            r["data"] = json.loads(r["data"])
        return rows

    # -- conversation --------------------------------------------------------
    def add_turn(self, session_id: str, speaker: str, text: str,
                 raw_text: str | None = None, source: str | None = None) -> int:
        cur = self.execute(
            "INSERT INTO turns(session_id, speaker, text, raw_text, source, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (session_id, speaker, text, raw_text, source, time.time()),
        )
        return int(cur.lastrowid)

    def history(self, session_id: str, limit: int = 8) -> list[dict]:
        rows = self.query(
            "SELECT id, speaker, text, created_at FROM turns WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        return list(reversed(rows))

    def phrase_time_prior(self, now: float | None = None) -> dict[int, float]:
        """How often each phrase was said at this time of day (night / morning / afternoon / evening),
        scaled 0..1 against the most-used phrase of that part of the day. Needs 2+ uses to count."""
        bucket = lambda ts: (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0)[time.localtime(ts).tm_hour]  # noqa: E731
        here = bucket(now or time.time())
        by_text = {r["text"]: r["id"] for r in self.query("SELECT id, text FROM phrases")}
        counts: dict[int, int] = {}
        for r in self.query("SELECT text, created_at FROM turns WHERE speaker='patient' ORDER BY id DESC LIMIT 1000"):
            pid = by_text.get(r["text"])
            if pid and bucket(r["created_at"]) == here:
                counts[pid] = counts.get(pid, 0) + 1
        counts = {k: v for k, v in counts.items() if v >= 2}
        top = max(counts.values(), default=0)
        return {k: round(v / top, 3) for k, v in counts.items()} if top else {}

    # -- phrases -------------------------------------------------------------
    def phrases(self) -> list[dict]:
        rows = self.query(
            "SELECT p.*, (SELECT COUNT(*) FROM audio_templates t WHERE t.phrase_id=p.id) "
            "AS n_templates FROM phrases p ORDER BY p.category, p.id"
        )
        for r in rows:
            r["aliases"] = json.loads(r["aliases"])
            r["keywords"] = json.loads(r["keywords"])
            r["is_emergency"] = bool(r["is_emergency"])
        return rows

    def add_phrase(self, text: str, language: str = "en", category: str = "general",
                   aliases: list[str] | None = None, keywords: list[str] | None = None,
                   is_emergency: bool = False) -> int:
        cur = self.execute(
            "INSERT INTO phrases(text, language, category, aliases, keywords, is_emergency, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (text, language, category, json.dumps(aliases or [], ensure_ascii=False),
             json.dumps(keywords or [], ensure_ascii=False), int(is_emergency), time.time()),
        )
        return int(cur.lastrowid)

    # -- cache stats -----------------------------------------------------------
    def cache_stats(self) -> dict:
        tts = self.one("SELECT COUNT(*) n, COALESCE(SUM(hits),0) hits, COALESCE(SUM(chars),0) chars, "
                       "COALESCE(SUM(LENGTH(audio)),0) bytes FROM tts_cache") or {}
        stt = self.one("SELECT COUNT(*) n, COALESCE(SUM(hits),0) hits, COALESCE(SUM(seconds),0) seconds "
                       "FROM stt_cache") or {}
        return {"tts": tts, "stt": stt}
