"""Storage: caches (TTS audio, STT transcripts), phrasebook, spectrogram templates,
ISL sign samples, consent records, voices, conversation turns and telemetry events.

Two backends behind one small API (execute / insert / query / one / script):
* SQLite (a file path): local development and tests, standard library only;
* Postgres (a postgres:// URL, e.g. Neon): the hosted site. Each account gets its own Postgres
  schema (`schema=`), so one patient's data can never show up in another patient's queries.

The SQL in this project is written for SQLite; `_to_pg` translates the few differences.
"""
from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import time
from decimal import Decimal
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
    audio       BLOB,                       -- trimmed 16 kHz int16 PCM, so features can be recomputed
    embedding   BLOB,                       -- float32 [emb_frames, emb_dim] speech-model embeddings
    emb_frames  INTEGER,
    emb_dim     INTEGER,
    emb_model   TEXT,                       -- which model made `embedding` (recomputed on change)
    created_at  REAL NOT NULL
);

-- Opt-in only: the patient's confirmed utterances, kept to fine-tune a personal speech recogniser.
CREATE TABLE IF NOT EXISTS training_clips (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase_id   INTEGER,
    text        TEXT NOT NULL,              -- the phrase the patient confirmed (a weak label)
    language    TEXT,
    audio       BLOB NOT NULL,              -- trimmed 16 kHz int16 PCM
    seconds     REAL NOT NULL,
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


# Primary keys, for turning SQLite's INSERT OR REPLACE into Postgres' ON CONFLICT ... DO UPDATE.
PRIMARY_KEYS = {"tts_cache": ["key"], "stt_cache": ["audio_hash", "model_id", "language"],
                "stt_fingerprints": ["audio_hash"], "voices": ["voice_id"], "settings": ["key"],
                "key_usage": ["key_id", "month"]}
# Tables with an auto-increment `id`, so an INSERT can return it.
ID_TABLES = {"phrases", "audio_templates", "training_clips", "sign_samples", "consent_records", "turns",
             "events", "users"}


def is_postgres(url: str) -> bool:
    return url.startswith(("postgres://", "postgresql://"))


def _schema_to_pg(sql: str) -> str:
    sql = re.sub(r"INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY", sql)
    sql = re.sub(r"\bBLOB\b", "BYTEA", sql)
    return re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql)


def _to_pg(sql: str) -> tuple[str, bool]:
    """Translate one SQLite statement. Returns (sql, returns_id)."""
    sql = sql.replace("%", "%%").replace("?", "%s")
    m = re.match(r"\s*INSERT OR (REPLACE|IGNORE) INTO (\w+)\s*\(([^)]*)\)(.*)$", sql, re.S)
    if m:
        verb, table, cols, rest = m.groups()
        sql = f"INSERT INTO {table}({cols}){rest}"
        if verb == "IGNORE":
            sql += " ON CONFLICT DO NOTHING"
        else:
            pk = PRIMARY_KEYS[table]
            upd = [c.strip() for c in cols.split(",") if c.strip() not in pk]
            sql += f" ON CONFLICT ({', '.join(pk)}) DO UPDATE SET " + ", ".join(f"{c}=EXCLUDED.{c}" for c in upd)
    ins = re.match(r"\s*INSERT INTO (\w+)", sql)
    returns_id = bool(ins and ins.group(1) in ID_TABLES and "RETURNING" not in sql.upper())
    if returns_id:
        sql += " RETURNING id"
    return sql, returns_id


class Result:
    def __init__(self, lastrowid: int | None, rowcount: int):
        self.lastrowid, self.rowcount = lastrowid, rowcount


def _clean(row: dict) -> dict:
    """Postgres returns Decimal for SUM() and memoryview for some bytea paths; normalise them."""
    for k, v in row.items():
        if isinstance(v, Decimal):
            row[k] = float(v)
        elif isinstance(v, memoryview):
            row[k] = v.tobytes()
    return row


class Database:
    def __init__(self, path: str, schema: str | None = None, ddl: str | None = None):
        """`path`: an SQLite file (or ':memory:') or a Postgres URL. `schema`: Postgres schema to use
        (created if missing). `ddl`: tables to create (default: one account's tables)."""
        self.path, self.schema, self.pg = path, schema, is_postgres(path)
        self.lock = threading.RLock()
        self.conn = None
        self._cache: dict[str, Any] = {}      # read caches, cleared when their tables are written
        self._events: queue.Queue | None = None
        with self.lock:
            self._connect()
            self.script(SCHEMA if ddl is None else ddl)
            if not self.pg and ddl is None:
                self._migrate_sqlite()

    # -- connection --------------------------------------------------------
    def _connect(self) -> None:
        if self.pg:
            import psycopg
            from psycopg.rows import dict_row
            if self.schema:
                if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", self.schema):
                    raise ValueError(f"bad schema name {self.schema!r}")
                with psycopg.connect(self.path, autocommit=True) as c:
                    c.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            path = self.schema or "public"
            # A session-level SET: use Neon's direct (non "-pooler") connection string, because a
            # transaction-mode pooler would hand later queries to connections without it.
            self.conn = psycopg.connect(self.path, autocommit=True, row_factory=dict_row)
            self.conn.execute(f"SET search_path TO {path}")
        else:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")

    def _run(self, fn):
        """Run fn(conn); on Postgres, reconnect once if the connection dropped (Neon suspends idle compute)."""
        with self.lock:
            if not self.pg:
                return fn(self.conn)
            import psycopg
            try:
                return fn(self.conn)
            except psycopg.OperationalError:
                self._connect()
                return fn(self.conn)

    def _migrate_sqlite(self) -> None:
        """Columns added after the first release (new SQLite files already have them)."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(voices)")}
        if "key_id" not in cols:   # added for multi-key fallback: the account that owns the voice
            self.conn.execute("ALTER TABLE voices ADD COLUMN key_id TEXT")
        for table, default in (("audio_templates", "enrolled"), ("sign_samples", "recorded")):
            if "source" not in {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN source TEXT NOT NULL DEFAULT '{default}'")
        tcols = {r[1] for r in self.conn.execute("PRAGMA table_info(audio_templates)")}
        for col, typ in (("audio", "BLOB"), ("embedding", "BLOB"), ("emb_frames", "INTEGER"),
                         ("emb_dim", "INTEGER"), ("emb_model", "TEXT")):
            if col not in tcols:
                self.conn.execute(f"ALTER TABLE audio_templates ADD COLUMN {col} {typ}")
        self.conn.commit()

    def close(self) -> None:
        with self.lock:
            if self.conn is not None:
                self.conn.close()

    # -- generic helpers -------------------------------------------------
    def script(self, sql: str) -> None:
        """Run DDL (several statements separated by ';')."""
        if not self.pg:
            self._run(lambda c: (c.executescript(sql), c.commit()))
            return
        ddl = re.sub(r"--[^\n]*", "", _schema_to_pg(sql))
        self._run(lambda c: c.execute(ddl))   # one round trip: several statements, no parameters

    # Reads repeated on every turn are cached; a write to a table clears what depends on it.
    _DEPENDS = {"phrases": ("phrases", "audio_templates"), "time_prior": ("phrases", "turns")}

    def _invalidate(self, sql: str) -> None:
        if self._cache and re.match(r"\s*(INSERT|UPDATE|DELETE|REPLACE)", sql, re.I):
            for key, tables in self._DEPENDS.items():
                if any(re.search(rf"\b{t}\b", sql) for t in tables):
                    self._cache.pop(key, None)

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        """Many rows in one go (one pipelined batch on Postgres instead of a round trip per row)."""
        rows = [tuple(r) for r in rows]
        self._invalidate(sql)
        if not self.pg:
            self._run(lambda c: (c.executemany(sql, rows), c.commit()))
            return
        pg_sql = sql.replace("%", "%%").replace("?", "%s")
        self._run(lambda c: c.cursor().executemany(pg_sql, rows))

    def execute(self, sql: str, params: Iterable[Any] = ()) -> Result:
        params = tuple(params)
        self._invalidate(sql)
        if not self.pg:
            def run(c):
                cur = c.execute(sql, params)
                c.commit()
                return Result(cur.lastrowid, cur.rowcount)
            return self._run(run)
        pg_sql, returns_id = _to_pg(sql)

        def run_pg(c):
            cur = c.execute(pg_sql, params)
            row = cur.fetchone() if returns_id else None
            return Result(row["id"] if row else None, cur.rowcount)
        return self._run(run_pg)

    def insert(self, sql: str, params: Iterable[Any] = ()) -> int:
        return int(self.execute(sql, params).lastrowid)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        params = tuple(params)
        if not self.pg:
            return self._run(lambda c: [dict(r) for r in c.execute(sql, params).fetchall()])
        pg_sql, _ = _to_pg(sql)
        return self._run(lambda c: [_clean(dict(r)) for r in c.execute(pg_sql, params).fetchall()])

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
        row = (session_id, kind, json.dumps(data, ensure_ascii=False), time.time())
        if not self.pg:
            self.execute("INSERT INTO events(session_id, kind, data, created_at) VALUES(?,?,?,?)", row)
            return
        # Telemetry is written off the request path: a network round trip per event adds up on Neon.
        if self._events is None:
            self._events = queue.Queue()
            threading.Thread(target=self._event_writer, daemon=True).start()
        self._events.put(row)

    def _event_writer(self) -> None:
        while True:
            rows = [self._events.get()]
            while not self._events.empty():
                rows.append(self._events.get_nowait())
            try:
                self.executemany("INSERT INTO events(session_id, kind, data, created_at) VALUES(?,?,?,?)", rows)
            except Exception:   # never let telemetry take the app down
                pass
            for _ in rows:
                self._events.task_done()

    def flush_events(self) -> None:
        if self._events is not None:
            self._events.join()

    def events(self, kind: str, limit: int = 1000) -> list[dict]:
        self.flush_events()
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
        cached = self._cache.get("time_prior")
        if cached and cached[0] == here:
            return dict(cached[1])
        prior = self._time_prior(bucket, here)
        self._cache["time_prior"] = (here, prior)
        return dict(prior)

    def _time_prior(self, bucket, here: int) -> dict[int, float]:
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
        if "phrases" in self._cache:
            return [dict(r, aliases=list(r["aliases"]), keywords=list(r["keywords"])) for r in self._cache["phrases"]]
        rows = self.query(
            "SELECT p.*, (SELECT COUNT(*) FROM audio_templates t WHERE t.phrase_id=p.id) "
            "AS n_templates FROM phrases p ORDER BY p.category, p.id"
        )
        for r in rows:
            r["aliases"] = json.loads(r["aliases"])
            r["keywords"] = json.loads(r["keywords"])
            r["is_emergency"] = bool(r["is_emergency"])
        self._cache["phrases"] = rows
        return [dict(r, aliases=list(r["aliases"]), keywords=list(r["keywords"])) for r in rows]

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
