"""Free-tier credit budget across one or more ElevenLabs API keys.

* Keys are tried in the order they are configured. A key is skipped once its
  estimated monthly use reaches (1 - reserve) of its limit, or as soon as
  ElevenLabs reports its quota as used up; the next key takes over.
* Use is estimated locally (TTS characters x credits per character, STT
  minutes x credits per minute) and corrected from the account's own counter
  (GET /v1/user/subscription) whenever that call succeeds.
* Keys are never stored: each one is identified by a short sha256 prefix.
* When every key is out, callers fall back (browser speech, typed input), so
  the app keeps working at zero cost.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from .db import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS key_usage (
    key_id      TEXT NOT NULL,             -- sha256(key)[:10], never the key itself
    month       TEXT NOT NULL,             -- 'YYYY-MM' (UTC)
    credits     REAL NOT NULL DEFAULT 0,   -- estimated (or synced) credits used this month
    credit_limit REAL,                     -- from the account, when known
    exhausted   INTEGER NOT NULL DEFAULT 0,
    synced_at   REAL,
    PRIMARY KEY (key_id, month)
);
"""


def key_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def _month() -> str:
    return time.strftime("%Y-%m", time.gmtime())


@dataclass
class Key:
    id: str
    secret: str


class KeyPool:
    def __init__(self, db: Database, keys: list[str], credits_per_key: float, reserve: float):
        self.db = db
        self.keys = [Key(key_id(k), k) for k in dict.fromkeys(k.strip() for k in keys) if k]
        self.credits_per_key = credits_per_key
        self.reserve = reserve
        with db.lock:
            db.conn.executescript(SCHEMA)
            db.conn.commit()

    def __bool__(self) -> bool:
        return bool(self.keys)

    def _row(self, kid: str) -> dict:
        month = _month()
        row = self.db.one("SELECT * FROM key_usage WHERE key_id=? AND month=?", (kid, month))
        if row is None:
            self.db.execute("INSERT OR IGNORE INTO key_usage(key_id, month) VALUES(?, ?)", (kid, month))
            row = {"key_id": kid, "month": month, "credits": 0.0, "credit_limit": None,
                   "exhausted": 0, "synced_at": None}
        return row

    def limit(self, kid: str) -> float:
        return self._row(kid)["credit_limit"] or self.credits_per_key

    def remaining(self, kid: str) -> float:
        row = self._row(kid)
        if row["exhausted"]:
            return 0.0
        usable = (row["credit_limit"] or self.credits_per_key) * (1.0 - self.reserve)
        return max(0.0, usable - row["credits"])

    def candidates(self, cost: float, only: str | None = None) -> list[Key]:
        """Keys that can still afford `cost` credits, in configured order."""
        return [k for k in self.keys if (only is None or k.id == only) and self.remaining(k.id) >= cost]

    def charge(self, kid: str, credits: float) -> None:
        self._row(kid)
        self.db.execute("UPDATE key_usage SET credits=credits+? WHERE key_id=? AND month=?",
                        (credits, kid, _month()))

    def mark_exhausted(self, kid: str) -> None:
        self._row(kid)
        self.db.execute("UPDATE key_usage SET exhausted=1 WHERE key_id=? AND month=?", (kid, _month()))
        self.db.log_event("credit", {"service": "budget", "action": "key_exhausted", "key_id": kid})

    def needs_sync(self, kid: str, every_s: float = 900) -> bool:
        synced = self._row(kid)["synced_at"]
        return synced is None or time.time() - synced > every_s

    def sync(self, kid: str, used: float | None, limit: float | None) -> None:
        """Replace the local estimate with the account's own counter."""
        row = self._row(kid)
        self.db.execute(
            "UPDATE key_usage SET credits=?, credit_limit=?, synced_at=? WHERE key_id=? AND month=?",
            (row["credits"] if used is None else float(used),
             row["credit_limit"] if limit is None else float(limit), time.time(), kid, _month()))

    def mark_synced(self, kid: str) -> None:
        self._row(kid)
        self.db.execute("UPDATE key_usage SET synced_at=? WHERE key_id=? AND month=?", (time.time(), kid, _month()))

    def fraction_left(self) -> float:
        """Share of the usable credits (all keys) still available, 0..1."""
        if not self.keys:
            return 0.0
        usable = sum(self.limit(k.id) * (1.0 - self.reserve) for k in self.keys)
        return round(sum(self.remaining(k.id) for k in self.keys) / usable, 3) if usable else 0.0

    def summary(self) -> list[dict]:
        out = []
        for k in self.keys:
            row = self._row(k.id)
            out.append({"key_id": k.id, "used": round(row["credits"], 1), "limit": self.limit(k.id),
                        "remaining": round(self.remaining(k.id), 1), "exhausted": bool(row["exhausted"]),
                        "synced": row["synced_at"] is not None})
        return out
