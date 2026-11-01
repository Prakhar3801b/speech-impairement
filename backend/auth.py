"""Accounts and login.

Each account (a patient, or the caregiver who sets Clio up for them) has its own data store:
on Postgres (Neon) a separate schema `u_<id>`, on SQLite a separate file data/users/u_<id>.db.
The shared "core" store only holds accounts, login sessions, API-key budgets and a server secret.

Passwords are hashed with scrypt (standard library). Login sessions are random tokens kept in an
HttpOnly cookie; only their SHA-256 is stored. The partner's phone gets a signed link instead of
a login: it can add partner turns to, and read, one conversation of one account.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from pathlib import Path

from .db import Database, is_postgres

CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    pw_hash     TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
"""
COOKIE = "clio_session"
SESSION_DAYS = 30
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(ValueError):
    pass


def hash_password(pw: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_password(pw: str, stored: str) -> bool:
    try:
        _, salt, _ = stored.split("$")
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(pw, bytes.fromhex(salt)), stored)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Accounts:
    def __init__(self, url: str):
        """`url`: DATABASE_URL (Postgres) or the core SQLite file path."""
        self.url = url
        self.core = Database(url, ddl=CORE_SCHEMA)
        secret = self.core.get_setting("server_secret")
        if not secret:
            secret = secrets.token_hex(32)
            self.core.execute("INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)", ("server_secret", secret))
            secret = self.core.get_setting("server_secret")
        self.secret = secret.encode()

    # ------------------------------------------------------------ accounts
    def signup(self, email: str, password: str, name: str) -> dict:
        email, name = email.strip().lower(), name.strip()
        if not EMAIL_RE.match(email):
            raise AuthError("Enter a valid email address.")
        if len(password) < 8:
            raise AuthError("Use a password of at least 8 characters.")
        if not name:
            raise AuthError("Enter a name.")
        if self.core.one("SELECT id FROM users WHERE email=?", (email,)):
            raise AuthError("An account with this email already exists. Sign in instead.")
        uid = self.core.insert("INSERT INTO users(email, name, pw_hash, created_at) VALUES(?,?,?,?)",
                               (email, name, hash_password(password), time.time()))
        return self.user(uid)

    def login(self, email: str, password: str) -> dict:
        row = self.core.one("SELECT * FROM users WHERE email=?", (email.strip().lower(),))
        if not row or not check_password(password, row["pw_hash"]):
            raise AuthError("Wrong email or password.")
        return self.user(row["id"])

    def user(self, uid: int) -> dict | None:
        return self.core.one("SELECT id, email, name, created_at FROM users WHERE id=?", (uid,))

    # ------------------------------------------------------------ sessions
    def start_session(self, uid: int) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self.core.execute("INSERT INTO auth_sessions(token_hash, user_id, created_at, expires_at) VALUES(?,?,?,?)",
                          (_token_hash(token), uid, now, now + SESSION_DAYS * 86400))
        return token

    def user_for_token(self, token: str | None) -> dict | None:
        if not token:
            return None
        row = self.core.one("SELECT user_id, expires_at FROM auth_sessions WHERE token_hash=?", (_token_hash(token),))
        if not row or row["expires_at"] < time.time():
            return None
        return self.user(row["user_id"])

    def end_session(self, token: str | None) -> None:
        if token:
            self.core.execute("DELETE FROM auth_sessions WHERE token_hash=?", (_token_hash(token),))

    # ------------------------------------------------------------ partner links
    def partner_key(self, uid: int, session_id: str) -> str:
        sig = hmac.new(self.secret, f"{uid}:{session_id}".encode(), hashlib.sha256).hexdigest()[:32]
        return f"{uid}.{sig}"

    def check_partner_key(self, key: str | None, session_id: str) -> int | None:
        """The account id the key was issued for, if it is valid for this conversation."""
        if not key or "." not in key:
            return None
        uid, _ = key.split(".", 1)
        if not uid.isdigit():
            return None
        return int(uid) if hmac.compare_digest(self.partner_key(int(uid), session_id), key) else None

    # ------------------------------------------------------------ per-account storage
    def open_user_db(self, uid: int, sqlite_dir: Path) -> Database:
        if is_postgres(self.url):
            return Database(self.url, schema=f"u_{int(uid)}")
        return Database(str(sqlite_dir / f"u_{int(uid)}.db"))
