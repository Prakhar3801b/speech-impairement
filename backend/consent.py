"""Consent enforcement for voice cloning.

Rules (PS16: "enforce consent for any voice cloning", "use only consented voices"):
* a clone request must reference an active (non-revoked) consent record;
* the consent statement must name the voice owner and the purpose, and the
  signer must actively agree (typed name must match + checkbox);
* revoking consent deletes every voice created under it (locally and at ElevenLabs);
* every consent action is written to the audit log (events table, kind='consent').
"""
from __future__ import annotations

import hashlib
import time

from .db import Database

STATEMENT_TEMPLATE = (
    "I, {signer}, {relation_text} {owner}, consent to creating a synthetic copy of "
    "{owner}'s voice for the purpose of: {purpose}. I understand the voice will be used only "
    "inside this assistive communication app, that I can revoke this consent at any time, and that "
    "revoking it deletes the cloned voice."
)


class ConsentError(ValueError):
    pass


def build_statement(owner: str, signer: str, relationship: str, purpose: str) -> str:
    relation_text = "being" if relationship == "self" else f"acting as {relationship} of"
    return STATEMENT_TEMPLATE.format(signer=signer, owner=owner, relation_text=relation_text,
                                     purpose=purpose)


def create(db: Database, owner: str, signer: str, relationship: str, purpose: str,
           typed_signature: str, agreed: bool, statement_audio: bytes | None = None) -> dict:
    owner, signer, purpose = owner.strip(), signer.strip(), purpose.strip()
    if not (owner and signer and purpose):
        raise ConsentError("voice owner, signer and purpose are required")
    if not agreed:
        raise ConsentError("the signer must explicitly agree to the consent statement")
    if typed_signature.strip().lower() != signer.lower():
        raise ConsentError("typed signature must match the signer's name")
    if relationship != "self" and relationship not in ("guardian", "legal representative", "caregiver"):
        raise ConsentError("relationship must be self, guardian, legal representative or caregiver")
    statement = build_statement(owner, signer, relationship, purpose)
    audio_hash = hashlib.sha256(statement_audio).hexdigest() if statement_audio else None
    cur = db.execute(
        "INSERT INTO consent_records(voice_owner, signer, relationship, purpose, statement, "
        "statement_audio_hash, created_at) VALUES(?,?,?,?,?,?,?)",
        (owner, signer, relationship, purpose, statement, audio_hash, time.time()))
    cid = int(cur.lastrowid)
    db.log_event("consent", {"action": "granted", "consent_id": cid, "owner": owner, "signer": signer})
    return get(db, cid)


def get(db: Database, consent_id: int) -> dict:
    row = db.one("SELECT * FROM consent_records WHERE id=?", (consent_id,))
    if not row:
        raise ConsentError(f"consent record {consent_id} not found")
    return row


def require_active(db: Database, consent_id: int | None) -> dict:
    if consent_id is None:
        raise ConsentError("voice cloning requires a consent record")
    row = get(db, consent_id)
    if row["revoked_at"]:
        raise ConsentError("consent has been revoked; this voice cannot be cloned")
    return row


def revoke(db: Database, consent_id: int) -> list[str]:
    """Mark consent revoked; return voice ids that must now be deleted."""
    get(db, consent_id)
    now = time.time()
    db.execute("UPDATE consent_records SET revoked_at=? WHERE id=? AND revoked_at IS NULL", (now, consent_id))
    voices = [r["voice_id"] for r in db.query(
        "SELECT voice_id FROM voices WHERE consent_id=? AND deleted_at IS NULL", (consent_id,))]
    db.execute("UPDATE voices SET deleted_at=? WHERE consent_id=?", (now, consent_id))
    for vid in voices:  # cached audio in the revoked voice must go too
        db.execute("DELETE FROM tts_cache WHERE voice_id=?", (vid,))
    db.log_event("consent", {"action": "revoked", "consent_id": consent_id, "deleted_voices": voices})
    return voices
