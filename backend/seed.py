"""Starter phrasebook (English, Hindi, Hinglish). Patients/caregivers edit it in the UI.

Each entry: text, language, category, aliases (other spellings/scripts the ASR or
sign spelling may produce), keywords (partner words that make it likely), emergency.
"""
from __future__ import annotations

import json
import time

from .db import Database

PHRASES: list[dict] = [
    # --- yes / no / control
    dict(text="Yes", language="en", category="answer", aliases=["yeah", "yes please", "haan", "हाँ", "ha"],
         keywords=[]),
    dict(text="No", language="en", category="answer", aliases=["nope", "nahi", "नहीं", "na"],
         keywords=[]),
    dict(text="Haan", language="hi", category="answer", aliases=["हाँ", "ha", "han", "yes"],
         keywords=[]),
    dict(text="Nahi", language="hi", category="answer", aliases=["नहीं", "nahin", "na", "no"],
         keywords=[]),
    dict(text="I don't know", language="en", category="answer", aliases=["dont know", "pata nahi"],
         keywords=["know", "where", "when", "what"]),
    dict(text="Please wait, I am still talking", language="en", category="control",
         aliases=["wait", "ruko", "ruk jao", "रुको"], keywords=[]),
    dict(text="Please repeat that", language="en", category="control",
         aliases=["repeat", "again", "phir se bolo", "dobara"], keywords=[]),
    dict(text="Thank you", language="en", category="social", aliases=["thanks", "dhanyavaad", "shukriya", "धन्यवाद"],
         keywords=["here", "done", "lijiye", "le lo"]),
    dict(text="Hello, how are you?", language="en", category="social", aliases=["hello", "hi", "namaste", "नमस्ते"],
         keywords=["hello", "hi", "namaste", "good morning"]),
    dict(text="I am fine", language="en", category="social", aliases=["fine", "theek hoon", "main theek hoon", "ठीक हूँ"],
         keywords=["how are you", "kaise ho", "kaise hain", "feeling", "kaisa"]),
    dict(text="Main theek hoon", language="hi", category="social", aliases=["मैं ठीक हूँ", "theek hoon", "fine"],
         keywords=["kaise ho", "kaise hain", "kaisa lag", "tabiyat"]),
    # --- needs
    dict(text="I want water", language="en", category="need", aliases=["water", "paani", "pani", "पानी", "drink"],
         keywords=["water", "drink", "thirsty", "paani", "pani", "pina", "peena"]),
    dict(text="Mujhe paani chahiye", language="hi", category="need",
         aliases=["paani chahiye", "pani", "मुझे पानी चाहिए", "paani", "water"],
         keywords=["paani", "pani", "pina", "peena", "chahiye", "pyaas"]),
    dict(text="I would like tea, please", language="en", category="need", aliases=["tea", "chai", "चाय"],
         keywords=["tea", "chai", "coffee", "drink"]),
    dict(text="Chai chahiye", language="mixed", category="need", aliases=["chai", "चाय चाहिए", "tea"],
         keywords=["chai", "tea", "peena", "chahiye"]),
    dict(text="I am hungry", language="en", category="need", aliases=["hungry", "food", "khana", "bhookh", "भूख"],
         keywords=["eat", "food", "lunch", "dinner", "breakfast", "khana", "hungry"]),
    dict(text="Mujhe bhookh lagi hai", language="hi", category="need",
         aliases=["bhookh", "khana", "मुझे भूख लगी है", "hungry"], keywords=["khana", "khaana", "bhookh", "nashta"]),
    dict(text="I need to use the toilet", language="en", category="need",
         aliases=["toilet", "bathroom", "washroom", "restroom", "bathroom jaana hai"],
         keywords=["toilet", "bathroom", "washroom"]),
    dict(text="Mujhe bathroom jaana hai", language="mixed", category="need",
         aliases=["bathroom", "toilet", "washroom jaana hai"], keywords=["bathroom", "toilet"]),
    dict(text="I am tired, I want to rest", language="en", category="need",
         aliases=["tired", "rest", "sleep", "thak gaya", "aaram", "neend"], keywords=["tired", "sleep", "rest", "bed"]),
    dict(text="Please turn on the fan", language="en", category="need", aliases=["fan", "pankha", "hot", "garmi"],
         keywords=["hot", "warm", "fan", "garmi"]),
    dict(text="I am cold, please give me a blanket", language="en", category="need",
         aliases=["cold", "blanket", "thand", "kambal"], keywords=["cold", "blanket", "thand", "sardi"]),
    dict(text="Please change my position", language="en", category="need",
         aliases=["position", "turn me", "move", "karvat"], keywords=["comfortable", "position", "lying"]),
    dict(text="Can you call my family?", language="en", category="need",
         aliases=["family", "call", "phone", "ghar", "ghar wale"], keywords=["call", "phone", "family", "visit"]),
    # --- medical / feelings
    dict(text="I am in pain", language="en", category="medical", aliases=["pain", "hurts", "dard", "दर्द", "ouch"],
         keywords=["pain", "hurt", "feeling", "dard", "how are you"]),
    dict(text="Mujhe dard ho raha hai", language="hi", category="medical",
         aliases=["dard", "dard ho raha hai", "मुझे दर्द हो रहा है", "pain"], keywords=["dard", "kaisa", "tabiyat"]),
    dict(text="The pain is in my chest", language="en", category="medical", aliases=["chest", "chest pain", "seene mein dard"],
         keywords=["where", "pain", "kahan"]),
    dict(text="The pain is in my head", language="en", category="medical", aliases=["head", "headache", "sar dard", "sir dard"],
         keywords=["where", "pain", "kahan"]),
    dict(text="I need my medicine", language="en", category="medical", aliases=["medicine", "dawai", "dawa", "दवाई", "tablet"],
         keywords=["medicine", "dawai", "tablet", "dose"]),
    dict(text="I feel better now", language="en", category="medical", aliases=["better", "accha", "behtar"],
         keywords=["how are you", "feeling", "better", "kaise"]),
    dict(text="I feel dizzy", language="en", category="medical", aliases=["dizzy", "chakkar", "चक्कर"],
         keywords=["feeling", "how", "kaisa"]),
    # --- emergency
    dict(text="Help! Please call the nurse now", language="en", category="emergency",
         aliases=["help", "nurse", "emergency", "madad", "bachao", "मदद"], keywords=[], is_emergency=True),
    dict(text="I can't breathe", language="en", category="emergency",
         aliases=["breathe", "breath", "saans", "सांस", "saans nahi"], keywords=[], is_emergency=True),
]


def seed_phrases(db: Database) -> int:
    """Insert the starter phrasebook if the table is empty. Returns rows inserted."""
    if db.one("SELECT COUNT(*) AS n FROM phrases")["n"]:
        return 0
    now = time.time()
    db.executemany(
        "INSERT INTO phrases(text, language, category, aliases, keywords, is_emergency, created_at) "
        "VALUES(?,?,?,?,?,?,?)",
        [(p["text"], p["language"], p["category"], json.dumps(p.get("aliases", []), ensure_ascii=False),
          json.dumps(p.get("keywords", []), ensure_ascii=False), int(p.get("is_emergency", False)), now)
         for p in PHRASES])
    return len(PHRASES)
