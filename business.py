"""Who the agent works for: a per-company BusinessProfile.

Everything that changes from one company to the next lives here instead of in the
prompts or the code: what callers book (a consultation, an appointment, a site
visit, a table), what Maya offers, address and opening hours, brand names to keep
in English, speech-to-text fixes for those names, and the persona's gender (Hindi
verbs change with it: "सकती हूँ" / "सकता हूँ").

Where a profile comes from, later sources winning:
  1. DEFAULTS: neutral values that work for any business.
  2. The knowledge base: one Gemini call reads the company's KB and fills the
     profile. Cached on disk per KB version (data/profiles/), so it runs once per
     KB change, never inside a call that already has a cached profile.
  3. business_profiles/<org_agent_id>.json (optional, in the repo): manual fixes,
     e.g. speech-to-text aliases the KB can't know about.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger("voice-agent.business")

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "profiles"
OVERRIDES_DIR = ROOT / "business_profiles"
PROFILE_VERSION = 2  # bump when the extraction prompt changes, to refresh caches


# Callback numbers: Indian 10-digit mobiles only for now ("landline", "international" exist but are off).
ALLOWED_PHONE_TYPES = ("mobile",)


@dataclass
class BusinessProfile:
    name: str = "our company"
    persona: str = "Maya"
    persona_hi: str = "माया"                 # how the persona's name is written in Hindi
    gender: str = "female"                   # persona gender: female / male (Hindi verb forms)
    summary: str = ""                        # one line: what the business does
    booking_en: str = "consultation"         # what a caller books ("appointment", "site visit", ...)
    booking_hi: str = "consultation"
    offer_en: str = "a callback from our team"   # what Maya offers interested callers
    offer_hi: str = "हमारी टीम से एक कॉल"
    discovery_en: str = "what they need"     # what to ask callers about to understand their need
    booking_words: list[str] = field(default_factory=list)  # extra caller words that mean "book it"
    address: str | None = None
    hours: str | None = None                 # opening hours, if the KB has them
    emergency_number: str | None = None      # emergency / 24x7 helpline, if the KB has one
    brands: list[str] = field(default_factory=list)          # names to keep as-is (English)
    aliases: dict[str, str] = field(default_factory=dict)    # STT mishearing regex -> brand name
    # Callback numbers accepted (limited to ALLOWED_PHONE_TYPES).
    phone_types: list[str] = field(default_factory=lambda: ["mobile"])

    @classmethod
    def from_dict(cls, data: dict, base: "BusinessProfile | None" = None) -> "BusinessProfile":
        """Overlay the known, non-empty keys of `data` on `base` (or the defaults)."""
        merged = asdict(base or cls())
        names = {f.name for f in fields(cls)}
        for k, v in (data or {}).items():
            if k in names and v not in (None, "", [], {}):
                merged[k] = v
        if merged["gender"] not in ("female", "male"):
            merged["gender"] = "female"
        # For now callbacks take Indian 10-digit mobiles only. contact_flow.clean_phone
        # also knows "landline" / "international": add them to ALLOWED_PHONE_TYPES to enable.
        merged["phone_types"] = [t for t in merged["phone_types"] if t in ALLOWED_PHONE_TYPES] or ["mobile"]
        return cls(**merged)

    def booking(self, lang: str) -> str:
        return self.booking_hi if lang == "hi" else self.booking_en

    def offer(self, lang: str) -> str:
        return self.offer_hi if lang == "hi" else self.offer_en

    @property
    def booking_regex(self) -> re.Pattern | None:
        """Caller words that mean "book it" for this business (e.g. "checkup", "टेबल")."""
        words = [w.strip() for w in self.booking_words if w and w.strip()]
        if not words:
            return None
        return re.compile("|".join(rf"(?<!\w){re.escape(w)}(?!\w)" for w in words), re.I)


DEFAULT_PROFILE = BusinessProfile()

# Female -> male first-person Hindi forms for the fixed lines Maya speaks.
_MALE_FORMS = [
    ("सकती", "सकता"), ("करूँगी", "करूँगा"), ("देखती", "देखता"), ("बताती", "बताता"), ("लेती", "लेता"),
    ("देती", "देता"), ("रही हूँ", "रहा हूँ"), ("समझ गई", "समझ गया"), ("चाहती", "चाहता"), ("पाई", "पाया"),
]


def gendered(text: str, gender: str) -> str:
    """Hindi first-person verb forms for the persona's gender (lines are written female)."""
    if gender != "male" or not text:
        return text
    for female, male in _MALE_FORMS:
        text = text.replace(female, male)
    return text


# --------------------------------------------------------------- KB extraction
_EXTRACT_PROMPT = """You set up a phone receptionist for a business. Read the business's knowledge base and
return ONE JSON object with exactly these keys (use null when the knowledge base doesn't say):

  "name":          the business name,
  "summary":       one short sentence: what the business does and for whom,
  "booking_en":    the ONE thing callers most often book or request, as a short English noun phrase
                   (examples: "consultation", "appointment", "site visit", "table reservation", "demo class",
                   "callback"),
  "booking_hi":    the same in natural spoken Hindi (Devanagari; English loanwords are fine),
  "offer_en":      what the receptionist should offer an interested caller, in English, starting with "a"/"an"
                   (e.g. "a free consultation" ONLY if the knowledge base says it is free, else "an appointment"),
  "offer_hi":      the same in natural spoken Hindi (Devanagari),
  "discovery_en":  what to ask a caller about to understand their need, as a short phrase
                   (e.g. "their dental problem", "their budget and preferred location", "their business"),
  "booking_words": 3-8 words or short phrases callers use to ask for that booking, English and Hindi
                   (e.g. ["appointment", "checkup", "अपॉइंटमेंट", "डॉक्टर से मिलना"]),
  "address":       the full address of the office / shop / clinic, exactly as written; if there are several
                   branches, list each as "Branch name: address" separated by " | ",
  "hours":         opening hours / timings exactly as written (null if not stated),
  "emergency_number": an emergency, ambulance or 24x7 helpline number, if one is given,
  "brands":        product, service and brand names that must be said exactly as written (max 15).

Return only the JSON object.

KNOWLEDGE BASE:
"""

_ADDRESS_LINE = re.compile(
    r"(?:^|\n|\s)(?:Office|Address|Location|Head office|Clinic|Shop|पता|ऑफिस)\s*[:\-]\s*(.+?)(?:\s+(?:Web|Website|Phone|Email|Hours|Timings)\s*:|\n|$)",
    re.I,
)
_HOURS_LINE = re.compile(
    r"(?:Hours|Timings?|Opening hours|Working hours|Open|समय|टाइमिंग)\s*[:\-]\s*(.+?)(?:\n|$)", re.I
)


def _from_kb_lines(kb_text: str) -> dict:
    """Cheap fallback when the LLM is unavailable: "Address: ..." / "Timings: ..." lines."""
    out = {}
    if m := _ADDRESS_LINE.search(kb_text or ""):
        out["address"] = m.group(1).strip(" .")
    if m := _HOURS_LINE.search(kb_text or ""):
        out["hours"] = m.group(1).strip(" .")
    return out


def _extract_with_llm(kb_text: str, model: str) -> dict:
    """One Gemini call over the KB -> profile fields. Blocking; run it in a thread."""
    from google import genai
    from google.genai import types

    from config import GOOGLE_API_KEY  # loads .env

    client = genai.Client(api_key=GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=model,
        contents=_EXTRACT_PROMPT + kb_text[:30000],
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    data = json.loads(resp.text or "{}")
    return data if isinstance(data, dict) else {}


def _cache_path(agent_id: str | None, kb_text: str) -> Path:
    digest = hashlib.sha1(f"{PROFILE_VERSION}:{kb_text}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{agent_id or 'default'}-{digest}.json"


def _overrides(agent_id: str | None) -> dict:
    path = OVERRIDES_DIR / f"{agent_id}.json" if agent_id else None
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.exception("bad business profile override %s", path)
        return {}


def _config_fields(name: str | None, persona: str | None) -> dict:
    """What the agent's own config (Supabase agents row) says: wins over the KB."""
    # A placeholder name ("our company") must not beat the real one read from the KB.
    out = {"name": name if name and name != DEFAULT_PROFILE.name else None, "persona": persona}
    if persona and persona != DEFAULT_PROFILE.persona:
        out["persona_hi"] = persona  # unknown spelling in Hindi: keep the name as is
    return out


def _layer(extracted: dict, agent_id: str | None, name: str | None, persona: str | None) -> BusinessProfile:
    """defaults < KB extraction < agent config < overrides file."""
    p = BusinessProfile.from_dict(extracted)
    p = BusinessProfile.from_dict(_config_fields(name, persona), p)
    return BusinessProfile.from_dict(_overrides(agent_id), p)


def cached_profile(agent_id: str | None, kb_text: str | None, name: str | None = None,
                   persona: str | None = None) -> BusinessProfile | None:
    """The profile if it's already on disk for this KB version (fast), else None."""
    path = _cache_path(agent_id, kb_text or "")
    if not kb_text or not path.exists():
        return None
    try:
        extracted = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _layer(extracted, agent_id, name, persona)


def build_profile(agent_id: str | None, kb_text: str | None, name: str | None = None,
                  persona: str | None = None, model: str | None = None) -> BusinessProfile:
    """Profile for one org agent. Blocking (may call Gemini once per KB version); never raises."""
    if not kb_text:
        return _layer({}, agent_id, name, persona)
    hit = cached_profile(agent_id, kb_text, name, persona)
    if hit:
        return hit
    extracted = _from_kb_lines(kb_text)
    try:
        extracted.update({k: v for k, v in _extract_with_llm(kb_text, model or "gemini-2.5-flash").items() if v})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(agent_id, kb_text).write_text(json.dumps(extracted, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("business profile extracted for %s: booking=%r", agent_id, extracted.get("booking_en"))
    except Exception:  # noqa: BLE001 - fall back to defaults + KB lines, retried next call
        log.exception("business profile extraction failed for %s; using defaults", agent_id)
    return _layer(extracted, agent_id, name, persona)
