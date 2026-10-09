"""Deterministic name + phone collection, driven by code instead of the LLM.

The LLM only decides *that* the caller wants a callback/consultation (by calling
capture_lead / book_consultation). From then on every caller turn is handled here
until the details are saved, so the agent can't loop, invent numbers or forget
the name:

    ask name -> "would you like to get a call back on this number or another?"
        same      -> read the caller ID back -> "is that correct?"
        different -> ask for ten digits; partial digits are collected across turns
                     ("okay, 7 6 7 7 6. Could you tell me the remaining 5 digits?")
                     -> read all ten back -> "is that correct?"
    yes -> save;  no -> ask for the full number again

All lines exist in English and Hindi; the caller's current language picks one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from business import gendered
from intents import (
    ALREADY_TOLD, DIFFERENT, NO, NOT_A_NAME, NOT_A_YES, REFUSE, SAME, TIME_OR_DATE, YES,
)

TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "ask_name": "May I know your name, please?",
        "sure": "Sure! ",
        "drop": "No problem, let's skip that. What else can I help you with?",
        "ask_name_again": "Sorry, I didn't catch your name. Could you tell me just your name?",
        "name_missed": "Sorry, I missed it. Could you tell me your name once more?",
        "name_updated": "Got it, {name}.",
        "no_caller_id": "Sorry, I can't see the number you're calling from. Could you tell me your ten-digit mobile number?",
        "time_noted": "Got it, I've noted that time. Could you tell me your ten-digit mobile number?",
        "ask_choice": "Thanks, {name}. Would you like to get a callback on this number, or another one?",
        "ask_choice_again": "Sorry, should we call you back on the number you're calling from, or on a different number?",
        "ask_digits": "Sure. Could you tell me your ten-digit mobile number?",
        "ask_digits_named": "Thanks, {name}. Could you please share your ten-digit mobile number for the callback?",
        "ask_rest": "Okay, {got}. Could you tell me the remaining {n} digit{s}?",
        "too_many": "Sorry, I got more than ten digits. Could you say your ten-digit number again from the start?",
        "invalid": "Sorry, I can only take a ten-digit Indian mobile number, starting with 6, 7, 8 or 9. Could you say it again?",
        "invalid_landline": "Sorry, that doesn't look like a valid number. Could you tell me a ten-digit mobile number, or a landline with its STD code?",
        "invalid_intl": "Sorry, that doesn't look like a valid number. Could you say it again? For a number outside India, start with the country code.",
        "skip_number": "No problem, let's leave the number for now. Is there anything else I can help you with?",
        "confirm": "Just to confirm, your number is {spoken}. Is that correct?",
        "confirm_again": "Sorry, is {spoken} the right number? Please say yes or no.",
        "retry_digits": "No problem. Could you tell me the full ten-digit number again?",
        "saved_lead": "Thank you, {name}! Your details are saved and the {business} team will contact you soon. Is there anything else I can help you with?",
        "saved_booking": "Thank you, {name}! Your {booking} request is booked and the {business} team will contact you soon. Is there anything else I can help you with?",
    },
    "hi": {
        "ask_name": "क्या मैं आपका नाम जान सकती हूँ?",
        "sure": "ज़रूर! ",
        "drop": "कोई बात नहीं, रहने देते हैं। बताइए, मैं और कैसे मदद कर सकती हूँ?",
        "ask_name_again": "माफ़ कीजिए, नाम ठीक से सुनाई नहीं दिया। क्या आप सिर्फ़ अपना नाम बता सकते हैं?",
        "name_missed": "माफ़ कीजिए, मुझसे छूट गया। क्या आप एक बार फिर अपना नाम बता सकते हैं?",
        "name_updated": "ठीक है, {name} जी।",
        "no_caller_id": "माफ़ कीजिए, मुझे आपका नंबर दिखाई नहीं दे रहा। क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "time_noted": "ठीक है, मैंने समय नोट कर लिया है। क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "ask_choice": "धन्यवाद {name} जी। क्या आप इसी नंबर पर कॉल बैक चाहेंगे, या किसी दूसरे नंबर पर?",
        "ask_choice_again": "माफ़ कीजिए, क्या हम आपको इसी नंबर पर कॉल करें, या किसी दूसरे नंबर पर?",
        "ask_digits": "ज़रूर। क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "ask_digits_named": "धन्यवाद {name} जी। कॉल बैक के लिए क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "ask_rest": "ठीक है, {got}। क्या आप बाकी के {n} अंक बता सकते हैं?",
        "too_many": "माफ़ कीजिए, मुझे दस से ज़्यादा अंक मिले। क्या आप अपना दस अंकों का नंबर शुरू से दोबारा बता सकते हैं?",
        "invalid": "माफ़ कीजिए, मैं सिर्फ़ दस अंकों का भारतीय मोबाइल नंबर ले सकती हूँ, जो 6, 7, 8 या 9 से शुरू हो। क्या आप दोबारा बता सकते हैं?",
        "invalid_landline": "माफ़ कीजिए, यह नंबर सही नहीं लग रहा। क्या आप दस अंकों का मोबाइल नंबर, या STD कोड के साथ लैंडलाइन नंबर बता सकते हैं?",
        "invalid_intl": "माफ़ कीजिए, यह नंबर सही नहीं लग रहा। क्या आप दोबारा बता सकते हैं? भारत के बाहर का नंबर हो तो country code से शुरू करें।",
        "skip_number": "कोई बात नहीं, नंबर अभी रहने देते हैं। क्या मैं आपकी और किसी चीज़ में मदद कर सकती हूँ?",
        "confirm": "एक बार कन्फ़र्म कर लूँ, आपका नंबर है {spoken}। क्या यह सही है?",
        "confirm_again": "माफ़ कीजिए, क्या {spoken} सही नंबर है? कृपया हाँ या ना बताइए।",
        "retry_digits": "कोई बात नहीं। क्या आप पूरा दस अंकों का नंबर दोबारा बता सकते हैं?",
        "saved_lead": "धन्यवाद {name} जी! आपकी जानकारी सेव हो गई है, {business} की टीम जल्द ही आपसे संपर्क करेगी। क्या मैं आपकी और किसी चीज़ में मदद कर सकती हूँ?",
        "saved_booking": "धन्यवाद {name} जी! आपकी {booking} request दर्ज हो गई है, {business} की टीम जल्द ही आपसे संपर्क करेगी। क्या मैं आपकी और किसी चीज़ में मदद कर सकती हूँ?",
    },
}

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Spoken digits: STT often writes "nine eight seven..." / "नौ आठ सात..." instead of numerals.
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "shunya": "0", "sunya": "0", "ek": "1", "do": "2", "teen": "3", "tin": "3", "char": "4", "chaar": "4",
    "paanch": "5", "panch": "5", "chhe": "6", "che": "6", "chhah": "6", "saat": "7", "sat": "7",
    # Not "no" for nau (9): "no no no" correcting a read-back became "999".
    "aath": "8", "ath": "8", "nau": "9",
    "शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4",
    "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6", "छे": "6", "छ": "6", "सात": "7", "आठ": "8", "नौ": "9",
}
_REPEAT_WORDS = {"double": 2, "triple": 3, "डबल": 2, "ट्रिपल": 3}
# Words that are also ordinary words ("do you", "no", "oh"): only digits inside a run of them.
_MIN_DIGIT_WORDS = 3


def _words_to_digits(text: str) -> str:
    """Turn runs of spoken digits into numerals: "double nine eight" -> "998"."""
    tokens = re.findall(r"[\wऀ-ॿ]+|\S", text)
    out: list[str] = []
    run: list[str] = []
    words = [t for t in tokens if re.match(r"[\wऀ-ॿ]", t)]
    only_digits = bool(words) and all(
        t.lower() in _DIGIT_WORDS or t.lower() in _REPEAT_WORDS or t.isdigit() for t in words
    )
    min_run = 1 if only_digits else _MIN_DIGIT_WORDS

    def flush() -> None:
        if len([t for t in run if t.lower() in _DIGIT_WORDS]) >= min_run or any(
            t.lower() in _REPEAT_WORDS for t in run
        ):
            i = 0
            while i < len(run):
                t = run[i].lower()
                if t in _REPEAT_WORDS and i + 1 < len(run) and run[i + 1].lower() in _DIGIT_WORDS:
                    out.append(_DIGIT_WORDS[run[i + 1].lower()] * _REPEAT_WORDS[t])
                    i += 2
                    continue
                out.append(_DIGIT_WORDS.get(t, run[i]))
                i += 1
        else:
            out.extend(run)
        run.clear()

    for tok in tokens:
        low = tok.lower()
        if low in _DIGIT_WORDS or low in _REPEAT_WORDS or tok.isdigit():
            run.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


# Robust Name extraction patterns
_NAME_PATTERNS = [
    # Not "call me X": "call me back please" / "call me tomorrow morning" became names.
    re.compile(r"(?:my name is|name is|name's|i am|i'm|this is|it's|myself|this side)\s+(.+)", re.I),
    re.compile(r"(?:mera|merra|meraa|meri|humara|my)?\s*(?:naam|nam|name)\s+(?:hai\s+|is\s+)?(.+)", re.I),
    re.compile(r"(?:मेरा|मेरी|हमारा)?\s*नाम\s+(?:है\s+)?(.+)"),
    re.compile(r"(?:main|mai|hum)\s+(.+?)\s+(?:bol raha|bol rahi|hoon|hun|baat kar)", re.I),
    re.compile(r"मैं\s+(.+?)\s+(?:हूँ|हूं|बोल\s+रह[ाी])"),
    re.compile(r"(.+?)\s+(?:here|this side|speaking|bol raha|bol rahi)", re.I),
    re.compile(r"(.+?)\s+(?:नाम\s+है|बोल\s+रह[ाी])"),
]

_FILLERS_PAT = re.compile(
    r"^(?:(?:\b(?:ji|haan|ha|yes|yeah|yep|sure|ok|okay|hello|hi|hey|sir|madam|मैडम|सर|जी|हाँ|हां|अच्छा|ठीक है|अरे|भाई|हेलो)\b|[!?,.।-])\s*)+",
    re.I,
)
_STOP_PAT = re.compile(
    r"\s+(?:here|this side|speaking|from|bol raha.*|bol rahi.*|baat kar.*|hai|hain|hoon|hun|ji|sir|madam|है|हैं|हे|हूँ|हूं|बोल\s*रह.*|और|and|but)(?=\s|$|[!?,.।]).*$",
    re.I,
)

_NOT_NAMES = {
    "yes", "no", "yeah", "yep", "nope", "nah", "ok", "okay", "sure", "fine", "good",
    "haan", "han", "nahi", "nahin", "theek", "sahi", "galat", "hello", "hi", "hey",
    "kya", "kyun", "kaise", "kab", "kahan", "who", "what", "why", "how", "when", "where",
    "nothing", "none", "user", "caller", "test", "demo", "callback", "consultation",
    "हाँ", "हां", "नहीं", "नही", "ठीक", "सही", "गलत", "नमस्ते", "हेलो"
}


def name_from_history(user_texts: list[str]) -> str | None:
    """A name the caller already gave earlier ("merra name Himanshu hai" to the LLM),
    newest first. Only explicit introductions count, not any short reply."""
    for t in reversed(user_texts[-6:]):
        if any(p.search(t or "") for p in _NAME_PATTERNS[:5]):
            name = extract_name(t)
            if name:
                return name
    return None


def spoken(digits: str) -> str:
    """'7677672641' -> '7 6 7 7 6 7 2 6 4 1' so TTS reads it digit by digit."""
    return " ".join(digits)


def clean_phone(phone: str | None, types=("mobile",)) -> str | None:
    """A valid phone number for this business, or None.

    types: "mobile" (10 digits, 6-9 first; +91 / 91 / 0 prefixes accepted),
    "landline" (Indian STD code + number, 10 digits without the leading 0, e.g.
    020 2543 1234), "international" (+country code, 8-15 digits, kept with "+")."""
    raw = (phone or "").translate(_DEV_DIGITS).strip()
    d = re.sub(r"\D", "", raw)
    is_plus = raw.startswith("+") or raw.startswith("00")
    if d.startswith("00"):
        d = d[2:]
    if is_plus and not d.startswith("91"):
        return f"+{d}" if "international" in types and 8 <= len(d) <= 15 else None
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    if re.fullmatch(r"[6-9]\d{9}", d):
        return d
    if "landline" in types and re.fullmatch(r"[1-9]\d{9}", d) and not _junk_number(d):
        return d
    return None


def _junk_number(d: str) -> bool:
    """1234567890 / 5555555555-style numbers: any 10 digits fit a landline's shape,
    so these jokes would otherwise pass."""
    return len(set(d)) <= 2 or d in "0123456789012345678" or d in "9876543210987654321"


def extract_digits(text: str) -> str:
    """Digits the caller spoke in this turn, without a leading +91 / 0 country/trunk prefix."""
    t = _words_to_digits((text or "").translate(_DEV_DIGITS))
    t = re.sub(r"\+\s*9\s*1", " ", t)
    return re.sub(r"\D", "", t)


def extract_name(text: str) -> str | None:
    """Extract caller name reliably across English, Hindi and Romanized Hindi."""
    t = (text or "").strip()
    if not t or ALREADY_TOLD.search(t):
        return None  # "abhi tho bola" is "I already told you", not a name
    if "?" in t:
        return None  # "And what's FinanceAuditor?" was saved as a name
    # Strip long numeric strings if caller provided digits in the same turn
    t_clean = re.sub(r"[\+\d\s-]{7,}", " ", t).strip()
    if not t_clean:
        return None

    # Strip leading fillers (e.g. 'Ji, ', 'Haan ', 'Yes, ')
    t_clean = _FILLERS_PAT.sub("", t_clean).strip(" .,।!?\"'")
    if not t_clean:
        return None

    candidate = None
    for pat in _NAME_PATTERNS:
        m = pat.search(t_clean)
        if m:
            candidate = m.group(1).strip()
            break

    if candidate is None:
        candidate = t_clean

    candidate = _STOP_PAT.sub("", candidate).strip(" .,।!?\"'")
    candidate = _FILLERS_PAT.sub("", candidate).strip(" .,।!?\"'")

    words = candidate.split()
    if not words or len(words) > 3:
        return None
    if NOT_A_NAME.search(candidate):
        return None  # "what's FinanceAuditor", "fine, bye", "lol kidding" are not names

    # Reject if single word is a filler / generic word
    if len(words) == 1 and words[0].lower() in _NOT_NAMES:
        return None
    if any(w.lower() in ("what", "how", "why", "when", "kya", "kaise", "kyun") for w in words):
        return None

    clean_res = " ".join(words)
    # Title-case Roman names
    if re.search(r"[A-Za-z]", clean_res) and not re.search(r"[ऀ-ॿ]", clean_res):
        clean_res = clean_res.title()
    return clean_res


@dataclass
class Step:
    """What the agent should do with this caller turn."""

    say: str | None = None   # speak this exact line (and skip the LLM)
    save: bool = False       # details confirmed: save them, then speak `say`
    reprompt: str | None = None  # not handled: let the LLM answer, then re-ask this


class ContactFlow:
    MAX_UNCLEAR = 2  # unclear answers per step before handing back to the LLM

    def __init__(self) -> None:
        # Set per business (tools.AppointmentTools.set_profile): what a booking is
        # called, and the persona's gender for Hindi verb forms.
        self.booking = {"en": "consultation", "hi": "consultation"}
        self.gender = "female"
        self.phone_types: tuple[str, ...] = ("mobile",)  # + "landline" / "international"
        self.reset()

    def reset(self) -> None:
        self.stage: str | None = None  # name | choice | digits | confirm
        self.name: str | None = None
        self.digits = ""
        self.phone: str | None = None
        self.tool = "capture_lead"
        self.requirement: str | None = None
        self.unclear = 0
        self.bad_numbers = 0
        self.caller_phone: str | None = None

    @property
    def active(self) -> bool:
        return self.stage is not None

    def _t(self, key: str, lang: str, **kw) -> str:
        kw.setdefault("booking", self.booking.get(lang, self.booking["en"]))
        return gendered(TEXTS.get(lang, TEXTS["en"])[key].format(**kw), self.gender)

    def start(
        self, tool: str, requirement: str | None, name: str | None, lang: str, caller: str | None,
        opener: bool = True,
    ) -> str:
        """First question. If name is already known, asks choice question immediately."""
        self.reset()
        self.tool, self.requirement = tool, requirement
        self.caller_phone = caller
        if name:
            self.name = name
            return self._after_name(lang, caller)
        self.stage = "name"
        return (self._t("sure", lang) if opener else "") + self._t("ask_name", lang)

    def question(self, lang: str, caller: str | None) -> str:
        """The question the flow is currently waiting on (to re-ask it)."""
        if self.stage == "name":
            return self._t("ask_name", lang)
        if self.stage == "choice":
            return self._t("ask_choice", lang, name=self.name or "")
        if self.stage == "confirm" and self.phone:
            return self._t("confirm", lang, spoken=spoken(self.phone))
        return self._t("ask_digits", lang)

    def saved_text(self, lang: str, business: str) -> str:
        key = "saved_booking" if self.tool == "book_consultation" else "saved_lead"
        return self._t(key, lang, name=self.name or "", business=business)

    # --- per-turn handling -------------------------------------------------
    def handle(self, text: str, lang: str, caller: str | None, is_question: bool = False) -> Step:
        if REFUSE.search(text or ""):
            # Caller wants to drop details: stop asking at once.
            self.reset()
            return Step(say=self._t("drop", lang))
        stage = self.stage
        if stage in ("choice", "digits", "confirm") and any(p.search(text or "") for p in _NAME_PATTERNS[:5]):
            new_name = extract_name(text)
            if new_name and new_name != self.name and not extract_digits(text):
                # "Okay my name is Sneha" / "it's Donald Trump" after the name step:
                # take the latest name, then repeat the pending question.
                self.name = new_name
                if stage == "choice":  # that question already thanks them by name
                    return Step(say=self.question(lang, caller))
                return Step(say=self._t("name_updated", lang, name=new_name) + " " + self.question(lang, caller))
        if stage == "name":
            return self._on_name(text, lang, caller, is_question)
        if stage == "choice":
            return self._on_choice(text, lang, caller, is_question)
        if stage == "digits":
            return self._on_digits(text, lang, is_question, caller)
        if stage == "confirm":
            return self._on_confirm(text, lang, is_question)
        return Step()

    def _unclear(self, lang: str, caller: str | None, again_key: str, is_question: bool, **kw) -> Step:
        self.unclear += 1
        if is_question:
            # The caller asked something else: the LLM answers, then re-asks.
            return Step(reprompt=self.question(lang, caller))
        if self.unclear > self.MAX_UNCLEAR:
            q = self.question(lang, caller)
            self.reset()  # stop insisting; the LLM takes over
            return Step(reprompt=q)
        return Step(say=self._t(again_key, lang, **kw))

    def _after_name(self, lang: str, caller: str | None) -> str:
        self.unclear = 0
        if caller:
            self.stage = "choice"
            return self._t("ask_choice", lang, name=self.name or "")
        self.stage = "digits"
        return self._t("ask_digits_named", lang, name=self.name or "")

    def _on_name(self, text: str, lang: str, caller: str | None, is_question: bool) -> Step:
        # 1. Check if caller said "bola to sahi" / "already told you"
        if ALREADY_TOLD.search(text):
            if self.name:
                return Step(say=self._after_name(lang, caller))
            # The agent looks the name up in the conversation first (name_from_history);
            # only if that found nothing do we get here: apologise and ask once more.
            return Step(say=self._t("name_missed", lang))

        # A bare "nahi" / "no" to "may I know your name?" means they won't give it:
        # stop asking instead of "didn't catch your name" again and again.
        if NO.search(text) and len(text.split()) <= 3 and not YES.search(text):
            self.reset()
            return Step(say=self._t("drop", lang))

        # 2. Check if caller gave a name (even if punctuation/question marks were transcribed)
        name = extract_name(text)
        if name:
            self.name = name
            # Check if caller also provided 10 digits in the same turn
            digits = extract_digits(text)
            if digits and clean_phone(digits, self.phone_types):
                phone = clean_phone(digits, self.phone_types)
                self.phone, self.stage = phone, "confirm"
                return Step(say=self._t("confirm", lang, spoken=spoken(phone)))
            return Step(say=self._after_name(lang, caller))

        return self._unclear(lang, caller, "ask_name_again", is_question)

    def _on_choice(self, text: str, lang: str, caller: str | None, is_question: bool) -> Step:
        if extract_digits(text):
            self.stage, self.digits = "digits", ""
            return self._on_digits(text, lang, False)
        if NO.search(text) and NOT_A_YES.search(text) and not SAME.search(text) and not DIFFERENT.search(text):
            # "नहीं धन्यवाद" / "no thanks" here is declining the callback, not
            # asking for another number: stop asking.
            self.reset()
            return Step(say=self._t("drop", lang))
        different = bool(DIFFERENT.search(text))
        same = bool(SAME.search(text)) or (bool(YES.search(text)) and not NO.search(text))
        if different or (NO.search(text) and not same):
            self.stage, self.digits, self.unclear = "digits", "", 0
            return Step(say=self._t("ask_digits", lang))
        if same and caller:
            self.phone, self.stage, self.unclear = caller, "confirm", 0
            return Step(say=self._t("confirm", lang, spoken=spoken(caller)))
        return self._unclear(lang, caller, "ask_choice_again", is_question)

    def _on_digits(self, text: str, lang: str, is_question: bool, caller: str | None = None) -> Step:
        if TIME_OR_DATE.search(text) and len(extract_digits(text)) < 5:
            # "tomorrow 8 am" is when to call, not the start of a phone number:
            # keep it as the preferred slot and ask for the number again.
            slot = f"Preferred: {text.strip()}"
            self.requirement = f"{self.requirement} ({slot})" if self.requirement else slot
            return Step(say=self._t("time_noted", lang))
        if "international" in self.phone_types and re.search(r"\+\s*\d|\bplus\b|प्लस", text, re.I):
            # "+44 20 7946 0958" / "plus four four...": the whole number in one go.
            spoken_plus = re.sub(r"\bplus\b|प्लस", "+", _words_to_digits(text), flags=re.I)
            m = re.search(r"\+[\d\s-]{6,}", spoken_plus)
            phone = clean_phone(m.group(0), self.phone_types) if m else None
            if phone:
                self.digits, self.phone, self.stage = "", phone, "confirm"
                return Step(say=self._t("confirm", lang, spoken=spoken(phone)))
        new = extract_digits(text)
        if not new:
            self.digits = ""
            if SAME.search(text) or ALREADY_TOLD.search(text):
                if caller:
                    # "fine, use this number" after giving up on another one.
                    self.phone, self.stage = caller, "confirm"
                    return Step(say=self._t("confirm", lang, spoken=spoken(caller)))
                # "same number" when we can't see the caller ID (console, hidden number):
                # say why we still need it instead of repeating the question.
                return Step(say=self._t("no_caller_id", lang))
            return self._unclear(lang, None, "ask_digits", is_question)
        self.unclear = 0
        buf = self.digits + new
        if len(buf) > 10 and clean_phone(new, self.phone_types):
            buf = clean_phone(new, self.phone_types)
        if not self.digits and len(buf) == 11 and buf.startswith("0"):
            buf = buf[1:]
        if len(buf) < 10:
            self.digits = buf
            n = 10 - len(buf)
            return Step(say=self._t("ask_rest", lang, got=spoken(buf), n=n, s="" if n == 1 else "s"))
        if len(buf) > 10:
            self.digits = ""
            return self._bad_number(lang, "too_many")
        phone = clean_phone(buf, self.phone_types)
        self.digits = ""
        if not phone:
            return self._bad_number(lang, self._invalid_key())
        self.phone, self.stage = phone, "confirm"
        return Step(say=self._t("confirm", lang, spoken=spoken(phone)))

    def _invalid_key(self) -> str:
        """The "that number isn't valid" line that matches what this business accepts."""
        if "international" in self.phone_types:
            return "invalid_intl"
        return "invalid_landline" if "landline" in self.phone_types else "invalid"

    def _bad_number(self, lang: str, key: str) -> Step:
        self.bad_numbers += 1
        if self.bad_numbers > self.MAX_UNCLEAR:
            self.reset()
            return Step(say=self._t("skip_number", lang))
        return Step(say=self._t(key, lang))

    def _on_confirm(self, text: str, lang: str, is_question: bool) -> Step:
        if extract_digits(text):
            self.stage, self.digits = "digits", ""
            return self._on_digits(text, lang, False)
        yes, no = bool(YES.search(text)), bool(NO.search(text))
        if yes and no:
            yes = bool(re.search(r"सही|correct|right", text, re.I))
            no = not yes
        if yes:
            return Step(save=True)
        if no:
            self.stage, self.digits, self.phone, self.unclear = "digits", "", None, 0
            return Step(say=self._t("retry_digits", lang))
        return self._unclear(lang, None, "confirm_again", is_question, spoken=spoken(self.phone or ""))
