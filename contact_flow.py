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

TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "ask_name": "May I know your name, please?",
        "sure": "Sure! ",
        "drop": "No problem, let's skip that. What else can I help you with?",
        "ask_name_again": "Sorry, I didn't catch your name. Could you tell me just your name?",
        "name_missed": "Sorry, I missed it. Could you tell me your name once more?",
        "no_caller_id": "Sorry, I can't see the number you're calling from. Could you tell me your ten-digit mobile number?",
        "time_noted": "Got it, I've noted that time. Could you tell me your ten-digit mobile number?",
        "ask_choice": "Thanks, {name}. Would you like to get a callback on this number, or another one?",
        "ask_choice_again": "Sorry, should we call you back on the number you're calling from, or on a different number?",
        "ask_digits": "Sure. Could you tell me your ten-digit mobile number?",
        "ask_digits_named": "Thanks, {name}. Could you please share your ten-digit mobile number for the callback?",
        "ask_rest": "Okay, {got}. Could you tell me the remaining {n} digit{s}?",
        "too_many": "Sorry, I got more than ten digits. Could you say your ten-digit number again from the start?",
        "invalid": "Sorry, I can only take a ten-digit Indian mobile number, starting with 6, 7, 8 or 9. Could you say it again?",
        "skip_number": "No problem, let's leave the number for now. Is there anything else I can help you with?",
        "confirm": "Just to confirm, your number is {spoken}. Is that correct?",
        "confirm_again": "Sorry, is {spoken} the right number? Please say yes or no.",
        "retry_digits": "No problem. Could you tell me the full ten-digit number again?",
        "saved_lead": "Thank you, {name}! Your details are saved and the {business} team will contact you soon. Is there anything else I can help you with?",
        "saved_booking": "Thank you, {name}! Your consultation request is booked and the {business} team will contact you soon. Is there anything else I can help you with?",
    },
    "hi": {
        "ask_name": "क्या मैं आपका नाम जान सकती हूँ?",
        "sure": "ज़रूर! ",
        "drop": "कोई बात नहीं, रहने देते हैं। बताइए, मैं और कैसे मदद कर सकती हूँ?",
        "ask_name_again": "माफ़ कीजिए, नाम ठीक से सुनाई नहीं दिया। क्या आप सिर्फ़ अपना नाम बता सकते हैं?",
        "name_missed": "माफ़ कीजिए, मुझसे छूट गया। क्या आप एक बार फिर अपना नाम बता सकते हैं?",
        "no_caller_id": "माफ़ कीजिए, मुझे आपका नंबर दिखाई नहीं दे रहा। क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "time_noted": "ठीक है, मैंने समय नोट कर लिया है। क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "ask_choice": "धन्यवाद {name} जी। क्या आप इसी नंबर पर कॉल बैक चाहेंगे, या किसी दूसरे नंबर पर?",
        "ask_choice_again": "माफ़ कीजिए, क्या हम आपको इसी नंबर पर कॉल करें, या किसी दूसरे नंबर पर?",
        "ask_digits": "ज़रूर। क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "ask_digits_named": "धन्यवाद {name} जी। कॉल बैक के लिए क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "ask_rest": "ठीक है, {got}। क्या आप बाकी के {n} अंक बता सकते हैं?",
        "too_many": "माफ़ कीजिए, मुझे दस से ज़्यादा अंक मिले। क्या आप अपना दस अंकों का नंबर शुरू से दोबारा बता सकते हैं?",
        "invalid": "माफ़ कीजिए, मैं सिर्फ़ दस अंकों का भारतीय मोबाइल नंबर ले सकती हूँ, जो 6, 7, 8 या 9 से शुरू हो। क्या आप दोबारा बता सकते हैं?",
        "skip_number": "कोई बात नहीं, नंबर अभी रहने देते हैं। क्या मैं आपकी और किसी चीज़ में मदद कर सकती हूँ?",
        "confirm": "एक बार कन्फ़र्म कर लूँ, आपका नंबर है {spoken}। क्या यह सही है?",
        "confirm_again": "माफ़ कीजिए, क्या {spoken} सही नंबर है? कृपया हाँ या ना बताइए।",
        "retry_digits": "कोई बात नहीं। क्या आप पूरा दस अंकों का नंबर दोबारा बता सकते हैं?",
        "saved_lead": "धन्यवाद {name} जी! आपकी जानकारी सेव हो गई है, {business} की टीम जल्द ही आपसे संपर्क करेगी। क्या मैं आपकी और किसी चीज़ में मदद कर सकती हूँ?",
        "saved_booking": "धन्यवाद {name} जी! आपकी consultation request दर्ज हो गई है, {business} की टीम जल्द ही आपसे संपर्क करेगी। क्या मैं आपकी और किसी चीज़ में मदद कर सकती हूँ?",
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


_YES = re.compile(
    r"\b(?:yes|yeah|yep|yup|correct|right|sure|ok|okay|perfect|absolutely|haan|han|ha|ji|jee|sahi|theek)\b"
    r"|हाँ|हां|हा\b|जी|सही|ठीक|बिल्कुल|बिलकुल|ओके|करेक्ट",
    re.I,
)
_NO = re.compile(r"\b(?:no|nope|nah|wrong|incorrect|nahi|nahin|galat)\b|नहीं|नही|ना\b|गलत|ग़लत", re.I)
_SAME = re.compile(
    r"\b(?:same|this number|this one|current|calling from|this|here|callback)\b|"
    r"यही|इसी|इस\s*(?:नंबर|नम्बर|number)|जिससे|इसपे|इस\s*पे|इसी\s*पे|इसी\s*पर|कॉल\s*बैक",
    re.I,
)
_DIFFERENT = re.compile(
    r"\b(?:different|another|other|new|second)\b|दूसर|अलग|नया|नए|और\s*(?:नंबर|नम्बर|number)",
    re.I,
)

# Caller indicates they already stated their name ("bola to sahi", "already told you")
_ALREADY_TOLD = re.compile(
    r"\b(?:already (?:told|said|gave)|told you|said it|i told you|already|said)\b|"
    r"बोला\s*त[ोॉ]\s*सही|bola\s*th?o\s*sahi|बोल\s*तो\s*दिया|bol\s*to\s*diya|"
    r"पहले\s*(?:ही\s*)?बता(?:या| दिया| चुके)|pehle\s*hi\s*bata|"
    r"abhi\s*t?h?o\s*(?:bola|bataya|kaha)|bata(?:ya)?\s*(?:to|na|tha)\b|bola\s*(?:to|na|tha)\b|"
    r"अभी\s*तो\s*(?:बोला|बताया|कहा)|बताया\s*(?:तो|ना|था)|बोला\s*(?:ना|था)",
    re.I,
)

# --- intent patterns shared with agent.py / tools.py ---------------------------
_BOOKING_WORDS = re.compile(
    r"\b(?:demo|consultation|meeting|appointment)\b|डेमो|कंसल्टेशन|मीटिंग|अपॉइंटमेंट|बुक|\bbook",
    re.I,
)
_CALLBACK_WORDS = re.compile(
    r"\b(?:call ?back|call me|contact me|reach me|"
    r"(?:contact|talk|speak|connect)(?: with| to)? (?:the |your )?team)\b|कॉल\s*बैक|वापस कॉल|कॉल कर(?:ना|ें|िए|ो)|संपर्क कर",
    re.I,
)
_WANT = re.compile(
    r"\b(?:want|need|like|book|schedule|can i|could i|i'd|please|get)\b"
    r"|चाहिए|चाहता|चाहती|मिल सकता|मिल सकती|मिलेगा|मिलेगी|करवा|करा|कर दो|कर दीजिए|कीजिए|बुक",
    re.I,
)
_NOT_A_YES = re.compile(
    r"\b(?:thanks|thank you|bye|later|not now|maybe)\b|धन्यवाद|शुक्रिया|बाय|बाद में|अभी नहीं|फिर कभी",
    re.I,
)
_REFUSE = re.compile(
    r"\b(?:(?:don'?t|do not) want to (?:give|share|tell)|not (?:giving|sharing)|no need|skip (?:it|that)|"
    r"leave it|forget it|stop asking|again and again|not interested)\b"
    r"|नहीं बताना|नहीं बताऊँगा|नहीं बताऊंगा|नहीं बताऊँगी|नहीं देना|रहने दो|रहने दीजिए|छोड़ो|छोड़िए|"
    r"बार बार|बार-बार|ज़रूरत नहीं|जरूरत नहीं",
    re.I,
)
_HUMAN = re.compile(
    r"\b(?:human|real person|a person|representative|someone from (?:the|your) team|team member|"
    r"manager|executive|customer care)\b|इंसान|असली (?:व्यक्ति|आदमी)|किसी (?:व्यक्ति|आदमी)|"
    r"टीम (?:के किसी|से किसी|मेंबर)|मैनेजर|एग्ज़ीक्यूटिव",
    re.I,
)
NO_TRANSFER: dict[str, str] = {
    "en": "I can't connect you to someone on this call, but our team will call you back. ",
    "hi": "मैं अभी इस कॉल पर किसी से कनेक्ट नहीं कर सकती, लेकिन हमारी टीम आपको वापस कॉल करेगी। ",
}

_ACCEPT = re.compile(
    r"\b(?:interested|sounds good|go ahead|let's do it|lets do it|why not|please do|sure|jarur|jaroor|zaroor|zarur|bilkul)\b"
    r"|ज़रूर|जरूर|चलिए|कर दीजिए|कर दो|बिल्कुल|बिलकुल",
    re.I,
)
_OFFER = re.compile(
    r"consultation|demo|meeting|team (?:to )?(?:call|reach|contact)|call you back|"
    r"कंसल्टेशन|डेमो|मीटिंग|टीम.*(?:कॉल|संपर्क|बात)|बुक",
    re.I,
)

# Robust Name extraction patterns
_NAME_PATTERNS = [
    re.compile(r"(?:my name is|name is|name's|i am|i'm|this is|it's|myself|this side|call me)\s+(.+)", re.I),
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


# A preferred date/time ("tomorrow 8 am", "kal shaam 5 baje"), not phone digits.
_TIME_OR_DATE = re.compile(
    r"\b(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|baje|o'?clock)|tomorrow|today|tonight|"
    r"morning|evening|afternoon|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"kal|parso|aaj|subah|shaam|dopahar)\b|बजे|कल|परसों|आज|सुबह|शाम|दोपहर",
    re.I,
)


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


def clean_phone(phone: str | None) -> str | None:
    """Valid ten-digit Indian mobile (accepts +91 / 91 / 0 prefixes), or None."""
    d = re.sub(r"\D", "", (phone or "").translate(_DEV_DIGITS))
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return d if re.fullmatch(r"[6-9]\d{9}", d) else None


def extract_digits(text: str) -> str:
    """Digits the caller spoke in this turn, without a leading +91 / 0 country/trunk prefix."""
    t = _words_to_digits((text or "").translate(_DEV_DIGITS))
    t = re.sub(r"\+\s*9\s*1", " ", t)
    return re.sub(r"\D", "", t)


def extract_name(text: str) -> str | None:
    """Extract caller name reliably across English, Hindi and Romanized Hindi."""
    t = (text or "").strip()
    if not t or _ALREADY_TOLD.search(t):
        return None  # "abhi tho bola" is "I already told you", not a name
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
        return TEXTS.get(lang, TEXTS["en"])[key].format(**kw)

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
        if _REFUSE.search(text or ""):
            # Caller wants to drop details: stop asking at once.
            self.reset()
            return Step(say=self._t("drop", lang))
        stage = self.stage
        if stage == "name":
            return self._on_name(text, lang, caller, is_question)
        if stage == "choice":
            return self._on_choice(text, lang, caller, is_question)
        if stage == "digits":
            return self._on_digits(text, lang, is_question)
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
        if _ALREADY_TOLD.search(text):
            if self.name:
                return Step(say=self._after_name(lang, caller))
            # The agent looks the name up in the conversation first (name_from_history);
            # only if that found nothing do we get here: apologise and ask once more.
            return Step(say=self._t("name_missed", lang))

        # A bare "nahi" / "no" to "may I know your name?" means they won't give it:
        # stop asking instead of "didn't catch your name" again and again.
        if _NO.search(text) and len(text.split()) <= 3 and not _YES.search(text):
            self.reset()
            return Step(say=self._t("drop", lang))

        # 2. Check if caller gave a name (even if punctuation/question marks were transcribed)
        name = extract_name(text)
        if name:
            self.name = name
            # Check if caller also provided 10 digits in the same turn
            digits = extract_digits(text)
            if digits and clean_phone(digits):
                phone = clean_phone(digits)
                self.phone, self.stage = phone, "confirm"
                return Step(say=self._t("confirm", lang, spoken=spoken(phone)))
            return Step(say=self._after_name(lang, caller))

        return self._unclear(lang, caller, "ask_name_again", is_question)

    def _on_choice(self, text: str, lang: str, caller: str | None, is_question: bool) -> Step:
        if extract_digits(text):
            self.stage, self.digits = "digits", ""
            return self._on_digits(text, lang, False)
        different = bool(_DIFFERENT.search(text))
        same = bool(_SAME.search(text)) or (bool(_YES.search(text)) and not _NO.search(text))
        if different or (_NO.search(text) and not same):
            self.stage, self.digits, self.unclear = "digits", "", 0
            return Step(say=self._t("ask_digits", lang))
        if same and caller:
            self.phone, self.stage, self.unclear = caller, "confirm", 0
            return Step(say=self._t("confirm", lang, spoken=spoken(caller)))
        return self._unclear(lang, caller, "ask_choice_again", is_question)

    def _on_digits(self, text: str, lang: str, is_question: bool) -> Step:
        if _TIME_OR_DATE.search(text) and len(extract_digits(text)) < 5:
            # "tomorrow 8 am" is when to call, not the start of a phone number:
            # keep it as the preferred slot and ask for the number again.
            slot = f"Preferred: {text.strip()}"
            self.requirement = f"{self.requirement} ({slot})" if self.requirement else slot
            return Step(say=self._t("time_noted", lang))
        new = extract_digits(text)
        if not new:
            self.digits = ""
            if _SAME.search(text) or _ALREADY_TOLD.search(text):
                # "same number" when we can't see the caller ID (console, hidden number):
                # say why we still need it instead of repeating the question.
                return Step(say=self._t("no_caller_id", lang))
            return self._unclear(lang, None, "ask_digits", is_question)
        self.unclear = 0
        buf = self.digits + new
        if len(buf) > 10 and clean_phone(new):
            buf = clean_phone(new)
        if not self.digits and len(buf) == 11 and buf.startswith("0"):
            buf = buf[1:]
        if len(buf) < 10:
            self.digits = buf
            n = 10 - len(buf)
            return Step(say=self._t("ask_rest", lang, got=spoken(buf), n=n, s="" if n == 1 else "s"))
        if len(buf) > 10:
            self.digits = ""
            return self._bad_number(lang, "too_many")
        phone = clean_phone(buf)
        self.digits = ""
        if not phone:
            return self._bad_number(lang, "invalid")
        self.phone, self.stage = phone, "confirm"
        return Step(say=self._t("confirm", lang, spoken=spoken(phone)))

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
        yes, no = bool(_YES.search(text)), bool(_NO.search(text))
        if yes and no:
            yes = bool(re.search(r"सही|correct|right", text, re.I))
            no = not yes
        if yes:
            return Step(save=True)
        if no:
            self.stage, self.digits, self.phone, self.unclear = "digits", "", None, 0
            return Step(say=self._t("retry_digits", lang))
        return self._unclear(lang, None, "confirm_again", is_question, spoken=spoken(self.phone or ""))
