"""Deterministic name + phone collection, driven by code instead of the LLM.

The LLM only decides *that* the caller wants a callback/consultation (by calling
capture_lead / book_consultation). From then on every caller turn is handled here
until the details are saved, so the agent can't loop, invent numbers or forget
the name:

    ask name -> "this same number, or another one?"
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
        "ask_choice": "Thanks, {name}. Should our team reach you on this same number, or would you like to give a different one?",
        "ask_choice_again": "Sorry, should we call you on the number you're calling from, or on a different number?",
        "ask_digits": "Sure. Could you tell me your ten-digit mobile number?",
        "ask_digits_named": "Thanks, {name}. Could you tell me your ten-digit mobile number?",
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
        "ask_choice": "धन्यवाद {name} जी। क्या हमारी टीम आपसे इसी नंबर पर संपर्क करे, या आप कोई दूसरा नंबर देना चाहेंगे?",
        "ask_choice_again": "माफ़ कीजिए, क्या हम आपको इसी नंबर पर कॉल करें, या किसी दूसरे नंबर पर?",
        "ask_digits": "ज़रूर। क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
        "ask_digits_named": "धन्यवाद {name} जी। क्या आप अपना दस अंकों का मोबाइल नंबर बता सकते हैं?",
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
    """Turn runs of spoken digits into numerals: "double nine eight" -> "998".
    A word becomes a digit only inside a run of at least three digit words (or after
    double/triple), so "do you" or "no" in a sentence stay words."""
    tokens = re.findall(r"[\wऀ-ॿ]+|\S", text)
    out: list[str] = []
    run: list[str] = []
    words = [t for t in tokens if re.match(r"[\wऀ-ॿ]", t)]
    # A reply made only of digit words ("six five") is digits even if short:
    # callers often give the last few digits on their own.
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
    r"\b(?:yes|yeah|yep|yup|correct|right|sure|ok|okay|perfect|absolutely|haan|han|ha|ji|sahi|theek)\b"
    r"|हाँ|हां|हा\b|जी|सही|ठीक|बिल्कुल|बिलकुल|ओके|करेक्ट",
    re.I,
)
_NO = re.compile(r"\b(?:no|nope|nah|wrong|incorrect|nahi|nahin|galat)\b|नहीं|नही|ना\b|गलत|ग़लत", re.I)
_SAME = re.compile(
    r"\b(?:same|this number|this one|current|calling from)\b|\bthis\b|यही|इसी|इस\s*(?:नंबर|नम्बर|number)|जिससे|इसपे|इस पे",
    re.I,
)
_DIFFERENT = re.compile(
    r"\b(?:different|another|other|new|second)\b|दूसर|अलग|नया|नए|और\s*(?:नंबर|नम्बर|number)",
    re.I,
)

# --- intent patterns shared with agent.py / tools.py ---------------------------
# Caller asks for a demo / consultation / meeting / callback (with a "want" word,
# so "what's in the demo?" doesn't count).
_BOOKING_WORDS = re.compile(
    r"\b(?:demo|consultation|meeting|appointment)\b|डेमो|कंसल्टेशन|मीटिंग|अपॉइंटमेंट|बुक|\bbook",
    re.I,
)
# A callback is a lead (capture_lead), not a consultation booking.
_CALLBACK_WORDS = re.compile(
    r"\b(?:call ?back|call me|contact me|reach me)\b|कॉल\s*बैक|वापस कॉल|कॉल कर(?:ना|ें|िए|ो)|संपर्क कर",
    re.I,
)
_WANT = re.compile(
    r"\b(?:want|need|like|book|schedule|can i|could i|i'd|please|get)\b"
    r"|चाहिए|चाहता|चाहती|मिल सकता|मिल सकती|मिलेगा|मिलेगी|करवा|करा|कर दो|कर दीजिए|कीजिए|बुक",
    re.I,
)
# Thanks / goodbye words: "okay thanks" or "ok bye" to an offer means no.
_NOT_A_YES = re.compile(
    r"\b(?:thanks|thank you|bye|later|not now|maybe)\b|धन्यवाद|शुक्रिया|बाय|बाद में|अभी नहीं|फिर कभी",
    re.I,
)
# Caller doesn't want to give details / is annoyed by the questions: stop the flow.
_REFUSE = re.compile(
    r"\b(?:don'?t want to (?:give|share|tell)|not (?:giving|sharing)|no need|skip (?:it|that)|"
    r"leave it|forget it|stop asking|again and again|not interested)\b"
    r"|नहीं बताना|नहीं बताऊँगा|नहीं बताऊंगा|नहीं बताऊँगी|नहीं देना|रहने दो|रहने दीजिए|छोड़ो|छोड़िए|"
    r"बार बार|बार-बार|ज़रूरत नहीं|जरूरत नहीं",
    re.I,
)
# Caller asks for a person ("AI agent" alone doesn't count).
_HUMAN = re.compile(
    r"\b(?:human|real person|a person|representative|someone from (?:the|your) team|team member|"
    r"manager|executive|customer care)\b|इंसान|असली (?:व्यक्ति|आदमी)|किसी (?:व्यक्ति|आदमी)|"
    r"टीम (?:के किसी|से किसी|मेंबर)|मैनेजर|एग्ज़ीक्यूटिव",
    re.I,
)
# Said before the name question when the caller asked for a person.
NO_TRANSFER: dict[str, str] = {
    "en": "I can't connect you to someone on this call, but our team will call you back. ",
    "hi": "मैं अभी इस कॉल पर किसी से कनेक्ट नहीं कर सकती, लेकिन हमारी टीम आपको वापस कॉल करेगी। ",
}

# Agreeing to Maya's offer without a plain yes: "sounds good, I'm interested".
_ACCEPT = re.compile(
    r"\b(?:interested|sounds good|go ahead|let's do it|lets do it|why not|please do|sure)\b"
    r"|ज़रूर|जरूर|चलिए|कर दीजिए|कर दो|बिल्कुल|बिलकुल",
    re.I,
)
# Maya's last reply offered a consultation / demo / team callback.
_OFFER = re.compile(
    r"consultation|demo|meeting|team (?:to )?(?:call|reach|contact)|call you back|"
    r"कंसल्टेशन|डेमो|मीटिंग|टीम.*(?:कॉल|संपर्क|बात)|बुक",
    re.I,
)


# Name extraction: "my name is X", "मेरा नाम X है", "मैं X बोल रहा हूँ", or a short bare reply.
_NAME_PATTERNS = [
    re.compile(r"(?:my name is|name is|name's|i am|i'm|this is)\s+(.+)", re.I),
    re.compile(r"(?:मेरा|मेरी|हमारा)?\s*नाम\s+(?:है\s+)?(.+)"),
    re.compile(r"मैं\s+(.+?)\s+(?:हूँ|हूं|बोल\s+रह[ाी])"),
]
# (?=\s|$) instead of \b: \b fails after a Devanagari vowel sign ("है" ends in one).
_NAME_STOP = re.compile(
    r"\s+(?:है|हैं|हे|हूँ|हूं|और|and|but|my|मेरा|मेरी|from|here|speaking|बोल)(?=\s|$).*$|[,.।!?].*$", re.I
)
_FILLERS = re.compile(
    r"^(?:(?:जी|हाँ|हां|अच्छा|ठीक है|ok|okay|yes|yeah|sir|madam|मैडम|सर|अरे|भाई|हेलो|hello|hi)\s*)+", re.I
)


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
    """The name in a reply to "may I know your name?", or None if it isn't clear."""
    t = (text or "").strip()
    if not t or re.search(r"\d", t):
        return None
    candidate = None
    for pat in _NAME_PATTERNS:
        m = pat.search(t)
        if m:
            candidate = m.group(1)
            break
    if candidate is None:
        if len(t.split()) > 4:
            return None  # a sentence, not a name
        candidate = t
    candidate = _FILLERS.sub("", _NAME_STOP.sub("", candidate.strip())).strip(" .,।!?\"'")
    words = candidate.split()
    if not words or len(words) > 3:
        return None
    if _NO.fullmatch(candidate) or _YES.fullmatch(candidate):
        return None
    return " ".join(words)


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

    @property
    def active(self) -> bool:
        return self.stage is not None

    def _t(self, key: str, lang: str, **kw) -> str:
        return TEXTS.get(lang, TEXTS["en"])[key].format(**kw)

    def start(
        self, tool: str, requirement: str | None, name: str | None, lang: str, caller: str | None,
        opener: bool = True,
    ) -> str:
        """First question. `opener` adds "Sure! " (not after an apology)."""
        self.reset()
        self.tool, self.requirement = tool, requirement
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
            return self._t("ask_choice", lang, name=self.name)
        if self.stage == "confirm" and self.phone:
            return self._t("confirm", lang, spoken=spoken(self.phone))
        return self._t("ask_digits", lang)

    def saved_text(self, lang: str, business: str) -> str:
        key = "saved_booking" if self.tool == "book_consultation" else "saved_lead"
        return self._t(key, lang, name=self.name, business=business)

    # --- per-turn handling -------------------------------------------------
    def handle(self, text: str, lang: str, caller: str | None, is_question: bool = False) -> Step:
        if _REFUSE.search(text or ""):
            # "बार बार एक ही बात", "I don't want to give my number": stop asking at once.
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
            return self._t("ask_choice", lang, name=self.name)
        self.stage = "digits"
        return self._t("ask_digits_named", lang, name=self.name)

    def _on_name(self, text, lang, caller, is_question) -> Step:
        name = None if is_question else extract_name(text)
        if not name:
            return self._unclear(lang, caller, "ask_name_again", is_question)
        self.name = name
        return Step(say=self._after_name(lang, caller))

    def _on_choice(self, text, lang, caller, is_question) -> Step:
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

    def _on_digits(self, text, lang, is_question) -> Step:
        new = extract_digits(text)
        if not new:
            # We re-ask for the whole number, so drop any partial digits.
            self.digits = ""
            return self._unclear(lang, None, "ask_digits", is_question)
        self.unclear = 0
        buf = self.digits + new
        if len(buf) > 10 and clean_phone(new):
            buf = clean_phone(new)  # the caller restarted with the full number
        if not self.digits and len(buf) == 11 and buf.startswith("0"):
            buf = buf[1:]  # "0 98765 43210"
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
        """Invalid / too-long number. After a few tries, stop asking instead of looping
        (landline or foreign numbers will never pass)."""
        self.bad_numbers += 1
        if self.bad_numbers > self.MAX_UNCLEAR:
            self.reset()
            return Step(say=self._t("skip_number", lang))
        return Step(say=self._t(key, lang))

    def _on_confirm(self, text, lang, is_question) -> Step:
        if extract_digits(text):
            # A correction ("no, 98...") replaces the number.
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
