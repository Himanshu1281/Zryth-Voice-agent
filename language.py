"""Language handling for English / Hindi calls.

Detecting the caller's language (Devanagari, Latin and romanized Hindi), spotting a
reply in the wrong language, and the reminders that keep Gemini in the caller's
language. Sarvam codemix STT writes Hindi in Devanagari and English in Latin.
"""

from __future__ import annotations

import re


DEVANAGARI = re.compile(r"[ऀ-ॿ]")


LATIN = re.compile(r"[A-Za-z]")


# Other Indic scripts (Bengali through Malayalam blocks).
FOREIGN_SCRIPT = re.compile(r"[ঀ-ൿ]")


# Letters of a reply read before deciding its language (see llm_node). ~6 words:
# enough to get past a leading English product name ("FinanceAuditor.ai एक...").
LANG_CHECK_LETTERS = 30


def wrong_language(head: str, code: str) -> bool:
    """True when the start of a reply is clearly not in the caller's language."""
    dev, lat = len(DEVANAGARI.findall(head)), len(LATIN.findall(head))
    if code == "hi":
        return dev == 0  # not a single Hindi letter in the first ~6 words
    return dev > lat


# Sent as the caller's next message when a reply came back in the wrong language.
# Written in the target language itself: Gemini anchors on the script it reads last.
RETRY_IN_LANGUAGE: dict[str, str] = {
    "en": "(Please answer my last message again, in English only.)",
    "hi": "(कृपया मेरे पिछले सवाल का जवाब दोबारा दीजिए, सिर्फ़ हिंदी में, देवनागरी लिपि में। अंग्रेज़ी में नहीं।)",
}


# Tagged onto the caller's latest message on every LLM call (see llm_node).
LANGUAGE_REMINDER: dict[str, str] = {
    "en": "Reply in English only.",
    "hi": "Reply ONLY in Hindi, in Devanagari script (product names may stay in English). "
          "Even if the knowledge, the tool output or your earlier replies are in English, your reply must be Hindi.",
}


# Short transcripts below this STT confidence are treated as noise ("कि आछे?" at 0.18).
MIN_TRANSCRIPT_CONFIDENCE = 0.3


# Text in another Indic script below this confidence is noise, above it a real caller.
FOREIGN_MIN_CONFIDENCE = 0.5


# Spoken when the caller talks in a language we don't support (never silence).
UNSUPPORTED_LANGUAGE: dict[str, str] = {
    "en": "Sorry, I can only help in English or Hindi. Could you tell me in English or Hindi, please?",
    "hi": "माफ़ कीजिए, मैं सिर्फ़ हिंदी या अंग्रेज़ी में मदद कर सकती हूँ। क्या आप हिंदी या अंग्रेज़ी में बता सकते हैं?",
}


# Hindi written in Latin letters ("mujhe price batao"): common function words / verbs
# that never appear in English sentences.
ROMAN_HINDI = re.compile(
    r"^(?:mujhe|mujhko|mera|meri|mere|hum|humein|hamara|aap|aapka|aapki|aapke|tum|kya|kaise|kaisa|"
    r"kab|kahan|kyun|kitna|kitne|kitni|hai|hain|tha|thi|ho|hoga|hogi|nahi|nahin|haan|ji|"
    r"batao|bataiye|bata|chahiye|chahta|chahti|karna|karo|kariye|kar|raha|rahi|rahe|"
    r"ka|ki|ke|ko|se|mein|par|pe|bhi|toh|aur|lekin|abhi|achha|accha|theek|thik|samjha|samajh|"
    r"baare|wala|wali|kuch|sab|matlab|bolo|boliye|dijiye|sakte|sakti|sakta|"
    # Seen in console calls ("jee jarur", "kuch batao apni company key bare mey",
    # "abhi tho bola"). Words that are also English (to, me, main, key) are left out.
    r"jee|jarur|jaroor|zaroor|zarur|han|haa|haanji|apni|apna|apne|bare|mey|tho|bola|bataya|"
    r"merra|meraa|naam|bilkul|bhai|yaar|chalo|sahi|galat|hoon|hun|liye|kaun|kaunsa|kuchh|nhi|"
    r"yeh|woh|iske|uske|isse|usse|dena|lena|karenge|karega|chahenge|pata|wahi|yahi|"
    # Hinglish abuse / slang ("abe chup, bakwas mat kar") got English replies.
    r"abe|chup|bakwas|saale|chal|teri|tera|karti|karta|bol|bas|kyu|kyon|accha|arre|arey)$",
    re.I,
)


def detect_language(text: str) -> str | None:
    """'hi' / 'en' from the transcript's script, or None when too short or mixed
    to tell. Sarvam codemix writes Hindi in Devanagari and English in Latin, so
    "Zryth के बारे में बताओ" -> hi and "tell me about your products" -> en."""
    words = text.split()
    if len(words) < 2:
        return None  # "ok", "haan" -- not enough to switch on
    # Count words, not letters: Hindi callers use many English nouns ("मेरे पास
    # Instagram पे influencer account है"), which swamped a letter ratio. An English
    # speaker never produces Devanagari, so two Devanagari words already mean Hindi.
    dev = sum(1 for w in words if DEVANAGARI.search(w))
    lat = sum(1 for w in words if LATIN.search(w))
    if dev >= 2 or (dev == 1 and lat <= 1):
        return "hi"
    if dev == 0 and lat >= 2:
        # Romanized Hindi ("mujhe aapke chatbot ka price batao") is still Hindi
        roman = sum(1 for w in words if ROMAN_HINDI.match(re.sub(r"[^\w]", "", w)))
        if roman >= 2 and roman * 3 >= len(words):
            return "hi"
        return "en"
    return None
