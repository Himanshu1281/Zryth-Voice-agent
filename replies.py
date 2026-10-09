"""Fixed lines Maya speaks, and rules that shape the LLM's replies.

Lines spoken by code (fallbacks, fillers, callback offers, office hours, goodbyes)
in English and Hindi, plus the reply guards: the sentence cap, the repeat check and
markdown cleaning.
"""

from __future__ import annotations

import os
import re


# Spoken when the LLM returns nothing, so the caller never hears dead air.
# Used instead when the empty reply follows a tool result (e.g. a rejected phone
# number): "say that once more" makes no sense to the caller there.
TOOL_REPLY_FALLBACK: dict[str, str] = {
    "en": "Sorry, I didn't get the full number. Could you tell me all ten digits, please?",
    "hi": "माफ़ कीजिए, पूरा नंबर नहीं मिला। क्या आप दसों अंक एक साथ बता सकते हैं?",
}


GENERIC_TOOL_FALLBACK: dict[str, str] = {
    "en": "Sorry, give me just a moment. Could you say that again?",
    "hi": "माफ़ कीजिए, एक पल। क्या आप फिर से बता सकते हैं?",
}


# Played before hanging up on a number that isn't connected to an agent yet.
NUMBER_NOT_ACTIVE: dict[str, str] = {
    "en": "Sorry, this number is not active yet. Please try again later. Thank you for calling.",
    "hi": "माफ़ कीजिए, यह नंबर अभी चालू नहीं है। कृपया थोड़ी देर बाद फिर से कॉल करें। धन्यवाद।",
}


EMPTY_REPLY_FALLBACK: dict[str, str] = {
    "en": "Sorry, could you say that once more?",
    "hi": "माफ़ कीजिए, क्या आप एक बार फिर से बता सकते हैं?",
}


# Spoken only when the knowledge lookup is slow, so the caller never hears dead air.
# Not added to the chat history, so the LLM doesn't repeat or react to it.
FILLERS: dict[str, tuple[str, ...]] = {
    "en": ("Let me check.", "One moment.", "Sure, let me see."),
    "hi": ("एक सेकंड, देखती हूँ।", "जी, बस एक पल।", "ठीक है, देखती हूँ।"),
}


# Markdown the LLM may emit despite instructions: **bold**, *, #, `, and "- " bullets.
MARKDOWN = re.compile(r"[*#`_]+|^\s*[-•]\s+", re.M)


# Spoken replies are cut after this many sentences (see llm_node).
MAX_REPLY_SENTENCES = int(os.getenv("MAX_REPLY_SENTENCES", "3"))


# ...and once this many characters are out (~10-12 s of speech), at most one more
# sentence: the closing question.
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "260"))


# A sentence ends at . ! ? or । followed by a space or newline ("FinanceAuditor.ai"
# and "2.5" are not sentence ends).
SENTENCE_END = re.compile(r"[.!?।](?=\s)")


# Sentences shorter than this (openers like "Got it.") don't count toward the cap.
MIN_SENTENCE_WORDS = 4


LENGTH_RULE: dict[str, str] = {
    "en": "Answer in 1-2 short sentences (never more than 3), then at most one short question.",
    "hi": "जवाब 1-2 छोटे वाक्यों में दें (3 से ज़्यादा कभी नहीं), फिर ज़्यादा से ज़्यादा एक छोटा सवाल।",
}


OFFICE_HOURS_ASK: dict[str, str] = {
    "en": "I'll have our team confirm the office timings with you. ",
    "hi": "ऑफिस की टाइमिंग हमारी टीम आपको कॉल करके बता देगी। ",
}


OFFICE_ADDRESS: dict[str, str] = {
    "en": "Our office is at {address}. ",
    "hi": "हमारा ऑफिस {address} में है। ",
}


def office_address(kb_text: str | None) -> str | None:
    """The "Office: ..." line from the knowledge base, so the address stays whatever
    the KB says (None when the KB isn't in the prompt or has no such line)."""
    m = re.search(r"Office:\s*(.+?)(?:\s+Web:|\n|$)", kb_text or "")
    return m.group(1).strip(" .") if m else None


# A caller in an emergency: help first, nothing else ({number}: the business's own
# emergency line from the KB, else 112).
EMERGENCY_LINE: dict[str, str] = {
    "en": "This sounds like an emergency. Please call {number} right away for immediate help.",
    "hi": "यह इमरजेंसी लग रही है। कृपया तुरंत {number} पर कॉल करें।",
}

# Any later turn of an emergency call that isn't a question: the number again.
EMERGENCY_REPEAT: dict[str, str] = {
    "en": "Please call {number} now, they can help you right away. Take care.",
    "hi": "कृपया अभी {number} पर कॉल करें, वहाँ तुरंत मदद मिलेगी। अपना ध्यान रखिए।",
}

OFFICE_HOURS_SAVED: dict[str, str] = {
    "en": "Our team will call you back about the office timings.",
    "hi": "ऑफिस की टाइमिंग के बारे में हमारी टीम आपको वापस कॉल करेगी।",
}


# An instruction-like aside the LLM sometimes appends to a reply, copying the "(...)"
# reminders tagged onto the caller's message: "(If the caller is done, call
# `end_call`. If they show interest, call `capture_lead`...)". Never spoken.
META_ASIDE = re.compile(
    r"\((?=[^)]{0,80}?(?:`|end_call|capture_lead|book_consultation|transfer_to_human|set_language|"
    r"the caller|caller's|reply (?:only )?in|in devanagari|tool\b|knowledge base))",
    re.I,
)
# How far past an open "(" we wait before deciding it's an ordinary aside.
META_LOOKAHEAD = 80


# Internal words the prompt uses that a caller should never hear ("The knowledge base
# does not contain information about monthly maintenance charges.").
JARGON: list = [
    (re.compile(r"\b(?:the |my |our )?knowledge(?: base)? (?:does not|doesn't|did not) (?:contain|have|mention|include|say)"
                r"(?: any)?(?: specific)? (?:information|details|info)(?: (?:about|on|regarding))?", re.I),
     "I don't have the details on"),
    # "According to the knowledge base, we open at 9." -> "We open at 9."
    (re.compile(r"\b(?:in|from|according to|based on) (?:the|my|our) knowledge(?: base)?,?\s*([a-z]?)", re.I),
     lambda m: m.group(1).upper()),
    (re.compile(r"\b(?:the|my|our) knowledge base\b", re.I), "my information"),
    (re.compile(r"नॉलेज\s*बेस में|knowledge base में"), "मेरी जानकारी में"),
    (re.compile(r"नॉलेज\s*बेस|knowledge base"), "जानकारी"),
]
# Words held back while streaming, so a phrase split across chunks is still caught.
JARGON_HOLD_WORDS = 4


def scrub_jargon(text: str) -> str:
    for pat, repl in JARGON:
        text = pat.sub(repl, text)
    return text


def repeats_earlier(head: str, replies: list[str]) -> bool:
    """True when this reply starts word-for-word like one Maya already gave
    ("कोई बात नहीं। क्या आप मुझे बता सकते हैं कि आपकी टीम का..." six times in a row)."""
    norm = lambda t: re.sub(r"\W+", " ", t).strip().lower()  # noqa: E731
    h = norm(head)
    return len(h) >= 25 and any(norm(r).startswith(h) for r in replies)


# Spoken by end_call itself, so every call ends the same polite way.
GOODBYES = {
    "en": "Thank you for calling {business}. Have a great day, goodbye!",
    "hi": "{business} को कॉल करने के लिए धन्यवाद। आपका दिन शुभ हो, नमस्ते!",
}


# Tool result that agent.llm_node speaks word for word instead of asking the LLM.
# (session.say() from inside a tool was queued behind the reply still playing:
# the name question came 4 turns late while Gemini invented "we've booked it".)
SAY_MARKER = "[[SAY]]"


def speak(line: str) -> str:
    return SAY_MARKER + line


# end_call's one-time offer before hanging up on a caller whose details weren't
# saved. A "yes" starts the contact flow (it matches contact_flow.OFFER).
CALLBACK_OFFER = {
    "en": "Before you go, should our team call you back with more details?",
    "hi": "जाने से पहले, क्या हमारी टीम आपको ज़्यादा जानकारी के लिए वापस कॉल करे?",
}


# Spoken by transfer_to_human itself when the transfer fails; ends with a callback
# offer, so a "yes" starts the contact flow.
TRANSFER_FAILED_LINES = {
    "en": "Sorry, nobody from the team is free right now. Should our team call you back?",
    "hi": "माफ़ कीजिए, अभी टीम में कोई उपलब्ध नहीं है। क्या हमारी टीम आपको वापस कॉल करे?",
}


NO_TRANSFER: dict[str, str] = {
    "en": "I can't connect you to someone on this call, but our team will call you back. ",
    "hi": "मैं अभी इस कॉल पर किसी से कनेक्ट नहीं कर सकती, लेकिन हमारी टीम आपको वापस कॉल करेगी। ",
}
