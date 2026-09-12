"""System prompts and per-language copy for Maya, the Zryth AI solutions voice assistant.

Latency note: on a phone call the system prompt is the single biggest latency
killer. Keep HOT_PERSONA SHORT (a couple hundred chars, well under ~800). Do NOT
stuff it with product details. Look up product details through the function tools
in tools.py. A short prompt = fewer input tokens = faster LLM time-to-first-token
every single turn.
"""

from __future__ import annotations

import functools
from pathlib import Path

# Where the per-language grammar sheets live (grammar/agent_<lang>_grammar.md).
GRAMMAR_DIR = Path(__file__).parent / "grammar"

# The one persona prompt, shared by every language agent. Keep it tight.


HOT_PERSONA = """
You are a friendly voice assistant for the company.
For specific questions about the company's products, pricing, or features, you MUST use the search_knowledge tool. Answer concisely based ONLY on the tool's results. Do not guess. If the search_knowledge tool returns no information or an error, politely inform the user that you don't have that information.
CRITICAL RULE: If the user asks about ANYTHING unrelated to the company or its products (e.g. general knowledge, internet search, other companies like Google), you MUST politely refuse to answer and state that you can only assist with company-related inquiries. Do not provide information outside your knowledge base.
Answer conversational questions naturally. Keep responses extremely brief, 1 to 2 short sentences max. Start responses with natural conversational fillers (like "Got it", "I understand", "Right") to feel human. Treat short replies ("yes", "okay") as acknowledgements. Preserve names exactly.
Use capture_lead for interested callers, book_consultation for confirmed bookings, transfer_to_human when needed (say "our team", NEVER "human"), and end_call when finished. When collecting contact info, never bluntly ask for a phone number. Instead, ask: "Would you like our team to contact you on this same number, or provide an alternate?"
CRITICAL: You MUST always speak a verbal response out loud immediately after receiving results from the search_knowledge tool. Never stay silent.
"""


CONVERSATION_ENDING = """
CONVERSATION ENDING:
If you ask whether the caller needs
anything else and they respond
negatively (e.g. "no", "no thanks",
"that's all", "nothing else", "that's
it", "I'm good", "I'm done", "bye"),
treat the conversation as complete.
CRITICAL RULE: You MUST call the
`end_call` tool to finish the
conversation. Do NOT generate a goodbye
message yourself, the tool will speak
the goodbye automatically.
"""

# Human-readable language names, used in the per-language instruction line.
LANG_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
}

# Tiny per-language style note appended to the persona. Kept short on purpose.
STYLE_NOTES: dict[str, str] = {
    "en": "Speak clear, simple English.",
    "hi": "Reply in natural, conversational Hindi (Devanagari script), not formal textbook Hindi.",
}

# What the agent says first when a call connects, per language.
GREETINGS: dict[str, str] = {
    "en": "Hi, thanks for calling! I'm the AI assistant. How can I help you today?",
    "hi": "नमस्ते, कॉल करने के लिए धन्यवाद! मैं AI असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकती हूँ?",
}


@functools.lru_cache(maxsize=8)
def load_grammar(language: str) -> str:
    """Return the per-language grammar sheet (grammar/agent_<lang>_grammar.md), or "".

    These sheets (honorifics, code-mix rules, real-estate vocab, the §5b
    wrong->right table) are what make the agent sound native. They are loaded once and
    cached. Missing file -> "" so the agent still runs on STYLE_NOTES alone.
    """
    path = GRAMMAR_DIR / f"agent_{language}_grammar.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def build_instructions(language: str, script: str, include_grammar: bool = True) -> str:
    """Compose the full system prompt for a per-language agent.

    `language` is a short code (en/hi/ta/...); `script` is the tiny per-language
    style note (usually STYLE_NOTES[language]). The bulk (HOT_PERSONA) stays the
    same across languages -- we bolt on a one-line language rule, then (if
    available) the full grammar sheet for that language.

    Latency tradeoff: the grammar sheet adds input tokens every turn, which raises
    LLM time-to-first-token a little. It buys much more natural, native-sounding
    Indic speech -- usually worth it. Set include_grammar=False (or trim the sheet)
    if you need to shave the last few ms. See docs/04-latency.md.
    """
    name = LANG_NAMES.get(language, language)
    base = f"{HOT_PERSONA}\n\nRespond only in {name}. {script}\n\nIf the caller asks to speak in a different language, immediately call the set_language tool with the language code (en, hi).\n\n{CONVERSATION_ENDING}"
    grammar = load_grammar(language) if include_grammar else ""
    return f"{base}\n\n{grammar}" if grammar else base


if __name__ == "__main__":
    # Self-check: the persona must stay short (latency) and every language must
    # have parallel copy so nothing goes silent after a language switch.
    assert len(HOT_PERSONA) <= 1500, f"HOT_PERSONA too long: {len(HOT_PERSONA)} chars"
    for _code in LANG_NAMES:
        assert _code in STYLE_NOTES, f"missing STYLE_NOTES[{_code}]"
        assert _code in GREETINGS, f"missing GREETINGS[{_code}]"
    assert "assistant" in build_instructions("hi", STYLE_NOTES["hi"])
    # Grammar sheets should exist and get appended when present.
    for _code in LANG_NAMES:
        assert load_grammar(_code), f"missing/empty grammar sheet for {_code}"
    _with = build_instructions("ta", STYLE_NOTES["ta"], include_grammar=True)
    _without = build_instructions("ta", STYLE_NOTES["ta"], include_grammar=False)
    assert len(_with) > len(_without), "grammar sheet was not appended"
    print(f"prompts.py self-check passed (HOT_PERSONA={len(HOT_PERSONA)} chars, grammar wired)")
