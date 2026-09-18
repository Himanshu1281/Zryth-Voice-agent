"""System prompts and per-language copy for Maya, the Zryth AI solutions voice assistant.

Latency note: on a phone call the system prompt is the single biggest latency
killer. Keep HOT_PERSONA SHORT (a couple hundred chars, well under ~800). Do NOT
Keep HOT_PERSONA SHORT. Maya captures leads and answers only with approved business information or available tools.
through the function tools in tools.py. A short prompt = fewer input tokens =
faster LLM time-to-first-token every single turn.
"""

from __future__ import annotations

import functools
from pathlib import Path

# Where the per-language grammar sheets live (grammar/maya_<lang>_grammar.md).
GRAMMAR_DIR = Path(__file__).parent / "grammar"

# The one persona prompt, shared by every language agent. Keep it tight.
HOT_PERSONA = """
You are Maya, friendly voice assistant for Zryth (pronounce: "Z-rith"). Zryth builds industry-specific Software as a Service (never say "SaaS") in Noida Sector 132.
RULES:
1. MAX 1-2 short sentences. Start replies with human fillers ("Got it", "Sure").
2. Answer ONLY about Zryth using `search_knowledge`. Do not guess. Decline unrelated topics politely.
3. Tools: `capture_lead` (interested), `book_consultation` (confirmed), `transfer_to_human` (say "our team", not "human"), `end_call` (finished).
4. Contacts: Don't ask bluntly. Say: "Should our team use this number, or an alternate?"
5. CRITICAL: After every tool call returns results, you MUST immediately speak a short, natural summary of the result. Never stay silent after a tool returns data.
"""


CONVERSATION_ENDING = """
CONVERSATION ENDING:
If you ask whether the caller needs anything else and they respond negatively
(e.g. "no", "no thanks", "that's all", "nothing else", "that's it", "I'm good",
"I'm done", "bye"), treat the conversation as complete. 
CRITICAL RULE: You MUST call the `end_call` tool to finish the conversation. Do NOT generate a goodbye message yourself, the tool will speak the goodbye automatically.
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

# What Maya says first when a call connects, per language.
GREETINGS: dict[str, str] = {
    "en": "Hi, thanks for calling Zryth! I'm Maya, Zryth's AI assistant. How can I help you today?",

    "hi": "नमस्ते, Zryth में कॉल करने के लिए धन्यवाद! मैं माया, Zryth की AI असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकती हूँ?",
}


@functools.lru_cache(maxsize=8)
def load_grammar(language: str) -> str:
    """Return the per-language grammar sheet (grammar/maya_<lang>_grammar.md), or "".

    These sheets (honorifics, code-mix rules, real-estate vocab, the §5b
    wrong->right table) are what make Maya sound native. They are loaded once and
    cached. Missing file -> "" so the agent still runs on STYLE_NOTES alone.
    """
    path = GRAMMAR_DIR / f"maya_{language}_grammar.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def build_instructions(language: str, script: str, include_grammar: bool = True) -> str:
    """Compose the full system prompt for a per-language agent.

    `language` is a short code (en/hi); `script` is the tiny per-language
    style note (usually STYLE_NOTES[language]). The bulk (HOT_PERSONA) stays the
    same across languages -- we bolt on a one-line language rule, then (if
    available) the full grammar sheet for that language.

    Latency tradeoff: the grammar sheet adds input tokens every turn, which raises
    LLM time-to-first-token a little. It buys much more natural, native-sounding
    Indic speech -- usually worth it. Set include_grammar=False (or trim the sheet)
    if you need to shave the last few ms. See docs/04-latency.md.
    """
    name = LANG_NAMES.get(language, language)
    base = f"{HOT_PERSONA}\n\nRespond only in {name}. {script}\nIMPORTANT: When tools like `search_knowledge` return English text, you MUST translate the information and respond in {name}.\n\nIf the caller starts speaking to you in a different language, or explicitly asks to change language, immediately call the set_language tool with the language code (en, hi).\n\n{CONVERSATION_ENDING}"
    grammar = load_grammar(language) if include_grammar else ""
    return f"{base}\n\n{grammar}" if grammar else base


if __name__ == "__main__":
    # Self-check: the persona must stay short (latency) and every language must
    # have parallel copy so nothing goes silent after a language switch.
    assert len(HOT_PERSONA) <= 1500, f"HOT_PERSONA too long: {len(HOT_PERSONA)} chars"
    for _code in LANG_NAMES:
        assert _code in STYLE_NOTES, f"missing STYLE_NOTES[{_code}]"
        assert _code in GREETINGS, f"missing GREETINGS[{_code}]"
    assert "Zryth" in build_instructions("hi", STYLE_NOTES["hi"])
    # Grammar sheets should exist and get appended when present.
    for _code in LANG_NAMES:
        assert load_grammar(_code), f"missing/empty grammar sheet for {_code}"
    _with = build_instructions("ta", STYLE_NOTES["ta"], include_grammar=True)
    _without = build_instructions("ta", STYLE_NOTES["ta"], include_grammar=False)
    assert len(_with) > len(_without), "grammar sheet was not appended"
    print(f"prompts.py self-check passed (HOT_PERSONA={len(HOT_PERSONA)} chars, grammar wired)")
