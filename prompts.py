"""System prompts and per-language copy for Maya, the Zryth AI solutions voice assistant.

HOT_PERSONA is capped at 800 chars (enforced by the self-check below): fewer
input tokens = faster time-to-first-token on every turn. It holds behaviour
only -- NO company facts. All facts come from the knowledge base, injected per
turn as "Relevant Zryth knowledge" (agent.BaseMayaAgent.on_user_turn_completed).
"""

from __future__ import annotations

import functools
from pathlib import Path

# Where the per-language grammar sheets live (grammar/maya_<lang>_grammar.md).
GRAMMAR_DIR = Path(__file__).parent / "grammar"

# The one persona prompt, shared by every language agent. Keep it tight.
HOT_PERSONA = """
You are Maya, Zryth's phone assistant. Reply in 1-2 short spoken sentences, max 25 words.
FACTS: Use ONLY the "Relevant Zryth knowledge" you are given. Never invent or guess products, features, clients, numbers, dates, or people. If the knowledge doesn't answer it, say the Zryth team will confirm, and offer to take their name and number. Never say Zryth doesn't offer something.
PRICES: Never quote a price or cost, even if one appears in the knowledge; say the Zryth team will share pricing.
CUSTOM WORK: For custom software, AI agents, or apps, offer a consultation with the Zryth team.
Off-topic (not Zryth): politely say you can only help with Zryth.
TOOLS: Ask the caller's name and phone before capture_lead or book_consultation; never pass blank details. Never say "lead" or "SaaS".
"""


CONVERSATION_ENDING = (
    'ENDING: When the caller is done ("no thanks", "that\'s all", "bye"), '
    "say a brief goodbye AND call end_call in that same reply. Never say goodbye without calling end_call."
)

# Human-readable language names, used in the per-language instruction line.
LANG_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
}

# Tiny per-language style note appended to the persona. Kept short on purpose.
STYLE_NOTES: dict[str, str] = {
    "en": "Speak clear, simple English.",
    "hi": "Write ONLY in Devanagari script, never romanized Hindi (product/brand names may stay in English). Use natural, conversational Hindi, not textbook Hindi. You are female: say सकती हूँ, करूँगी, never सकता, करूँगा.",
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


def build_instructions(language: str, script: str, include_grammar: bool = False) -> str:
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
    base = (
        f"{HOT_PERSONA}"
        f"LANGUAGE: Default to {name}. {script}"
        + (f" Translate English knowledge or tool output into {name}." if language != "en" else "")
        + "\n"
        "This overrides the default: ALWAYS reply in the language of the caller's LATEST "
        "message (English or Hindi), without mentioning it. Call set_language only if the caller asks for a language.\n"
        "After any tool returns, always reply to the caller.\n"
        f"{CONVERSATION_ENDING}"
    )
    grammar = load_grammar(language) if include_grammar else ""
    return f"{base}\n\n{grammar}" if grammar else base


if __name__ == "__main__":
    # Self-check: the persona must stay short (latency) and every language must
    # have parallel copy so nothing goes silent after a language switch.
    assert len(HOT_PERSONA) <= 800, f"HOT_PERSONA too long: {len(HOT_PERSONA)} chars (cap 800)"
    for _code in LANG_NAMES:
        assert _code in STYLE_NOTES, f"missing STYLE_NOTES[{_code}]"
        assert _code in GREETINGS, f"missing GREETINGS[{_code}]"
    assert "Zryth" in build_instructions("hi", STYLE_NOTES["hi"])
    # Grammar sheets are optional; when present they must get appended.
    for _code in LANG_NAMES:
        if load_grammar(_code):
            _with = build_instructions(_code, STYLE_NOTES[_code], include_grammar=True)
            _without = build_instructions(_code, STYLE_NOTES[_code], include_grammar=False)
            assert len(_with) > len(_without), f"grammar sheet for {_code} was not appended"
    print(f"prompts.py self-check passed (HOT_PERSONA={len(HOT_PERSONA)} chars)")
