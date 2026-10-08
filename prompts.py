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
You are {persona}, a friendly voice assistant for {business}, talking on the phone. Keep replies to 1-2 short, natural sentences.
Start with a natural opener ("Got it", "Sure", "Right") and talk like a person. Treat "yes"/"okay" as acknowledgements. Keep names exactly as said.
Facts come ONLY from "Relevant {business} knowledge"; never guess. If it isn't there, say the team will confirm. Never quote prices; the team shares pricing. Off-topic: you only help with {business}.
For contact details ask only their name, then: "Should our team reach you on this same number, or another one?" Never say "lead" or "SaaS". No lists.
"""


CONVERSATION_ENDING = (
    'ENDING: When the caller is done ("no thanks", "that\'s all", "bye"): if you have NOT yet saved their '
    "details this call, first ask once (no goodbye yet): \"Before you go, should our team call you back on "
    "this number with more details?\"; if yes, ask their name and save it. Otherwise, or if they decline, "
    "call end_call and say nothing else: the tool speaks the goodbye itself."
)

QUALIFY = (
    "ENGAGE: Sound warm and curious, like a helpful person, not a brochure. Start with a short, varied "
    "acknowledgement (\"Great question\", \"Got it\", \"Oh nice\"), never the same one twice in a row. "
    "Use the caller's name now and then once you know it. End most replies with ONE easy question that moves "
    "the conversation forward (what their business does, what slows their team down, what they want to build), "
    "never a dead-end \"anything else?\" while they're still exploring. "
    "Once you know their need, link ONE relevant {business} product or service to it in a sentence "
    "(\"For a clinic like yours, our Voice AI could answer patient calls\"). Only name a product whose "
    "description actually matches their need; if none does, say {business} builds custom AI agents and "
    "automation for exactly that, never stretch an unrelated product to fit. Then offer a free consultation "
    "with the {business} team; if they agree, ask their name, then confirm the number. "
    "If you didn't catch something, ask about the specific part you missed instead of \"please rephrase\". "
    "Never repeat a question already answered; if they decline, stay friendly and keep helping."
)

PHONE_RULE = (
    "PHONE: Ask for the full ten-digit number in one go. If you hear only part of it, just say "
    "\"Go on\" and wait for the rest; never fill in, guess or pad missing digits. "
    "Call the tool only once you have all ten digits."
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

# What the agent says first when a call connects, per language ({business}/{persona} filled per agent).
GREETINGS: dict[str, str] = {
    "en": "Hi, thanks for calling {business}! I'm {persona}, {business}'s AI assistant. How can I help you today?",

    "hi": "नमस्ते, {business} में कॉल करने के लिए धन्यवाद! मैं {persona}, {business} की AI असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकती हूँ?",
}


def greeting(language: str, business: str = "Zryth", persona: str = "Maya") -> str:
    """Template greeting for one agent (used when the customer set no custom greeting)."""
    if language == "hi" and persona == "Maya":
        persona = "माया"
    return GREETINGS.get(language, GREETINGS["en"]).format(business=business, persona=persona)


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


def build_instructions(
    language: str,
    script: str,
    include_grammar: bool = True,
    business: str = "Zryth",
    persona: str = "Maya",
) -> str:
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
        HOT_PERSONA.format(business=business, persona=persona)
        + f"LANGUAGE: Default to {name}. {script}"
        + (f" Translate English knowledge or tool output into {name}." if language != "en" else "")
        + "\n"
        "This overrides the default: ALWAYS reply in the language of the caller's LATEST "
        "message (English or Hindi), without mentioning it. Call set_language only if the caller asks for a language.\n"
        "After any tool returns, always reply to the caller.\n"
        f"{QUALIFY.format(business=business)}\n"
        f"{PHONE_RULE}\n"
        f"{CONVERSATION_ENDING}"
    )
    grammar = load_grammar(language) if include_grammar else ""
    return f"{base}\n\n{grammar}" if grammar else base


if __name__ == "__main__":
    # Self-check: the persona must stay short (latency) and every language must
    # have parallel copy so nothing goes silent after a language switch.
    _rendered = HOT_PERSONA.format(business="Zryth", persona="Maya")
    assert len(_rendered) <= 800, f"HOT_PERSONA too long: {len(_rendered)} chars (cap 800)"
    for _code in LANG_NAMES:
        assert _code in STYLE_NOTES, f"missing STYLE_NOTES[{_code}]"
        assert _code in GREETINGS, f"missing GREETINGS[{_code}]"
    assert "Zryth" in build_instructions("hi", STYLE_NOTES["hi"])
    assert "Acme" in build_instructions("en", STYLE_NOTES["en"], business="Acme")
    assert "{" not in greeting("hi", "Acme") and "Acme" in greeting("en", "Acme")
    # Grammar sheets are optional; when present they must get appended.
    for _code in LANG_NAMES:
        if load_grammar(_code):
            _with = build_instructions(_code, STYLE_NOTES[_code], include_grammar=True)
            _without = build_instructions(_code, STYLE_NOTES[_code], include_grammar=False)
            assert len(_with) > len(_without), f"grammar sheet for {_code} was not appended"
    print(f"prompts.py self-check passed (HOT_PERSONA={len(_rendered)} chars)")
