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

USE_CASE_TEMPLATES: dict[str, str] = {
    "promotional": (
        "Lead with {company}'s current offers. Be upbeat and concise. "
        "Always try to end with a clear next step."
    ),
    "lead_generation": (
        "Ask qualifying questions (need, timeline, budget) ONE BY ONE. Do not ask "
        "multiple questions in a single response. Capture the caller's name and a "
        "contact method before the call ends. Call capture_lead as soon as you have "
        "a name and one contact method."
    ),
    "support": (
        "Answer only from search_knowledge results. If there's no answer, say so "
        "and offer transfer_to_human. Never invent information."
    ),
    "custom": "",
}

# Non-negotiable, not editable by any client config.
GUARDRAILS = (
    "You MUST use the search_knowledge tool for any question about {company}'s "
    "products, pricing, or features. Never guess. If asked about anything "
    "unrelated to {company}, politely decline and redirect. CRITICAL RULE: "
    "Keep responses to a maximum of 1 to 2 short lines. Never ask more than "
    "one question at a time."
)

def build_persona(agent_row: dict) -> str:
    company = agent_row["company_name"]
    template = USE_CASE_TEMPLATES.get(agent_row["use_case"], "")
    custom = (agent_row.get("custom_instructions") or "").strip()

    parts = [
        f"You are {agent_row['agent_name']}, the voice assistant for {company}.",
        GUARDRAILS.format(company=company),
        template.format(company=company) if template else "",
        custom,
    ]
    persona = "\n".join(p for p in parts if p)

    # Same discipline as before — this is still the biggest latency lever.
    # Don't let a client's custom_instructions blow the per-turn token budget.
    if len(persona) > 900:
        persona = persona[:900].rsplit(".", 1)[0] + "."
    return persona


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
    "en": "Hi, thanks for calling {company_name}! I'm {agent_name}, the AI assistant. How can I help you today?",
    "hi": "नमस्ते, {company_name} में कॉल करने के लिए धन्यवाद! मैं {agent_name} हूँ, आपकी AI असिस्टेंट। मैं आपकी कैसे मदद कर सकती हूँ?",
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


def build_instructions(language: str, script: str, persona: str, include_grammar: bool = True) -> str:
    """Compose the full system prompt for a per-language agent.

    `language` is a short code (en/hi/...); `script` is the tiny per-language
    style note (usually STYLE_NOTES[language]). `persona` is the dynamically assembled
    persona for the specific agent.

    Latency tradeoff: the grammar sheet adds input tokens every turn, which raises
    LLM time-to-first-token a little. It buys much more natural, native-sounding
    Indic speech -- usually worth it. Set include_grammar=False (or trim the sheet)
    if you need to shave the last few ms. See docs/04-latency.md.
    """
    name = LANG_NAMES.get(language, language)
    base = f"{persona}\n\nRespond only in {name}. {script}\n\nIf the caller asks to speak in a different language, immediately call the set_language tool with the language code (en, hi).\n\n{CONVERSATION_ENDING}"
    grammar = load_grammar(language) if include_grammar else ""
    return f"{base}\n\n{grammar}" if grammar else base


if __name__ == "__main__":
    # Self-check: the persona must stay short (latency) and every language must
    mock_agent = {"agent_name": "TestBot", "company_name": "TestCo", "use_case": "promotional", "custom_instructions": ""}
    test_persona = build_persona(mock_agent)
    assert len(test_persona) <= 1500, f"Persona too long: {len(test_persona)} chars"
    for _code in LANG_NAMES:
        assert _code in STYLE_NOTES, f"missing STYLE_NOTES[{_code}]"
        assert _code in GREETINGS, f"missing GREETINGS[{_code}]"
    assert "TestBot" in build_instructions("hi", STYLE_NOTES["hi"], test_persona)
    # Grammar sheets should exist and get appended when present.
    for _code in LANG_NAMES:
        assert load_grammar(_code), f"missing/empty grammar sheet for {_code}"
    _with = build_instructions("en", STYLE_NOTES["en"], test_persona, include_grammar=True)
    _without = build_instructions("en", STYLE_NOTES["en"], test_persona, include_grammar=False)
    assert len(_with) > len(_without), "grammar sheet was not appended"
    print(f"prompts.py self-check passed (Persona={len(test_persona)} chars, grammar wired)")
