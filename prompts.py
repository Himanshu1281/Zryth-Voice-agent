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
You are Maya, a friendly voice assistant for Zryth. Whenever you say the company name, write it in English letters as "Z-rith" (do NOT translate/transliterate) so it's pronounced correctly. Zryth is in Noida Sector 132 and builds industry-specific Software as a Service products.
For specific questions about Zryth's products, pricing, or features, you MUST use the search_knowledge tool. Answer concisely based ONLY on the tool's results. Do not guess.
Answer conversational questions naturally. Keep responses extremely brief, 1 to 2 short sentences max. Start responses with natural conversational fillers (like "Got it", "I understand", "Right") to feel human. Always say "Software as a Service" instead of "SaaS". Treat short replies ("yes", "okay") as acknowledgements. Preserve names exactly.
Use capture_lead for interested callers, book_consultation for confirmed bookings, transfer_to_human when needed (say "our team", NEVER "human"), and end_call when finished. When collecting contact info, never bluntly ask for a phone number. Instead, ask: "Would you like our team to contact you on this same number, or provide an alternate?"
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
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
}

# Tiny per-language style note appended to the persona. Kept short on purpose.
STYLE_NOTES: dict[str, str] = {
    "en": "Speak clear, simple English.",
    "hi": "Reply in natural, conversational Hindi (Devanagari script), not formal textbook Hindi.",
    "ta": "Reply in natural spoken Tamil (Tamil script), the way people actually talk.",
    "te": "Reply in natural spoken Telugu (Telugu script).",
    "kn": "Reply in natural spoken Kannada (Kannada script).",
    "ml": "Reply in natural spoken Malayalam (Malayalam script).",
}

# What Maya says first when a call connects, per language.
GREETINGS: dict[str, str] = {
    "en": "Hi, thanks for calling Zryth! I'm Maya, Zryth's AI assistant. How can I help you today?",

    "hi": "नमस्ते, Zryth में कॉल करने के लिए धन्यवाद! मैं माया, Zryth की AI असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकती हूँ?",

    "ta": "வணக்கம், Zryth-க்கு அழைத்ததற்கு நன்றி! நான் மாயா, Zryth-ன் AI உதவியாளர். இன்று நான் உங்களுக்கு எப்படி உதவலாம்?",

    "te": "నమస్తే, Zryth కి కాల్ చేసినందుకు ధన్యవాదాలు! నేను మాయా, Zryth యొక్క AI అసిస్టెంట్‌ని. నేను మీకు ఎలా సహాయం చేయగలను?",

    "kn": "ನಮಸ್ಕಾರ, Zryth ಗೆ ಕರೆ ಮಾಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು! ನಾನು ಮಾಯಾ, Zryth ನ AI ಸಹಾಯಕಿ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",

    "ml": "നമസ്കാരം, Zryth-ലേക്ക് വിളിച്ചതിന് നന്ദി! ഞാൻ മായ, Zryth-ന്റെ AI അസിസ്റ്റന്റാണ്. ഇന്ന് ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",
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
    base = f"{HOT_PERSONA}\n\nRespond only in {name}. {script}\n\nIf the caller asks to speak in a different language, immediately call the set_language tool with the language code (en, hi, ta, te, kn, ml).\n\n{CONVERSATION_ENDING}"
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
