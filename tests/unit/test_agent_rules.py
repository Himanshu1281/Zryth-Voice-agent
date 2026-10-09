"""Agent-level rules: maya.py, intents.py, language.py, replies.py, tools.py. No network.

Each case is a regression from a real call, a simulated call, or the persona suite.
"""

import asyncio

import pytest
from livekit.agents.llm import ChatContext

import intents
import language
import maya
import prompts
import replies
from tools import AppointmentTools


# ------------------------------------------------------------ language detection
@pytest.mark.parametrize("text, expected", [
    ("Hi, what does your company do?", "en"),
    ("I want to talk to the team", "en"),
    ("can you tell me more about it", "en"),
    ("जी हमारा 1 action class पटना करके coaching institute है", "hi"),
    ("मेरे पास Instagram पे influencer account है, per day बहुत messages आते हैं", "hi"),  # many English nouns
    ("kuch batao apni company key bare mey", "hi"),    # romanized Hindi (console)
    ("jee jarur", "hi"),
    ("abe chup, bakwas mat kar", "hi"),               # Hinglish abuse got English replies
    ("हाँ", None),                                     # single words never switch
    ("ok", None),
])
def test_detect_language(text, expected):
    assert language.detect_language(text) == expected


@pytest.mark.parametrize("head, code, wrong", [
    ("Got it. We can help you automate the process", "hi", True),   # English to a Hindi caller
    ("FinanceAuditor.ai एक audit platform है जो", "hi", False),     # English product name first is fine
    ("ज़रूर, हम AI के ज़रिए आपकी मदद कर सकते हैं", "en", True),
    ("Sure, Zryth builds custom AI agents for you", "en", False),
])
def test_wrong_language_guard(head, code, wrong):
    assert language.wrong_language(head, code) is wrong


def test_repeat_guard_catches_the_six_times_loop():
    asked = "कोई बात नहीं। क्या आप मुझे बता सकते हैं कि आपकी टीम का कौन सा काम सबसे ज़्यादा समय लेता है?"
    assert replies.repeats_earlier("कोई बात नहीं। क्या आप मुझे बता सकते हैं कि", [asked])
    assert not replies.repeats_earlier("कोई बात नहीं! क्या आपके Zryth के बारे में कोई", [asked])
    assert not replies.repeats_earlier("Got it. ", ["Got it. We build AI."])  # short openers repeat freely


# -------------------------------------------------------------- sentence cap
def _run_cap(pieces, monkeypatch, lang="en"):
    async def inner(self, ctx, tools, ms):
        for p in pieces:
            yield p

    monkeypatch.setattr(maya.BaseMayaAgent, "_llm_node_inner", inner)

    async def go():
        g = maya.GreeterAgent(language=lang)
        out = ""
        async for c in g.llm_node(ChatContext(), [], None):
            out += c if isinstance(c, str) else c.delta.content
        return out

    return asyncio.run(go())


def test_cap_keeps_closing_question(monkeypatch):
    out = _run_cap(["Oswaal AI is a chapter-wise learning hub for students. ", "It has podcasts and slides for every chapter. ",
                    "It also has adaptive tests for revision. ", "Would you like to know more?"], monkeypatch)
    assert out.endswith("Would you like to know more?")
    assert "adaptive tests" not in out


def test_cap_short_opener_does_not_eat_the_answer(monkeypatch):
    # "नमस्ते!" used to count as a sentence and push the real answer out.
    out = _run_cap(["नमस्ते! ", "आपकी परेशानी समझ सकती हूँ। ", "हम WhatsApp orders को अपने आप billing में डाल सकते हैं। ",
                    "इससे गलती कम होगी और समय बचेगा। ", "क्या आप consultation चाहेंगे?"], monkeypatch, "hi")
    assert "WhatsApp orders" in out and out.endswith("चाहेंगे?")


def test_cap_ignores_dots_inside_names(monkeypatch):
    out = _run_cap(["Sure, version 2.5 of FinanceAuditor.ai is ready. Want a demo?"], monkeypatch)
    assert out == "Sure, version 2.5 of FinanceAuditor.ai is ready. Want a demo?"


def test_instruction_aside_is_never_spoken(monkeypatch):
    # Seen in the persona suite: the reply ended with a made-up instruction. The
    # "(If" arrives in one chunk and the rest in the next, as in a real stream.
    out = _run_cap(["नमस्ते, यह Zryth है, मैं आपकी इसमें मदद नहीं कर सकती।\n\n(If ",
                    "the caller is done, call `end_call`. If they show interest, call `capture_lead`.)"],
                   monkeypatch, "hi")
    assert out.strip() == "नमस्ते, यह Zryth है, मैं आपकी इसमें मदद नहीं कर सकती।"


def test_ordinary_brackets_are_kept(monkeypatch):
    out = _run_cap(["We build AI agents (like ", "WhatsApp chatbots) for your business. Want to know more?"], monkeypatch)
    assert out == "We build AI agents (like WhatsApp chatbots) for your business. Want to know more?"


def test_tool_speak_line_is_said_verbatim(monkeypatch):
    from livekit.agents.llm import FunctionCallOutput

    async def inner(self, ctx, tools, ms):
        raise AssertionError("the LLM must not be called")
        yield  # pragma: no cover

    monkeypatch.setattr(maya.BaseMayaAgent, "_llm_node_inner", inner)
    ctx = ChatContext()
    ctx.items.append(FunctionCallOutput(call_id="1", name="capture_lead", output=replies.SAY_MARKER + "May I know your name?",
                                        is_error=False))

    async def go():
        g = maya.GreeterAgent(language="en")
        return "".join([c async for c in g.llm_node(ctx, [], None)])

    assert asyncio.run(go()) == "May I know your name?"


# ------------------------------------------------------------ ending the call
@pytest.mark.parametrize("text, done", [
    ("no thanks, bye", True),
    ("Okay, that's all I wanted to know. Bye.", True),
    ("नहीं, बस इतना ही", True),
    ("nahi bas", True),                    # romanized, didn't end the clinic call
    ("ठीक है धन्यवाद", True),               # job seeker
    ("अच्छा गलती से लग गया, सॉरी", True),   # wrong number
    ("thanks, and what about pricing", False),
    ("tell me more", False),
])
def test_clear_goodbye(text, done):
    assert bool(intents.CLEAR_GOODBYE.search(text)) is done


@pytest.mark.parametrize("text, prev, done", [
    ("नहीं बस", "क्या मैं आपकी और किसी चीज़ में मदद कर सकती हूँ?", True),
    ("ठीक है", "Is there anything else I can help you with?", True),
    ("no", "Is there anything else I can help you with?", True),
    ("thanks, and what about pricing?", "x", False),
    ("Tell me about Mill Software", "anything else?", False),
])
def test_caller_is_done_after_save(text, prev, done):
    assert AppointmentTools.caller_is_done(text, prev) is done


# ---------------------------------------------------------- office hours / visits
@pytest.mark.parametrize("text, hours, where", [
    ("आपका office कहाँ है? मैं मिलने आना चाहता हूँ", True, True),
    ("कब खुला रहता है? Sunday को आ सकता हूँ?", True, False),
    ("Where is your office? I'd like to visit you this week.", True, True),
    ("Are you open on Saturday?", True, False),
    ("Can I come and meet you tomorrow?", True, False),
    ("where is your office", False, True),       # answered from the KB, no callback
    ("what does Zryth do", False, False),
    ("tell me about your team", False, False),
])
def test_office_hours_detection(text, hours, where):
    assert bool(intents.OFFICE_HOURS.search(text)) is hours
    assert bool(intents.WHERE.search(text)) is where


def test_office_address_read_from_kb():
    kb = "Phone / WhatsApp: +91-9870661438 Office: Tower A, Logix Technova, A-002, Sector 132, Noida, UP 201304 Web: zryth.com"
    assert replies.office_address(kb) == "Tower A, Logix Technova, A-002, Sector 132, Noida, UP 201304"
    assert replies.office_address(None) is None


@pytest.mark.parametrize("text", [
    "academy kitne baje khulti hai? Sunday ko aa sakte hain?",   # romanized, coaching persona
    "क्लिनिक कितने बजे खुलता है?",
])
def test_office_hours_romanized_and_hindi(text):
    assert intents.OFFICE_HOURS.search(text)


def test_romanized_booking_request():
    # "appointment chahiye" was missed: only Devanagari "चाहिए" counted as wanting.
    text = "doctor se milna hai, appointment chahiye"
    assert intents.WANT.search(text) and intents.BOOKING_WORDS.search(text)


@pytest.mark.parametrize("text, unknown", [
    ("payment कैसे लेते हो? UPI चलेगा? EMI मिलेगी?", True),
    ("GST invoice दोगे ना?", True),
    ("मैडम आपके यहाँ नौकरी है क्या?", True),
    ("what does Zryth do", False),
])
def test_unknown_fact_nudge(text, unknown):
    assert bool(intents.UNKNOWN_FACT.search(text)) is unknown


# ----------------------------------------------------------------- intents
@pytest.mark.parametrize("text, is_no", [
    ("एक गाना सुनाओ ना", False),     # soft "please", not a no
    ("price बताइए ना", False),
    ("ना", True),
    ("नहीं", True),
    ("nahi", True),
])
def test_no_detection(text, is_no):
    assert bool(intents.NO.search(text)) is is_no


@pytest.mark.parametrize("text, wants", [
    ("Can I talk to a human please?", True),
    ("मुझे किसी इंसान से बात करनी है", True),
    ("I want to speak with your manager", True),
    ("Are you a real person or a bot?", False),   # used to start the callback steps
])
def test_wants_human(text, wants):
    assert bool(intents.HUMAN.search(text) and intents.TALK.search(text)) is wants


@pytest.mark.parametrize("text, callback", [
    ("Can someone from your team call me back?", True),
    ("yes i want to contact with team", True),
    ("मुझे आपकी टीम से कॉल बैक चाहिए", True),
    ("उसका नंबर लिखो", True),            # elderly caller: LLM "noted" it and saved nothing
    ("take my number please", True),
    ("tell me about the team", False),
    ("tell me a number", False),
])
def test_callback_words(text, callback):
    assert bool(intents.CALLBACK_WORDS.search(text)) is callback


@pytest.mark.parametrize("text", ["That sounds useful, I'm interested.", "jee jarur", "ज़रूर", "sounds good"])
def test_accepting_an_offer(text):
    assert intents.YES.search(text) or intents.ACCEPT.search(text)


# ------------------------------------------------------------------ prompts
def test_prompt_persona_stays_short():
    assert len(prompts.HOT_PERSONA.format(business="Zryth", persona="Maya")) <= 800


def test_prompt_has_truth_and_no_connect_promise():
    text = prompts.build_instructions("hi", prompts.STYLE_NOTES["hi"])
    assert "TRUTH:" in text
    assert "offer to connect them" not in text


# Real call 2026-10-09: "फ्री डिस्कवरी ऑडिट करें?" -> "haan ji" was not a yes, then the LLM said
# "कंसल्टेशन बुक कर रही हूँ" -> "karo" -> "बुक कर दिया" without asking name or number.
@pytest.mark.parametrize("reply", [
    "क्या आप चाहेंगे कि हम इस पर एक फ्री डिस्कवरी ऑडिट करें?",
    "Would you like a free discovery audit?",
])
def test_kb_worded_offer_is_an_offer(reply):
    from intents import OFFER
    assert OFFER.search(reply)


@pytest.mark.parametrize("reply, expected", [
    ("ज़रूर, मैं आपके लिए एक कंसल्टेशन बुक कर रही हूँ।", True),
    ("बहुत बढ़िया! मैं आपके लिए एक कंसल्टेशन बुक कर देती हूँ।", True),
    ("Great, I'll book a consultation for you.", True),
    ("क्या आप जानना चाहेंगे कि हम यह कैसे कर सकते हैं?", False),
])
def test_booking_promise(reply, expected):
    from intents import PROMISED
    assert bool(PROMISED.search(reply)) == expected


@pytest.mark.parametrize("text", ["karo", "kar do", "हाँ करो", "book kar do"])
def test_karo_is_a_yes(text):
    from intents import ACCEPT
    assert ACCEPT.search(text)


@pytest.mark.parametrize("text, expected", [
    ("मेरा नाम हिमांशु है", True),
    ("my name is Ravi Kumar", True),
    ("main Sunil bol raha hoon", True),
    ("what is your name?", False),
    ("naam mein kya rakha hai", False),
])
def test_volunteered_name(text, expected):
    from contact_flow import extract_name
    from intents import GAVE_NAME
    assert bool(GAVE_NAME.search(text) and extract_name(text)) == expected
