"""The agent works for any business: profile layering, gender, booking words, and
no Zryth-specific text leaking into another company's prompt. No network."""

import json

import pytest

import business
import prompts
from business import BusinessProfile, gendered
from contact_flow import ContactFlow
from tools import AppointmentTools, _labelled
from routing import CallConfig

CLINIC = BusinessProfile.from_dict({
    "name": "Smile Care Dental Clinic", "persona": "Riya", "persona_hi": "रिया",
    "booking_en": "appointment", "booking_hi": "अपॉइंटमेंट", "offer_en": "an appointment",
    "offer_hi": "एक अपॉइंटमेंट", "discovery_en": "their dental problem",
    "booking_words": ["checkup", "डॉक्टर से मिलना"], "hours": "Mon-Sat 10 am to 8 pm",
    "brands": ["Smile Care"],
})

ZRYTH_ONLY = ["Zryth", "Oswaal", "Mill Software", "FinanceAuditor", "Document AI", "Voice AI",
              "discovery audit", "AI seminar", "custom AI agents"]


def test_defaults_are_neutral():
    p = BusinessProfile()
    assert p.name == "our company" and p.gender == "female" and p.hours is None


def test_from_dict_ignores_empty_and_unknown_keys():
    p = BusinessProfile.from_dict({"name": "Acme", "hours": "", "nonsense": 1, "gender": "robot"})
    assert p.name == "Acme" and p.hours is None and p.gender == "female"


def test_layering_config_beats_kb_and_override_beats_both(tmp_path, monkeypatch):
    monkeypatch.setattr(business, "OVERRIDES_DIR", tmp_path)
    (tmp_path / "agent-1.json").write_text(json.dumps({"booking_en": "trial class"}), encoding="utf-8")
    p = business._layer({"name": "Name From KB", "booking_en": "demo class", "address": "Kota"},
                        "agent-1", "Name From Config", "Maya")
    assert p.name == "Name From Config"      # the agents row wins over the KB
    assert p.booking_en == "trial class"     # the override file wins over everything
    assert p.address == "Kota"               # KB facts kept


def test_profile_without_kb_never_calls_the_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no KB: no LLM call")

    monkeypatch.setattr(business, "_extract_with_llm", boom)
    assert business.build_profile("x", None, "Acme", "Maya").name == "Acme"


def test_failed_llm_falls_back_to_kb_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(business, "CACHE_DIR", tmp_path)

    def down(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(business, "_extract_with_llm", down)
    kb = "Spice Route\nTimings: 11:30 am to 11 pm\nAddress: 12, Vijay Nagar, Indore 452010\nWeb: x"
    p = business.build_profile("rest", kb, "Spice Route")
    assert p.hours == "11:30 am to 11 pm" and p.address == "12, Vijay Nagar, Indore 452010"


@pytest.mark.parametrize("female, male", [
    ("मैं आपकी मदद कर सकती हूँ", "मैं आपकी मदद कर सकता हूँ"),
    ("ठीक है, देखती हूँ।", "ठीक है, देखता हूँ।"),
])
def test_gendered_hindi(female, male):
    assert gendered(female, "male") == male
    assert gendered(female, "female") == female


def test_booking_words_from_profile():
    assert CLINIC.booking_regex.search("bas checkup karwana hai")
    assert CLINIC.booking_regex.search("मुझे डॉक्टर से मिलना है")
    assert not CLINIC.booking_regex.search("what are your timings")
    assert BusinessProfile().booking_regex is None


def test_prompt_for_another_business_has_no_zryth_text():
    for lang in ("en", "hi"):
        text = prompts.build_instructions(lang, prompts.STYLE_NOTES[lang], profile=CLINIC)
        for term in ZRYTH_ONLY:
            assert term not in text, f"{term!r} leaked into a dental clinic's {lang} prompt"
        assert "Smile Care Dental Clinic" in text and "appointment" in text and "their dental problem" in text


def test_male_persona_prompt_and_greeting():
    male = BusinessProfile.from_dict({"name": "Acme", "persona": "Arjun", "persona_hi": "अर्जुन", "gender": "male"})
    assert "You are male" in prompts.build_instructions("hi", prompts.STYLE_NOTES["hi"], profile=male)
    hi = prompts.greeting("hi", profile=male)
    assert "अर्जुन" in hi and "सकता हूँ" in hi and "सकती" not in hi


def test_contact_flow_uses_business_booking_word_and_gender():
    f = ContactFlow()
    f.booking, f.gender = {"en": "table reservation", "hi": "टेबल रिज़र्वेशन"}, "male"
    f.start("book_consultation", None, "Prakash", "hi", "7677672641")
    f.handle("इसी नंबर पर", "hi", "7677672641")
    assert f.handle("हाँ", "hi", "7677672641").save
    line = f.saved_text("hi", "Spice Route")
    assert "टेबल रिज़र्वेशन" in line and "सकता हूँ" in line


def test_saved_label_uses_booking_word():
    assert _labelled("book_consultation", "x", "site visit") == "[Site visit request] x"
    assert _labelled("capture_lead", None, "site visit") == "[Callback request]"


def test_tools_take_the_profile():
    t = AppointmentTools(config=CallConfig(business_name="Skyline Homes", persona_name="Maya"))
    assert t.profile.name == "Skyline Homes"
    t.set_profile(CLINIC)
    assert t.contact.booking["en"] == "appointment"


def test_read_only_cache_still_returns_extracted_profile(tmp_path, monkeypatch):
    # VPS (systemd ProtectSystem=strict): data/profiles was read-only and the cache write
    # raised. The extracted profile must still be used.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setattr(business, "CACHE_DIR", blocker / "profiles")
    monkeypatch.setattr(business, "_extract_with_llm", lambda kb, model: {"booking_en": "site visit"})
    p = business.build_profile("agent-x", "Skyline Homes. Book a site visit.")
    assert p.booking_en == "site visit"


@pytest.mark.parametrize("llm, saved, expected", [
    ("lead_captured", None, "unresolved"),          # real call: no lead, marked lead_captured
    ("consultation_booked", None, "unresolved"),
    ("answered", None, "answered"),
    ("answered", "capture_lead", "lead_captured"),
    ("lead_captured", "book_consultation", "consultation_booked"),
])
def test_call_outcome_comes_from_what_was_saved(llm, saved, expected):
    from tools import call_outcome
    assert call_outcome(llm, saved) == expected
