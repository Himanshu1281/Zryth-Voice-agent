"""Phone number types, emergencies, department visits and profile warm-up. No network."""

import pytest

import business
import tools
from business import BusinessProfile
from contact_flow import ContactFlow, clean_phone
from intents import BOOKING_WORDS, EMERGENCY, INTERESTED, WANT
from replies import scrub_jargon

CALLER = "7677672641"


# --------------------------------------------------------------- phone types
@pytest.mark.parametrize("phone, types, expected", [
    ("+91 98765 43210", ("mobile",), "9876543210"),
    ("09876543210", ("mobile",), "9876543210"),
    ("020 2543 1234", ("mobile",), None),                         # landline not allowed by default
    ("020 2543 1234", ("mobile", "landline"), "2025431234"),
    ("0141-4567890", ("mobile", "landline"), "1414567890"),
    ("1234567890", ("mobile", "landline"), None),                 # joke numbers fit a landline's shape
    ("5555555555", ("mobile", "landline"), None),
    ("+44 20 7946 0958", ("mobile",), None),
    ("+44 20 7946 0958", ("mobile", "international"), "+442079460958"),
    ("0044 20 7946 0958", ("mobile", "international"), "+442079460958"),
    ("+1 23", ("mobile", "international"), None),                  # too short
    ("+91 98765 43210", ("mobile", "international"), "9876543210"),  # +91 stays an Indian mobile
])
def test_clean_phone_types(phone, types, expected):
    assert clean_phone(phone, types) == expected


def _flow(types):
    f = ContactFlow()
    f.phone_types = list(types)
    f.start("capture_lead", "test", "Priya", "en", CALLER)
    f.handle("another number", "en", CALLER)
    return f


def test_international_number_in_one_go():
    f = _flow(("mobile", "international"))
    turn = f.handle("plus four four two zero seven nine four six zero nine five eight", "en", CALLER)
    assert "4420" in (turn.say or "").replace(" ", "") or f.phone == "+442079460958"


def test_landline_accepted_only_when_allowed():
    f = _flow(("mobile", "landline"))
    f.handle("0 2 0 2 5 4 3 1 2 3 4", "en", CALLER)
    assert f.phone == "2025431234"
    g = _flow(("mobile",))
    g.handle("0 2 0 2 5 4 3 1 2 3 4", "en", CALLER)
    assert g.phone != "2025431234"


# --------------------------------------------------------------- emergencies
@pytest.mark.parametrize("text", [
    "मेरे पापा को सीने में बहुत दर्द हो रहा है, साँस नहीं ले पा रहे",
    "My husband just fainted and he is not breathing properly",
    "accident ho gaya hai road pe",
    "someone had an accident outside, there is heavy bleeding",
    "papa behosh ho gaye",
    "घर में आग लग गई है",
])
def test_emergency_detected(text):
    assert EMERGENCY.search(text)


@pytest.mark.parametrize("text", [
    "मेरे दाँत में बहुत दर्द है",                             # ordinary pain: book a visit
    "I want to claim insurance for my car accident last month",
    "Do you have a cardiologist?",
    "knee pain hai, ortho ko dikhana hai",
])
def test_not_an_emergency(text):
    assert not EMERGENCY.search(text)


@pytest.mark.parametrize("text", ["cardiologist ko dikhana hai", "मेरी माँ को डॉक्टर को दिखाना है", "heart checkup chahiye"])
def test_department_visit_is_a_want(text):
    assert WANT.search(text)


def test_emergency_number_in_profile():
    p = BusinessProfile.from_dict({"name": "Shanti Hospital", "emergency_number": "0141-4567890"})
    assert p.emergency_number == "0141-4567890"
    assert BusinessProfile().emergency_number is None


# --------------------------------------------------------------- warm-up
def test_warm_profiles_without_kb_is_a_no_op(monkeypatch, tmp_path):
    import database
    monkeypatch.setattr(database, "LANCEDB_PATH", str(tmp_path / "missing"))
    called = []
    monkeypatch.setattr(tools, "build_profile", lambda *a, **k: called.append(a))
    assert tools.warm_profiles() == 0 and not called


def test_profile_version_invalidates_old_cache():
    assert business.PROFILE_VERSION >= 2


# --------------------------------------------------------------- interest / doctor visits


@pytest.mark.parametrize("text, expected", [
    ("That sounds useful, I'm interested.", True),
    ("main interested hoon", True),
    ("मुझे इसमें रुचि है", True),
    ("I'm not interested", False),
    ("I'm interested in knowing the price?", False),
])
def test_clear_interest(text, expected):
    assert bool(INTERESTED.search(text)) == expected


@pytest.mark.parametrize("text, expected", [
    ("मेरी माँ को cardiologist को दिखाना है", True),
    ("doctor se milna hai", True),
    ("Do you have a cardiologist?", False),   # a question, not a booking
])
def test_doctor_visit_starts_booking(text, expected):
    assert bool(BOOKING_WORDS.search(text) and WANT.search(text)) == expected


# --------------------------------------------------------------- internal jargon


@pytest.mark.parametrize("text, expected", [
    ("The knowledge base does not contain information about monthly maintenance charges.",
     "I don't have the details on monthly maintenance charges."),
    ("According to the knowledge base, we open at 9.", "We open at 9."),
    ("We sell knowledge products.", "We sell knowledge products."),
])
def test_scrub_jargon(text, expected):
    assert scrub_jargon(text) == expected


def test_scrub_jargon_keeps_mid_sentence_case():
    # Streamed chunks start mid-sentence: "for all departments" must stay lowercase.
    assert scrub_jargon("for all departments, at our Malviya Nagar branch") == "for all departments, at our Malviya Nagar branch"


def test_only_indian_mobiles_for_now():
    # Landline / international can't be switched on from a profile file yet.
    p = BusinessProfile.from_dict({"phone_types": ["mobile", "landline", "international"]})
    assert p.phone_types == ["mobile"]
