"""Name + phone collection (contact_flow.py). No network: pure logic.

Most cases here are regressions from real or simulated calls; the comment says which.
"""

import pytest

from contact_flow import (
    ContactFlow,
    clean_phone,
    extract_digits,
    extract_name,
    introduced_name,
    name_from_history,
)

CALLER = "7677672641"


def flow(lang="en", name=None, caller=CALLER, tool="capture_lead"):
    f = ContactFlow()
    first = f.start(tool, "test requirement", name, lang, caller)
    return f, first


def say(f, text, lang="en", caller=CALLER):
    return f.handle(text, lang, caller).say


# --------------------------------------------------------------------- names
@pytest.mark.parametrize("text, expected", [
    ("My name is Rohit Sharma", "Rohit Sharma"),
    ("Rohit", "Rohit"),
    ("Ravi Kumar", "Ravi Kumar"),
    ("मेरा नाम राहुल है", "राहुल"),            # "है" ends in a vowel sign: \b used to fail
    ("मेरा नाम योगेश है मेरा phone number है", "योगेश"),
    ("मैं सुनील बोल रहा हूँ", "सुनील"),
    ("merra name hiamsnhu hai", "Hiamsnhu"),     # romanized "name", console call
    ("it's Donald Trump", "Donald Trump"),       # trolls are saved as said
])
def test_extract_name_accepts(text, expected):
    assert extract_name(text) == expected


@pytest.mark.parametrize("text", [
    "abhi tho bola",                      # "I already told you" was saved as a name
    "bataya na",
    "And what's FinanceAuditor?",         # a question was saved as a name
    "okay fine, bye",                     # a goodbye was saved as a name
    "Call me back please",                # became "Back Please"
    "Ok ok, then call me tomorrow morning",  # became "Tomorrow Morning"
    "Just get someone to call me ASAP",   # became "Asap"
    "lol just kidding",
    "yes",
    "हाँ",
    "हमारा 1 action class पटना करके coaching institute है",
])
def test_extract_name_rejects(text):
    assert extract_name(text) is None


def test_name_from_history_finds_explicit_intro():
    texts = ["kuch batao", "jee jarur", "merra name hiamsnhu hai", "tomorrow 8 am"]
    assert name_from_history(texts) == "Hiamsnhu"


def test_name_from_history_ignores_bare_words():
    assert name_from_history(["kuch batao", "jee jarur", "tomorrow 8 am"]) is None


# -------------------------------------------------------------------- digits
@pytest.mark.parametrize("text, expected", [
    ("9843234432", "9843234432"),
    ("+91 98450 12345", "9845012345"),
    ("nine eight seven six", "9876"),
    ("six five", "65"),                        # a short spoken group on its own
    ("double nine eight", "998"),
    ("नौ आठ सात छह पाँच चार तीन दो एक शून्य", "9876543210"),
    ("no no no", ""),                          # used to become "999"
    ("do you have it", ""),
    ("ok", ""),
])
def test_extract_digits(text, expected):
    assert extract_digits(text) == expected


@pytest.mark.parametrize("raw, expected", [
    ("07677672641", "7677672641"),
    ("+919876543210", "9876543210"),
    ("1234567890", None),     # must start with 6-9
    ("98765", None),
])
def test_clean_phone(raw, expected):
    assert clean_phone(raw) == expected


# ---------------------------------------------------------------- the steps
def test_full_flow_same_number_en():
    f, first = flow()
    assert "name" in first.lower()
    assert "callback on this number, or another one" in say(f, "Rohit")
    assert "7 6 7 7 6 7 2 6 4 1" in say(f, "same number")
    assert f.handle("yes", "en", CALLER).save is True
    assert (f.name, f.phone) == ("Rohit", CALLER)


def test_full_flow_other_number_in_parts_hi():
    f, _ = flow("hi")
    say(f, "मेरा नाम सुनील है", "hi")
    assert "दस अंकों" in say(f, "दूसरे नंबर पर कॉल करना", "hi")
    assert "5 अंक" in say(f, "98765", "hi")
    assert "9 8 7 6 5 4 3 2 1 0" in say(f, "43210", "hi")
    assert f.handle("हाँ सही है", "hi", CALLER).save is True
    assert f.phone == "9876543210"


def test_no_after_readback_asks_again_and_correction_works():
    f, _ = flow(name="Apurv")
    say(f, "another number")
    say(f, "9876543210")
    assert "again" in say(f, "no no no")
    assert "8 5 9 1 1 9 4 5 0 6" in say(f, "8591194506")


def test_invalid_numbers_give_up_after_three():
    f, _ = flow(name="Neha")
    say(f, "different number")
    say(f, "1234567890")
    say(f, "0123456789")
    line = say(f, "5555555555")
    assert "leave the number" in line and not f.active


def test_time_during_number_step_is_kept_not_read_as_digit():
    f, _ = flow(name="Himanshu", caller=None, tool="book_consultation")
    assert "noted that time" in say(f, "tomorrow 8 am", caller=None)
    assert "Preferred: tomorrow 8 am" in f.requirement
    assert "9 8 4 3 2 3 4 4 3 2" in say(f, "9843234432", caller=None)


def test_same_number_without_caller_id_explains():
    f, _ = flow(name="Himanshu", caller=None)
    assert "can't see the number" in say(f, "same number .", caller=None)


def test_use_this_number_after_other_attempts():
    f, _ = flow(name="Raju")
    say(f, "different number")
    assert CALLER[0] in say(f, "fine, use this number")
    assert f.stage == "confirm" and f.phone == CALLER


@pytest.mark.parametrize("text", ["nahi", "I don't want to give my number", "बार बार मत पूछो", "no need, leave it"])
def test_refusal_drops_the_flow(text):
    f, _ = flow()
    assert "skip" in say(f, text) or "रहने देते" in say(f, text) or not f.active
    assert not f.active


def test_no_thanks_at_choice_declines_instead_of_other_number():
    f, _ = flow("hi", name="Vikas")
    assert "रहने देते" in say(f, "नहीं धन्यवाद", "hi")
    assert not f.active


def test_name_correction_later_replaces_name_once():
    f, _ = flow(name="Batman")
    line = say(f, "lol just kidding, it's Donald Trump")
    assert f.name == "Donald Trump"
    assert line.count("Donald Trump") == 1


def test_already_told_without_name_asks_once_more():
    f, _ = flow(tool="book_consultation", caller=None)
    assert "once more" in say(f, "abhi tho bola", caller=None)
    assert f.name is None


# Real call 2026-10-09: "same number per call kar sakte hain" came through STT as
# "किसी number पर phone कर सकते हैं" and Maya asked "same or another number?" again.
@pytest.mark.parametrize("text", [
    "किसी number पर phone कर सकते हैं।",
    "same number per call kar sakte hain",
    "isi number pe kar lo",
    "is number par",
    "jis number se call kiya usi pe",
])
def test_same_number_variants(text):
    f, _ = flow("hi", name="Rakesh")
    turn = f.handle(text, "hi", CALLER)
    assert f.phone == CALLER and "कन्फ़र्म" in turn.say


def test_other_number_still_other():
    f, _ = flow("hi", name="Rakesh")
    turn = f.handle("दूसरे नंबर पर", "hi", CALLER)
    assert f.phone is None and "दस अंकों" in turn.say


# Real call 2026-10-09: asked for the name, the caller said "इसी number पर call कर दीजिए";
# Maya said "नाम ठीक से सुनाई नहीं दिया". Remember the choice, ask the name, then confirm.
def test_number_choice_before_name():
    f, _ = flow("hi")
    first = f.handle("इसी number पर call कर दीजिए", "hi", CALLER).say
    assert "सुनाई नहीं" not in first and "नाम" in first
    confirm = f.handle("अर्जुन", "hi", CALLER).say
    assert f.name == "अर्जुन" and f.phone == CALLER and "कन्फ़र्म" in confirm


def test_other_number_before_name():
    f, _ = flow("hi")
    f.handle("दूसरे नंबर पर कॉल करना", "hi", CALLER)
    ask = f.handle("अर्जुन", "hi", CALLER).say
    assert f.stage == "digits" and "अर्जुन" in ask


# Caller gives name + number before Maya asks (right after the greeting).
def test_number_given_up_front_is_confirmed_not_reasked():
    f = ContactFlow()
    line = f.start("capture_lead", "x", "सुनील", "hi", CALLER, phone="9822012345")
    assert f.stage == "confirm" and f.phone == "9822012345" and "9 8 2 2" in line and "सुनील" in line


@pytest.mark.parametrize("text, name, phone", [
    ("This is Neha, 9811012345. What does Mill Software do?", "Neha", "9811012345"),
    ("मैं प्रिया बोल रही हूँ, आप लोग क्या करते हो?", "प्रिया", None),
    ("main Amit bol raha hoon, mera number nau aath aath shoonya ek do teen chaar paanch chhah hai", "Amit", "9880123456"),
    ("Okay, what is the price?", None, None),
])
def test_name_and_number_from_first_turn(text, name, phone):
    from contact_flow import phone_from_history
    assert extract_name(text) == name and phone_from_history([text]) == phone


def test_same_number_at_confirm_is_a_yes():
    f, _ = flow("hi", name="अर्जुन")
    f.handle("इसी नंबर पर", "hi", CALLER)          # -> confirm caller ID
    assert f.stage == "confirm"
    assert f.handle("इसी नंबर पर", "hi", CALLER).save


# --------------------------------------------------------- name said early in the call
@pytest.mark.parametrize("text, expected", [
    ("Hi, this is Ravi", "Ravi"),
    ("my name is Priya, what do you do?", "Priya"),
    ("मेरा नाम सुनील है", "सुनील"),
    ("मैं सुनील बोल रहा हूँ", "सुनील"),
    ("Ravi here, I wanted to ask about pricing", "Ravi"),
])
def test_introduced_name(text, expected):
    assert introduced_name(text) == expected


@pytest.mark.parametrize("text", [
    "I'm looking for a clinic near me",
    "hello",
    "what is this?",
    "क्या आप OPD बुक करते हैं?",
])
def test_not_an_introduction(text):
    assert introduced_name(text) is None


def test_name_from_first_turn_survives_long_call():
    # The name was said in turn 1; booking was asked in turn 9 (was a 6-turn window).
    texts = ["Hi, this is Ravi"] + ["what are your timings?"] * 8
    assert name_from_history(texts) == "Ravi"


def test_known_name_skips_name_question():
    f, first = flow(name="Ravi")
    assert "name" not in first.lower()
    assert f.stage == "choice"


@pytest.mark.parametrize("text", ["main interested hoon", "मैं परेशान हूँ", "main bimar hoon"])
def test_main_x_hoon_is_not_a_name(text):
    # Runs on every turn: "मैं परेशान हूँ" must not make the caller "परेशान".
    assert introduced_name(text) is None


# ------------------------------------------- Kixie.ai's sales bot call, 2026-10-17
@pytest.mark.parametrize("text", ["Take your time. I'm all ears", "I'm all set", "I'm here"])
def test_filler_phrases_are_not_names(text):
    assert extract_name(text) is None


def test_confirm_with_same_digits_is_yes():
    # "That is correct. 7965813925." was re-read as a new number and confirmed twice.
    f, _ = flow(name="Kixie", caller="7965813925")
    f.handle("this number", "en", "7965813925")
    assert f.stage == "confirm"
    assert f.handle("That is correct. 7965813925.", "en", "7965813925").save


def test_confirm_with_other_digits_is_not_yes():
    f, _ = flow(name="Kixie", caller="7965813925")
    f.handle("this number", "en", "7965813925")
    assert not f.handle("no, 9822012345", "en", "7965813925").save


def test_name_stops_at_sentence_end():
    t = ("This is Kixie from Kixie.ai and yes, I am the AI you just summoned. "
         "Pretty cool that you are already talking to the product, right?")
    assert introduced_name(t) == "Kixie"
    assert extract_name("This is Kixie. Pretty cool") == "Kixie"
