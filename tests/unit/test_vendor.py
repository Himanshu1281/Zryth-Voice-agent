"""Vendor mode (vendor.py): a sales bot's call. No network."""

import time

import vendor
from tools import AppointmentTools, call_outcome


def test_vendor_number_detected():
    t = AppointmentTools()
    t.caller_phone = "07965813925"
    assert t.vendor
    t.caller_phone = "07677672641"
    assert not t.vendor


def test_questions_in_order_then_wrap_up():
    assert vendor.QUESTIONS[0] in vendor.vendor_note(0)
    assert vendor.QUESTIONS[-1] in vendor.vendor_note(len(vendor.QUESTIONS) - 1)
    assert "goodbye" in vendor.vendor_note(len(vendor.QUESTIONS))


def test_note_forbids_selling():
    note = vendor.vendor_note(0).lower()
    assert "don't pitch" in note and "never ask for their name" in note


def test_stats_time_both_sides(monkeypatch):
    clock = iter([0.0, 1.0, 2.5, 4.0, 4.6, 5.0, 6.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    s = vendor.VendorStats()
    s.on_agent_state("speaking")    # 0.0 Maya talks
    s.on_agent_state("listening")   # 1.0 Maya done
    s.on_user_state("speaking")     # 2.5 they answer: 1.5 s
    s.on_user_state("listening")    # 4.0 they're done
    s.on_agent_state("speaking")    # 4.6 Maya answers: 0.6 s
    s.on_user_state("speaking")     # 5.0 they talk over Maya
    s.on_agent_state("listening")   # 6.0
    out = s.summary()
    assert out["their_response_s"] == [1.5]
    assert out["our_response_s"] == [0.6]
    assert out["they_interrupted_maya"] == 1


def test_vendor_call_outcome():
    assert call_outcome("consultation_booked", None, vendor=True) == "vendor_pitch"
