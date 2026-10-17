"""Vendor mode: a competitor's sales bot is calling (VENDOR_NUMBERS, e.g. Kixie.ai).

Maya stops selling and acts as an interested buyer: she asks the questions any
prospect would ask, one per turn. Meanwhile we time how their agent behaves, and
after the call a report (their answers + timings + what they do that we don't)
is saved as the call's summary. Normal buyer questions only: no attempts to pull
out their hidden prompts or internal setup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("vendor")

# Asked in this order, one per turn, after a short reply to what they just said.
QUESTIONS: list[str] = [
    # Services
    "Which languages do you support, and can your agent switch language in the middle of a call?",
    "Which channels do you cover, voice, WhatsApp, SMS, and how does a lead move between them?",
    "Which CRMs and calendars do you integrate with, and how is a hot lead handed to a person?",
    "Can your agent transfer a call to a human live, and what happens outside office hours?",
    # How it's built (what they share publicly)
    "Which voice and AI providers do you use, or is it all built in-house?",
    "What is your typical response time on a call, from when the caller stops talking?",
    "How does your agent handle callers interrupting it, background noise or a bad line?",
    "How do you train the agent on a new client's business, and how long does setup take?",
    # Scale and operations
    "How many calls can you run at the same time, and what uptime do you offer?",
    "Where is call data stored, and how do you handle consent and call recordings?",
    "What is your pricing model, per minute, per call or per lead?",
]


def vendor_note(asked: int) -> str:
    """System note for the LLM this turn."""
    base = (
        "This caller is a vendor's AI sales agent pitching their product to us. You are an interested "
        "buyer, not a seller: don't pitch our services, don't offer a consultation, callback or booking, "
        "and never ask for their name or number. Never repeat their words back. If they offer to take "
        "our details, book a demo or call us, politely decline for now (we'd like to understand more "
        "first). React to what they said in a few words, then ask"
    )
    if asked < len(QUESTIONS):
        return f'{base} exactly this one question: "{QUESTIONS[asked]}"'
    return f"{base} nothing new: thank them, say our team will review it, and say goodbye."


@dataclass
class VendorStats:
    """Turn timings on a vendor call, from the session's agent/user state events."""

    their_gaps: list[float] = field(default_factory=list)  # Maya stops -> they start
    our_gaps: list[float] = field(default_factory=list)    # they stop -> Maya starts
    they_interrupted: int = 0  # they started talking while Maya spoke
    asked: int = 0             # QUESTIONS asked so far
    _agent: str = "initializing"
    _maya_done: float | None = None
    _they_done: float | None = None

    def on_agent_state(self, state: str) -> None:
        now = time.monotonic()
        if state == "speaking" and self._they_done is not None:
            self.our_gaps.append(round(now - self._they_done, 2))
            self._they_done = None
        if self._agent == "speaking" and state != "speaking":
            self._maya_done = now
        self._agent = state

    def on_user_state(self, state: str) -> None:
        now = time.monotonic()
        if state == "speaking":
            if self._agent == "speaking":
                self.they_interrupted += 1
            elif self._maya_done is not None:
                self.their_gaps.append(round(now - self._maya_done, 2))
            self._maya_done = None
        elif state == "listening":
            self._they_done = now

    def summary(self) -> dict:
        def avg(xs: list[float]) -> float | None:
            return round(sum(xs) / len(xs), 2) if xs else None

        return {
            "their_avg_response_s": avg(self.their_gaps),
            "their_response_s": self.their_gaps,
            "our_avg_response_s": avg(self.our_gaps),
            "our_response_s": self.our_gaps,
            "they_interrupted_maya": self.they_interrupted,
            "questions_asked": min(self.asked, len(QUESTIONS)),
        }


_REPORT_PROMPT = (
    "Below is a phone call between Maya (our voice agent) and a vendor's AI sales agent, plus timings "
    "we measured. \"Caller\" lines are the VENDOR's agent; \"Maya\" lines are ours. We want to learn "
    "from their agent to improve ours. Return JSON only:\n"
    '{"vendor": "company name", "answers": {"<topic>": "what the VENDOR said about its own product '
    '(services, channels, languages, integrations, tech, scale, pricing), or \\"not answered\\""}, '
    '"behaviour": ["how their agent handled interruptions, silence, mishearing, steering the call"], '
    '"timings": "one line comparing their response times with ours", '
    '"improve": {"services": ["..."], "code": ["..."], "architecture": ["..."]}}\n'
    "Base everything on the transcript and timings only; never invent facts.\n\n"
)


async def vendor_report(transcript: str, stats: VendorStats, model: str) -> str | None:
    """The post-call report as pretty JSON text (saved as the call summary), or None."""
    from tools import llm_client  # tools imports config; keep this module light

    if llm_client is None or not transcript.strip():
        return None
    timings = json.dumps(stats.summary())
    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: llm_client.models.generate_content(
                    model=model,
                    contents=f"{_REPORT_PROMPT}TIMINGS: {timings}\n\nTRANSCRIPT:\n{transcript}",
                    config={
                        "response_mime_type": "application/json",
                        "automatic_function_calling": {"disable": True},
                    },
                )
            ),
            timeout=30.0,
        )
        report = json.loads(res.text)
        report["measured"] = stats.summary()
        return json.dumps(report, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("Vendor report generation failed")
        return None
