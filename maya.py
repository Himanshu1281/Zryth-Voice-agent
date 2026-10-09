"""Maya: the conversational agent for one call.

BaseMayaAgent wraps the LLM with the rules that keep a phone call on track:
language following, the name/number steps and booking starts decided in code
(contact_flow.py), ending the call, office hours, and the reply guards (language,
repeats, sentence cap). GreeterAgent is the agent every call starts with.
The worker, pipeline and entrypoint live in agent.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time

from livekit import agents
from livekit.agents import Agent, RunContext, function_tool

from business import gendered
from config import BCP47, SUPPORTED_LANGUAGES
from contact_flow import extract_digits, extract_name, name_from_history
from database import update_call_lead
from intents import (
    ACCEPT, ALREADY_TOLD, BOOKING_WORDS, CALLBACK_WORDS, CLEAR_GOODBYE, EMERGENCY, HOLD, GAVE_NAME, HUMAN, INTERESTED, NO, NOT_A_YES, OFFER, PROMISED,
    OFFICE_HOURS, QUESTION, SAME, TALK, TIME_OR_DATE, TOPIC, UNKNOWN_FACT, WANT, WHERE, YES,
)
from language import (
    DEVANAGARI, FOREIGN_MIN_CONFIDENCE, FOREIGN_SCRIPT, LANG_CHECK_LETTERS, LANGUAGE_REMINDER, LATIN,
    MIN_TRANSCRIPT_CONFIDENCE, RETRY_IN_LANGUAGE, UNSUPPORTED_LANGUAGE, detect_language, wrong_language,
)
from prompts import LANG_NAMES, STYLE_NOTES, build_instructions
from replies import (
    CALLBACK_OFFER, EMERGENCY_LINE, EMERGENCY_REPEAT, EMPTY_REPLY_FALLBACK, FILLERS, GENERIC_TOOL_FALLBACK, LENGTH_RULE, MARKDOWN,
    MAX_REPLY_SENTENCES, MIN_SENTENCE_WORDS, NO_TRANSFER, OFFICE_ADDRESS, OFFICE_HOURS_ASK, OFFICE_HOURS_SAVED,
    JARGON_HOLD_WORDS, META_ASIDE, META_LOOKAHEAD, SAY_MARKER, SENTENCE_END, TOOL_REPLY_FALLBACK, office_address,
    repeats_earlier, scrub_jargon,
)
from routing import CallConfig
from tools import AppointmentTools

log = logging.getLogger("voice-agent")

FILLER_DELAY_S = 1.2  # lookups (normally ~0.6 s) faster than this get no filler


def _with_caller(instructions: str, caller_phone: str | None) -> str:
    """The caller's number is deliberately NOT given to the LLM: it used to pass it
    (or one from the knowledge base) to the tools unasked. contact_flow reads it
    back to the caller itself."""
    return instructions


def compose_instructions(code: str, caller_phone: str | None, kb: "AppointmentTools | None") -> str:
    """Persona + language rules, then the whole KB (small-KB fast path), then the
    per-call caller line last so the long stable prefix stays cacheable."""
    cfg = kb.config if kb is not None else CallConfig()
    profile = kb.profile if kb is not None else None
    text = build_instructions(code, STYLE_NOTES[code], business=cfg.business_name, persona=cfg.persona_name,
                              profile=profile)
    business = profile.name if profile is not None else cfg.business_name
    if kb is not None and kb.full_kb:
        text += f"\n\nRelevant {business} knowledge (the ONLY source of facts):\n" + kb.full_kb
    return _with_caller(text, caller_phone)


# Live transfer to a person (tools.to_tools); off unless ENABLE_HUMAN_TRANSFER=1.
TRANSFER_ON = os.getenv("ENABLE_HUMAN_TRANSFER", "").strip().lower() in ("1", "true", "yes")


class _CodeReply(Exception):
    """A typed turn answered by code (see BaseMayaAgent._reply_by_code)."""

    def __init__(self, line: str | None, goodbye: bool = False) -> None:
        super().__init__(line)
        self.line, self.goodbye = line, goodbye


class BaseMayaAgent(Agent):
    """Base class: language switching + knowledge prefetch before every reply."""

    caller_phone: str | None = None
    kb: AppointmentTools | None = None
    code: str = "en"

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Cap every spoken reply at MAX_REPLY_SENTENCES: the prompt asks for 1-2
        sentences but Gemini still gave 4-5 sentence answers (15-18 s of audio).
        Text is cut at a sentence boundary; tool calls pass through untouched."""
        # A tool asked for an exact line (tools.speak): say it, no LLM call.
        last = chat_ctx.items[-1] if chat_ctx.items else None
        if getattr(last, "type", None) == "function_call_output" and str(last.output).startswith(SAY_MARKER):
            yield str(last.output)[len(SAY_MARKER):]
            return
        live = MAX_REPLY_SENTENCES - 1  # sentences spoken as they stream
        sentences = 0
        held = ""   # text after the last space: a "." there may be "FinanceAuditor.ai"
        tail = None  # once `live` sentences are out, the rest is buffered here
        muted = False  # the model started an instruction-like aside: drop the rest
        async for chunk in self._llm_node_inner(chat_ctx, tools, model_settings):
            if isinstance(chunk, str):
                text = chunk
            elif getattr(chunk, "delta", None) is not None and not chunk.delta.tool_calls:
                text = chunk.delta.content or ""
            else:
                yield chunk  # tool call (or anything else): never cut
                continue
            if muted:
                continue
            if tail is not None:
                tail += text
                if (m := META_ASIDE.search(tail)):
                    tail, muted = tail[: m.start()], True
                continue
            buf = scrub_jargon(held + text)
            if (m := META_ASIDE.search(buf)):
                # "...सकती।\n\n(If the caller is done, call `end_call`...)": Gemini copies
                # the "(...)" reminders we tag onto the caller's message. Never speak it.
                log.warning("dropped an instruction-like aside from the reply: %r", buf[m.start(): m.start() + 60])
                buf, muted = buf[: m.start()].rstrip(), True
            start = 0
            for m in SENTENCE_END.finditer(buf):
                piece, start = buf[start: m.end()], m.end()
                if len(piece.split()) < MIN_SENTENCE_WORDS:
                    continue  # "नमस्ते!" / "Got it." would push the real answer out
                sentences += 1
                if sentences >= live:
                    yield buf[: m.end()]
                    tail, held = buf[m.end():], ""
                    break
            if tail is not None:
                continue
            # Emit up to the last whitespace; hold the tail until we know whether
            # its "." / "?" ends a sentence (needs the following space).
            split = max(buf.rfind(" "), buf.rfind("\n")) + 1
            # Also hold the last few words: "The knowledge" + " base does not..." must
            # reach scrub_jargon together.
            spaces = [i for i, c in enumerate(buf) if c in " \n"]
            if len(spaces) >= JARGON_HOLD_WORDS:
                split = min(split, spaces[-JARGON_HOLD_WORDS] + 1)
            else:
                split = 0
            paren = buf.rfind("(")
            if paren != -1 and ")" not in buf[paren:] and len(buf) - paren < META_LOOKAHEAD:
                # An open "(" may be the start of an instruction-like aside: hold it
                # until we can tell (META_ASIDE needs the words after it).
                split = min(split, paren)
            if split > 0:
                yield buf[:split]
                held = buf[split:]
            else:
                held = buf
        if tail is None:
            if held:
                yield held
            return
        # One more sentence at most. If the model went on longer, keep its closing
        # question ("Would you like to know more?"), which moves the call forward.
        rest = [s for s in re.split(r"(?<=[.!?।])\s+", scrub_jargon(tail).strip()) if s.strip()]
        if len(rest) > 1:
            log.info("reply capped at %d sentences (dropped %d)", MAX_REPLY_SENTENCES, len(rest) - 1)
            last = rest[-1]
            rest = [last] if last.rstrip().endswith("?") else [rest[0]]
        if rest:
            yield " " + rest[0]

    async def _llm_node_inner(self, chat_ctx, tools, model_settings):
        """Pass the LLM stream through. Gemini intermittently (~5%) returns an
        empty reply: retry once silently, then fall back to a short line rather
        than leave the caller in dead air."""
        # Typed (console/text) input skips on_user_turn_completed, so re-check the
        # caller's language here too -- otherwise a Hindi->English switch is missed.
        last_user = next(
            (m.text_content or "" for m in reversed(chat_ctx.items)
             if getattr(m, "role", None) == "user" and m.text_content),
            "",
        )
        lang = detect_language(last_user)
        chat_ctx = chat_ctx.copy()
        # Replies the caller cut off are fragments ("I can help"); the model copies
        # them verbatim on later turns, so keep them out of what it sees.
        chat_ctx.items = [
            m for m in chat_ctx.items
            if not (getattr(m, "role", None) == "assistant" and getattr(m, "interrupted", False))
        ]
        if lang and lang != self.code:
            await self._switch_language(lang)
            chat_ctx.add_message(
                role="system", content=f"The caller is now speaking {LANG_NAMES[lang]}; reply in {LANG_NAMES[lang]}."
            )
        # Typed input (console) skips on_user_turn_completed, so the name/phone steps,
        # booking starts and goodbyes never ran there: the LLM "noted" names it never
        # saved. Run the same code for a caller message the hook hasn't seen.
        last_msg = next(
            (m for m in reversed(chat_ctx.items)
             if getattr(m, "role", None) == "user" and getattr(m, "type", None) == "message"),
            None,
        )
        if self.kb is not None and last_msg is not None and last_msg.id not in self._seen_user_ids():
            self._seen_user_ids().add(last_msg.id)
            await self._apply_profile()
            self._typed_turn = True
            try:
                await self._code_turn(chat_ctx, last_msg, last_msg.text_content or "")
            except _CodeReply as r:
                if r.goodbye:
                    self.kb.say_goodbye(self.session, self.code)
                elif r.line:
                    yield r.line
                return
            finally:
                self._typed_turn = False
        # Last two replies identical = the model is stuck in a loop: nudge it out.
        replies = [
            (m.text_content or "").strip() for m in chat_ctx.items
            if getattr(m, "role", None) == "assistant" and m.text_content
        ]
        # System messages don't work reliably with Gemini's API because all system
        # messages are merged before the conversation. We tag the caller's latest
        # message itself -- it is always the last thing the model reads before generating.
        nudges = []
        reminder = LANGUAGE_REMINDER.get(self.code, LANGUAGE_REMINDER["en"])
        if reminder:
            nudges.append(reminder)

        # Loop / repetition detection:
        if len(replies) >= 2 and any(r == replies[-1] for r in replies[:-1]):
            log.warning("LLM repeating itself (%r); adding anti-loop nudge", replies[-1])
            nudges.append("Do NOT repeat any previous question or sentence. Say something completely new.")

        # Caller declined or said "nahi" / "no":
        if NO.search(last_user) and len(last_user.split()) <= 4:
            nudges.append("The caller said no/declined. Do NOT ask the same question again. Say 'No problem!' and ask what questions they have, or offer " + (self.kb.profile.offer(self.code) if self.kb else "help") + ".")

        # Facts callers often ask about that aren't in the knowledge: the TRUTH prompt
        # rule alone didn't stop "रविवार को हमारा ऑफिस बंद रहता है".
        if UNKNOWN_FACT.search(last_user):
            nudges.append(
                "This question (timings, holidays, payment, EMI, GST, jobs) is NOT answered in the knowledge: "
                "do not guess; say the team will confirm."
            )
        # Last thing the model reads, so it keeps answers short (llm_node also cuts).
        nudges.append(LENGTH_RULE.get(self.code, LENGTH_RULE["en"]))
        nudge_text = " ".join(nudges)
        for i in range(len(chat_ctx.items) - 1, -1, -1):
            m = chat_ctx.items[i]
            if getattr(m, "role", None) == "user" and getattr(m, "type", None) == "message":
                chat_ctx.items[i] = m.model_copy(update={"content": [*m.content, f"\n\n({nudge_text})"]})
                break
        regens = 0  # wrong-language / repeated replies thrown away this turn (max 2)
        lang_regens = 0
        for attempt in range(3):
            produced = False
            # Third try after two wrong-language replies: hold the whole reply and
            # translate it if it's still wrong ("price बताइए ना" got English 3 times).
            translate_last = regens >= 2 and lang_regens >= 2
            # Hold the first few words back until we know the reply's language:
            # Gemini sometimes copies an earlier English answer to a Hindi question.
            pending: list = []
            head = ""
            checking = regens < 2
            wrong_lang = False
            repeated = False
            async with contextlib.aclosing(
                Agent.default.llm_node(self, chat_ctx, tools, model_settings)
            ) as stream:
                async for chunk in stream:
                    text, has_tool = "", False
                    if isinstance(chunk, str):
                        text = chunk
                    elif getattr(chunk, "delta", None) is not None:
                        text = chunk.delta.content or ""
                        has_tool = bool(chunk.delta.tool_calls)
                    produced = produced or bool(text.strip() or has_tool)
                    if translate_last and not has_tool:
                        pending.append(chunk)
                        head += text
                        continue
                    translate_last = False
                    if not checking or has_tool:
                        checking = False
                        for p in pending:
                            yield p
                        pending.clear()
                        yield chunk
                        continue
                    pending.append(chunk)
                    head += text
                    if len(LATIN.findall(head)) + len(DEVANAGARI.findall(head)) >= LANG_CHECK_LETTERS:
                        if wrong_language(head, self.code):
                            wrong_lang = True
                            break
                        if repeats_earlier(head, replies):
                            repeated = True
                            break
                        checking = False
                        for p in pending:
                            yield p
                        pending.clear()
            if (checking and not wrong_lang and not repeated and regens < 2
                    and len(LATIN.findall(head)) + len(DEVANAGARI.findall(head)) >= 10
                    and wrong_language(head, self.code)):
                # A short reply ("Sorry, this is Zryth.") ended before the 30-letter
                # check: still don't send a Hindi caller an English line.
                wrong_lang = True
            if repeated:
                regens += 1
                log.warning("LLM repeating an earlier reply (%r...); regenerating", head[:40])
                chat_ctx = chat_ctx.copy()
                chat_ctx.add_message(
                    role="user",
                    content=(
                        "(You already said that. I don't want to answer that question. Don't ask it again: "
                        f"say something new, in {LANG_NAMES[self.code]}, e.g. ask if I have any question about "
                        f"{self.kb.profile.name if self.kb else 'the company'}.)"
                    ),
                )
                continue
            if wrong_lang:
                # Up to two tries: one regeneration alone still came back in English
                # sometimes ("Got it. We can help you automate..." to a Hindi caller).
                regens += 1
                lang_regens += 1
                log.warning("LLM replied in the wrong language (%r...); regenerating in %s", head[:40], self.code)
                chat_ctx = chat_ctx.copy()
                chat_ctx.add_message(
                    role="user",
                    content=RETRY_IN_LANGUAGE.get(self.code, RETRY_IN_LANGUAGE["en"]),
                )
                continue
            if translate_last and head.strip() and wrong_language(head, self.code):
                log.warning("LLM still in the wrong language after 2 retries; translating to %s", self.code)
                translated = await self._translate(head, model_settings)
                if translated:
                    yield translated
                    return
            for p in pending:  # short reply that ended before the check
                yield p
            if produced:
                return
            log.warning("LLM returned an empty reply (attempt %d)", attempt + 1)
        last = chat_ctx.items[-1] if chat_ctx.items else None
        if getattr(last, "type", None) == "function_call_output":
            # A refused tool call ("Not started: ...") never asked for a number, so
            # "पूरा नंबर नहीं मिला" would make no sense there.
            phone_tool = (
                getattr(last, "name", "") in ("capture_lead", "book_consultation")
                and not str(getattr(last, "output", "")).startswith("Not started")
            )
            table = TOOL_REPLY_FALLBACK if phone_tool else GENERIC_TOOL_FALLBACK
        else:
            table = EMPTY_REPLY_FALLBACK
        yield table.get(self.code, table["en"])

    async def tts_node(self, text, model_settings):
        """Strip markdown (bullets, **bold**, #) the LLM sometimes emits, so the
        TTS never reads symbols aloud."""
        async def _clean(stream):
            async for chunk in stream:
                chunk = MARKDOWN.sub("", chunk)
                if chunk:
                    yield chunk
        async for frame in Agent.default.tts_node(self, _clean(text), model_settings):
            yield frame

    async def _apply_profile(self) -> None:
        """When the business profile finished loading in the background (first call
        after a KB change), rebuild the prompt with it. Waits at most 3 s for it."""
        if self.kb is None:
            return
        task = getattr(self.kb, "profile_task", None)
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), 3)
        if getattr(self, "_profile_used", None) is not self.kb.profile:
            self._profile_used = self.kb.profile
            await self.update_instructions(compose_instructions(self.code, self.caller_phone, self.kb))

    async def on_user_turn_completed(
        self, turn_ctx: agents.llm.ChatContext, new_message: agents.llm.ChatMessage
    ) -> None:
        """Inject the agent's relevant knowledge so the LLM answers in ONE round trip."""
        await self._apply_profile()
        text = new_message.text_content or ""
        self._seen_user_ids().add(new_message.id)  # llm_node must not handle it again
        if self.kb is not None and HOLD.search(text):
            self.kb.hold_requested_at = time.monotonic()  # "hold on": wait longer before checking in
        # Drop near-silent noise ("Hmm", "Oh bro" at ~0.05 confidence) instead of answering it.
        conf = getattr(new_message, "transcript_confidence", None)
        last_reply = next(
            (m.text_content or "" for m in reversed(turn_ctx.items)
             if getattr(m, "role", None) == "assistant" and m.text_content),
            "",
        )
        # Never noise: digits (part of a phone number), yes/no, or any answer to a
        # question Maya just asked ("haan", "Ravi" on a bad line).
        expected_answer = (
            "?" in last_reply[-80:]
            or (self.kb is not None and self.kb.contact.active)
            or bool(YES.search(text) or NO.search(text))
        )
        if (conf is not None and conf < MIN_TRANSCRIPT_CONFIDENCE and len(text.split()) <= 3
                and not re.search(r"\d", text) and not expected_answer):
            log.info("Ignoring low-confidence transcript %r (%.2f)", text, conf)
            raise agents.StopResponse()
        # Only English/Hindi are supported. Low-confidence text in another script is
        # STT misreading noise; a clear sentence is a caller who deserves an answer.
        if FOREIGN_SCRIPT.search(text) and not (DEVANAGARI.search(text) or LATIN.search(text)):
            if conf is not None and conf < FOREIGN_MIN_CONFIDENCE:
                log.info("Ignoring transcript in an unsupported script %r (%.2f)", text, conf)
                raise agents.StopResponse()
            log.info("Caller spoke an unsupported language: %r", text)
            self.session.say(gendered(UNSUPPORTED_LANGUAGE.get(self.code, UNSUPPORTED_LANGUAGE["en"]),
                                      self.kb.profile.gender if self.kb else "female"))
            raise agents.StopResponse()
        lang = detect_language(text)
        if lang and lang != self.code:
            await self._switch_language(lang)
            turn_ctx.add_message(
                role="system", content=f"The caller is now speaking {LANG_NAMES[lang]}; reply in {LANG_NAMES[lang]}."
            )
        if self.kb is None:
            return
        await self._code_turn(turn_ctx, new_message, text)
        if self.kb.full_kb:
            # Whole KB is already in the system prompt: nothing to wait for. Run
            # the vector lookup in the background only to log knowledge gaps.
            asyncio.create_task(self.kb.retrieve_for_turn(text))
            return
        # Only look things up when the caller asks about something. Names, "okay",
        # numbers and small talk get a plain conversational reply: no facts block
        # for the LLM to recite, and no lookup wait.
        asking = bool(QUESTION.search(text) or TOPIC.search(text))
        if not asking:
            return
        # Short follow-ups ("what's its price?") need the previous question for context.
        if len(text.split()) < 6:
            prev = [
                m.text_content for m in turn_ctx.items
                if getattr(m, "role", None) == "user" and m is not new_message and m.text_content
            ]
            if prev:
                text = f"{prev[-1]} {text}"
        filler = asyncio.create_task(self._filler_after(FILLER_DELAY_S))
        try:
            chunks = await self.kb.retrieve_for_turn(text)
        finally:
            filler.cancel()
        if chunks:
            # Must be "system": as "assistant", Gemini continues the message and reads
            # the raw chunks aloud ("[Zryth Company Profile] Zryth's AI products are...").
            turn_ctx.add_message(
                role="system",
                content=(
                    f"Relevant {self.kb.profile.name} knowledge (reference only; never read it out verbatim, "
                    "answer only what was asked in at most 2 short spoken sentences, no lists or markdown):\n"
                    + "\n\n".join(chunks)
                ),
            )

    async def _code_turn(
        self, turn_ctx: agents.llm.ChatContext, new_message: agents.llm.ChatMessage, text: str
    ) -> None:
        """Turns decided by code, not the LLM: the name/phone steps, hanging up once
        details are saved, and starting the steps on a clear request. Raises (via
        _reply_by_code) when code answers; returns when the LLM should reply."""
        last_reply = next(
            (m.text_content or "" for m in reversed(turn_ctx.items)
             if getattr(m, "role", None) == "assistant" and m.text_content),
            "",
        )
        # Count plain "nahi"/"no" replies in a row. After two, asking yet another
        # question just loops ("कोई बात नहीं! कोई सवाल हैं?" x5): wrap up instead.
        plain_no = bool(NO.search(text)) and len(text.split()) <= 3 and not YES.search(text)
        self._no_streak = (getattr(self, "_no_streak", 0) + 1) if plain_no else 0
        if self.kb.contact.active and CLEAR_GOODBYE.search(text) and not extract_digits(text):
            # "okay fine, bye" while we ask for a name/number: they're leaving.
            self.kb.contact.reset()
            self._reply_by_code(new_message, goodbye=True)
        number = self.kb.profile.emergency_number or "112"
        if EMERGENCY.search(text):
            # "सीने में दर्द है, साँस नहीं ले पा रहे": help first. No booking, no pitch.
            self.kb.contact.reset()
            self.kb.emergency = True
            self._reply_by_code(new_message, EMERGENCY_LINE.get(self.code, EMERGENCY_LINE["en"]).format(number=number))
        if (
            getattr(self.kb, "emergency", False) and len(text.split()) <= 4 and "?" not in text
            and not QUESTION.search(text) and not WHERE.search(text)
        ):
            # "अच्छा ठीक है" after the emergency line: Gemini asked "since when is the
            # pain?". Keep pointing them to the emergency number instead.
            self._reply_by_code(new_message, EMERGENCY_REPEAT.get(self.code, EMERGENCY_REPEAT["en"]).format(number=number))
        if self.kb.contact.active:
            await self._contact_turn(turn_ctx, new_message, text)
        elif OFFICE_HOURS.search(text) and not self.kb.profile.hours:
            # Timings / visits that the knowledge doesn't cover (Gemini invented "closed on
            # Sunday"): the team confirms them by phone, so take the details. When the KB
            # does list timings, the LLM answers from it instead.
            if self.kb.saved_contact is not None:
                self._reply_by_code(new_message, OFFICE_HOURS_SAVED.get(self.code, OFFICE_HOURS_SAVED["en"]))
            line = self.kb.contact.start(
                "capture_lead", f"Office hours / visit: {text.strip()}"[:300],
                getattr(self.kb, "known_name", None) or name_from_history(self._user_texts(turn_ctx, new_message)),
                self.code, self.kb.clean_caller(), opener=False,
            )
            # "आपका office कहाँ है? मैं मिलने आना चाहता हूँ": answer the where first.
            address = (self.kb.profile.address or office_address(self.kb.full_kb)) if WHERE.search(text) else None
            where = OFFICE_ADDRESS.get(self.code, OFFICE_ADDRESS["en"]).format(address=address) if address else ""
            self._reply_by_code(new_message, where + OFFICE_HOURS_ASK.get(self.code, OFFICE_HOURS_ASK["en"]) + line)
        elif self._no_streak >= 2:
            if self.kb.saved_contact is None and not self.kb.callback_offered:
                self.kb.callback_offered = True
                self._reply_by_code(new_message, CALLBACK_OFFER.get(self.code, CALLBACK_OFFER["en"]))
            self._reply_by_code(new_message, goodbye=True)
        elif self.kb.saved_contact is not None and (
            self.kb.caller_is_done(text, last_reply) or (CLEAR_GOODBYE.search(text) and "?" not in text)
        ):
            # Details saved and the caller is done ("नहीं, बस इतना ही"): hang up from
            # code. Gemini sometimes just says "ज़रूर" and never calls end_call.
            self._reply_by_code(new_message, goodbye=True)
        elif self.kb.saved_contact is None and (
            CLEAR_GOODBYE.search(text) or (plain_no and self.kb.callback_offered)
        ) and "?" not in text:
            # Caller is leaving with nothing saved: offer a callback once (not when they
            # are turning down an offer Maya just made), then hang up. Done in code: the
            # LLM's "caller said no" nudge made it ask "any other questions?" forever.
            wrong_number = re.search(r"wrong number|galti se|गलती से|ग़लती से|गलत नंबर", text, re.I)
            if not self.kb.callback_offered and not OFFER.search(last_reply) and not wrong_number:
                self.kb.callback_offered = True
                self._reply_by_code(new_message, CALLBACK_OFFER.get(self.code, CALLBACK_OFFER["en"]))
            self._reply_by_code(new_message, goodbye=True)
        elif self.kb.saved_contact is None:
            self._maybe_start_contact(turn_ctx, new_message, text)

    async def _translate(self, text: str, model_settings) -> str | None:
        """The reply in the caller's language (last resort of the wrong-language guard)."""
        ctx = agents.llm.ChatContext.empty()
        ctx.add_message(
            role="user",
            content=(
                f"Translate this phone reply into natural spoken {LANG_NAMES[self.code]}"
                + (" in Devanagari script (keep product names in English)" if self.code == "hi" else "")
                + f". Output only the translation.\n\n{text.strip()}"
            ),
        )
        out = ""
        try:
            async with contextlib.aclosing(Agent.default.llm_node(self, ctx, [], model_settings)) as stream:
                async for chunk in stream:
                    if isinstance(chunk, str):
                        out += chunk
                    elif getattr(chunk, "delta", None) is not None:
                        out += chunk.delta.content or ""
        except Exception:  # noqa: BLE001 - the original reply is better than silence
            log.exception("translation failed")
            return None
        out = out.strip()
        return out if out and not wrong_language(out, self.code) else None

    def _reply_by_code(
        self, new_message: agents.llm.ChatMessage, line: str | None = None, goodbye: bool = False
    ) -> None:
        """Answer this turn with a fixed line (or the goodbye) instead of the LLM.
        Spoken turns: keep the caller's message and stop the LLM. Typed turns (console)
        come through llm_node instead, which catches _CodeReply and outputs the line."""
        if getattr(self, "_typed_turn", False):
            raise _CodeReply(gendered(line, self.kb.profile.gender) if line else line, goodbye)
        self._keep_user_turn(new_message)
        if goodbye:
            self.kb.say_goodbye(self.session, self.code)
        else:
            self.session.say(gendered(line, self.kb.profile.gender))
        raise agents.StopResponse()

    @staticmethod
    def _user_texts(turn_ctx: agents.llm.ChatContext, new_message: agents.llm.ChatMessage) -> list[str]:
        """Caller messages so far, oldest first, including this turn's (once)."""
        texts = [
            m.text_content for m in turn_ctx.items
            if getattr(m, "role", None) == "user" and m.text_content and m.id != new_message.id
        ]
        return texts + [new_message.text_content or ""]

    def _seen_user_ids(self) -> set:
        """Caller messages already handled by on_user_turn_completed."""
        if not hasattr(self, "_seen_ids"):
            self._seen_ids: set = set()
        return self._seen_ids

    def _keep_user_turn(self, msg: agents.llm.ChatMessage) -> None:
        """Record the caller's words before a StopResponse. LiveKit drops the user
        message when on_user_turn_completed raises StopResponse, so without this the
        name/number turns were missing from the transcript (Supabase + summary), from
        what the LLM sees afterwards and from end_call's "what did the caller just say".
        Same calls LiveKit makes itself when it keeps a turn (agent_activity.py)."""
        self._chat_ctx.items.append(msg)
        self.session._conversation_item_added(msg)

    def _maybe_start_contact(
        self, turn_ctx: agents.llm.ChatContext, new_message: agents.llm.ChatMessage, text: str
    ) -> None:
        """Start the contact flow from code when the caller clearly wants a
        demo/consultation/callback, or says yes to Maya's offer of one. Gemini
        flash-lite often asks "what time suits you?" itself instead of calling
        book_consultation, which never gets the details saved."""
        last_reply = next(
            (m.text_content or "" for m in reversed(turn_ctx.items)
             if getattr(m, "role", None) == "assistant" and m.text_content),
            "",
        )
        own = self.kb.profile.booking_regex  # this business's words: "checkup", "site visit", "टेबल"
        books = lambda t: bool(BOOKING_WORDS.search(t) or (own and own.search(t)))  # noqa: E731
        asked = (
            bool(books(text) and WANT.search(text)) or bool(CALLBACK_WORDS.search(text))
            or bool(INTERESTED.search(text.strip()))
            # "मेरा नाम हिमांशु है" out of the blue: they want a callback.
            or (bool(GAVE_NAME.search(text)) and bool(extract_name(text)) and self.kb.saved_contact is None)
        )
        # "Okay thanks, bye" after an offer is a goodbye, not a yes.
        # "मैं कंसल्टेशन बुक कर रही हूँ" (no tool called, nothing saved) then "karo":
        # Gemini said "बुक कर दिया" without ever asking name or number.
        promised = bool(PROMISED.search(last_reply)) and self.kb.saved_contact is None
        accepted = (
            bool(OFFER.search(last_reply) or books(last_reply)) and ("?" in last_reply or promised)
            and len(text.split()) <= 8
            # "इसी नंबर पर" / "same number" to "should our team call you?" is a yes too.
            and bool(YES.search(text) or ACCEPT.search(text) or SAME.search(text))
            and not NO.search(text)
            and not NOT_A_YES.search(text)
        )
        # Asking for a person while live transfer is off: offer a callback instead.
        wants_human = bool(HUMAN.search(text) and TALK.search(text)) and not TRANSFER_ON
        if not (asked or accepted or wants_human):
            return
        # Demo/consultation -> booking; a callback ("call me back", or yes to "should our
        # team call you back?") -> lead.
        if wants_human:
            booking = False
        elif books(text):
            booking = True
        elif CALLBACK_WORDS.search(text):
            booking = False
        else:
            booking = books(last_reply) or bool(PROMISED.search(last_reply))
        tool = "book_consultation" if booking else "capture_lead"
        recent = [
            m.text_content for m in turn_ctx.items
            if getattr(m, "role", None) == "user" and m.text_content and m.id != new_message.id
        ][-2:] + [text]
        requirement = " | ".join(recent)[:300] or None
        if wants_human:
            requirement = f"Wanted to talk to a person | {requirement}"
        if TIME_OR_DATE.search(text):
            # "call me tomorrow morning": keep the slot they asked for.
            requirement = f"{requirement} (Preferred: {text.strip()})"
        known_name = (
            getattr(self.kb, "known_name", None) or self.kb.contact.name
            or name_from_history(self._user_texts(turn_ctx, new_message))
        )
        log.info("%s started from code (asked=%s, accepted offer=%s, human=%s, name=%s)", tool, asked, accepted, wants_human, known_name)
        line = self.kb.contact.start(
            tool, requirement, known_name, self.code, self.kb.clean_caller(), opener=not wants_human
        )
        if wants_human:
            line = NO_TRANSFER.get(self.code, NO_TRANSFER["en"]) + line
        self._reply_by_code(new_message, line)

    async def _contact_turn(
        self, turn_ctx: agents.llm.ChatContext, new_message: agents.llm.ChatMessage, text: str
    ) -> None:
        """One caller turn while taking name + number. Speaks the flow's line and
        stops the LLM, or (caller asked something else) lets the LLM answer and re-ask."""
        kb = self.kb
        caller = kb.clean_caller()
        if kb.contact.stage == "name" and not kb.contact.name and ALREADY_TOLD.search(text):
            # "abhi tho bola": they gave it to the LLM earlier; find it instead of re-asking.
            kb.contact.name = kb.known_name or name_from_history(self._user_texts(turn_ctx, new_message))
        step = kb.contact.handle(
            text, self.code, caller, is_question=bool(QUESTION.search(text)) and not re.search(r"\d", text)
        )
        if kb.contact.name:
            kb.known_name = kb.contact.name
            if kb.call_id:
                asyncio.create_task(
                    asyncio.to_thread(
                        update_call_lead,
                        call_id=kb.call_id,
                        customer_name=kb.contact.name,
                        phone=kb.contact.phone or caller or None,
                        requirement=kb.contact.requirement,
                    )
                )
        if step.save:
            line = await kb.finish_contact(self.code)
        else:
            line = step.say
        if line:
            self._reply_by_code(new_message, line)
        if step.reprompt:
            turn_ctx.add_message(
                role="system",
                content=(
                    "You are taking the caller's contact details. Answer what they just said in one short "
                    f"sentence, then ask exactly this: \"{step.reprompt}\""
                ),
            )

    async def _filler_after(self, delay: float) -> None:
        """Say a short "let me check" if the knowledge lookup is still running after `delay`."""
        await asyncio.sleep(delay)
        options = FILLERS.get(self.code, FILLERS["en"])
        self._filler_i = (getattr(self, "_filler_i", -1) + 1) % len(options)
        try:
            self.session.say(gendered(options[self._filler_i], self.kb.profile.gender if self.kb else "female"),
                             add_to_chat_ctx=False)
        except Exception:
            log.exception("filler failed")

    async def _switch_language(self, code: str) -> None:
        """Reply language + TTS voice follow the caller; STT keeps auto-detecting."""
        self.code = code
        tts = self.session.tts
        if tts is not None and hasattr(tts, "update_options"):
            tts.update_options(target_language_code=BCP47[code])
        await self.update_instructions(compose_instructions(code, self.caller_phone, self.kb))
        log.info("language -> %s", code)

    @function_tool
    async def set_language(self, context: RunContext, language: str):
        """Switch the reply language ONLY when the caller explicitly asks to speak a
        language ("speak in Hindi", "Tamil mein baat karo"). Never call it otherwise:
        replying in the caller's own language already happens automatically.
        Also call it when they ask for any other language (Tamil, Bengali...), so
        they get told which languages are available.

        Args:
            language: language code the caller asked for: en, hi, or any other (e.g. ta, bn).
        """
        code = language.strip().lower()
        if code not in SUPPORTED_LANGUAGES:
            return (
                f"'{language}' is not available. Politely tell the caller, in the language they are "
                "using now, that you can help in English or Hindi only, and ask which they prefer."
            )
        if code != self.code:
            await self._switch_language(code)
        return {"status": "switched", "language": code}


class GreeterAgent(BaseMayaAgent):
    """Greets the caller in English and detects their language."""

    def __init__(
        self, caller_phone: str | None = None, kb: AppointmentTools | None = None,
        language: str = "en",
    ) -> None:
        # Full per-language prompt (incl. call-ending + tool-result rules) --
        # most calls never leave the greeter.
        super().__init__(
            instructions=compose_instructions(language, caller_phone, kb)
        )
        self.caller_phone = caller_phone
        self.kb = kb
        self.code = language
