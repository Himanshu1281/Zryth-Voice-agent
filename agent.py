"""inbound AI voice agent (LiveKit Agents worker).

Run it:
    python agent.py start          # register with LiveKit and take calls
    python agent.py console         # talk to Maya in your terminal (no phone)

Pipeline (cascade), tuned for ~700 ms-1.2 s perceived turn latency on one India VPS:
    Silero VAD -> Sarvam Saaras STT (codemix, 8 kHz) -> Google Gemini
    -> Sarvam Bulbul TTS.

Language flow: STT auto-detects English/Hindi on every utterance; each turn the
agent switches the reply language and TTS voice to match the caller
(BaseMayaAgent._switch_language).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# --- TLS / cert guard (optional) --------------------------------------------
# Only when the OS has no CA bundle (some minimal images) point at certifi's.
# Setting SSL_CERT_FILE unconditionally makes every new HTTP client re-read the
# bundle on the event loop (~200 ms each, seen as "event loop blocked" at call
# start, delaying the greeting).
try:
    import ssl

    _paths = ssl.get_default_verify_paths()
    if not (
        (_paths.cafile and os.path.exists(_paths.cafile))
        or (_paths.capath and os.path.isdir(_paths.capath))
    ):
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:  # pragma: no cover - purely defensive
    pass

from dotenv import load_dotenv

load_dotenv()

from livekit import agents
from livekit import api as lk_api
from livekit.agents import (
    Agent,
    TurnHandlingOptions,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RunContext,
    WorkerOptions,
    function_tool,
    metrics,
)
from livekit.agents import NOT_GIVEN
from livekit.agents.llm import FallbackAdapter
from livekit.plugins import openai, sarvam, silero

from config import (
    AUDIO_SAMPLE_RATE,
    BCP47,
    DEFAULT_LANGUAGE,
    MAX_ENDPOINTING_DELAY,
    MAX_TOKENS,
    MIN_ENDPOINTING_DELAY,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SARVAM_STT_MODEL,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_VOICE,
    SUPPORTED_LANGUAGES,
    VAD_MIN_SILENCE_S,
    VAD_ACTIVATION_THRESHOLD,
    USER_AWAY_TIMEOUT_S,
)
from prompts import (
    LANG_NAMES,
    STYLE_NOTES,
    build_instructions,
    greeting,
)
from routing import CallConfig, resolve_call_config
from tools import AppointmentTools, build_dynamic_tools, summarize_transcript

from database import (
    create_call,
    save_message,
    finish_call,
    save_call_summary,
    sync_knowledge_to_lancedb,
    fetch_dynamic_prompt,
    fetch_assigned_tools,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice-agent")

# Per-stage latency lands here as JSONL, one row per metric. The dashboard
# (dashboard/build_dashboard.py) reads this to colour-code each call.
METRICS_LOG = Path(__file__).parent / "logs" / "metrics.jsonl"

# Spoken when the LLM returns nothing, so the caller never hears dead air.
# Used instead when the empty reply follows a tool result (e.g. a rejected phone
# number): "say that once more" makes no sense to the caller there.
TOOL_REPLY_FALLBACK: dict[str, str] = {
    "en": "Sorry, I didn't get the full number. Could you tell me all ten digits, please?",
    "hi": "माफ़ कीजिए, पूरा नंबर नहीं मिला। क्या आप दसों अंक एक साथ बता सकते हैं?",
}
GENERIC_TOOL_FALLBACK: dict[str, str] = {
    "en": "Sorry, give me just a moment. Could you say that again?",
    "hi": "माफ़ कीजिए, एक पल। क्या आप फिर से बता सकते हैं?",
}

EMPTY_REPLY_FALLBACK: dict[str, str] = {
    "en": "Sorry, could you say that once more?",
    "hi": "माफ़ कीजिए, क्या आप एक बार फिर से बता सकते हैं?",
}

# Spoken only when the knowledge lookup is slow, so the caller never hears dead air.
# Not added to the chat history, so the LLM doesn't repeat or react to it.
FILLERS: dict[str, tuple[str, ...]] = {
    "en": ("Let me check.", "One moment.", "Sure, let me see."),
    "hi": ("एक सेकंड, देखती हूँ।", "जी, बस एक पल।", "ठीक है, देखती हूँ।"),
}
FILLER_DELAY_S = 1.2  # lookups (normally ~0.6 s) faster than this get no filler


# --- pipeline wiring ---------------------------------------------------------
def _build_llm():
    """Gemini only for now. Groq (primary, Gemini fallback) is commented out below."""
    gemini = openai.LLM(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=LLM_TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
    )
    return gemini
    # --- Groq disabled for now: re-enable to make Groq primary, Gemini fallback ---
    # if not GROQ_API_KEY:
    #     return gemini
    # groq = openai.LLM(
    #     model=GROQ_MODEL,
    #     api_key=GROQ_API_KEY,
    #     base_url="https://api.groq.com/openai/v1",
    #     temperature=LLM_TEMPERATURE,
    #     max_completion_tokens=MAX_TOKENS,
    #     reasoning_effort=GROQ_REASONING_EFFORT or NOT_GIVEN,
    #     # No SDK retries: a 429 (rate limit) must fail over to Gemini at once,
    #     # not sit in exponential backoff while the caller waits.
    #     max_retries=0,
    # )
    # # Groq normally starts in ~0.3 s; if it hasn't within 1.5 s, use Gemini.
    # return FallbackAdapter([groq, gemini], attempt_timeout=1.5, max_retry_per_llm=0)


def _build_session(
    language: str,
    job_ctx: JobContext,
    call_id: str,
    dynamic_tool_specs: list[dict],
    config: CallConfig,
) -> tuple["AgentSession", "AppointmentTools"]:

    """Wire the whole cascade for a starting language. Returns (session, tools_instance).
    The tools instance is returned separately so the caller can reset per-turn counters.
    """
    bcp47 = BCP47[language]
    tools_instance = AppointmentTools(job_ctx, call_id, config)
    
    tools_list = tools_instance.to_tools()
    tools_list.extend(build_dynamic_tools(dynamic_tool_specs))
    return AgentSession(
        vad=job_ctx.proc.userdata.get("vad") or silero.VAD.load(
            min_silence_duration=VAD_MIN_SILENCE_S,
            activation_threshold=VAD_ACTIVATION_THRESHOLD,
           #min_speech_duration=VAD_MIN_SPEECH_DURATION,
            # min_speech_duration removed: high activation_threshold (0.8) is sufficient
            # noise rejection for telephony; removing this saves ~300ms TTFT.
            sample_rate=AUDIO_SAMPLE_RATE,
        ),
        stt=sarvam.STT(
            model=SARVAM_STT_MODEL,       # saaras:v3
            mode="codemix",               # keep English words spoken mid-sentence
            language="unknown",           # auto-detect en/hi per utterance (no extra latency)
            sample_rate=AUDIO_SAMPLE_RATE,  # 8 kHz telephony
        ),
        llm=_build_llm(),
        user_away_timeout=USER_AWAY_TIMEOUT_S,
        tts=sarvam.TTS(
            model=SARVAM_TTS_MODEL,        # bulbul:v3
            target_language_code=bcp47,
            speaker=SARVAM_TTS_VOICE,      # Sirman
            pace=1.10,                     # 15% faster speech pace
            min_buffer_size=30,
            max_chunk_length=80,      
        ),
	tools=tools_list,
        turn_handling=TurnHandlingOptions(
            endpointing={
                "mode": "fixed",
                "min_delay": MIN_ENDPOINTING_DELAY,
                "max_delay": MAX_ENDPOINTING_DELAY,
            },
        ),  # 1.0 s
    ), tools_instance




# --- agents ------------------------------------------------------------------


def _with_caller(instructions: str, caller_phone: str | None) -> str:
    """Let Maya offer the number the caller is dialling from."""
    if not caller_phone:
        return instructions
    return (
        f"{instructions}\n\nCALLER: dialling from {caller_phone}. When you need contact "
        "details, ask whether the team should use this number; if yes, pass it as `phone`."
    )


def _compose_instructions(code: str, caller_phone: str | None, kb: "AppointmentTools | None") -> str:
    """Persona + language rules, then the whole KB (small-KB fast path), then the
    per-call caller line last so the long stable prefix stays cacheable."""
    cfg = kb.config if kb is not None else CallConfig()
    text = build_instructions(code, STYLE_NOTES[code], business=cfg.business_name, persona=cfg.persona_name)
    if kb is not None and kb.full_kb:
        text += f"\n\nRelevant {cfg.business_name} knowledge (the ONLY source of facts):\n" + kb.full_kb
    return _with_caller(text, caller_phone)


# Markdown the LLM may emit despite instructions: **bold**, *, #, `, and "- " bullets.
_MARKDOWN = re.compile(r"[*#`_]+|^\s*[-•]\s+", re.M)

# Caller is asking something (worth a "let me check" if the lookup is slow).
_QUESTION = re.compile(
    r"\?|\b(?:what|which|how|why|when|where|who|can you|could you|tell me|explain|do you|does)\b"
    r"|क्या|कैसे|कौन|कब|कहाँ|क्यों|बताइए|बताओ|बता",
    re.I,
)

# Statements that still need facts ("I want to know about the mill software").
_TOPIC = re.compile(
    r"\b(?:about|know|service|services|product|products|software|price|pricing|cost|offer|"
    r"build|create|make|develop|automate|custom|chatbot|app|website|AI|features?|company|contact|"
    r"whatsapp|email|address|located|clients?)\b"
    r"|बारे|सर्विस|प्रोडक्ट|सॉफ्टवेयर|कीमत|कंपनी|company|software",
    re.I,
)

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
# Other Indic scripts (Bengali through Malayalam blocks).
_FOREIGN_SCRIPT = re.compile(r"[ঀ-ൿ]")

# Short transcripts below this STT confidence are treated as noise ("कि आछे?" at 0.18).
MIN_TRANSCRIPT_CONFIDENCE = 0.3


def _detect_language(text: str) -> str | None:
    """'hi' / 'en' from the transcript's script, or None when too short or mixed
    to tell. Sarvam codemix writes Hindi in Devanagari and English in Latin, so
    "Zryth के बारे में बताओ" -> hi and "tell me about your products" -> en."""
    if len(text.split()) < 2:
        return None  # "ok", "haan" -- not enough to switch on
    dev, lat = len(_DEVANAGARI.findall(text)), len(_LATIN.findall(text))
    if dev + lat < 4:
        return None
    ratio = dev / (dev + lat)
    if ratio >= 0.5:
        return "hi"
    if ratio <= 0.15:
        return "en"
    return None


class BaseMayaAgent(Agent):
    """Base class: language switching + knowledge prefetch before every reply."""

    caller_phone: str | None = None
    kb: AppointmentTools | None = None
    code: str = "en"

    async def llm_node(self, chat_ctx, tools, model_settings):
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
        lang = _detect_language(last_user)
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
        # Last two replies identical = the model is stuck in a loop: nudge it out.
        replies = [
            (m.text_content or "").strip() for m in chat_ctx.items
            if getattr(m, "role", None) == "assistant" and m.text_content
        ]
        if len(replies) >= 2 and replies[-1] == replies[-2]:
            log.warning("LLM repeating itself (%r); adding a nudge", replies[-1])
            chat_ctx.add_message(
                role="system",
                content=(
                    f"Do NOT repeat your previous reply. Answer the caller's latest message directly in "
                    f"{LANG_NAMES[self.code]}; if it was unclear, ask them what they meant."
                ),
            )
        for attempt in range(2):
            produced = False
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                if isinstance(chunk, str):
                    produced = produced or bool(chunk.strip())
                elif getattr(chunk, "delta", None) is not None:
                    d = chunk.delta
                    produced = produced or bool((d.content or "").strip() or d.tool_calls)
                yield chunk
            if produced:
                return
            log.warning("LLM returned an empty reply (attempt %d)", attempt + 1)
        last = chat_ctx.items[-1] if chat_ctx.items else None
        if getattr(last, "type", None) == "function_call_output":
            phone_tool = getattr(last, "name", "") in ("capture_lead", "book_consultation")
            table = TOOL_REPLY_FALLBACK if phone_tool else GENERIC_TOOL_FALLBACK
        else:
            table = EMPTY_REPLY_FALLBACK
        yield table.get(self.code, table["en"])

    async def tts_node(self, text, model_settings):
        """Strip markdown (bullets, **bold**, #) the LLM sometimes emits, so the
        TTS never reads symbols aloud."""
        async def _clean(stream):
            async for chunk in stream:
                chunk = _MARKDOWN.sub("", chunk)
                if chunk:
                    yield chunk
        async for frame in Agent.default.tts_node(self, _clean(text), model_settings):
            yield frame

    async def on_user_turn_completed(
        self, turn_ctx: agents.llm.ChatContext, new_message: agents.llm.ChatMessage
    ) -> None:
        """Inject the agent's relevant knowledge so the LLM answers in ONE round trip."""
        text = new_message.text_content or ""
        # Drop near-silent noise ("Hmm", "Oh bro" at ~0.05 confidence) instead of answering it.
        conf = getattr(new_message, "transcript_confidence", None)
        if conf is not None and conf < MIN_TRANSCRIPT_CONFIDENCE and len(text.split()) <= 3:
            log.info("Ignoring low-confidence transcript %r (%.2f)", text, conf)
            raise agents.StopResponse()
        # Only English/Hindi are supported: text in another script (Bengali, Tamil...)
        # is STT misreading noise or a stray word, not something to answer.
        if _FOREIGN_SCRIPT.search(text) and not (_DEVANAGARI.search(text) or _LATIN.search(text)):
            log.info("Ignoring transcript in an unsupported script %r", text)
            raise agents.StopResponse()
        lang = _detect_language(text)
        if lang and lang != self.code:
            await self._switch_language(lang)
            turn_ctx.add_message(
                role="system", content=f"The caller is now speaking {LANG_NAMES[lang]}; reply in {LANG_NAMES[lang]}."
            )
        if self.kb is None:
            return
        if self.kb.full_kb:
            # Whole KB is already in the system prompt: nothing to wait for. Run
            # the vector lookup in the background only to log knowledge gaps.
            asyncio.create_task(self.kb.retrieve_for_turn(text))
            return
        # Only look things up when the caller asks about something. Names, "okay",
        # numbers and small talk get a plain conversational reply: no facts block
        # for the LLM to recite, and no lookup wait.
        asking = bool(_QUESTION.search(text) or _TOPIC.search(text))
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
                    f"Relevant {self.kb.config.business_name} knowledge (reference only; never read it out verbatim, "
                    "answer only what was asked in at most 2 short spoken sentences, no lists or markdown):\n"
                    + "\n\n".join(chunks)
                ),
            )

    async def _filler_after(self, delay: float) -> None:
        """Say a short "let me check" if the knowledge lookup is still running after `delay`."""
        await asyncio.sleep(delay)
        options = FILLERS.get(self.code, FILLERS["en"])
        self._filler_i = (getattr(self, "_filler_i", -1) + 1) % len(options)
        try:
            self.session.say(options[self._filler_i], add_to_chat_ctx=False)
        except Exception:
            log.exception("filler failed")

    async def _switch_language(self, code: str) -> None:
        """Reply language + TTS voice follow the caller; STT keeps auto-detecting."""
        self.code = code
        tts = self.session.tts
        if tts is not None and hasattr(tts, "update_options"):
            tts.update_options(target_language_code=BCP47[code])
        await self.update_instructions(_compose_instructions(code, self.caller_phone, self.kb))
        log.info("language -> %s", code)

    @function_tool
    async def set_language(self, context: RunContext, language: str):
        """Switch the reply language when the caller explicitly asks for one.

        Args:
            language: one of en, hi.
        """
        code = language.strip().lower()
        if code not in SUPPORTED_LANGUAGES:
            return f"Language '{language}' is not supported. Supported: {', '.join(SUPPORTED_LANGUAGES)}."
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
            instructions=_compose_instructions(language, caller_phone, kb)
        )
        self.caller_phone = caller_phone
        self.kb = kb
        self.code = language


# --- metrics -> JSONL --------------------------------------------------------
def _make_metrics_handler(call_id: str):
    """Build a metrics_collected handler that logs per-stage latency as JSONL."""

    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        m = ev.metrics
        # Pull the one latency number that matters per stage.
        if isinstance(m, metrics.EOUMetrics):
            row = {"stage": "eou", "seconds": m.end_of_utterance_delay, "speech_id": m.speech_id}
        elif isinstance(m, metrics.STTMetrics):
            row = {"stage": "stt", "seconds": m.duration}
        elif isinstance(m, metrics.LLMMetrics):
            row = {"stage": "llm_ttft", "seconds": m.ttft, "speech_id": m.speech_id}
        elif isinstance(m, metrics.TTSMetrics):
            row = {"stage": "tts_ttfb", "seconds": m.ttfb, "speech_id": m.speech_id}
        else:
            return

        row["call_id"] = call_id
        row["ts"] = time.time()
        try:
            METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with METRICS_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # never let metrics logging break a live call
            log.exception("failed to write metrics row")
        metrics.log_metrics(m)  # also to the console log

    return _on_metrics


# --- entrypoint --------------------------------------------------------------
def prewarm(proc) -> None:
    """Runs once per worker process before any call: load the VAD model and the
    OpenAI SDK here so the first call doesn't block the event loop (~0.6 s + ~1.8 s)."""
    import openai.resources  # noqa: F401  (lazy import otherwise happens mid-call)
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=VAD_MIN_SILENCE_S,
        activation_threshold=VAD_ACTIVATION_THRESHOLD,
        sample_rate=AUDIO_SAMPLE_RATE,
    )


async def entrypoint(ctx: JobContext) -> None:
    start_time = time.time()

    # Who is calling, and which of our numbers did they dial? The dialled number
    # decides which organization's agent answers (routing.resolve_call_config).
    await ctx.connect()
    phone = dialed = None
    try:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=3.0)
        phone = participant.attributes.get("sip.phoneNumber") or None
        dialed = participant.attributes.get("sip.trunkPhoneNumber") or None
    except Exception:
        log.warning("Could not read SIP participant attributes; continuing without them")
    if not phone:
        # Fallback: some dispatch setups put the caller number in the room name/metadata
        if ctx.room.metadata and "+" in ctx.room.metadata:
            phone = ctx.room.metadata
        elif "+" in ctx.room.name:
            phone = ctx.room.name

    config = await asyncio.to_thread(resolve_call_config, dialed)
    if config.blocked:
        # One of our numbers, but the customer hasn't connected it to an agent yet
        log.warning("Number %s is not connected to an agent; dropping call", config.dialed_number)
        try:
            await ctx.api.room.delete_room(lk_api.DeleteRoomRequest(room=ctx.room.name))
        finally:
            ctx.shutdown(reason="number not configured")
        return

    # Create one Supabase record for this call.
    # Fall back to a local UUID if Supabase is unavailable so the call continues.
    try:
        call_id = await asyncio.to_thread(
            create_call,
            livekit_room=ctx.room.name,
            phone=phone,
            language=config.language,
            org_id=config.org_id,
            org_agent_id=config.org_agent_id,
            dialed_number=config.dialed_number,
        )
    except Exception:
        import uuid
        call_id = str(uuid.uuid4())
        log.exception("Supabase create_call failed; using in-memory call_id=%s", call_id)

    log.info(
        "Supabase call created: %s for room %s (org=%s agent=%s)",
        call_id, ctx.room.name, config.org_id, config.org_agent_id,
    )

    default_greeting = config.greeting or greeting(config.language, config.business_name, config.persona_name)
    dynamic_greeting, dynamic_tool_specs = await asyncio.gather(
        # A template-level greeting_prompt applies only when the customer set no greeting
        asyncio.to_thread(fetch_dynamic_prompt, "greeting_prompt", default_greeting, config.template_id)
        if not config.greeting else asyncio.sleep(0, result=config.greeting),
        asyncio.to_thread(fetch_assigned_tools, config.template_id),
    )

    session, tools_instance = _build_session(
        config.language,
        ctx,
        call_id,
        dynamic_tool_specs,
        config,
    )
    await tools_instance.prepare_kb()
    session.on(
        "metrics_collected",
        _make_metrics_handler(ctx.room.name),
    )

    # Save customer and Maya messages.
    def on_conversation_item(event) -> None:
        item = event.item

        role = getattr(item, "role", None)
        text = getattr(item, "raw_text_content", None)

        if not role or not text or not text.strip():
            return

        if role == "user":
            speaker = "customer"
        elif role == "assistant":
            speaker = "maya"
        else:
            return

        async def _save():
            for attempt in range(3):
                try:
                    await asyncio.to_thread(save_message, call_id, speaker, text)
                    log.info("Saved %s message", speaker)
                    return
                except Exception:
                    if attempt == 2:
                        log.exception("Failed to save %s message after 3 attempts — dropping", speaker)
                    else:
                        await asyncio.sleep(2 ** attempt)  # 1s, 2s backoff
        
        asyncio.create_task(_save())

    session.on(
        "conversation_item_added",
        on_conversation_item,
    )

    # This runs when the LiveKit job shuts down.
    async def on_shutdown(reason: str = "") -> None:
        try:
            duration_seconds = int(time.time() - start_time)
            await asyncio.to_thread(finish_call, call_id, duration_seconds)
            log.info(
                "Supabase call finished: %s reason=%s",
                call_id,
                reason,
            )
        except Exception:
            log.exception("Failed to finish Supabase call")

        # Post-call summary / intent / outcome for the dashboard and KB review.
        lines = []
        for item in session.history.items:
            role = getattr(item, "role", None)
            text = getattr(item, "text_content", None)
            if role in ("user", "assistant") and text:
                lines.append(f"{'Caller' if role == 'user' else 'Maya'}: {text}")
        result = await summarize_transcript("\n".join(lines), LLM_MODEL)
        if result:
            try:
                await asyncio.to_thread(
                    save_call_summary,
                    call_id,
                    result.get("summary", ""),
                    result.get("intent"),
                    result.get("outcome"),
                )
            except Exception:
                log.exception("Failed to save call summary")

    ctx.add_shutdown_callback(on_shutdown)

    _GOODBYE_MARKERS = ("goodbye", "bye", "din shubh", "दिन शुभ", "have a great day")

    # Caller went silent: check in once, then end the call on a second timeout.
    _away_prompts = [0]

    def on_user_state_changed(ev) -> None:
        if ev.new_state != "away" or tools_instance.is_ending or tools_instance.is_transferring:
            return
        _away_prompts[0] += 1
        # Maya already said goodbye but the LLM skipped end_call: just hang up.
        last = next(
            (m.text_content or "" for m in reversed(session.history.items)
             if getattr(m, "role", None) == "assistant" and m.text_content),
            "",
        ).lower()
        if any(w in last for w in _GOODBYE_MARKERS):
            tools_instance.is_ending = True
            tools_instance.on_end_requested()
            asyncio.create_task(_hang_up())
            return
        if _away_prompts[0] == 1:
            session.generate_reply(
                instructions="The caller has gone quiet. Briefly and warmly ask if they are still there."
            )
        else:
            tools_instance.is_ending = True
            tools_instance.on_end_requested()
            session.generate_reply(
                instructions="The caller is still silent. Say a short, polite goodbye and invite them to call back."
            )

    session.on("user_state_changed", on_user_state_changed)
    
    # Hang up once end_call was requested AND the goodbye has finished playing
    # (speaking -> listening). A fallback timer covers a missing goodbye.
    _hangup = {"spoke_goodbye": False, "started": False}

    async def _hang_up() -> None:
        if _hangup["started"]:
            return
        _hangup["started"] = True
        # Close the session first so it stops streaming events into a room we're
        # about to delete (otherwise: "failed to send session event").
        try:
            await session.aclose()
        except Exception:
            log.exception("session.aclose failed during hang-up")
        try:
            await ctx.api.room.delete_room(lk_api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:
            log.exception("delete_room failed; shutting down job instead")
        ctx.shutdown(reason="call ended by agent")

    async def _hang_up_fallback() -> None:
        # Backstop only: normally we hang up the moment the goodbye finishes.
        # The fixed goodbye is ~4-4.5 s of audio (+ TTS start), so 7 s never cuts it off.
        await asyncio.sleep(7)
        await _hang_up()

    def on_agent_state_changed(ev) -> None:
        if not tools_instance.is_ending:
            return
        if ev.new_state == "speaking":
            _hangup["spoke_goodbye"] = True
        elif ev.new_state == "listening" and _hangup["spoke_goodbye"]:
            asyncio.create_task(_hang_up())

    session.on("agent_state_changed", on_agent_state_changed)
    tools_instance.on_end_requested = lambda: asyncio.create_task(_hang_up_fallback())
    tools_instance.caller_phone = phone

    await session.start(
        agent=GreeterAgent(caller_phone=phone, kb=tools_instance, language=config.language),
        room=ctx.room,
    )

    # The greeting is persisted by on_conversation_item like every other turn.
    await session.say(dynamic_greeting)


if __name__ == "__main__":

    import sys

    # Only commands that run the worker need the KB sync, heartbeat and status rows;
    # "download-files" (Docker build) and "--help" must work without Supabase.
    runs_worker = len(sys.argv) > 1 and sys.argv[1] in ("dev", "start", "console", "connect")
    if runs_worker:
        log.info("Performing initial LanceDB sync...")
        sync_knowledge_to_lancedb()

        from database import _init_supabase
        import socket

        # One status row per worker host, so several hosts don't overwrite each
        # other (the backend counts the platform online if any row is fresh).
        WORKER_ID = f"{os.getenv('AGENT_NAME', 'maya')}@{socket.gethostname()}"

        try:
            supabase = _init_supabase()
            supabase.table("agent_status").upsert({
                "agent_id": WORKER_ID,
                "status": "active",
                "last_heartbeat": datetime.now(timezone.utc).isoformat()
            }).execute()
            log.info("Agent status set to active in Supabase.")
        except Exception as e:
            log.error(f"Failed to set agent status to active: {e}")


        import threading
        from database import realtime_sync_loop

        def heartbeat_loop():
            while True:
                try:
                    sb = _init_supabase()
                    sb.table("agent_status").upsert({
                        "agent_id": WORKER_ID,
                        "status": "active",
                        "last_heartbeat": datetime.now(timezone.utc).isoformat()
                    }).execute()
                except Exception as e:
                    log.error(f"Heartbeat failed: {e}")
                time.sleep(10)

        def realtime_sync_worker():
            asyncio.run(realtime_sync_loop())

        t1 = threading.Thread(target=heartbeat_loop, daemon=True)
        t1.start()

        t2 = threading.Thread(target=realtime_sync_worker, daemon=True)
        t2.start()

    try:
        agents.cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm,
                agent_name=os.getenv("AGENT_NAME", "maya"),
            )
        )
    finally:
        if runs_worker:
            try:
                supabase = _init_supabase()
                supabase.table("agent_status").upsert({
                    "agent_id": WORKER_ID,
                    "status": "inactive"
                }).execute()
                log.info("Agent status set to inactive in Supabase.")
            except Exception as e:
                log.error(f"Failed to set agent status to inactive: {e}")
