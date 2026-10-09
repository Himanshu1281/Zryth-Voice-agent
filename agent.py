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
    WorkerOptions,
    metrics,
)
from livekit.agents.llm import FallbackAdapter
from livekit.plugins import google, openai, sarvam, silero

from config import (
    AUDIO_SAMPLE_RATE,
    BCP47,
    DEFAULT_LANGUAGE,
    MAX_ENDPOINTING_DELAY,
    MAX_TOKENS,
    MIN_ENDPOINTING_DELAY,
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SARVAM_STT_MODEL,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_VOICE,
    VAD_MIN_SILENCE_S,
    VAD_ACTIVATION_THRESHOLD,
    USER_AWAY_TIMEOUT_S,
)
from prompts import greeting
from contact_flow import clean_phone as _clean_phone
from maya import GreeterAgent
from replies import NUMBER_NOT_ACTIVE
from routing import CallConfig, resolve_call_config
from tools import AppointmentTools, build_dynamic_tools, summarize_transcript

from database import (
    create_call,
    save_message,
    update_call_lead,
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


# Extra silence allowed after the caller says "hold on" before Maya checks in.
HOLD_GRACE_S = 40


# --- pipeline wiring ---------------------------------------------------------
def _build_llm():
    """Gemini via the native Google plugin, with Gemini's OpenAI-compatible endpoint
    as a fallback. (The OpenAI-compatible stream sometimes sends a chunk with no
    `id`, which fails the whole turn: "ChatChunk id Input should be a valid string".)
    Groq (primary, Gemini fallback) is commented out below."""
    gemini_native = google.LLM(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )
    gemini = openai.LLM(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=LLM_TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
    )
    # attempt_timeout is passed to Google as the request deadline: it must be >= 10 s.
    return FallbackAdapter([gemini_native, gemini], attempt_timeout=10.0, max_retry_per_llm=0)
    # --- Groq disabled for now (re-enabling needs: from livekit.agents import NOT_GIVEN;
    # from config import GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT): re-enable to make Groq primary, Gemini fallback ---
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
        else:
            m = re.search(r"(\+?\d{10,12})", ctx.room.name or "")
            if m:
                phone = m.group(1)

    config = await asyncio.to_thread(resolve_call_config, dialed)
    if config.blocked:
        # One of our numbers, but the customer hasn't connected it to an agent yet.
        # Tell the caller instead of cutting the line silently.
        log.warning("Number %s is not connected to an agent; dropping call", config.dialed_number)
        try:
            notice = AgentSession(tts=sarvam.TTS(
                model=SARVAM_TTS_MODEL, target_language_code=BCP47[DEFAULT_LANGUAGE],
                speaker=SARVAM_TTS_VOICE, pace=1.10,
            ))
            await notice.start(agent=Agent(instructions=""), room=ctx.room)
            await asyncio.wait_for(
                notice.say(NUMBER_NOT_ACTIVE.get(DEFAULT_LANGUAGE, NUMBER_NOT_ACTIVE["en"]),
                           allow_interruptions=False),
                timeout=15,
            )
            await notice.aclose()
        except Exception:
            log.exception("Could not play the inactive-number message")
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

        # Fallback for dashboard: ensure lead details are recorded
        final_name = getattr(tools_instance, "known_name", None) or tools_instance.contact.name
        final_phone = tools_instance.contact.phone or _clean_phone(tools_instance.caller_phone)
        if final_name or final_phone:
            try:
                await asyncio.to_thread(
                    update_call_lead,
                    call_id=call_id,
                    customer_name=final_name or "Caller",
                    phone=final_phone,
                    requirement=tools_instance.contact.requirement,
                )
            except Exception:
                log.exception("Failed to update final call lead on shutdown")

    ctx.add_shutdown_callback(on_shutdown)

    _GOODBYE_MARKERS = ("goodbye", "bye", "din shubh", "दिन शुभ", "have a great day")

    # Caller went silent: check in once, then end the call on a second timeout.
    # The count restarts whenever the caller speaks, so every silence gets a check-in.
    _away_prompts = [0]
    _hold_wait: list[asyncio.Task | None] = [None]

    async def _after_hold() -> None:
        # Caller asked us to wait: give them HOLD_GRACE_S more before the usual check-in.
        await asyncio.sleep(HOLD_GRACE_S)
        if session.user_state == "away":
            _on_silence()

    def on_user_state_changed(ev) -> None:
        if ev.new_state == "speaking":
            _away_prompts[0] = 0
            if _hold_wait[0]:
                _hold_wait[0].cancel()
                _hold_wait[0] = None
            return
        if ev.new_state != "away" or tools_instance.is_ending or tools_instance.is_transferring:
            return
        held = tools_instance.hold_requested_at
        if held is not None and time.monotonic() - held < HOLD_GRACE_S + 30:
            tools_instance.hold_requested_at = None
            _hold_wait[0] = asyncio.create_task(_after_hold())
            return
        _on_silence()

    def _on_silence() -> None:
        if tools_instance.is_ending or tools_instance.is_transferring:
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
