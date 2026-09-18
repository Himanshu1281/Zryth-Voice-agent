"""inbound AI voice agent (LiveKit Agents worker).

Run it:
    python agent.py start          # register with LiveKit and take calls
    python agent.py console         # talk to Maya in your terminal (no phone)

Pipeline (cascade), tuned for ~700 ms-1.2 s perceived turn latency on one India VPS:
    Silero VAD -> Sarvam Saaras STT (codemix, 8 kHz) -> Google Gemini
    -> Sarvam Bulbul TTS.

Language flow: a GreeterAgent greets in English and detects the caller's
language, then hands off to a per-language LangAgent that locks STT + TTS to
that language. See GreeterAgent.set_language for the (important) silent-swap fix.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

# --- TLS / cert guard (optional) --------------------------------------------
# On some minimal VPS images the system CA bundle is missing and Sarvam/Google
# TLS handshakes fail. Pointing at certifi's bundle avoids that. Best-effort.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:  # pragma: no cover - purely defensive
    pass

from dotenv import load_dotenv

load_dotenv()

from livekit import agents
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
from livekit.agents.llm import FallbackAdapter
from livekit.plugins import google, openai, sarvam, silero

from config import (
    AGENT_NAME,
    AUDIO_SAMPLE_RATE,
    BCP47,
    DEFAULT_LANGUAGE,
    MAX_ENDPOINTING_DELAY,
    MAX_TOKENS,
    MIN_ENDPOINTING_DELAY,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SARVAM_STT_MODEL,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_VOICE,
    SUPPORTED_LANGUAGES,
    VAD_MIN_SILENCE_S,
    VAD_ACTIVATION_THRESHOLD,
    VAD_MIN_SPEECH_DURATION,
)
from prompts import (
    GREETINGS,
    HOT_PERSONA,
    STYLE_NOTES,
    build_instructions,
)
from tools import AppointmentTools

from database import (
    create_call, 
    save_message, 
    finish_call, 
    sync_knowledge_to_lancedb, 
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice-agent")

# Per-stage latency lands here as JSONL, one row per metric. The dashboard
# (dashboard/build_dashboard.py) reads this to colour-code each call.
METRICS_LOG = Path(__file__).parent / "logs" / "metrics.jsonl"

# Confirmation phrase spoken the moment we switch language, IN that language.
# These MUST exist for every supported language -- see set_language for why.
CONFIRMATIONS: dict[str, str] = {
    "en": "Perfect, we'll continue in English.",
    "hi": "ठीक है, अब हम हिंदी में बात करेंगे।",
}


# --- pipeline wiring ---------------------------------------------------------
def _build_session(
    language: str,
    job_ctx: JobContext,
    call_id: str,
) -> AgentSession:

    """Wire the whole cascade for a starting language and return the session.
    Silero VAD + Sarvam STT/TTS locked to `language`, Google Gemini LLM, Maya's tools,
    and tight endpointing. The session owns the tools and default pipeline;
    each LangAgent later overrides only STT + TTS to relock the language.
    """
    bcp47 = BCP47[language]
    return AgentSession(
        vad=silero.VAD.load(
            min_silence_duration=VAD_MIN_SILENCE_S,
            activation_threshold=VAD_ACTIVATION_THRESHOLD,
            min_speech_duration=VAD_MIN_SPEECH_DURATION,
            sample_rate=AUDIO_SAMPLE_RATE,
        ),
        stt=sarvam.STT(
            model=SARVAM_STT_MODEL,       # saaras:v3
            mode="codemix",               # keep English words spoken mid-sentence
            language=bcp47,
            sample_rate=AUDIO_SAMPLE_RATE,  # 8 kHz telephony
        ),
        llm=FallbackAdapter([
            openai.LLM(
                model=LLM_MODEL,
                api_key=GOOGLE_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                temperature=LLM_TEMPERATURE,
            )
            # openai.LLM(
            #     model="openai/gpt-oss-20b",
            #     api_key=GROQ_API_KEY,
            #     base_url="https://api.groq.com/openai/v1",
            #     temperature=LLM_TEMPERATURE,
            #     max_completion_tokens=MAX_TOKENS,
            # ),
            # openai.LLM(
            #     model="openai/gpt-oss-safeguard-20b",
            #     api_key=GROQ_API_KEY,
            #     base_url="https://api.groq.com/openai/v1",
            #     temperature=LLM_TEMPERATURE,
            #     max_completion_tokens=MAX_TOKENS,
            # )
        ]),
        tts=sarvam.TTS(
            model=SARVAM_TTS_MODEL,        # bulbul:v3
            target_language_code=bcp47,
            speaker=SARVAM_TTS_VOICE,      # Sirman
            pace=1.10,                     # 15% faster speech pace
            min_buffer_size=30,
            max_chunk_length=80,      
        ),
	tools=AppointmentTools(job_ctx, call_id).to_tools(),
        turn_handling=TurnHandlingOptions(
            endpointing={
                "mode": "fixed",
                "min_delay": MIN_ENDPOINTING_DELAY,
                "max_delay": MAX_ENDPOINTING_DELAY,
            },
        ),  # 1.0 s
    )


# --- agents ------------------------------------------------------------------
GREETER_INSTRUCTIONS = (
    HOT_PERSONA
    + "\n\nYou are speaking with the caller in English. "
    "If the caller starts speaking to you in a different language, or explicitly asks to change language, "
    "immediately call the set_language tool with the language code (en, hi)."
)


class BaseMayaAgent(Agent):
    """Base class providing language switching capabilities."""
    
    @function_tool
    async def set_language(self, context: RunContext, language: str) -> None:
        """Switch the whole conversation to the caller's preferred language.

        Args:
            language: one of en, hi.
        """     
        code = language.strip().lower()
        if code not in SUPPORTED_LANGUAGES:
            # Don't swap on an unknown code -- tell the LLM so it can re-ask.
            return f"Language '{language}' is not supported. Supported: {', '.join(SUPPORTED_LANGUAGES)}."

        # Swap to the new language agent so TTS uses the native language model
        context.session.update_agent(LangAgent(code))
        
        # Return instructions to the LLM so it generates an answer to the user's question
        return f"Language successfully switched to {code}. You MUST now answer the user's previous question naturally in {code} without saying 'language switched' or acknowledging the switch. CRITICAL: You MUST respond with spoken text now. DO NOT call any further tools."


class GreeterAgent(BaseMayaAgent):
    """Greets the caller in English and detects their language."""

    def __init__(self) -> None:
        super().__init__(instructions=GREETER_INSTRUCTIONS)


class LangAgent(BaseMayaAgent):
    """One agent parameterised by language code; STT + TTS locked to it."""

    def __init__(self, code: str) -> None:
        bcp47 = BCP47[code]
        super().__init__(
            instructions=build_instructions(code, STYLE_NOTES[code]),
            # Override the session's STT/TTS to relock the language for this agent.
            stt=sarvam.STT(
                model=SARVAM_STT_MODEL,
                mode="codemix",
                language=bcp47,
                sample_rate=AUDIO_SAMPLE_RATE,
            ),
            tts=sarvam.TTS(
                model=SARVAM_TTS_MODEL,
                target_language_code=bcp47,
                speaker=SARVAM_TTS_VOICE,
                pace=1.10,
		min_buffer_size=30,
    		max_chunk_length=80,
            ),
        )
        self.code = code


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
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # Try to extract phone from room name or metadata
    phone = None
    if ctx.room.metadata and "+" in ctx.room.metadata:
        phone = ctx.room.metadata
    elif "+" in ctx.room.name:
        phone = ctx.room.name
        
    start_time = time.time()

    # Create one Supabase record for this call.
    call_id = await asyncio.to_thread(
        create_call,
        livekit_room=ctx.room.name,
        phone=phone,
        language=DEFAULT_LANGUAGE,
    )

    log.info(
        "Supabase call created: %s for room %s",
        call_id,
        ctx.room.name,
    )

    session = _build_session(
        DEFAULT_LANGUAGE,
        ctx,
        call_id,
    )
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
            try:
                await asyncio.to_thread(save_message, call_id, speaker, text)
                log.info("Saved %s message", speaker)
            except Exception:
                # Never allow database issues to break the live call.
                log.exception("Failed to save conversation message")
        
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

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(
        agent=GreeterAgent(),
        room=ctx.room,
    )

    # Save Maya's opening greeting.
    try:
        save_message(
            call_id=call_id,
            speaker="maya",
            message=GREETINGS[DEFAULT_LANGUAGE],
        )
    except Exception:
        log.exception("Failed to save opening greeting")

    await session.say(
        GREETINGS[DEFAULT_LANGUAGE]
    )


if __name__ == "__main__":

    log.info("Performing initial LanceDB sync...")
    sync_knowledge_to_lancedb()

    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("AGENT_NAME", "maya"),
        )
    )
