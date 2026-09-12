"""AI voice agent (LiveKit Agents worker).

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

import json
import logging
import os
import time
import uuid
from pathlib import Path

# --- TLS / cert guard (optional) --------------------------------------------
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
from livekit.plugins import openai, sarvam, silero

from config import (
    AUDIO_SAMPLE_RATE,
    BCP47,
    MAX_ENDPOINTING_DELAY,
    MAX_TOKENS,
    MIN_ENDPOINTING_DELAY,
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
    STYLE_NOTES,
    build_instructions,
    build_persona,
)
from tools import AppointmentTools

from database import create_call, save_message, finish_call, get_agent_by_did, get_default_agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice-agent")

METRICS_LOG = Path(__file__).parent / "logs" / "metrics.jsonl"

def prewarm(proc: agents.JobProcess):
    """Run once per worker to load heavy models and persistent clients."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=VAD_MIN_SILENCE_S,
        activation_threshold=VAD_ACTIVATION_THRESHOLD,
        min_speech_duration=VAD_MIN_SPEECH_DURATION,
        sample_rate=AUDIO_SAMPLE_RATE,
    )
    proc.userdata["groq_client"] = __import__("openai").AsyncClient(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

def _build_session(
    language: str,
    job_ctx: JobContext,
    call_id: str,
    agent_id: str,
) -> AgentSession:
    bcp47 = BCP47[language]
    return AgentSession(
        vad=job_ctx.proc.userdata["vad"],
        stt=sarvam.STT(
            model=SARVAM_STT_MODEL,
            mode="codemix",
            language=bcp47,
            sample_rate=AUDIO_SAMPLE_RATE,
        ),
        llm=openai.LLM(
            model=LLM_MODEL,
            client=job_ctx.proc.userdata["groq_client"],
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
        ),
        tts=sarvam.TTS(
            model=SARVAM_TTS_MODEL,
            target_language_code=bcp47,
            speaker=SARVAM_TTS_VOICE,
            pace=1.10,
            min_buffer_size=30,
            max_chunk_length=80,      
        ),
        tools=AppointmentTools(job_ctx, call_id, agent_id).to_tools(),
        turn_handling=TurnHandlingOptions(
            endpointing={
                "mode": "fixed",
                "min_delay": MIN_ENDPOINTING_DELAY,
                "max_delay": MAX_ENDPOINTING_DELAY,
            },
        ),
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
            return f"Language '{language}' is not supported. Supported: {', '.join(SUPPORTED_LANGUAGES)}."

        context.session.update_agent(LangAgent(code, self.persona, self.agent_row))
        
        return f"Language swapped to {code}. You MUST now answer the user's question in {code} without mentioning the language swap."


class GreeterAgent(BaseMayaAgent):
    """Greets the caller in English and detects their language."""

    def __init__(self, persona: str, agent_row: dict) -> None:
        self.persona = persona
        self.agent_row = agent_row
        instructions = (
            persona
            + "\n\nCRITICAL: Do NOT ask the caller which language they prefer. "
            "Automatically detect their language based on their speech. If they speak "
            "Hindi, immediately call set_language('hi'). If they speak English, "
            "immediately call set_language('en'). Do not dive into property questions "
            "before the language is set. IMPORTANT: Do NOT use the search_knowledge tool "
            "until you have successfully called set_language."
        )
        super().__init__(instructions=instructions)


class LangAgent(BaseMayaAgent):
    """One agent parameterised by language code; STT + TTS locked to it."""

    def __init__(self, code: str, persona: str, agent_row: dict) -> None:
        self.code = code
        self.persona = persona
        self.agent_row = agent_row
        bcp47 = BCP47[code]
        super().__init__(
            instructions=build_instructions(code, STYLE_NOTES[code], persona),
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


def _make_metrics_handler(call_id: str):
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        m = ev.metrics
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
        except Exception:
            log.exception("failed to write metrics row")
        metrics.log_metrics(m)

    return _on_metrics


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    phone = None
    if ctx.room.metadata and "+" in ctx.room.metadata:
        phone = ctx.room.metadata
    elif "+" in ctx.room.name:
        phone = ctx.room.name
        
    start_time = time.time()

    # Determine DID from metadata or room name. 
    did = phone if phone else None
    
    if did:
        agent_row = await get_agent_by_did(did)
    else:
        # If testing via LiveKit Sandbox or Console without a phone number, 
        # fallback to any available live agent so testing still works!
        agent_row = await get_default_agent()
        log.info("No DID found in room. Falling back to default live agent: %s", 
                 agent_row["agent_name"] if agent_row else "None")

    if agent_row is None or agent_row["status"] != "live":
        await ctx.room.local_participant.publish_data(b"unavailable")
        ctx.shutdown(reason="no live agent for this number")
        return

    persona = build_persona(agent_row)
    call_id = str(uuid.uuid4())
    
    import asyncio
    call_task = asyncio.create_task(create_call(
        call_id=call_id,
        livekit_room=ctx.room.name,
        agent_id=agent_row["id"],
        phone=phone,
        language=agent_row["default_language"],
    ))

    session = _build_session(
        agent_row["default_language"],
        ctx,
        call_id,
        agent_row["id"],
    )
    session.on(
        "metrics_collected",
        _make_metrics_handler(ctx.room.name),
    )

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

        async def _do_save():
            try:
                await call_task
                await save_message(
                    call_id=call_id,
                    speaker=speaker,
                    message=text,
                )
            except Exception:
                log.exception("Failed to save conversation message")
                
        import asyncio
        asyncio.create_task(_do_save())

    session.on(
        "conversation_item_added",
        on_conversation_item,
    )

    async def on_shutdown(reason: str = "") -> None:
        try:
            duration_seconds = int(time.time() - start_time)
            await finish_call(call_id, duration_seconds)
        except Exception:
            log.exception("Failed to finish Supabase call")

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(
        agent=GreeterAgent(persona, agent_row),
        room=ctx.room,
    )

    greeting = GREETINGS.get(agent_row["default_language"], GREETINGS["en"]).format(
        company_name=agent_row["company_name"],
        agent_name=agent_row["agent_name"]
    )
    await session.say(greeting)

    try:
        await call_task
        await save_message(
            call_id=call_id,
            speaker="maya",
            message=greeting,
        )
    except Exception:
        log.exception("Failed to save opening greeting")


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=os.getenv("AGENT_NAME", "maya"),
        )
    )
