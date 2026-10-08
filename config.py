"""Central configuration for the voice-agent-starter-kit.

Everything the agent pipeline needs is read from environment variables here and
exposed as plain, typed module-level constants. Import from this module instead
of calling os.getenv() all over the codebase, so there is exactly one place that
maps env -> value and one place to change a default.

Defaults mirror `.env.example`. No secrets live in this file; missing secrets
simply fall back to obvious placeholders so the module still imports.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load a local .env if present. In production the values come from the systemd
# EnvironmentFile instead, so this is a no-op there.
load_dotenv(override=True)


# --- LiveKit -----------------------------------------------------------------
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://your-project.livekit.cloud")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "YOUR_LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "YOUR_LIVEKIT_API_SECRET")
SIP_URI = os.getenv("SIP_URI", "<your-project-id>.sip.livekit.cloud:5060")


# --- Sarvam AI (STT + TTS) ---------------------------------------------------
# The plugin reads SARVAM_API_KEY from the environment itself; we surface it here
# only so a missing key fails loudly and early rather than deep in the audio loop.
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "YOUR_SARVAM_API_KEY")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
# Sarvam's TTS constructor calls this the "speaker"; the env var is *_VOICE.
SARVAM_TTS_VOICE = os.getenv("SARVAM_TTS_VOICE", "roopa")


# --- Google Gemini (LLM) ------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# When GROQ_API_KEY is set, Groq is the primary LLM and Gemini (LLM_MODEL) the
# automatic fallback. Measured TTFT with the real prompt (India, 2026-10):
#   qwen/qwen3.8-27b, reasoning off   ~170-300 ms
#   gemini-2.5-flash-lite             ~1000-1600 ms
#   openai/gpt-oss-20b / 120b         reasoning models: slower and add odd unicode
# Groq's free tier allows only 8000 tokens/min (~4-8 turns/min in total): on a
# 429 we fail over to Gemini immediately (no retries). Use a paid Groq tier in
# production to stay on the fast path.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "none")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
# Safety cap only: the prompt keeps replies to 1-2 sentences. Devanagari costs ~3-4x the
# tokens of English, so 150 cut Hindi answers mid-sentence. Replies stream, so a higher
# cap does not add latency.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))


# --- Agent behaviour ---------------------------------------------------------
AGENT_NAME = os.getenv("AGENT_NAME", "maya")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
# Silero VAD minimum silence before end-of-utterance. Stored as ms in the env
# (matching .env.example) and converted to seconds for the plugin.
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "250"))
VAD_MIN_SILENCE_S = VAD_MIN_SILENCE_MS / 1000.0

# VAD thresholds to prevent background noise from interrupting Maya.
VAD_ACTIVATION_THRESHOLD = float(os.getenv("VAD_ACTIVATION_THRESHOLD", "0.8"))
VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.3"))

# Endpointing window (how long to wait for the caller to resume before treating
# the turn as finished). Too tight (<0.3 s) cuts callers off mid-sentence.
MIN_ENDPOINTING_DELAY = float(os.getenv("MIN_ENDPOINTING_DELAY", "0.4"))
MAX_ENDPOINTING_DELAY = float(os.getenv("MAX_ENDPOINTING_DELAY", "1.0"))

# Seconds of caller silence before Maya checks "are you still there?".
USER_AWAY_TIMEOUT_S = float(os.getenv("USER_AWAY_TIMEOUT_S", "20"))

# Telephony sample rate. 8 kHz end-to-end is the latency recipe for phone audio.
AUDIO_SAMPLE_RATE = 8000


# --- Telephony / transfer ----------------------------------------------------
DEFAULT_TRANSFER_NUMBER = os.getenv("DEFAULT_TRANSFER_NUMBER", "+918591194506")


# --- Language map ------------------------------------------------------------
# Every supported language -> its BCP-47 tag. Both Sarvam STT (`language`) and
# Sarvam TTS (`target_language_code`) are locked to one of these per call.
BCP47: dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
   
}
SUPPORTED_LANGUAGES: list[str] = list(BCP47.keys())
