"""Function tools AI solutions voice assistant."""

from __future__ import annotations

import json
import logging
import asyncio
import re
import time
from typing import Optional
import os

from google import genai
from livekit.agents import JobContext, RunContext, function_tool
from livekit.agents.beta.workflows import WarmTransferTask

from routing import DEFAULT_CONFIG, CallConfig
from database import log_knowledge_gap, update_call_lead
from dynamic_executor import execute_tool_call, is_executable

# Initialize the Gemini client for embeddings
if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
    llm_client = None
else:
    llm_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

log = logging.getLogger("voice-agent.tools")

_cached_db = None

# Knowledge prefetch tuning (see AppointmentTools.retrieve_for_turn).
# gemini-embedding-2 cosine scores on this KB: on-topic ~0.54-0.65, off-topic
# ~0.43-0.53 -- too close for a hard cutoff. So we always hand the LLM the top
# chunks above a low floor (the prompt forbids using anything they don't
# support) and only treat a weak best match as a knowledge gap.
_RELEVANCE_FLOOR = 0.40    # below this a chunk is pure noise
_GAP_THRESHOLD = 0.55      # best match below this = KB probably can't answer
_MIN_GAP_WORDS = 3         # don't log fragments as knowledge gaps
_RETRIEVAL_TIMEOUT_S = 1.5 # never hold the reply longer than this
# Pure small talk: no lookup needed.
_SMALL_TALK = re.compile(
    r"^(?:hi|hello|hey|hii|ok|okay|yes|yeah|yep|no|nope|thanks|thank you|bye|"
    r"hmm+|haan|ha|nahi|theek hai|accha|achha|namaste|हाँ|नहीं|ठीक है|अच्छा|नमस्ते)[\s.!?,]*$",
    re.I,
)

# STT mis-hearings of Zryth names -> spelling used in the knowledge base.
# Sarvam STT has no keyterm boosting, so we normalise before retrieval.
_NAME_ALIASES = [
    # Hindi STT of "Oswaal AI": "उस वाले ऐसे", "उस वाले आए", "एशो oil" (seen on live calls).
    (re.compile(r"(?:उस ?वाले?|ओ[सस्]+वाल|एशो)\s+(?:ए\s?आई|एआई|ऐसे|आए|ए|AI|oil)(?=[\s।.?!,]|$)", re.I), "Oswaal AI"),
    (re.compile(r"ओस्?वाल"), "Oswaal"),
    (re.compile(r"\b(?:oswal|oswall|osval|oswaal)\b", re.I), "Oswaal"),
    (re.compile(r"\b(?:zyrth|zrith|zerith|zareth|zarith|zarid|zerid|zaret)\b", re.I), "Zryth"),
]


_PRICING_Q = re.compile(
    r"\b(?:price|prices|pricing|cost|costs|charge|charges|fee|fees|rate|rates|quote|budget|"
    r"how much|kitna|kitne|keemat|daam|paisa|paise)\b|कीमत|दाम|कितना|कितने|शुल्क",
    re.I,
)


def normalise_names(text: str) -> str:
    for pattern, canonical in _NAME_ALIASES:
        text = pattern.sub(canonical, text)
    return text


_SEARCH_TOP_K = 3


_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _kb_filter(agent_id: str | None) -> str:
    """LanceDB where-clause for one agent's knowledge ('' = rows without an agent)."""
    if agent_id and not _UUID.match(agent_id):
        raise ValueError(f"bad agent_id {agent_id!r}")
    return f"agent_id = '{agent_id or ''}'"


def _vector_search(emb: list[float], agent_id: str | None = None) -> list[dict]:
    """Blocking LanceDB lookup, limited to one agent's knowledge base. Never re-syncs
    mid-call: the startup sync and the realtime_sync_loop keep the table fresh; a
    missing table just means no results."""
    global _cached_db
    import lancedb
    from database import LANCEDB_PATH

    if _cached_db is None:
        if not os.path.exists(LANCEDB_PATH):
            return []
        _cached_db = lancedb.connect(LANCEDB_PATH)

    if "knowledge" not in _cached_db.table_names():
        log.warning("'knowledge' table not found in LanceDB; returning no results")
        _cached_db = None  # reconnect next time, after the background sync lands
        return []

    table = _cached_db.open_table("knowledge")
    return (
        table.search(emb).metric("cosine")
        .where(_kb_filter(agent_id), prefilter=True)
        .limit(_SEARCH_TOP_K).to_list()
    )


# Small-KB fast path: while the whole knowledge base fits in this many chars
# (~6k tokens), it goes into the system prompt and no per-turn retrieval runs.
# That removes the ~600 ms Gemini embedding call from every turn (measured:
# Gemini TTFT 1.0 s full-KB vs 1.6 s embed+search+LLM). Beyond this size we
# switch to per-turn vector retrieval automatically.
KB_FULL_MAX_CHARS = int(os.getenv("KB_FULL_MAX_CHARS", "1500"))


def load_full_kb(agent_id: str | None = None) -> str | None:
    """One agent's whole KB from the local LanceDB copy (kept fresh by the realtime
    sync), or None if it is empty/missing or too large for the prompt."""
    import lancedb
    from database import LANCEDB_PATH

    try:
        if not os.path.exists(LANCEDB_PATH):
            return None
        db = lancedb.connect(LANCEDB_PATH)
        if "knowledge" not in db.table_names():
            return None
        table = db.open_table("knowledge")
        rows = (
            table.search().where(_kb_filter(agent_id), prefilter=True)
            .select(["id", "content"]).limit(10_000).to_list()
        )
    except Exception:
        log.exception("Could not load knowledge base for prompt; using retrieval")
        return None
    text = "\n\n".join(r["content"] for r in sorted(rows, key=lambda r: r["id"]))
    if not text or len(text) > KB_FULL_MAX_CHARS:
        return None
    return text


_search_cache = {}
_search_cache_lock = asyncio.Lock()

async def _cached_search(query: str, agent_id: str | None = None) -> tuple[tuple[str, ...], float]:
    """Return (top chunks above the relevance floor, best similarity score) from one agent's KB."""
    key = (agent_id or "", query)
    async with _search_cache_lock:
        if key in _search_cache:
            return _search_cache[key]

    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: llm_client.models.embed_content(
                    model='gemini-embedding-2',
                    contents=query,
                )
            ),
            timeout=_RETRIEVAL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log.warning("Gemini embedding timed out for query: %s", query)
        return tuple(), 0.0

    emb = res.embeddings[0].values

    # LanceDB calls are synchronous -- run them off the event loop so audio never stalls.
    results = await asyncio.to_thread(_vector_search, emb, agent_id)

    scored = [
        (1.0 - row["_distance"], row["content"])
        for row in results
        if row.get("_distance") is not None
    ]
    best = max((sc for sc, _ in scored), default=0.0)
    # Chunks are sentence-aligned and <= ~1000 chars; don't cut them mid-sentence.
    # dict.fromkeys drops duplicate rows (same chunk ingested twice), keeping order.
    result = (tuple(dict.fromkeys(c for sc, c in scored if sc >= _RELEVANCE_FLOOR)), best)
    
    async with _search_cache_lock:
        _search_cache[key] = result
        if len(_search_cache) > 256:
            # simple prune
            _search_cache.pop(next(iter(_search_cache)))
        
    return result


_SUMMARY_PROMPT = (
    "Summarise this phone call between Maya (Zryth's voice assistant) and a caller. "
    'Return JSON only: {"summary": "2-3 sentences", "intent": "short label, e.g. '
    'product_enquiry / pricing / custom_dev / support / other", "outcome": "one of '
    'lead_captured, consultation_booked, transferred, answered, unresolved, dropped"}.\n\n'
)


async def summarize_transcript(transcript: str, model: str) -> dict | None:
    """Post-call summary via Gemini. Returns None on any failure."""
    if llm_client is None or not transcript.strip():
        return None
    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: llm_client.models.generate_content(
                    model=model,
                    contents=_SUMMARY_PROMPT + transcript,
                    config={
                        "response_mime_type": "application/json",
                        "automatic_function_calling": {"disable": True},
                    },
                )
            ),
            timeout=20.0,
        )
        return json.loads(res.text)
    except Exception:
        log.exception("Call summary generation failed")
        return None


def _with_preferred_slot(
    requirement: Optional[str], date: Optional[str], time: Optional[str]
) -> Optional[str]:
    """Append the preferred slot only when the caller actually gave one."""
    slot = " ".join(p for p in (date, time) if p)
    if not slot:
        return requirement
    return f"{requirement} (Preferred: {slot})" if requirement else f"Preferred: {slot}"


def build_dynamic_tools(tool_specs: list[dict]) -> list:
    """Convert Supabase JSON tool specs into LiveKit tools using the dynamic executor."""
    dynamic_tools = []
    
    for spec in tool_specs:
        tool_name = spec.get("name")
        json_spec = spec.get("json_spec", {})
        instruction = spec.get("execution_instruction", "")
        
        if not tool_name:
            continue
        if not is_executable(json_spec):
            # e.g. an OpenAI-style {"type": "function"} spec: nothing to call
            log.warning("Skipping tool %r: json_spec has no https http.url_template", tool_name)
            continue

        def make_wrapper(name, spec, instr):
            async def _dynamic_wrapper(context: RunContext, raw_arguments: dict) -> str:
                return await asyncio.to_thread(
                    execute_tool_call, name, spec, instr, raw_arguments
                )
            return _dynamic_wrapper
            
        wrapper_func = make_wrapper(tool_name, json_spec, instruction)
            
        # LiveKit's function_tool supports `raw_schema` precisely for this!
        raw_schema = {
            "name": tool_name,
            # LLM-facing text: json_spec.description if set, else the executor instruction.
            "description": json_spec.get("description") or spec.get("description") or instruction,
            "parameters": json_spec.get("input_schema", {"type": "object", "properties": {}})
        }
        
        tool = function_tool(wrapper_func, raw_schema=raw_schema)
        dynamic_tools.append(tool)
        
    return dynamic_tools


def _clean_phone(phone: str | None) -> str | None:
    """Return a valid 10-digit Indian mobile number, or None.

    Accepts spaces, dashes, brackets and a +91 / 91 / 0 prefix
    ("+91 98765-43210" -> "9876543210"). Must start with 6-9.
    """
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None


_BAD_PHONE_MSG = (
    "That phone number is not a valid 10-digit mobile number. Read back what you heard "
    "and ask the caller to repeat their 10-digit number. Do not save until it is valid."
)


class AppointmentTools:
    """Tools the agent can use during a customer call."""

    def __init__(
        self,
        job_ctx: JobContext | None = None,
        call_id: str | None = None,
        config: CallConfig = DEFAULT_CONFIG,
    ) -> None:
        self.job_ctx = job_ctx
        self.call_id = call_id
        # Which org agent answers: business name, knowledge base, transfer number
        self.config = config
        self.is_ending = False
        self.is_transferring = False
        # Set by the entrypoint; arms the hang-up fallback timer.
        self.on_end_requested = None
        # Whole KB for the system prompt (small-KB fast path); None = retrieve per turn.
        self.full_kb: str | None = None

    async def prepare_kb(self) -> None:
        """Load the KB once per call (local disk, a few ms)."""
        self.full_kb = await asyncio.to_thread(load_full_kb, self.config.org_agent_id)
        log.info(
            "knowledge mode: %s",
            f"full KB in prompt ({len(self.full_kb)} chars)" if self.full_kb else "per-turn retrieval",
        )

    def to_tools(self) -> list:
        return [
            self.capture_lead,
            self.book_consultation,
            self.transfer_to_human,
            self.end_call,
        ]

    @function_tool
    async def capture_lead(
        self,
        context: RunContext,
        name: str,
        phone: str,
        requirement: Optional[str] = None,
    ) -> dict:
        """Save an interested caller's name and phone number.

        Use when the caller shows interest but has NOT asked to book a meeting
        (for a date/time or "book a call", use `book_consultation`).
        Ask the caller ONLY for their name and 10-digit phone number. Never ask
        for email, company, or anything else.

        Args:
            name: Caller's name.
            phone: Caller's 10-digit mobile number.
            requirement: What they want, summarised by you from the conversation
                (do NOT ask the caller for this).
        """

        if not name or not name.strip():
            return {
                "status": "failed",
                "message": "Name is required. Please ask the caller for their name before proceeding.",
            }
        clean = _clean_phone(phone)
        if not clean:
            return {"status": "failed", "message": _BAD_PHONE_MSG}

        if self.call_id:
            await asyncio.to_thread(
                update_call_lead,
                call_id=self.call_id,
                customer_name=name.strip(),
                phone=clean,
                requirement=requirement,
            )

        log.info("capture_lead -> %s", name)

        return {
            "status": "saved",
            "message": "The customer enquiry has been recorded successfully.",
        }

    async def _log_gap(self, query: str) -> None:
        """Fire-and-forget: record an unanswered question so the KB can be extended."""
        try:
            await asyncio.to_thread(log_knowledge_gap, self.call_id, query)
        except Exception:
            log.exception("Failed to log knowledge gap: %s", query)

    async def retrieve_for_turn(self, text: str) -> tuple[str, ...]:
        """Prefetch knowledge for the caller's finished utterance (RAG before the LLM).

        Runs in on_user_turn_completed so the answer needs ONE LLM round trip
        instead of tool-call -> search -> second LLM call (~1.3 s saved).
        Pure small talk ("hello", "okay") is skipped. Questions whose best match
        is weak are logged as knowledge gaps.
        """
        text = normalise_names((text or "").strip())
        if not text or _SMALL_TALK.match(text):
            return tuple()
        t0 = time.perf_counter()
        try:
            chunks, best = await asyncio.wait_for(_cached_search(text, self.config.org_agent_id), timeout=_RETRIEVAL_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("knowledge prefetch timed out for: %s", text)
            return tuple()
        except Exception:
            log.exception("knowledge prefetch failed for: %s", text)
            return tuple()
        log.info(
            "knowledge prefetch: %r -> %d chunks, best=%.3f, %.0f ms",
            text, len(chunks), best, (time.perf_counter() - t0) * 1000,
        )
        # Prices are deliberately not in the KB (team follows up), so they aren't gaps.
        if (
            best < _GAP_THRESHOLD
            and len(text.split()) >= _MIN_GAP_WORDS
            and not _PRICING_Q.search(text)
        ):
            asyncio.create_task(self._log_gap(text))
        return chunks

    @function_tool
    async def book_consultation(
        self,
        context: RunContext,
        name: str,
        phone: str,
        requirement: Optional[str] = None,
        preferred_date: Optional[str] = None,
        preferred_time: Optional[str] = None,
    ) -> dict:
        """Record a consultation request for the team.

        Use ONLY when the caller asks to schedule or book a meeting/call, or agrees
        when you offer one; otherwise use `capture_lead`.
        Ask the caller ONLY for their name, 10-digit phone number and (optionally)
        a preferred date/time. Never ask for email, company, or anything else.

        Args:
            name: Caller's name.
            phone: Caller's 10-digit mobile number.
            requirement: What they want, summarised by you from the conversation
                (do NOT ask the caller for this).
            preferred_date: Preferred date, if the caller gave one.
            preferred_time: Preferred time, if the caller gave one.
        """

        if not name or not name.strip():
            return {
                "status": "failed",
                "message": "Name is required. Please ask the caller for their name before proceeding.",
            }
        clean = _clean_phone(phone)
        if not clean:
            return {"status": "failed", "message": _BAD_PHONE_MSG}

        if self.call_id:
            await asyncio.to_thread(
                update_call_lead,
                call_id=self.call_id,
                customer_name=name.strip(),
                phone=clean,
                requirement=_with_preferred_slot(requirement, preferred_date, preferred_time),
            )

        log.info("book_consultation -> %s", name)

        return {
            "status": "requested",
            "message": (
                "The consultation request has been recorded. "
                f"The {self.config.business_name} team will follow up to confirm the appointment."
            ),
        }

    @function_tool
    async def transfer_to_human(
        self,
        context: RunContext,
    ) -> dict:
        """Transfer the caller to a human team member.

        Use when the caller asks to be transferred, to speak with a human or
        someone from the team, or the request requires human assistance.
        In the same reply, briefly tell the caller you're connecting them now
        and to please hold.
        """

        sip_trunk_id = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")
        sip_number = os.getenv("LIVEKIT_SIP_NUMBER")

        transfer_to = self.config.transfer_number
        if not sip_trunk_id or not transfer_to:
            log.error(
                "transfer_to_human unavailable (outbound trunk set=%s, transfer number set=%s)",
                bool(sip_trunk_id), bool(transfer_to),
            )
            return (
                "I'm sorry, I can't transfer calls right now. "
                f"The {self.config.business_name} team will call you back directly."
            )

        # Blocks the silence check / hang-up logic while the caller is on hold.
        self.is_transferring = True
        try:
            # Let the "connecting you now" line (spoken with this tool call) finish
            # before dialling, so the caller isn't left in silence.
            await context.wait_for_playout()

            result = await WarmTransferTask(
                sip_call_to=transfer_to,
                sip_trunk_id=sip_trunk_id,
                sip_number=sip_number,
                chat_ctx=context.session.history,
                ringing_timeout=30.0,
            )

            log.info("transfer_to_human -> success, agent: %s", result.human_agent_identity)
            # The team member is now in the caller's room; Maya must leave so she
            # doesn't talk over them. The room (and the call) stays up.
            if self.job_ctx is not None:
                self.job_ctx.shutdown(reason="transferred to human")
            return "Transfer completed. Stay silent; the team member is now on the line."

        except Exception:
            self.is_transferring = False
            log.exception(
                "Human transfer failed (trunk=%s, to=%s)",
                sip_trunk_id,
                transfer_to,
            )
            return (
                "I was unable to connect you to a team member right now. "
                "Please try again in a moment."
            )


    @function_tool
    async def end_call(
        self,
        context: RunContext,
    ) -> str:
        """Initiates the call termination sequence. Use when the conversation is finished."""

        log.info("end_call requested by Maya")

        if self.job_ctx is None:
            log.warning("Cannot end call: JobContext is not available")
            return ""
            
        if hasattr(context.session, "_closed") and context.session._closed:
            return ""

        if not self.is_ending:
            self.is_ending = True
            if self.on_end_requested:
                self.on_end_requested()
        return "Call will end after your goodbye. Say a brief, polite goodbye now."


if __name__ == "__main__":
    print("tools.py self-check passed")
    print("Zryth tools available:")
    print("- capture_lead")
    print("- book_consultation")
    print("- transfer_to_human")
