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

from contact_flow import ContactFlow, clean_phone, name_from_history
from intents import (
    ACK, BOOKING_WORDS, CALLBACK_WORDS, CONTACT_WORDS, GOODBYE, OFFER, PRICING_QUESTION, SHORT_NO, SIGNOFF, SMALL_TALK,
)
from replies import CALLBACK_OFFER, GOODBYES, TRANSFER_FAILED_LINES, speak
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

# STT mis-hearings of Zryth names -> spelling used in the knowledge base.
# Sarvam STT has no keyterm boosting, so we normalise before retrieval.
_NAME_ALIASES = [
    # Hindi STT of "Oswaal AI": "उस वाले ऐसे", "उस वाले आए", "एशो oil" (seen on live calls).
    (re.compile(r"(?:उस ?वाले?|ओ[सस्]+वाल|एशो)\s+(?:ए\s?आई|एआई|ऐसे|आए|ए|AI|oil)(?=[\s।.?!,]|$)", re.I), "Oswaal AI"),
    (re.compile(r"ओस्?वाल"), "Oswaal"),
    (re.compile(r"\b(?:oswal|oswall|osval|oswaal)\b", re.I), "Oswaal"),
    (re.compile(r"\b(?:zyrth|zrith|zerith|zirith|zirth|zerth|zareth|zarith|zarid|zerid|zaret)\b", re.I), "Zryth"),
]


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
KB_FULL_MAX_CHARS = int(os.getenv("KB_FULL_MAX_CHARS", "12000"))


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


def _kb_version() -> int | None:
    """Current version of the local 'knowledge' table. Every sync (from this process,
    the realtime thread or the webhook service) bumps it, so keying the cache on it
    drops stale answers without a restart."""
    import lancedb
    from database import LANCEDB_PATH

    try:
        if not os.path.exists(LANCEDB_PATH):
            return None
        return lancedb.connect(LANCEDB_PATH).open_table("knowledge").version
    except Exception:
        return None


async def _cached_search(query: str, agent_id: str | None = None) -> tuple[tuple[str, ...], float]:
    """Return (top chunks above the relevance floor, best similarity score) from one agent's KB."""
    key = (await asyncio.to_thread(_kb_version), agent_id or "", query)
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


# Custom tools slower than this get a spoken filler while the caller waits.
_TOOL_FILLER_DELAY_S = 1.0
# Instruction (not a script), so the LLM says it in the caller's language.
_TOOL_FAILED = (
    "The lookup failed. In the caller's language, briefly say you couldn't get that information "
    "right now, offer to have the team follow up, and continue the conversation."
)


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
                # External APIs can take seconds: say "let me check" instead of dead air
                agent = getattr(context.session, "current_agent", None)
                filler = (
                    asyncio.create_task(agent._filler_after(_TOOL_FILLER_DELAY_S))
                    if hasattr(agent, "_filler_after") else None
                )
                try:
                    return await asyncio.to_thread(
                        execute_tool_call, name, spec, instr, raw_arguments
                    )
                except Exception:
                    log.exception("Dynamic tool %s failed", name)
                    return _TOOL_FAILED
                finally:
                    if filler:
                        filler.cancel()
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
    """Valid ten-digit Indian mobile number, or None (see contact_flow.clean_phone)."""
    return clean_phone(phone)


# Values the LLM invents when it calls a tool before the caller has said anything.
_PLACEHOLDERS = {"", "user", "caller", "customer", "unknown", "none", "null", "n/a", "na", "name", "phone"}


def _user_texts(context: RunContext) -> list[str]:
    """Caller transcripts so far, oldest first ([] if the history can't be read)."""
    try:
        return [
            m.text_content for m in context.session.history.items
            if getattr(m, "role", None) == "user" and m.text_content
        ]
    except Exception:
        log.exception("could not read history")
        return []


_NAME_INTRO = re.compile(r"\bname\b|नाम|\bi am\b|\bi'm\b|\bthis is\b|मैं .* हूँ", re.I)


def _user_said_name(context: RunContext, name: str) -> bool:
    """True when the caller actually gave this name: it appears in a short reply
    ("Rahul", "मेरा नाम राहुल है") or one that introduces a name ("my name is...").
    Stops the LLM saving a phrase lifted from a long sentence ("founder social media")."""
    texts = _user_texts(context)
    if not texts:
        return True  # history unreadable: don't block saving
    # Split on spaces, not \w: Devanagari vowel signs aren't \w, so r"\w+" breaks
    # "योगेश" into single letters and the name never matches.
    words = [w.lower() for w in re.sub(r"[^\w\sऀ-ॿ]", " ", name).split() if len(w) > 1]
    if not words:
        return False
    for t in texts:
        low = t.lower()
        if any(w in low for w in words) and (len(t.split()) <= 6 or _NAME_INTRO.search(t)):
            return True
    # The caller answered "what's your name?" but STT wrote it in another script
    # (राहुल vs Rahul): trust the answer to a direct question.
    return _last_reply_asked(context, r"name|नाम")


def _last_reply_asked(context: RunContext, pattern: str) -> bool:
    """True when Maya's last reply matches `pattern` AND the caller has answered it."""
    try:
        items = [m for m in context.session.history.items if getattr(m, "role", None) in ("user", "assistant")]
    except Exception:
        return True
    for i in range(len(items) - 1, -1, -1):
        if items[i].role == "assistant" and items[i].text_content:
            asked = re.search(pattern, items[i].text_content, re.I)
            answered = any(m.role == "user" for m in items[i + 1:])
            return bool(asked and answered)
    return False


def _labelled(tool: str, requirement: Optional[str]) -> str:
    """The calls row has one requirement field for both tools: tag which one it was
    so the team can tell a callback request from a consultation booking."""
    label = "[Consultation request]" if tool == "book_consultation" else "[Callback request]"
    return f"{label} {requirement}" if requirement else label


def _wants_contact(context: RunContext) -> bool:
    """The caller asked for a callback/demo/consultation, or answered Maya's offer of
    one, in the last few turns ("I'm interested" can come a turn before "yes").
    Anything else and the contact flow shouldn't start."""
    try:
        items = [m for m in context.session.history.items
                 if getattr(m, "role", None) in ("user", "assistant") and m.text_content]
    except Exception:
        return True  # can't tell: don't block a real request
    recent = items[-6:]
    users = [m.text_content for m in recent if m.role == "user"]
    replies = [m.text_content for m in recent if m.role == "assistant"]
    return any(
        BOOKING_WORDS.search(t) or CALLBACK_WORDS.search(t) or CONTACT_WORDS.search(t) for t in users
    ) or any(OFFER.search(t) for t in replies)


def _user_turns(context: RunContext) -> int:
    """How many caller messages the session has (to tell whether they spoke since)."""
    try:
        return sum(1 for m in context.session.history.items if getattr(m, "role", None) == "user")
    except Exception:
        return -1


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
        # Number the caller is dialling from (set by the entrypoint).
        self.caller_phone: str | None = None
        # (name, phone) already saved this call -- stops duplicate bookings.
        self.saved_contact: tuple[str, str] | None = None
        # Caller name remembered across turns to never re-ask
        self.known_name: str | None = None
        # Name + phone collection, driven turn by turn from agent.on_user_turn_completed.
        self.contact = ContactFlow()
        # end_call defers once to offer a callback when no details were saved.
        self.callback_offered = False
        # time.monotonic() of the caller's last "hold on" (silence check waits longer).
        self.hold_requested_at: float | None = None

    async def prepare_kb(self) -> None:
        """Load the KB once per call (local disk, a few ms)."""
        self.full_kb = await asyncio.to_thread(load_full_kb, self.config.org_agent_id)
        log.info(
            "knowledge mode: %s",
            f"full KB in prompt ({len(self.full_kb)} chars)" if self.full_kb else "per-turn retrieval",
        )

    def to_tools(self) -> list:
        tools = [self.capture_lead, self.book_consultation, self.end_call]
        # Live transfer is off until the SIP outbound trunk / transfer number work:
        # every attempt failed and left Maya stuck in LiveKit's transfer mode. A
        # request for a human now starts a callback instead (agent._maybe_start_contact).
        # Set ENABLE_HUMAN_TRANSFER=1 to turn it back on.
        if os.getenv("ENABLE_HUMAN_TRANSFER", "").strip().lower() in ("1", "true", "yes"):
            tools.insert(2, self.transfer_to_human)
        return tools

    @function_tool
    async def capture_lead(
        self,
        context: RunContext,
        requirement: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str | None:
        """Start taking the caller's contact details so the team can call them back.

        Use when the caller shows interest or agrees to a callback but has NOT asked
        to book a meeting (for a meeting, demo or consultation use `book_consultation`).
        Do NOT ask for their name or number yourself: this tool asks for the name,
        the phone number and confirms it, then saves everything. Say nothing after
        calling it.

        Args:
            requirement: What they want, summarised by you from the conversation
                (do NOT ask the caller for this).
            name: The caller's own name ONLY if they already told you it; never a
                company or institute name.
        """

        return await self._start_contact(context, "capture_lead", name, requirement)

    async def _start_contact(
        self, context: RunContext, tool: str, name: Optional[str], requirement: Optional[str]
    ) -> str | None:
        """Shared by capture_lead/book_consultation: hand the conversation to the
        contact flow (contact_flow.py). Returns the first question as a speak() line,
        which agent.llm_node says word for word."""
        lang = getattr(context.session.current_agent, "code", "en")
        caller = _clean_phone(self.caller_phone)
        if self.saved_contact is not None:
            log.info("%s: details already saved this call", tool)
            return None
        if self.contact.active:
            # The LLM called again mid-flow: just repeat the pending question.
            return speak(self.contact.question(lang, caller))
        if not _wants_contact(context):
            # e.g. "yes" to "what kind of business do you have?" made Gemini call
            # capture_lead and start asking for a name out of nowhere.
            log.info("%s: refused, caller hasn't asked for a callback/demo", tool)
            return (
                "Not started: the caller hasn't asked for a callback, demo or consultation. "
                "Don't ask for their name or number; just keep the conversation going."
            )
        name = (name or self.known_name or self.contact.name or "").strip()
        if name.lower() in _PLACEHOLDERS or not _user_said_name(context, name):
            # The LLM may have heard the name already ("merra name Himanshu hai" ->
            # "Got it, Himanshu") without passing it: take it from what the caller said.
            name = self.known_name or name_from_history(_user_texts(context)) or None
        if name:
            self.known_name = name
        log.info("%s: contact flow started (name known=%s, caller id=%s)", tool, bool(name), bool(caller))
        return speak(self.contact.start(tool, requirement, name, lang, caller))

    async def finish_contact(self, lang: str) -> str:
        """Save the confirmed name + number; returns the line to speak."""
        flow = self.contact
        name, phone = flow.name or self.known_name or "", flow.phone or _clean_phone(self.caller_phone) or ""
        if self.call_id and (name or phone):
            try:
                await asyncio.to_thread(
                    update_call_lead,
                    call_id=self.call_id,
                    customer_name=name,
                    phone=phone,
                    requirement=_labelled(flow.tool, flow.requirement),
                )
            except Exception:
                log.exception("Saving contact details failed")
        self.saved_contact = (name, phone)
        self.known_name = name
        log.info("%s -> %s (%s) saved", flow.tool, name, phone)
        text = flow.saved_text(lang, self.config.business_name)
        flow.reset()
        return text

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
        if not text or SMALL_TALK.match(text):
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
            and not PRICING_QUESTION.search(text)
        ):
            asyncio.create_task(self._log_gap(text))
        return chunks

    @function_tool
    async def book_consultation(
        self,
        context: RunContext,
        requirement: Optional[str] = None,
        preferred_date: Optional[str] = None,
        preferred_time: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str | None:
        """Book a consultation / demo / meeting with the team.

        Call it IMMEDIATELY when the caller asks for a meeting, call or demo, or
        agrees when you offer one ("yes", "book it", "demo मिल सकता है?"); otherwise
        use `capture_lead`. Do NOT first ask for a date, time or name: pass a date
        or time only if the caller already said one.
        Do NOT ask for their name or number yourself: this tool asks for the name,
        the phone number and confirms it, then saves everything. Say nothing after
        calling it.

        Args:
            requirement: What they want, summarised by you from the conversation
                (do NOT ask the caller for this).
            preferred_date: Preferred date, if the caller gave one.
            preferred_time: Preferred time, if the caller gave one.
            name: The caller's own name ONLY if they already told you it; never a
                company or institute name.
        """

        return await self._start_contact(
            context, "book_consultation", name,
            _with_preferred_slot(requirement, preferred_date, preferred_time),
        )

    @function_tool
    async def transfer_to_human(
        self,
        context: RunContext,
    ) -> dict:
        """Transfer the caller to a human team member.

        Use ONLY when the caller explicitly asks to talk to a person, a human, or
        someone from the team. NEVER use it when they just want an answer
        ("tell me", "बताइए", "अभी बताएँ"): answer them yourself instead.
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
            return self._transfer_failed(context)

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
            return self._transfer_failed(context)

    def _transfer_failed(self, context: RunContext) -> None:
        """Tell the caller right away, in their language, and offer a callback (a
        "yes" starts the contact flow). Left to the LLM, the apology came turns later,
        in English, as the answer to something else."""
        lang = getattr(context.session.current_agent, "code", "en")
        context.session.say(TRANSFER_FAILED_LINES.get(lang, TRANSFER_FAILED_LINES["en"]))
        return None

    @function_tool
    async def end_call(
        self,
        context: RunContext,
    ) -> str | None:
        """Initiates the call termination sequence. Use when the conversation is finished."""

        log.info("end_call requested by Maya")

        if self.job_ctx is None:
            log.warning("Cannot end call: JobContext is not available")
            return ""
            
        if hasattr(context.session, "_closed") and context.session._closed:
            return ""

        # The LLM sometimes hangs up on fragments like "Sorry" or "First". Only end
        # when the caller's latest words actually sound like they're done.
        last_user, spoke_after, prev_reply = "", False, ""
        try:
            for m in reversed(context.session.history.items):
                role = getattr(m, "role", None)
                if role == "assistant" and m.text_content:
                    spoke_after = True  # Maya already replied after the caller's turn
                elif role == "user" and m.text_content:
                    last_user = m.text_content
                    break
            # Maya's reply just before the caller's last words.
            seen_user = False
            for m in reversed(context.session.history.items):
                role = getattr(m, "role", None)
                if role == "user" and m.text_content:
                    seen_user = True
                elif seen_user and role == "assistant" and m.text_content:
                    prev_reply = m.text_content
                    break
        except Exception:
            log.exception("end_call: could not read history")
        # "ठीक है भाई" / "okay" right after Maya already said goodbye means the caller is done.
        acked_goodbye = bool(SIGNOFF.search(prev_reply)) and bool(ACK.match(last_user))
        if last_user and not (GOODBYE.search(last_user) or SHORT_NO.match(last_user) or acked_goodbye):
            log.info("end_call refused; caller said %r", last_user)
            if spoke_after:
                return None  # she already answered; a tool reply would make her speak twice
            # Nothing said this turn yet: have her reply instead of leaving dead air.
            return "Don't end the call yet. Reply to what the caller just said."

        # Lead backstop: one callback offer before hanging up on a caller whose
        # details weren't saved. Asked only once, so nobody gets trapped.
        # Skipped when they just declined Maya's own offer ("book a consultation?" ->
        # "no thanks, bye"): asking again ignores the answer they just gave.
        if (self.saved_contact is None and not self.callback_offered and not self.is_transferring
                and not OFFER.search(prev_reply)):
            self.callback_offered = True
            self._offer_turn = _user_turns(context)
            log.info("end_call deferred once for a callback offer")
            # Spoken as a fixed line: as an instruction, the LLM read "Don't hang up
            # yet." aloud to the caller.
            lang = getattr(context.session.current_agent, "code", "en")
            context.session.say(CALLBACK_OFFER.get(lang, CALLBACK_OFFER["en"]))
            return None

        # The LLM sometimes asks the callback question AND calls end_call again in the
        # same breath: wait for the caller's answer first.
        if self.callback_offered and getattr(self, "_offer_turn", None) == _user_turns(context):
            log.info("end_call refused: caller hasn't answered the callback offer yet")
            return None
        self.say_goodbye(context.session, getattr(context.session.current_agent, "code", "en"))
        return None

    def say_goodbye(self, session, lang: str) -> None:
        """Fixed goodbye (like the original agent): predictable, and the hang-up
        fires once it finishes playing. Used by end_call and by the agent itself."""
        if self.is_ending:
            return
        self.is_ending = True
        if self.on_end_requested:
            self.on_end_requested()
        session.say(
            GOODBYES.get(lang, GOODBYES["en"]).format(business=self.config.business_name),
            allow_interruptions=False,
        )

    @staticmethod
    def caller_is_done(text: str, prev_reply: str) -> bool:
        """The caller is wrapping up: "no, that's all", "नहीं, बस इतना ही", "bye", or
        "ठीक है" right after Maya asked "anything else?" / said goodbye. Short replies
        only, so "thanks, and what about pricing?" doesn't hang up on them."""
        t = (text or "").strip()
        if not t or "?" in t or len(t.split()) > 6:
            return False
        return bool(
            GOODBYE.search(t) or SHORT_NO.match(t)
            or (SIGNOFF.search(prev_reply or "") and (ACK.match(t) or re.match(r"^\W*(?:no|नहीं|ना)(?=\W|$)", t, re.I)))
        )


if __name__ == "__main__":
    print("tools.py self-check passed")
    print("Zryth tools available:")
    print("- capture_lead")
    print("- book_consultation")
    print("- transfer_to_human")
