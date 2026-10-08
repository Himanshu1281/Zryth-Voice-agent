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
    (re.compile(r"\b(?:zyrth|zrith|zerith|zirith|zirth|zerth|zareth|zarith|zarid|zerid|zaret)\b", re.I), "Zryth"),
]

# A bare "no" answering "anything else?" also means the caller is done.
_SHORT_NO = re.compile(
    r"^\W*(?:no|nope|nah|no no|nothing|not really|nahi|nahin|नहीं|ना|जी नहीं)\W*$", re.I
)

# Spoken by end_call itself, so every call ends the same polite way.
_GOODBYES = {
    "en": "Thank you for calling {business}. Have a great day, goodbye!",
    "hi": "{business} को कॉल करने के लिए धन्यवाद। आपका दिन शुभ हो, नमस्ते!",
}

# Caller is wrapping up: only then may end_call actually hang up.
_GOODBYE = re.compile(
    r"\b(?:bye|goodbye|good night|that's all|thats all|that is all|no thanks?|nothing else|"
    r"not now|that's it|thats it|i'm done|im done|hang up|cut the call|done|thank you|thanks|"
    r"ok bye|okay bye|chalo|rakhta|rakhti|take care|see you|see ya|alright|all right|"
    r"have a (?:good|nice|great) day)\b|धन्यवाद|शुक्रिया|बाय|बस इतना|और कुछ नहीं|रखता|रखती",
    re.I,
)


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
    """Return a valid ten-digit Indian mobile number, or None.

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
    "That phone number is not a valid ten-digit mobile number. Read back what you heard "
    "and ask the caller to repeat their ten-digit number. Do not save until it is valid."
)

# Values the LLM invents when it calls a tool before the caller has said anything.
_PLACEHOLDERS = {"", "user", "caller", "customer", "unknown", "none", "null", "n/a", "na", "name", "phone"}


def _missing_details(name: str | None, phone: str | None) -> dict | None:
    """Failure result when name/phone were never actually given, else None.

    Keeps Maya from telling the caller their number is invalid when they
    haven't given one yet (the LLM fills placeholders like "User").
    """
    missing = []
    if (name or "").strip().lower() in _PLACEHOLDERS:
        missing.append("name")
    if not re.search(r"\d", phone or ""):
        missing.append("phone number")
    if not missing:
        return None
    return {
        "status": "failed",
        "message": (
            f"Nothing was saved: the caller has not given their {' or '.join(missing)} yet. "
            "Do NOT say anything was invalid or that they provided it. Politely ask for the "
            f"{missing[0]} now (one question only), then call this tool again."
        ),
    }


def _spoken_digits(phone: str) -> str:
    """'8591194506' -> '8 5 9 1 1 9 4 5 0 6' so Maya reads it digit by digit."""
    return " ".join(phone)


def _user_said_digits(context: RunContext, digits: str) -> bool:
    """True when the caller's own transcripts contain this number."""
    try:
        said = "".join(
            re.sub(r"\D", "", m.text_content or "")
            for m in context.session.history.items
            if getattr(m, "role", None) == "user"
        )
    except Exception:
        log.exception("could not read history for phone check")
        return True  # don't block saving on an SDK change
    return digits in said


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
    words = [w.lower() for w in re.findall(r"\w+", name) if len(w) > 1]
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


def _last_reply_asked_number(context: RunContext) -> bool:
    """Maya's last reply asked about / read back a number, and the caller answered."""
    return _last_reply_asked(context, r"number|नंबर|नम्बर|\d")


# An instruction, not a script: verbatim English text got read out to Hindi callers.
_TRANSFER_FAILED = (
    "Transfer failed: no team member could be reached. In the caller's language (the one they "
    "are speaking now), say sorry that nobody is free right now, that the {business} team will "
    "call them back, and offer to keep helping."
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
        # Number the caller is dialling from (set by the entrypoint).
        self.caller_phone: str | None = None
        # (name, phone) already saved this call -- stops duplicate bookings.
        self.saved_contact: tuple[str, str] | None = None
        # A spoken number waiting for the caller's "yes" after Maya reads it back.
        self.pending_phone: str | None = None
        # end_call defers once to offer a callback when no details were saved.
        self.callback_offered = False

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
        Ask the caller ONLY for their name and ten-digit phone number. Never ask
        for email, company, or anything else.

        Args:
            name: Caller's name.
            phone: Caller's ten-digit mobile number.
            requirement: What they want, summarised by you from the conversation
                (do NOT ask the caller for this).
        """

        return await self._save_contact(context, "capture_lead", name, phone, requirement)

    async def _save_contact(
        self, context: RunContext, tool: str, name: str, phone: str, requirement: Optional[str]
    ) -> dict:
        """Shared by capture_lead/book_consultation: validate, de-duplicate, save,
        then have Maya confirm the name and read the number back."""
        missing = _missing_details(name, phone)
        if missing:
            return missing
        # The LLM sometimes lifts a "name" from a sentence ("founder social media")
        # without ever asking for it.
        if not _user_said_name(context, name):
            log.warning("%s: rejected name %r not given by caller", tool, name)
            return {
                "status": "failed",
                "message": (
                    "Nothing saved: the caller has not told you their name. Politely ask for their "
                    "name now (one question only), then call this tool again."
                ),
            }
        clean = _clean_phone(phone)
        if not clean:
            return {"status": "failed", "message": _BAD_PHONE_MSG}
        # The LLM sometimes invents a number (e.g. the 9876543210 example). Accept
        # only the caller ID or digits the caller actually said. Don't reveal the
        # caller ID here, or the LLM just resubmits it without asking.
        caller = _clean_phone(self.caller_phone)
        if clean != caller and not _user_said_digits(context, clean):
            log.warning("%s: rejected number %s not said by caller", tool, clean)
            offer = (
                "Ask whether the team should call them on the number they're calling from, "
                "or on another number, and wait for their answer."
                if caller else "Ask the caller for their ten-digit mobile number."
            )
            return {
                "status": "failed",
                "message": f"Nothing saved: the caller never gave that number. {offer}",
            }

        # The caller ID is used only after Maya asked about the number and the
        # caller answered (not just because the LLM decided to use it).
        if clean == caller and not _last_reply_asked_number(context):
            log.warning("%s: caller ID used without asking the caller", tool)
            return {
                "status": "confirm_first",
                "message": (
                    "Not saved yet. First ask the caller whether the team should reach them on the "
                    "number they're calling from, or another one. Call this tool again only after they answer."
                ),
            }

        # Spoken numbers are often misheard or split across turns ("8221" ... "5216"),
        # so read a new one back and save only when the LLM calls again after a "yes".
        if clean != caller and (self.pending_phone != clean or not _last_reply_asked_number(context)):
            self.pending_phone = clean
            return {
                "status": "confirm_first",
                "message": (
                    f"Not saved yet. Read the number back digit by digit ({_spoken_digits(clean)}) "
                    "and ask if it is correct. Call this tool again with the same number only after "
                    "the caller says yes; if they correct it, ask for the full ten-digit number again."
                ),
            }
        self.pending_phone = None

        name = name.strip()
        if self.saved_contact == (name, clean):
            return {
                "status": "already_saved",
                "message": "This is already recorded. Do not mention it again; just reply to the caller.",
            }
        updating = self.saved_contact is not None

        if self.call_id:
            await asyncio.to_thread(
                update_call_lead,
                call_id=self.call_id,
                customer_name=name,
                phone=clean,
                requirement=requirement,
            )
        self.saved_contact = (name, clean)
        log.info("%s -> %s", tool, name)

        if updating:
            return {
                "status": "updated",
                "message": f"Updated to {name}, {_spoken_digits(clean)}. Briefly confirm the change.",
            }
        what = "consultation request" if tool == "book_consultation" else "details"
        if clean != caller:
            # The caller already confirmed this number on the read-back.
            return {
                "status": "saved",
                "message": (
                    f"Saved. In one short reply tell {name} their {what} is recorded and the "
                    f"{self.config.business_name} team will contact them on that number. "
                    "Don't read the number again."
                ),
            }
        return {
            "status": "saved",
            "message": (
                f"Saved. In one reply: tell {name} their {what} is recorded and the "
                f"{self.config.business_name} team will contact them, read the number back digit by digit "
                f"({_spoken_digits(clean)}), and ask if they'd prefer a different number. "
                "If they give another number, call this tool again with it."
            ),
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
        Ask the caller ONLY for their name, ten-digit phone number and (optionally)
        a preferred date/time. Never ask for email, company, or anything else.

        Args:
            name: Caller's name.
            phone: Caller's ten-digit mobile number.
            requirement: What they want, summarised by you from the conversation
                (do NOT ask the caller for this).
            preferred_date: Preferred date, if the caller gave one.
            preferred_time: Preferred time, if the caller gave one.
        """

        return await self._save_contact(
            context, "book_consultation", name, phone,
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
            return _TRANSFER_FAILED.format(business=self.config.business_name)

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
            return _TRANSFER_FAILED.format(business=self.config.business_name)


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
        last_user, spoke_after = "", False
        try:
            for m in reversed(context.session.history.items):
                role = getattr(m, "role", None)
                if role == "assistant" and m.text_content:
                    spoke_after = True  # Maya already replied after the caller's turn
                elif role == "user" and m.text_content:
                    last_user = m.text_content
                    break
        except Exception:
            log.exception("end_call: could not read history")
        if last_user and not (_GOODBYE.search(last_user) or _SHORT_NO.match(last_user)):
            log.info("end_call refused; caller said %r", last_user)
            if spoke_after:
                return None  # she already answered; a tool reply would make her speak twice
            # Nothing said this turn yet: have her reply instead of leaving dead air.
            return "Don't end the call yet. Reply to what the caller just said."

        # Lead backstop: one callback offer before hanging up on a caller whose
        # details weren't saved. Asked only once, so nobody gets trapped.
        if self.saved_contact is None and not self.callback_offered and not self.is_transferring:
            self.callback_offered = True
            log.info("end_call deferred once for a callback offer")
            ask = (
                "should our team call you back on this number with more details?"
                if _clean_phone(self.caller_phone)
                else "may I take your number so our team can call you back with more details?"
            )
            return (
                "Don't hang up yet. If you haven't already offered a callback, ask in one short sentence: "
                f"\"Before you go, {ask}\" If you already offered and they declined, just say a brief "
                "goodbye and call end_call again."
            )

        if self.is_ending:
            return None
        self.is_ending = True
        if self.on_end_requested:
            self.on_end_requested()
        # Fixed goodbye (like the original agent): predictable, and the hang-up
        # fires once it finishes playing. None = the LLM adds nothing after it.
        lang = getattr(context.session.current_agent, "code", "en")
        context.session.say(
            _GOODBYES.get(lang, _GOODBYES["en"]).format(business=self.config.business_name),
            allow_interruptions=False,
        )
        return None


if __name__ == "__main__":
    print("tools.py self-check passed")
    print("Zryth tools available:")
    print("- capture_lead")
    print("- book_consultation")
    print("- transfer_to_human")
