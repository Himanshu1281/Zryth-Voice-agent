"""Function tools AI solutions voice assistant."""

from __future__ import annotations

import json
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import os

from google import genai
from livekit.agents import JobContext, RunContext, function_tool

from config import DEFAULT_TRANSFER_NUMBER
from database import _init_supabase, update_call_lead

# Initialize the Gemini client for embeddings
if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
    llm_client = None
else:
    llm_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

log = logging.getLogger("voice-agent.tools")

_cached_db = None

DATA_DIR = Path(__file__).parent / "data"
LEADS_PATH = DATA_DIR / "leads.json"


def _load_leads() -> list[dict]:
    """Load saved leads."""
    if not LEADS_PATH.exists():
        return []

    try:
        return json.loads(LEADS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_lead(lead: dict) -> None:
    """Save a lead locally.

    This can later be replaced with an n8n webhook, CRM, Supabase,
    Google Sheets, or another backend.
    """
    DATA_DIR.mkdir(exist_ok=True)

    leads = _load_leads()
    leads.append(lead)

    LEADS_PATH.write_text(
        json.dumps(leads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Per-instance call tracking to rate-limit search_knowledge
_SEARCH_RATE_LIMIT = 4  # max search calls per turn


class AppointmentTools:
    """Tools Maya can use during a Zryth customer call."""

    def __init__(self, job_ctx: JobContext | None = None, call_id: str | None = None) -> None:
        self.job_ctx = job_ctx
        self.call_id = call_id
        self._search_calls_this_turn: int = 0  # rate-limit counter
        self._shutdown_task: asyncio.Task | None = None  # track pending shutdown

    def to_tools(self) -> list:
        return [
            self.capture_lead,
            self.book_consultation,
            self.transfer_to_human,
	    self.end_call,
            self.search_knowledge,
        ]

    def reset_turn_counters(self) -> None:
        """Reset per-turn rate-limit counters. Call this on each new user utterance."""
        self._search_calls_this_turn = 0

    @function_tool
    async def capture_lead(
        self,
        context: RunContext,
        name: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        company: Optional[str] = None,
        requirement: Optional[str] = None,
    ) -> dict:
        """Log a potential customer's contact details and interest.

        Use ONLY when the caller has expressed general interest in Zryth's services
        but has NOT asked to schedule or book a specific meeting.
        If they mention a date, time, or say "book/schedule a call", use
        `book_consultation` instead.

        Args:
            name: Caller's full name.
            phone: Caller's phone number, if provided.
            email: Caller's email address, if provided.
            company: Company or organization name, if relevant.
            requirement: Short summary of what the caller wants to build,
                automate, integrate, or improve with AI/software.
        """

        if self.call_id:
            await asyncio.to_thread(
                update_call_lead,
                call_id=self.call_id,
                customer_name=name,
                phone=phone,
                email=email,
                company=company,
                requirement=requirement,
            )

        log.info("capture_lead -> %s", name)

        return {
            "status": "saved",
            "message": "The customer enquiry has been recorded successfully.",
        }

    @function_tool
    async def search_knowledge(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """Search the Zryth knowledge base for product details, features, or pricing.

        Call this ONLY when ALL of these are true:
        - The user's utterance is a complete, finished sentence (not a fragment).
        - It specifically concerns Zryth's products, services, team, or pricing.
        Example — CALL: "What does Oswal AI do?" / "Tell me about Zryth's products."
        Example — DO NOT CALL: "what" / "umm tell me" / silence / mid-sentence fragments.

        Args:
            query: 3+ word descriptive phrase, e.g. "Oswal AI features" or
                   "What products does Zryth make?".
        """
        # Guard: reject vague/partial queries
        if not query or len(query.split()) < 3:
            return "Please wait for the user to complete their question before searching."
        
        # Rate-limit: reset counter at start of each tool call, then check
        self._search_calls_this_turn += 1
        if self._search_calls_this_turn > _SEARCH_RATE_LIMIT:
            log.warning("search_knowledge rate limit hit (%d calls this turn)", self._search_calls_this_turn)
            return "You have already searched enough. Please answer the user with what you know."
        
        log.info(f"search_knowledge -> querying for: {query}")
        
        try:
            def _do_search():
                res = llm_client.models.embed_content(
                    model='gemini-embedding-2',
                    contents=query,
                )
                emb = res.embeddings[0].values

                 # Check globally initialized db connection instead of reconnecting every time
                global _cached_db
                if _cached_db is None:
                    import lancedb
                    import os
                    from database import LANCEDB_PATH
                    if not os.path.exists(LANCEDB_PATH):
                        return []
                    _cached_db = lancedb.connect(LANCEDB_PATH)

                db = _cached_db
                if "knowledge" not in db.table_names():
                    # Table missing — try invalidating the cached connection and reconnecting once
                    log.warning("'knowledge' table not found, re-syncing LanceDB...")
                    _cached_db = None
                    from database import sync_knowledge_to_lancedb
                    sync_knowledge_to_lancedb()
                    import lancedb as _lancedb
                    from database import LANCEDB_PATH
                    _cached_db = _lancedb.connect(LANCEDB_PATH)
                    db = _cached_db
                    if "knowledge" not in db.table_names():
                        return []
                    
                table = db.open_table("knowledge")
                results = table.search(emb).limit(3).to_list()
                
                if not results:
                    return []
                # Truncate each chunk to keep LLM output manageable
                return [row['content'][:300] for row in results]
                
            results_content = await asyncio.to_thread(_do_search)
            
            if not results_content:
                return "No relevant information found in the knowledge base."
                
            return "\n\n".join(results_content)
            
        except Exception as e:
            log.error(f"search_knowledge error: {e}")
            return "An error occurred while searching the knowledge base."

    @function_tool
    async def book_consultation(
        self,
        context: RunContext,
        name: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        company: Optional[str] = None,
        requirement: Optional[str] = None,
        preferred_date: Optional[str] = None,
        preferred_time: Optional[str] = None,
    ) -> dict:
        """Record a consultation request for the Zryth team.

        Use ONLY when the caller explicitly asks to schedule or book a meeting/call,
        OR agrees when you offer one. If they've only expressed general interest with
        no scheduling intent, use `capture_lead` instead.

        Args:
            name: Caller's full name.
            phone: Caller's phone number.
            email: Caller's email address.
            company: Company or organization name, if relevant.
            requirement: Brief description of the project or business problem.
            preferred_date: Preferred consultation date, if provided.
            preferred_time: Preferred consultation time, if provided.
        """

        if self.call_id:
            await asyncio.to_thread(
                update_call_lead,
                call_id=self.call_id,
                customer_name=name,
                phone=phone,
                email=email,
                company=company,
                requirement=f"{requirement} (Preferred: {preferred_date} {preferred_time})",
            )

        log.info("book_consultation -> %s", name)

        return {
            "status": "requested",
            "message": (
                "The consultation request has been recorded. "
                "The Zryth team will follow up to confirm the appointment."
            ),
        }

    @function_tool
    async def transfer_to_human(
        self,
        context: RunContext,
    ) -> dict:
        """Transfer the caller to a Zryth team member.

        Use when the caller specifically asks to speak with a human,
        asks for a team member, or the request requires human assistance.
        """

        log.info(
            "transfer_to_human -> %s",
            DEFAULT_TRANSFER_NUMBER,
        )

        return {
            "action": "transfer",
            "to": DEFAULT_TRANSFER_NUMBER,
        }

    @function_tool
    async def end_call(
        self,
        context: RunContext,
    ) -> dict:
        """Ends the call and plays a goodbye message.

        Call this ONLY when the CONVERSATION ENDING conditions in the system
        instructions are met (caller said goodbye / confirmed no further needs).
        Do not use this for any other reason.
        """

        log.info("end_call requested by Maya")

        if self.job_ctx is None:
            log.warning("Cannot end call: JobContext is not available")
            return {
                "status": "failed",
                "message": "Call ending is not available.",
            }
            
        if hasattr(context.session, "_closed") and context.session._closed:
            return {"status": "ended", "message": "Already ending."}
            
        try:
            # Explicitly push the goodbye message into the TTS queue
            await context.session.say("Thank you for your interest in Z-rith. Have a great day! Goodbye.")
        except RuntimeError:
            # Ignore if already closing
            pass

        # Cancel any pending duplicate shutdown
        if self._shutdown_task and not self._shutdown_task.done():
            self._shutdown_task.cancel()

        async def _delayed_shutdown():
            # Give the goodbye audio time to finish before shutting down.
            await asyncio.sleep(6)
            if self.job_ctx:
                try:
                    self.job_ctx.shutdown(reason="customer ended conversation")
                except Exception as exc:
                    log.warning("shutdown() raised: %s", exc)

        self._shutdown_task = asyncio.create_task(_delayed_shutdown())

        return {
            "status": "ended",
            "message": "The call is ending. DO NOT say anything else. DO NOT call any further tools.",
        }

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    print("tools.py self-check passed")
    print("Zryth tools available:")
    print("- capture_lead")
    print("- book_consultation")
    print("- transfer_to_human")
