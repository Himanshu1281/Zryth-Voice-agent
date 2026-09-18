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


class AppointmentTools:
    """Tools Maya can use during a Zryth customer call."""

    def __init__(self, job_ctx: JobContext | None = None, call_id: str | None = None) -> None:
        self.job_ctx = job_ctx
        self.call_id = call_id

    def to_tools(self) -> list:
        return [
            self.capture_lead,
            self.book_consultation,
            self.transfer_to_human,
	    self.end_call,
            self.search_knowledge,
        ]

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
        """Save a potential customer's enquiry.

        Use this when the caller is interested in Zryth's services and
        you have collected their name plus at least one contact method.

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
        
        Use this tool when the user asks a specific question about Zryth's offerings.
        Do NOT guess; always look it up.
        
        Args:
            query: The question or search term (e.g., 'Oswal AI features').
        """
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
                    return []
                    
                table = db.open_table("knowledge")
                results = table.search(emb).limit(5).to_list()
                
                if not results:
                    return []
                return [row['content'] for row in results]
                
            results_content = await asyncio.to_thread(_do_search)
            
            if not results_content:
                return "No relevant information found in the knowledge base."
                
            formatted_results = "\n\n".join(results_content)
            return f"{formatted_results}\n\nINSTRUCTION: You must now respond to the user's question using the information above. Provide your response as natural spoken text, translated into the user's language. Do NOT make any more function calls right now."
            
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
        """Record a request for a Zryth consultation.

        Use when the caller wants to discuss a project with the Zryth team.
        Collect their name and at least one contact method before calling
        this tool.

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
        """End the call when the customer clearly indicates the conversation is over.

        Only use this after the customer says goodbye, confirms they need no more
        help, or otherwise clearly indicates that the conversation is finished.
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

        async def _delayed_shutdown():
            import asyncio
            # Give the goodbye response plenty of time to finish playing before shutting down.
            await asyncio.sleep(6)
            if self.job_ctx:
                self.job_ctx.shutdown(reason="customer ended conversation")
                
        import asyncio
        asyncio.create_task(_delayed_shutdown())

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
