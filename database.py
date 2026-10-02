from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client, create_async_client
import json
import asyncio
import logging
import lancedb
import threading
import time

log = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANCEDB_PATH = os.path.join(BASE_DIR, "data", "lancedb")

_lancedb_sync_lock = threading.Lock()

# PostgREST returns at most 1000 rows per request (Supabase default): page through.
PAGE_SIZE = 1000
# Safety net when Realtime events are missed: full resync at least this often.
KB_RESYNC_INTERVAL_S = int(os.getenv("KB_RESYNC_INTERVAL_S", "600"))


def fetch_all(table: str, columns: str, apply=None) -> list[dict]:
    """Every row of a query, paged so results are never silently capped at 1000."""
    rows: list[dict] = []
    start = 0
    while True:
        q = _init_supabase().table(table).select(columns)
        if apply is not None:
            q = apply(q)
        batch = q.order("id").range(start, start + PAGE_SIZE - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE

# Re-chunk knowledge written by other chunkers. Enable on ONE host only.
KB_AUTO_HEAL = os.getenv("KB_AUTO_HEAL", "false").strip().lower() in ("1", "true", "yes")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

# Only validate when actually creating the client
supabase: Client | None = None
realtime_supabase = None

def _init_supabase() -> Client:
    global supabase
    if supabase is not None:
        return supabase
    
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not SUPABASE_SECRET_KEY:
        raise RuntimeError("SUPABASE_SECRET_KEY is not configured")
    
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    return supabase


async def _init_realtime_supabase():
    global realtime_supabase

    if realtime_supabase is not None:
        return realtime_supabase

    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not SUPABASE_SECRET_KEY:
        raise RuntimeError("SUPABASE_SECRET_KEY is not configured")

    realtime_supabase = await create_async_client(
        SUPABASE_URL,
        SUPABASE_SECRET_KEY,
    )

    return realtime_supabase


IST = timezone(timedelta(hours=5, minutes=30))

def create_call(
    livekit_room: str,
    phone: Optional[str] = None,
    language: str = "en",
    org_id: Optional[str] = None,
    org_agent_id: Optional[str] = None,
    dialed_number: Optional[str] = None,
) -> str:
    """Create a call record (tagged with the answering org/agent) and return its UUID."""

    data = {
        "livekit_room": livekit_room,
        "phone": phone,
        "language": language,
        "started_at": datetime.now(IST).isoformat(),
        "org_id": org_id,
        "agent_id": org_agent_id,  # calls.agent_id -> agents.id
        "dialed_number": dialed_number,
    }

    response = (
        _init_supabase()
        .table("calls")
        .upsert(data, on_conflict="livekit_room")
        .execute()
    )

    return response.data[0]["id"]


def save_message(
    call_id: str,
    speaker: str,
    message: str,
) -> None:
    """Save one customer or Maya message."""

    if not message or not message.strip():
        return

    (
        _init_supabase()
        .table("messages")
        .insert(
            {
                "call_id": call_id,
                "speaker": speaker,
                "message": message.strip(),
            }
        )
        .execute()
    )


def update_call_lead(
    call_id: str,
    customer_name: str,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    company: Optional[str] = None,
    requirement: Optional[str] = None,
) -> None:
    """Update a call record with the customer's lead information."""
    
    update_data = {"customer_name": customer_name}
    if phone: update_data["phone"] = phone
    if email: update_data["email"] = email
    if company: update_data["company"] = company
    if requirement: update_data["requirement"] = requirement
    
    (
        _init_supabase()
        .table("calls")
        .update(update_data)
        .eq("id", call_id)
        .execute()
    )


def finish_call(call_id: str, duration_seconds: int = 0) -> None:
    """Mark a call as finished."""

    (
        _init_supabase()
        .table("calls")
        .update(
            {
                "ended_at": datetime.now(IST).isoformat(),
                "duration_seconds": duration_seconds,
            }
        )
        .eq("id", call_id)
        .execute()
    )

def set_call_phone(call_id: str, phone: str) -> None:
    """Store the caller's real SIP number on the call record."""
    _init_supabase().table("calls").update({"phone": phone}).eq("id", call_id).execute()


def log_knowledge_gap(call_id: Optional[str], query: str) -> None:
    """Record a caller question the knowledge base couldn't answer."""
    _init_supabase().table("knowledge_gaps").insert(
        {"call_id": call_id, "query": query.strip()}
    ).execute()


def save_call_summary(
    call_id: str,
    summary: str,
    intent: Optional[str] = None,
    outcome: Optional[str] = None,
) -> None:
    """Attach the post-call summary to the call record."""
    _init_supabase().table("calls").update(
        {"summary": summary, "intent": intent, "outcome": outcome}
    ).eq("id", call_id).execute()


# Template prompts/tools are read at every call start, before the greeting: cache them
# briefly so a call costs no Supabase round trips (dashboard edits apply within the TTL).
_TEMPLATE_CACHE_TTL_S = 30.0
_template_cache: dict[tuple, tuple[float, object]] = {}
_template_cache_lock = threading.Lock()


def _cached(key: tuple, load):
    now = time.monotonic()
    with _template_cache_lock:
        hit = _template_cache.get(key)
        if hit and now - hit[0] < _TEMPLATE_CACHE_TTL_S:
            return hit[1]
    value = load()
    if value is not None:
        with _template_cache_lock:
            _template_cache[key] = (now, value)
    return value


def fetch_dynamic_prompt(tag: str, fallback_content: str, agent_id: str = "maya_v2") -> str:
    """Fetch a custom prompt from Supabase if it's explicitly assigned to the agent."""
    content = _cached(("prompt", agent_id, tag), lambda: _load_prompt(tag, agent_id))
    return content or fallback_content


def _load_prompt(tag: str, agent_id: str) -> str | None:
    """Assigned prompt's content, '' when not assigned, None on error (not cached)."""
    try:
        sb = _init_supabase()
        
        # 1. Check if the prompt is assigned to this agent
        assignment = sb.table("agent_prompts").select("*").eq("agent_id", agent_id).eq("prompt_tag", tag).execute()
        
        # If it's not explicitly assigned, immediately return the default hardcoded greeting
        if not assignment.data:
            return ""
        # 2. If it IS assigned, fetch the custom content from the prompts table
        response = sb.table("prompts").select("content").eq("tag", tag).execute()
        return response.data[0]["content"] if response.data else ""
    except Exception as e:
        log.error(f"Failed to fetch prompt '{tag}': {e}")
        return None


def fetch_assigned_tools(agent_id: str = "maya_v2") -> list[dict]:
    """Fetch JSON specs for tools explicitly assigned to the agent."""
    return _cached(("tools", agent_id), lambda: _load_tools(agent_id)) or []


def _load_tools(agent_id: str) -> list[dict] | None:
    try:
        sb = _init_supabase()
        
        # 1. Find assigned tool names
        assignments = sb.table("agent_tools").select("tool_name").eq("agent_id", agent_id).execute()
        if not assignments.data:
            return []
            
        names = [row["tool_name"] for row in assignments.data]
        
        # 2. Fetch the actual tool JSON specs
        response = sb.table("tools").select("*").in_("name", names).execute()
        return response.data if response.data else []
        
    except Exception as e:
        log.error(f"Failed to fetch assigned tools for '{agent_id}': {e}")
        return None

def sync_knowledge_to_lancedb() -> None:
    """Sync Supabase knowledge base to local LanceDB."""

    if not _lancedb_sync_lock.acquire(blocking=False):
        log.info("LanceDB sync already in progress. Skipping duplicate sync.")
        return

    try:
        data = fetch_all("zryth_knowledge", "id, content, embedding, agent_id")

        if not data:
            # Knowledge base is empty (e.g. last document deleted): drop the local
            # table too, otherwise Maya keeps answering from stale chunks.
            log.info("No data in Supabase zryth_knowledge; clearing local LanceDB.")
            if os.path.exists(LANCEDB_PATH):
                local = lancedb.connect(LANCEDB_PATH)
                if "knowledge" in local.table_names():
                    local.drop_table("knowledge")
            return

        lancedb_data = []

        for row in data:
            try:
                vec = json.loads(row["embedding"])

                lancedb_data.append(
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "vector": vec,
                        # '' = rows without an agent; searches filter on this per call
                        "agent_id": row.get("agent_id") or "",
                    }
                )

            except Exception as e:
                log.warning(
                    f"Failed to parse embedding for row "
                    f"{row['id']}: {e}"
                )

        if not lancedb_data:
            log.warning("No valid knowledge rows to write to LanceDB.")
            return

        os.makedirs(LANCEDB_PATH, exist_ok=True)

        log.info(
            "Synchronizing %d rows to LanceDB at %s...",
            len(lancedb_data),
            LANCEDB_PATH,
        )

        db = lancedb.connect(LANCEDB_PATH)

        db.create_table(
            "knowledge",
            data=lancedb_data,
            mode="overwrite",
        )

        log.info(
            "Successfully synced %d rows to local LanceDB.",
            len(lancedb_data),
        )

    except Exception as e:
        log.exception(
            f"Error syncing knowledge to LanceDB: {e}"
        )

    finally:
        _lancedb_sync_lock.release()


async def realtime_sync_loop() -> None:
    """Keep the local LanceDB in sync with zryth_knowledge.

    Realtime events trigger a (debounced) resync; if the subscription drops or never
    connects it is retried with backoff, and a full resync runs every
    KB_RESYNC_INTERVAL_S regardless, so a missed event can't leave a host stale.
    """
    sync_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _realtime_callback(payload):
        try:
            loop.call_soon_threadsafe(sync_queue.put_nowait, True)
            log.info("Supabase Realtime event received: %s", payload.get("eventType", "unknown"))
        except Exception as e:
            log.warning(f"Could not queue Realtime sync event: {e}")

    async def _subscribe_forever() -> None:
        backoff = 5
        while True:
            try:
                client = await _init_realtime_supabase()
                channel = client.channel("zryth_knowledge_changes")
                channel.on_postgres_changes(
                    event="*", schema="public", table="zryth_knowledge", callback=_realtime_callback,
                )
                await channel.subscribe()
                log.info("Subscribed to Supabase Realtime for public.zryth_knowledge.")
                # Catch up on anything that changed while we were disconnected
                sync_queue.put_nowait(True)
                return
            except Exception:
                log.exception("Realtime subscribe failed; retrying in %ss", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)

    async def _periodic() -> None:
        while True:
            await asyncio.sleep(KB_RESYNC_INTERVAL_S)
            sync_queue.put_nowait(True)

    asyncio.create_task(_subscribe_forever())
    asyncio.create_task(_periodic())

    while True:
        await sync_queue.get()
        await asyncio.sleep(5)  # debounce bursts (e.g. a 200-chunk ingest)
        while not sync_queue.empty():
            try:
                sync_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Rows written by another chunker get rebuilt sentence-aware. Opt-in
        # (KB_AUTO_HEAL=true) on exactly ONE host, or hosts would insert duplicates.
        if KB_AUTO_HEAL:
            try:
                from knowledge_ingest import heal_legacy_chunks

                fixed = await asyncio.to_thread(heal_legacy_chunks)
                if fixed:
                    log.info("Re-chunked legacy knowledge sources: %s", fixed)
            except Exception:
                log.exception("Legacy chunk healing failed; syncing rows as they are")

        try:
            await asyncio.to_thread(sync_knowledge_to_lancedb)
        except Exception:
            log.exception("Error syncing LanceDB")


if __name__ == "__main__":
    print("Testing Supabase connection...")

    (
        _init_supabase()
        .table("calls")
        .select("id")
        .limit(1)
        .execute()
    )

    print("Supabase connection OK")
    print("calls table is accessible")
