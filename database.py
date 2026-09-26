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

log = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANCEDB_PATH = os.path.join(BASE_DIR, "data", "lancedb")

_lancedb_sync_lock = threading.Lock()

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
) -> str:
    """Create a call record and return its Supabase UUID."""

    data = {
        "livekit_room": livekit_room,
        "phone": phone,
        "language": language,
        "started_at": datetime.now(IST).isoformat(),
    }

    response = (
        _init_supabase()
        .table("calls")
        .insert(data)
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

def fetch_dynamic_prompt(tag: str, fallback_content: str, agent_id: str = "maya_v2") -> str:
    """Fetch a custom prompt from Supabase if it's explicitly assigned to the agent."""
    try:
        sb = _init_supabase()
        
        # 1. Check if the prompt is assigned to this agent
        assignment = sb.table("agent_prompts").select("*").eq("agent_id", agent_id).eq("prompt_tag", tag).execute()
        
        # If it's not explicitly assigned, immediately return the default hardcoded greeting
        if not assignment.data or len(assignment.data) == 0:
            return fallback_content 
        # 2. If it IS assigned, fetch the custom content from the prompts table
        response = sb.table("prompts").select("content").eq("tag", tag).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]["content"]
    except Exception as e:
        log.error(f"Failed to fetch prompt '{tag}': {e}")
        
    return fallback_content

def fetch_assigned_tools(agent_id: str = "maya_v2") -> list[dict]:
    """Fetch JSON specs for tools explicitly assigned to the agent."""
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
        return []

def sync_knowledge_to_lancedb() -> None:
    """Sync Supabase knowledge base to local LanceDB."""

    if not _lancedb_sync_lock.acquire(blocking=False):
        log.info("LanceDB sync already in progress. Skipping duplicate sync.")
        return

    try:
        response = (
            _init_supabase()
            .table("zryth_knowledge")
            .select("id, content, embedding")
            .execute()
        )

        data = response.data

        if not data:
            log.info("No data in Supabase zryth_knowledge to sync.")
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
    """Listen for Supabase Realtime changes and sync local LanceDB."""

    sync_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _realtime_callback(payload):
        """Queue a sync whenever zryth_knowledge changes."""
        try:
            loop.call_soon_threadsafe(
                sync_queue.put_nowait,
                True,
            )
            log.info(
                "Supabase Realtime event received: %s",
                payload.get("eventType", "unknown"),
            )
        except Exception as e:
            log.warning(
                f"Could not queue Realtime sync event: {e}"
            )

    try:
        supabase_client = await _init_realtime_supabase()

        channel = supabase_client.channel(
            "zryth_knowledge_changes"
        )

        # Using on_postgres_changes as required by the python AsyncRealtimeChannel
        channel.on_postgres_changes(
            event="*",
            schema="public",
            table="zryth_knowledge",
            callback=_realtime_callback,
        )

        await channel.subscribe()

        log.info(
            "Successfully subscribed to Supabase Realtime "
            "for public.zryth_knowledge."
        )

    except Exception as e:
        log.exception(
            f"Failed to subscribe to Supabase Realtime: {e}"
        )
        return

    while True:
        # Wait for the first database change.
        await sync_queue.get()

        # Debounce multiple rapid changes.
        await asyncio.sleep(5)

        # Drain additional events.
        while not sync_queue.empty():
            try:
                sync_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        log.info(
            "Supabase Realtime change detected. "
            "Synchronizing LanceDB..."
        )

        try:
            await asyncio.to_thread(
                sync_knowledge_to_lancedb
            )

            log.info(
                "LanceDB synchronization completed "
                "after Realtime event."
            )

        except Exception as e:
            log.exception(
                f"Error syncing LanceDB after Realtime event: {e}"
            )

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
