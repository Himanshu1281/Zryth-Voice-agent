from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client
import json
import asyncio
import logging
import lancedb

log = logging.getLogger(__name__)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

# Only validate when actually creating the client
supabase: Client | None = None

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

def sync_knowledge_to_lancedb() -> None:
    """Sync Supabase knowledge base to local LanceDB."""
    try:
        response = _init_supabase().table("zryth_knowledge").select("id, content, embedding").execute()
        data = response.data
        if not data:
            log.info("No data in Supabase zryth_knowledge to sync.")
            return

        lancedb_data = []
        for row in data:
            try:
                # Supabase returns the pgvector as a string, e.g. "[0.1, 0.2, ...]"
                vec = json.loads(row['embedding'])
                lancedb_data.append({
                    "id": row['id'],
                    "content": row['content'],
                    "vector": vec
                })
            except Exception as e:
                log.warning(f"Failed to parse embedding for row {row['id']}: {e}")

        if lancedb_data:
            os.makedirs("data", exist_ok=True)
            db = lancedb.connect("data/lancedb")
            db.create_table("knowledge", data=lancedb_data, mode="overwrite")
            log.info(f"Successfully synced {len(lancedb_data)} rows to local LanceDB.")
            
    except Exception as e:
        log.error(f"Error syncing knowledge to LanceDB: {e}")


sync_queue = asyncio.Queue()

def _realtime_callback(payload):
    """Callback for Supabase Realtime events."""
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(sync_queue.put_nowait, True)
    except Exception as e:
        # If no running loop, log or ignore
        log.warning(f"Could not queue sync event: {e}")

async def realtime_sync_loop() -> None:
    """Background task to sync LanceDB via Supabase Realtime with debouncing."""
    from supabase import create_async_client
    try:
        supabase_async = await create_async_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        
        # Subscribe to changes
        channel = supabase_async.channel("zryth_knowledge_changes")
        channel.on_postgres_changes(
            event="*", 
            schema="public", 
            table="zryth_knowledge", 
            callback=_realtime_callback
        )
        await channel.subscribe()
        log.info("Subscribed to Supabase Realtime for zryth_knowledge table.")
    except Exception as e:
        log.error(f"Failed to subscribe to Realtime: {e}")
        return

    while True:
        # Wait until an event is pushed to the queue
        await sync_queue.get()
        
        # Debounce: Wait 5 seconds to gather any subsequent rapid events
        await asyncio.sleep(5)
        
        # Clear the queue of any additional events that arrived during the wait
        while not sync_queue.empty():
            try:
                sync_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        log.info("Sync required! Changes detected via Realtime.")
        try:
            await asyncio.to_thread(sync_knowledge_to_lancedb)
        except Exception as e:
            log.error(f"Error syncing after realtime event: {e}")

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
