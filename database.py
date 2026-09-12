from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv
from supabase import create_async_client
from supabase.client import AsyncClient

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

# Only validate when actually creating the client
supabase: AsyncClient | None = None

async def _init_supabase() -> AsyncClient:
    global supabase
    if supabase is not None:
        return supabase
    
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not SUPABASE_SECRET_KEY:
        raise RuntimeError("SUPABASE_SECRET_KEY is not configured")
    
    supabase = await create_async_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    return supabase


IST = timezone(timedelta(hours=5, minutes=30))

async def create_call(
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

    client = await _init_supabase()
    response = await (
        client
        .table("calls")
        .insert(data)
        .execute()
    )

    return response.data[0]["id"]


async def save_message(
    call_id: str,
    speaker: str,
    message: str,
) -> None:
    """Save one customer or Maya message."""

    if not message or not message.strip():
        return

    client = await _init_supabase()
    await (
        client
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


async def update_call_lead(
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
    
    client = await _init_supabase()
    await (
        client
        .table("calls")
        .update(update_data)
        .eq("id", call_id)
        .execute()
    )


async def finish_call(call_id: str, duration_seconds: int = 0) -> None:
    """Mark a call as finished."""

    client = await _init_supabase()
    await (
        client
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


if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing Supabase connection...")
    
        client = await _init_supabase()
        await (
            client
            .table("calls")
            .select("id")
            .limit(1)
            .execute()
        )
    
        print("Supabase connection OK")
        print("calls table is accessible")
        
    asyncio.run(test())
