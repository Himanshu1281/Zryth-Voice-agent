from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client
import json
import asyncio
import logging

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
