import hmac
import logging
import os

from fastapi import FastAPI, Header, HTTPException
from database import sync_knowledge_to_lancedb

logging.basicConfig(level=logging.INFO)

log = logging.getLogger("knowledge-webhook")

app = FastAPI(
    title="Zryth Knowledge Webhook",
)

WEBHOOK_SECRET = os.getenv("SUPABASE_WEBHOOK_SECRET")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "zryth-knowledge-webhook",
    }


@app.post("/internal/supabase/knowledge-sync")
async def knowledge_sync(
    x_webhook_secret: str | None = Header(default=None),
):
    if not WEBHOOK_SECRET:
        log.error("SUPABASE_WEBHOOK_SECRET is not configured")
        raise HTTPException(
            status_code=500,
            detail="Webhook secret is not configured",
        )

    if not x_webhook_secret:
        raise HTTPException(
            status_code=401,
            detail="Missing webhook secret",
        )

    if not hmac.compare_digest(
        x_webhook_secret,
        WEBHOOK_SECRET,
    ):
        log.warning("Invalid webhook secret received")
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook secret",
        )

    log.info(
        "Supabase knowledge webhook received. "
        "Synchronizing LanceDB..."
    )

    try:
        sync_knowledge_to_lancedb()

        return {
            "status": "ok",
            "message": "LanceDB synchronization completed",
        }

    except Exception as exc:
        log.exception(
            "Knowledge synchronization failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="LanceDB synchronization failed",
        )
