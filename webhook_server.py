import hmac
import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import urllib.request
import json
from database import sync_knowledge_to_lancedb

logging.basicConfig(level=logging.INFO)

log = logging.getLogger("knowledge-webhook")

app = FastAPI(
    title="Zryth Knowledge Webhook",
)

WEBHOOK_SECRET = os.getenv("SUPABASE_WEBHOOK_SECRET")


# No CORS: only zryth-backend calls this server (server-to-server), never a browser.


def _verify(secret: str | None) -> None:
    """Every /internal route requires the shared secret zryth-backend sends."""
    if not WEBHOOK_SECRET:
        log.error("SUPABASE_WEBHOOK_SECRET is not configured")
        raise HTTPException(status_code=500, detail="Webhook secret is not configured")
    if not secret or not hmac.compare_digest(secret, WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


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
    _verify(x_webhook_secret)

    log.info(
        "Supabase knowledge webhook received. "
        "Synchronizing LanceDB..."
    )

    try:
        from database import KB_AUTO_HEAL

        # Re-chunk anything the uploader wrote with a different chunker first
        # (only on the host where KB_AUTO_HEAL=true, to avoid duplicate rows).
        fixed = []
        if KB_AUTO_HEAL:
            from knowledge_ingest import heal_legacy_chunks

            try:
                fixed = heal_legacy_chunks()
            except Exception:
                log.exception("Legacy chunk healing failed; syncing rows as they are")

        sync_knowledge_to_lancedb()

        return {
            "status": "ok",
            "message": "LanceDB synchronization completed",
            "rechunked": fixed,
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

class IngestRequest(BaseModel):
    filename: str                # object path in the knowledge_base bucket (.pdf or .txt)
    agent_id: str | None = None  # org agent that owns it; None = no agent


@app.post("/internal/knowledge/ingest")
async def knowledge_ingest(
    req: IngestRequest,
    x_webhook_secret: str | None = Header(default=None),
):
    """Chunk + embed one uploaded file the same way the agent expects.

    The dashboard should call this after uploading to the bucket instead of
    chunking itself. Replaces that file's existing rows.
    """
    _verify(x_webhook_secret)

    from knowledge_ingest import ingest_source
    from starlette.concurrency import run_in_threadpool

    try:
        count = await run_in_threadpool(ingest_source, req.filename, req.agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        log.exception("Ingest failed for %s", req.filename)
        raise HTTPException(status_code=500, detail="Ingest failed")

    # Refresh this host's LanceDB now instead of waiting for the realtime event.
    await run_in_threadpool(sync_knowledge_to_lancedb)
    return {"status": "ok", "filename": req.filename, "chunks": count}


class DeleteRequest(BaseModel):
    filename: str  # object name that was removed from the knowledge_base bucket


@app.post("/internal/knowledge/delete")
async def knowledge_delete(
    req: DeleteRequest,
    x_webhook_secret: str | None = Header(default=None),
):
    """Drop a deleted file's chunks so Maya stops answering from it."""
    _verify(x_webhook_secret)

    from knowledge_ingest import delete_source
    from starlette.concurrency import run_in_threadpool

    try:
        removed = await run_in_threadpool(delete_source, req.filename)
        await run_in_threadpool(sync_knowledge_to_lancedb)
    except Exception:
        log.exception("Delete failed for %s", req.filename)
        raise HTTPException(status_code=500, detail="Delete failed")

    return {"status": "ok", "filename": req.filename, "removed": removed}


class SummarizeRequest(BaseModel):
    transcript: str

SUMMARY_PROMPT = """You are an expert conversation analyst. Read the following customer service transcript and write a concise, professional summary paragraph (3-5 sentences).

Make sure to include:
1. The customer's specific questions or requests.
2. Any exact product names, features, or details the agent provided.
3. The final outcome of the call.

Write it as a fluid paragraph, without bullet points or markdown.

Transcript:
"""


@app.post("/internal/ai/summarize")
async def summarize_call(
    req: SummarizeRequest,
    x_webhook_secret: str | None = Header(default=None),
):
    _verify(x_webhook_secret)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GOOGLE_API_KEY in backend environment")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": "Summarize this call transcript accurately. Keep it concise. Only output the summary.\n\n" + req.transcript}]}]
    }
    
    data = json.dumps(payload).encode("utf-8")
    req_obj = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req_obj, timeout=60) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            candidates = res_body.get("candidates", [])
            if not candidates:
                return {"summary": "No summary available."}
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "No summary available.")
            return {"summary": text}
    except Exception as e:
        log.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail="Failed to summarize")
