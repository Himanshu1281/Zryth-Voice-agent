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


from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    filename: str  # object name in the knowledge_base bucket (.pdf or .txt)


@app.post("/internal/knowledge/ingest")
async def knowledge_ingest(
    req: IngestRequest,
    x_webhook_secret: str | None = Header(default=None),
):
    """Chunk + embed one uploaded file the same way the agent expects.

    The dashboard should call this after uploading to the bucket instead of
    chunking itself. Replaces that file's existing rows.
    """
    if not WEBHOOK_SECRET or not x_webhook_secret or not hmac.compare_digest(
        x_webhook_secret, WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    from knowledge_ingest import ingest_source
    from starlette.concurrency import run_in_threadpool

    try:
        count = await run_in_threadpool(ingest_source, req.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        log.exception("Ingest failed for %s", req.filename)
        raise HTTPException(status_code=500, detail="Ingest failed")

    return {"status": "ok", "filename": req.filename, "chunks": count}


class SummarizeRequest(BaseModel):
    transcript: str

@app.post("/internal/ai/summarize")
async def summarize_call(req: SummarizeRequest):
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
        with urllib.request.urlopen(req_obj) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            candidates = res_body.get("candidates", [])
            if not candidates:
                return {"summary": "No summary available."}
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "No summary available.")
            return {"summary": text}
    except Exception as e:
        log.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail="Failed to summarize")
