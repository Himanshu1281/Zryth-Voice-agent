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
