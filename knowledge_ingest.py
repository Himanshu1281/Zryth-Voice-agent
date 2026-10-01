"""Single ingestion path for zryth_knowledge (PDF or TXT in the knowledge_base bucket).

Used by:
  * webhook_server.py  POST /internal/knowledge/ingest   (dashboard can call this)
  * database.realtime_sync_loop -> heal_legacy_chunks()  (auto-fixes rows written
    by any other chunker, e.g. the dashboard's fixed-size one)
  * scripts/rechunk_knowledge.py                           (manual, with backup)

Always: sentence-aware chunks (scripts/chunking.py), gemini-embedding-2 (must
match the query model in tools.py), insert new rows BEFORE deleting old ones.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from chunking import chunk_text, title_from_filename  # noqa: E402
from database import _init_supabase  # noqa: E402

log = logging.getLogger("knowledge-ingest")

BUCKET = "knowledge_base"
EMBED_MODEL = "gemini-embedding-2"
CHUNKER_VERSION = "sentence-v1"
SUPPORTED_EXT = (".pdf", ".txt")


def _extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="replace")


def build_rows(filename: str) -> list[dict]:
    """Download one source file and return chunk rows (without embeddings)."""
    if not filename.lower().endswith(SUPPORTED_EXT):
        raise ValueError(f"Unsupported file type: {filename}")
    data = _init_supabase().storage.from_(BUCKET).download(filename)
    text = _extract_text(filename, data)
    if "[FILL IN" in text:
        # Never let template placeholders reach callers.
        raise ValueError(f"{filename} still contains [FILL IN] placeholders")
    chunks = chunk_text(text, title=title_from_filename(filename))
    return [
        {"content": c, "metadata": {"source": filename, "chunk_index": i, "chunker": CHUNKER_VERSION}}
        for i, c in enumerate(chunks)
    ]


def _embed(rows: list[dict]) -> None:
    import os

    from google import genai

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    for row in rows:
        row["embedding"] = client.models.embed_content(
            model=EMBED_MODEL, contents=row["content"]
        ).embeddings[0].values
        time.sleep(0.2)  # stay under embedding rate limits


def ingest_source(filename: str) -> int:
    """(Re)ingest one source file, replacing its existing rows. Returns new row count."""
    sb = _init_supabase()
    rows = build_rows(filename)
    if not rows:
        log.warning("No text extracted from %s; leaving existing rows untouched", filename)
        return 0
    _embed(rows)

    old_ids = [
        r["id"]
        for r in sb.table("zryth_knowledge").select("id").eq("metadata->>source", filename).execute().data
    ]
    sb.table("zryth_knowledge").insert(rows).execute()
    if old_ids:
        sb.table("zryth_knowledge").delete().in_("id", old_ids).execute()
    log.info("Ingested %s: %d chunks (replaced %d)", filename, len(rows), len(old_ids))
    return len(rows)


def legacy_sources() -> list[str]:
    """Sources that have any row not produced by the current chunker."""
    rows = _init_supabase().table("zryth_knowledge").select("metadata").execute().data
    return sorted({
        (r.get("metadata") or {}).get("source")
        for r in rows
        if (r.get("metadata") or {}).get("chunker") != CHUNKER_VERSION
        and (r.get("metadata") or {}).get("source")
    })


def heal_legacy_chunks() -> list[str]:
    """Re-ingest every source whose rows came from another chunker. Returns fixed sources."""
    fixed = []
    for src in legacy_sources():
        if not src.lower().endswith(SUPPORTED_EXT):
            log.warning("Legacy rows for %s but file type unsupported; skipping", src)
            continue
        try:
            ingest_source(src)
            fixed.append(src)
        except Exception:
            log.exception("Auto re-chunk failed for %s (rows left as they were)", src)
    return fixed
