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
import re
import sys
import time
from pathlib import Path

from scripts.chunking import chunk_text, title_from_filename  # noqa: E402
from database import _init_supabase, fetch_all  # noqa: E402

log = logging.getLogger("knowledge-ingest")

BUCKET = "knowledge_base"
EMBED_MODEL = "gemini-embedding-2"
CHUNKER_VERSION = "sentence-v2"  # v2: no "[Document title]" prefix in chunk text
SUPPORTED_EXT = (".pdf", ".txt")
_AGENT_FOLDER = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/")


def agent_from_path(filename: str) -> str | None:
    """The agent a bucket object belongs to: files live under "<agent_id>/"."""
    m = _AGENT_FOLDER.match(filename or "")
    return m.group(1) if m else None


def _extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="replace")


def build_rows(filename: str, agent_id: str | None = None) -> list[dict]:
    """Download one source file and return chunk rows (without embeddings).

    `filename` is the object path in the bucket: "<agent_id>/<file>" for an org
    agent's knowledge, or "<file>" for knowledge with no agent (agent_id None)."""
    if not filename.lower().endswith(SUPPORTED_EXT):
        raise ValueError(f"Unsupported file type: {filename}")
    data = _init_supabase().storage.from_(BUCKET).download(filename)
    text = _extract_text(filename, data)
    if "[FILL IN" in text:
        # Never let template placeholders reach callers.
        raise ValueError(f"{filename} still contains [FILL IN] placeholders")
    chunks = chunk_text(text)
    return [
        {
            "content": c,
            "agent_id": agent_id,
            "metadata": {"source": filename, "chunk_index": i, "chunker": CHUNKER_VERSION},
        }
        for i, c in enumerate(chunks)
    ]


EMBED_BATCH = 50  # chunks per embedding request


def _embed(rows: list[dict]) -> None:
    """Embed in batches (one request per EMBED_BATCH chunks) with retry on rate limits.
    A 300-chunk PDF takes a handful of requests instead of 300 sequential ones."""
    import os

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    for i in range(0, len(rows), EMBED_BATCH):
        batch = rows[i:i + EMBED_BATCH]
        for attempt in range(4):
            try:
                # One Content per chunk: a plain list of strings is embedded as ONE combined
                # input by gemini-embedding-2 (returns a single vector).
                res = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=[types.Content(parts=[types.Part(text=r["content"])]) for r in batch],
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                log.warning("Embedding batch %d failed (attempt %d); retrying", i // EMBED_BATCH, attempt + 1)
                time.sleep(2 ** attempt)
        if len(res.embeddings) != len(batch):
            raise RuntimeError(f"Embedding count mismatch: {len(res.embeddings)} for {len(batch)} chunks")
        for row, emb in zip(batch, res.embeddings):
            row["embedding"] = emb.values


def ingest_source(filename: str, agent_id: str | None = None) -> int:
    """(Re)ingest one source file into one agent's KB, replacing its existing rows.
    Returns the new row count. The agent defaults to the file's "<agent_id>/" folder."""
    agent_id = agent_id or agent_from_path(filename)
    sb = _init_supabase()
    rows = build_rows(filename, agent_id)
    if not rows:
        log.warning("No text extracted from %s; leaving existing rows untouched", filename)
        return 0
    _embed(rows)

    old_ids = [r["id"] for r in fetch_all("zryth_knowledge", "id", lambda q: q.eq("metadata->>source", filename))]
    for i in range(0, len(rows), 200):  # insert in chunks: big PDFs exceed request size limits
        sb.table("zryth_knowledge").insert(rows[i:i + 200]).execute()
    for i in range(0, len(old_ids), 500):
        sb.table("zryth_knowledge").delete().in_("id", old_ids[i:i + 500]).execute()
    log.info("Ingested %s: %d chunks (replaced %d)", filename, len(rows), len(old_ids))
    return len(rows)


def delete_source(filename: str) -> int:
    """Remove every chunk of one source file. Returns the number of rows removed."""
    sb = _init_supabase()
    ids = [r["id"] for r in fetch_all("zryth_knowledge", "id", lambda q: q.eq("metadata->>source", filename))]
    # delete by source in one statement (no huge IN list)
    if ids:
        sb.table("zryth_knowledge").delete().eq("metadata->>source", filename).execute()
    log.info("Deleted %s: %d chunks", filename, len(ids))
    return len(ids)


def legacy_sources() -> list[str]:
    """Sources that have any row not produced by the current chunker."""
    rows = fetch_all("zryth_knowledge", "id, metadata")
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
            # The file's folder decides the agent; rows written elsewhere may lack agent_id
            agent_id = agent_from_path(src)
            if not agent_id:
                owner = (
                    _init_supabase().table("zryth_knowledge").select("agent_id")
                    .eq("metadata->>source", src).limit(1).execute().data
                )
                agent_id = owner[0].get("agent_id") if owner else None
            ingest_source(src, agent_id)
            fixed.append(src)
        except Exception:
            log.exception("Auto re-chunk failed for %s (rows left as they were)", src)
    # Uploads whose ingest never ran (agent unreachable at upload time)
    for src in missing_sources():
        try:
            ingest_source(src)
            fixed.append(src)
        except Exception:
            log.exception("Ingest of un-ingested upload %s failed", src)
    return fixed


def missing_sources() -> list[str]:
    """Bucket files (root and agent folders) that have no rows in zryth_knowledge."""
    bucket = _init_supabase().storage.from_(BUCKET)
    known = {(r.get("metadata") or {}).get("source") for r in fetch_all("zryth_knowledge", "id, metadata")}
    files: list[str] = []
    for entry in bucket.list("", {"limit": 1000}):
        if entry.get("id") is None:  # a folder
            files += [f"{entry['name']}/{f['name']}" for f in bucket.list(entry["name"], {"limit": 1000}) if f.get("id")]
        else:
            files.append(entry["name"])
    return sorted(f for f in files if f.lower().endswith(SUPPORTED_EXT) and f not in known)
