"""Ingest only bucket files that have no rows in zryth_knowledge yet.

    python scripts/sync_storage_knowledge.py

Uses the database itself (not a local log file) to decide what is new, so it
behaves the same on every machine / EC2 host. Chunking/embedding lives in
knowledge_ingest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import _init_supabase  # noqa: E402
from knowledge_ingest import BUCKET, SUPPORTED_EXT, ingest_source  # noqa: E402


def main() -> None:
    sb = _init_supabase()
    ingested = {
        (r.get("metadata") or {}).get("source")
        for r in sb.table("zryth_knowledge").select("metadata").execute().data
    }
    new_files = [
        f["name"]
        for f in sb.storage.from_(BUCKET).list()
        if f["name"].lower().endswith(SUPPORTED_EXT) and f["name"] not in ingested
    ]
    if not new_files:
        print("No new knowledge files to ingest.")
        return
    for name in new_files:
        try:
            print(f"{name}: {ingest_source(name)} chunks")
        except Exception as exc:
            print(f"{name}: FAILED ({exc})")


if __name__ == "__main__":
    main()
