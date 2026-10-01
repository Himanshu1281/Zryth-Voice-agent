"""Ingest knowledge files from the Supabase `knowledge_base` bucket.

    python scripts/ingest_knowledge.py                 # every .pdf/.txt in the bucket
    python scripts/ingest_knowledge.py FILE [FILE...]  # specific files

Each file's rows are replaced independently (new rows inserted BEFORE old ones
are deleted), so live calls never see an empty knowledge base and other files
are untouched. Chunking/embedding lives in knowledge_ingest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import _init_supabase  # noqa: E402
from knowledge_ingest import BUCKET, SUPPORTED_EXT, ingest_source  # noqa: E402


def main() -> None:
    files = sys.argv[1:] or [
        f["name"]
        for f in _init_supabase().storage.from_(BUCKET).list()
        if f["name"].lower().endswith(SUPPORTED_EXT)
    ]
    if not files:
        print(f"No .pdf/.txt files found in bucket '{BUCKET}'.")
        return
    failed = []
    for name in files:
        try:
            print(f"{name}: {ingest_source(name)} chunks")
        except Exception as exc:
            failed.append(name)
            print(f"{name}: FAILED ({exc})")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
