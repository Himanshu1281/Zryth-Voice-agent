"""Re-chunk every source already in zryth_knowledge with the current chunker.

    python scripts/rechunk_knowledge.py            # dry run: print new chunks only
    python scripts/rechunk_knowledge.py --apply    # back up, then re-ingest per source

Backs up all current rows to data/backups/ first. Each source is then replaced
via knowledge_ingest.ingest_source (insert new rows before deleting old ones),
so the knowledge base is never empty. Agents refresh LanceDB via realtime sync.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database import _init_supabase  # noqa: E402
from knowledge_ingest import build_rows, ingest_source  # noqa: E402

BACKUP_DIR = ROOT / "data" / "backups"


def main() -> None:
    apply = "--apply" in sys.argv
    rows = _init_supabase().table("zryth_knowledge").select("*").order("id").execute().data
    sources = sorted({(r.get("metadata") or {}).get("source") for r in rows} - {None})
    print(f"Current rows: {len(rows)} from {len(sources)} source(s)")

    if not apply:
        for src in sources:
            new = build_rows(src)
            print(f"\n{src}: {len(new)} chunks")
            for r in new:
                c = r["content"]
                print(f"  [{r['metadata']['chunk_index']}] ({len(c)} chars) {c[:90]!r} ... {c[-50:]!r}")
        print("\nDry run only. Re-run with --apply to write to Supabase.")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"zryth_knowledge_{datetime.now():%Y%m%d_%H%M%S}.json"
    backup.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"Backed up {len(rows)} rows -> {backup}")

    for src in sources:
        print(f"{src}: {ingest_source(src)} chunks")


if __name__ == "__main__":
    main()
