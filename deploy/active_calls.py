"""Print how many calls are in progress (used by deploy/update.sh before a restart).

A call in progress has no ended_at yet. Only calls started in the last 20 minutes
count, so a call that crashed without finishing never blocks a deploy forever.
Prints -1 if Supabase can't be reached (update.sh then refuses to restart).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WINDOW = timedelta(minutes=20)


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        from database import _init_supabase

        since = (datetime.now(timezone.utc) - WINDOW).isoformat()
        rows = (
            _init_supabase().table("calls").select("id")
            .is_("ended_at", "null").gte("started_at", since).execute().data
        )
        print(len(rows))
    except Exception as e:  # noqa: BLE001 - any failure means "don't know": don't restart
        print(-1)
        print(f"active_calls: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
