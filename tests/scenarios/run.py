"""Run the live scenarios against Gemini (costs API calls; not part of CI).

    python -m tests.scenarios.run                   # all 67
    python -m tests.scenarios.run --suite core      # one suite
    python -m tests.scenarios.run --suite bad mischief --quiet
    python -m tests.scenarios.run good_ca_english phone_messy
    python -m tests.scenarios.run --list

Needs GOOGLE_API_KEY and the local knowledge base (data/lancedb, synced by the
agent). Exit code 1 if any scenario fails.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tests.scenarios.cases import CASES  # noqa: E402
from tests.scenarios.harness import run_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names (default: all)")
    ap.add_argument("--suite", nargs="*", help="only these suites (core, difficult, good, confused, bad, mischief, regional, practical, dental, realestate, coaching, restaurant)")
    ap.add_argument("--quiet", action="store_true", help="print only the results table")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args()

    cases = CASES
    if args.suite:
        cases = [c for c in cases if c.suite in args.suite]
    if args.names:
        unknown = set(args.names) - {c.name for c in CASES}
        if unknown:
            print("unknown scenarios:", ", ".join(sorted(unknown)))
            return 2
        cases = [c for c in cases if c.name in args.names]
    if args.list:
        for c in cases:
            print(f"{c.suite:10s} {c.name:40s} {c.note}")
        return 0

    async def go():
        return [await run_case(c, verbose=not args.quiet) for c in cases]

    results = asyncio.run(go())
    print("\nRESULTS")
    for r in results:
        slow = sum(1 for t in r.transcript if t[2] > 4)
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.case.suite:10s} {r.case.name:40s} "
              f"{'; '.join(r.failures)}{f'  (slow turns: {slow})' if slow else ''}")
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
