# Tests

Two layers.

## Unit tests (`tests/unit`): run on every push in CI

Fast (about 7 s), no API keys, no network. They cover the rules that decide a call
without the LLM: name and number extraction, the name/number steps, language
detection, the wrong-language and repeat guards, the 3-sentence cap, goodbye and
office-hours detection, and the intent patterns. Almost every case is a regression
from a real call.

```bash
pip install -r requirements-dev.txt
python -m pytest            # or: python -m pytest -q tests/unit/test_contact_flow.py
```

## Live scenarios (`tests/scenarios`): run by hand before a release

67 scripted calls through the real agent with live Gemini, the real tools and your
knowledge base, following the same path as a spoken phone turn. Nothing is written
to Supabase. They cost API calls, so they are not part of CI.

| Suite | Count | What |
|---|---|---|
| core | 15 | bookings, callbacks, numbers, language switching, endings |
| difficult | 11 | abuse, gibberish, jailbreaks, price pushing, number chaos, trolls |
| good / confused / bad / mischief / regional / practical | 41 | Indian caller personas |

```bash
python -m tests.scenarios.run --list            # what's there
python -m tests.scenarios.run --suite core      # one suite
python -m tests.scenarios.run --quiet           # all 67, results table only (~25 min)
python -m tests.scenarios.run phone_messy       # one scenario, full transcript
```

Needs `GOOGLE_API_KEY` and the local knowledge base (`data/lancedb`, which the agent
syncs on start). The knowledge base used is `TEST_KB_AGENT_ID`, else
`DEFAULT_ORG_AGENT_ID`. Every scenario is checked automatically for: lead saved or
not, call ended, replies in the caller's language, and things Maya must never say
(prices, "I am human", political opinions, promises, invented facts). Answers vary
run to run because the LLM does; a single failure is worth reading the transcript
for before treating it as a bug.

To add a case, append a `Case(...)` to `tests/scenarios/cases.py`. To lock in a fix
cheaply, add a unit test too.
