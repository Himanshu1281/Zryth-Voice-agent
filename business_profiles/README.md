# Business profiles: one agent, any company

The agent has no company built into it. Everything that differs between businesses
comes from a `BusinessProfile` (see `business.py`), built per org agent in three layers,
later ones winning:

1. **Defaults**: neutral values that work for any business.
2. **The company's knowledge base**: one Gemini call reads the KB and fills the profile:
   what callers book ("appointment", "site visit", "table reservation", "demo class"),
   what to offer ("a free site visit" only if the KB says free), what to ask callers about,
   booking words callers use (English and Hindi), address, opening hours, brand names.
   Cached in `data/profiles/` per KB version, so it runs once per KB change.
3. **The agents row in Supabase** (business name, persona) and then
   **`business_profiles/<org_agent_id>.json`**: manual fixes on top, optional.

## Onboarding a new company

1. Create the org agent in Supabase (business name, persona, language, phone number).
2. Upload the company's knowledge base. Plain sentences work best. Include, if they apply:
   - what you offer and what callers can book;
   - prices, if Maya may quote them (otherwise she says the team will share pricing);
   - opening hours / timings (otherwise she takes a callback when asked);
   - the address, written as `Address: ...`; for several branches, one line per branch;
   - an emergency / ambulance / 24x7 number, if there is one (clinics, hospitals).
     Callers describing an emergency ("chest pain", "साँस नहीं ले पा रहे") are told to
     call it at once; without one Maya says 112.
3. That's it. The worker re-checks every KB every 2 minutes and builds the profile in
   the background, so even the first call after a KB change normally finds it ready.
4. Optional: add `business_profiles/<org_agent_id>.json` to fix anything, e.g.

```json
{
  "booking_en": "consultation",
  "offer_en": "a free consultation",
  "gender": "male",
  "persona_hi": "अर्जुन",
  "aliases": { "\\b(?:skylane|sky line)\\b": "Skyline" }
}
```

`aliases` map a speech-to-text mishearing (regex) to the brand name. Any field of
`BusinessProfile` can be set. See `2ed55165-....json` (Zryth) for a real example.

Callback numbers: only Indian 10-digit mobile numbers are accepted (landline /
international support exists in `contact_flow.clean_phone` but is switched off in
`business.ALLOWED_PHONE_TYPES`). An emergency number can be set here too:

```json
{ "emergency_number": "0141-4567890" }
```

## Check a new company before going live

Put its KB in `tests/scenarios/kbs/<name>.txt`, add a few `Case(..., kb="<name>")` to
`tests/scenarios/cases.py` (copy the dental / real-estate / coaching / restaurant ones),
and run `python -m tests.scenarios.run --suite <suite>`.
