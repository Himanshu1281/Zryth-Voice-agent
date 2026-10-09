"""Which agent answers a call: dialled number -> phone_numbers -> agents -> agent_templates.

One worker pool serves every organization. At call start the entrypoint reads the
dialled number from the SIP participant (`sip.trunkPhoneNumber`) and resolves it here.

  * number bound to an org agent  -> that agent's business name, greeting, language,
                                     transfer number and knowledge base (agent_id)
  * number in phone_numbers but not bound/active -> blocked (call is dropped)
  * number unknown / not presented (console and dev rooms) -> the agent in
    DEFAULT_ORG_AGENT_ID (Zryth's own), or the bare template if that isn't set
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, replace

from config import DEFAULT_LANGUAGE, DEFAULT_TRANSFER_NUMBER, SUPPORTED_LANGUAGES

log = logging.getLogger("voice-agent.routing")

DEFAULT_TEMPLATE_ID = "maya_v2"
_CACHE_TTL_S = 30.0  # agent edits (greeting etc.) reach live calls within this window


@dataclass(frozen=True)
class CallConfig:
    template_id: str = DEFAULT_TEMPLATE_ID
    org_id: str | None = None
    org_agent_id: str | None = None      # None = no org agent (knowledge rows without agent_id)
    business_name: str = "our company"  # real name comes from the org agent row
    persona_name: str = "Maya"
    greeting: str | None = None          # None = template greeting
    transfer_number: str | None = DEFAULT_TRANSFER_NUMBER
    language: str = DEFAULT_LANGUAGE
    dialed_number: str | None = None
    blocked: bool = False                # our number, but not connected to an agent


DEFAULT_CONFIG = CallConfig()
# Org agent that answers calls to numbers not in phone_numbers (Zryth's own agent)
DEFAULT_ORG_AGENT_ID = os.getenv("DEFAULT_ORG_AGENT_ID", "").strip() or None
_AGENT_COLS = "id, org_id, template_id, business_name, greeting, transfer_number, language, agent_templates(persona_name)"

_cache: dict[str, tuple[float, CallConfig]] = {}
_cache_lock = threading.Lock()


def to_e164(number: str | None) -> str | None:
    """Vobiz presents the dialled number as 0XXXXXXXXXX / XXXXXXXXXX / 91... / +91..."""
    d = re.sub(r"\D", "", number or "")
    if len(d) == 11 and d.startswith("0"):
        d = d[1:]
    if len(d) == 10:
        d = "91" + d
    return f"+{d}" if re.fullmatch(r"91\d{10}", d) else None


def _from_agent(agent: dict, org_id: str | None, e164: str | None) -> CallConfig:
    lang = agent.get("language") or DEFAULT_LANGUAGE
    return CallConfig(
        template_id=agent["template_id"],
        org_id=org_id,
        org_agent_id=agent["id"],
        business_name=agent.get("business_name") or "our company",
        persona_name=(agent.get("agent_templates") or {}).get("persona_name") or "Maya",
        greeting=agent.get("greeting") or None,
        transfer_number=agent.get("transfer_number") or None,
        language=lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE,
        dialed_number=e164,
    )


def _cached(key: str) -> CallConfig | None:
    with _cache_lock:
        hit = _cache.get(key)
        return hit[1] if hit and time.monotonic() - hit[0] < _CACHE_TTL_S else None


def _store(key: str, cfg: CallConfig) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), cfg)
        if len(_cache) > 2048:
            _cache.pop(next(iter(_cache)))


def _default_config(dialed: str | None) -> CallConfig:
    """Config for numbers not in phone_numbers: DEFAULT_ORG_AGENT_ID, else the bare template."""
    if not DEFAULT_ORG_AGENT_ID:
        return replace(DEFAULT_CONFIG, dialed_number=dialed)
    cfg = _cached("default")
    if cfg is None:
        try:
            from database import _init_supabase

            rows = (_init_supabase().table("agents").select(_AGENT_COLS)
                    .eq("id", DEFAULT_ORG_AGENT_ID).limit(1).execute().data)
        except Exception:
            log.exception("Default agent lookup failed; using the bare template")
            return replace(DEFAULT_CONFIG, dialed_number=dialed)
        if not rows:
            log.error("DEFAULT_ORG_AGENT_ID %s not found; using the bare template", DEFAULT_ORG_AGENT_ID)
            return replace(DEFAULT_CONFIG, dialed_number=dialed)
        cfg = _from_agent(rows[0], rows[0]["org_id"], None)
        _store("default", cfg)
    return replace(cfg, dialed_number=dialed)


def resolve_call_config(dialed: str | None) -> CallConfig:
    """Blocking (run in a thread). Never raises: any failure falls back to the default agent."""
    e164 = to_e164(dialed)
    if not e164:
        return _default_config(dialed)

    hit = _cached(e164)
    if hit:
        return hit

    try:
        from database import _init_supabase

        rows = (
            _init_supabase()
            .table("phone_numbers")
            .select(f"org_id, status, agent_id, agents({_AGENT_COLS})")
            .eq("e164", e164)
            .limit(1)
            .execute()
            .data
        )
    except Exception:
        log.exception("Number lookup failed for %s; using the default agent", e164)
        return _default_config(e164)

    if not rows:
        cfg = _default_config(e164)
    else:
        row = rows[0]
        agent = row.get("agents")
        if row.get("status") != "active" or not agent:
            cfg = replace(DEFAULT_CONFIG, dialed_number=e164, org_id=row.get("org_id"), blocked=True)
        else:
            cfg = _from_agent(agent, row["org_id"], e164)

    _store(e164, cfg)
    log.info("number %s -> org=%s agent=%s blocked=%s", e164, cfg.org_id, cfg.org_agent_id, cfg.blocked)
    return cfg
