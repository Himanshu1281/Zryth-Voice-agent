"""Generic HTTP tool executor for tools defined on the dashboard's Tools page.

Arguments come from the LLM (and so, indirectly, from what the caller says), so they
are always escaped for the place they land in and can never change the request's host.
"""

import json
import re
from typing import Any, Dict
from urllib.parse import quote, urlsplit

import requests

try:
    import jsonschema
except ImportError:
    jsonschema = None

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _fill(value: str, args: Dict[str, Any], encode=lambda s: s) -> str:
    """Replace {name} with the argument (unknown placeholders are left as they are)."""
    if not isinstance(value, str):
        return value
    return _PLACEHOLDER.sub(lambda m: encode(str(args[m.group(1)])) if m.group(1) in args else m.group(0), value)


def _fill_json(node: Any, args: Dict[str, Any]) -> Any:
    """Fill placeholders inside a JSON body structurally (no string splicing into JSON).
    A string that is exactly "{name}" takes the argument's own type (number, bool...)."""
    if isinstance(node, dict):
        return {k: _fill_json(v, args) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill_json(v, args) for v in node]
    if isinstance(node, str):
        m = _PLACEHOLDER.fullmatch(node)
        if m and m.group(1) in args:
            return args[m.group(1)]
        return _fill(node, args)
    return node


def _safe_header(v: str) -> str:
    return v.replace("\r", " ").replace("\n", " ")


def is_executable(json_spec: dict) -> bool:
    """True when the spec has the shape this executor runs (https URL with a fixed host)."""
    http = (json_spec or {}).get("http") or {}
    host = urlsplit(http.get("url_template") or "").netloc
    return (http.get("url_template") or "").startswith("https://") and bool(host) and "{" not in host


def execute_tool_call(tool_name: str, json_spec: dict, execution_instruction: str, arguments: dict) -> str:
    """Run one tool call and return a JSON string for the LLM.

    json_spec:
    {
      "input_schema": { ... JSON Schema ... },
      "http": {
        "method": "GET" | "POST" | ...,
        "url_template": "https://fixed.host/path/{placeholder}",
        "query_params": { "param": "{arg_name}" },   # optional
        "headers": { "Header": "{arg_name}" },       # optional
        "body": { ... } | "..."                      # optional
      }
    }
    """
    if not is_executable(json_spec):
        raise ValueError(f"tool {tool_name!r} has no runnable https endpoint")
    arguments = arguments or {}

    input_schema = json_spec.get("input_schema")
    if input_schema and jsonschema is not None:
        jsonschema.validate(instance=arguments, schema=input_schema)

    http_cfg = json_spec["http"]
    method = http_cfg.get("method", "GET").upper()
    template = http_cfg["url_template"]

    # Path/query placeholders are percent-encoded; the host was checked to be fixed
    url = _fill(template, arguments, encode=lambda s: quote(s, safe=""))
    if urlsplit(url).netloc != urlsplit(template).netloc:
        raise ValueError(f"tool {tool_name!r}: arguments may not change the host")

    params = {k: _fill(v, arguments) for k, v in http_cfg.get("query_params", {}).items()} or None
    headers = {k: _safe_header(_fill(v, arguments)) for k, v in http_cfg.get("headers", {}).items()} or None

    data = None
    json_body = None
    if "body" in http_cfg:
        raw = http_cfg["body"]
        if isinstance(raw, (dict, list)):
            json_body = _fill_json(raw, arguments)
        else:
            data = _fill(str(raw), arguments)

    resp = requests.request(
        method=method,
        url=url,
        params=params,
        headers=headers,
        json=json_body,
        data=data,
        timeout=5,  # caller waits in silence during this call
        allow_redirects=False,
    )
    resp.raise_for_status()

    try:
        result = resp.json()
    except Exception:
        result = resp.text[:4000]

    return json.dumps({"tool_name": tool_name, "execution_instruction": execution_instruction, "result": result})
