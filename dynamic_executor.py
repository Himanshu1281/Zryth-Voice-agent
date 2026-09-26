import json
import requests
from typing import Any, Dict

try:
    import jsonschema
except ImportError:
    jsonschema = None


def _fill_template(value: str, args: Dict[str, Any]) -> str:
    if not isinstance(value, str):
        return value
    result = value
    for k, v in args.items():
        result = result.replace(f"{{{k}}}", str(v))
    return result


def _resolve_dict_templates(d: Dict[str, str], args: Dict[str, Any]) -> Dict[str, str]:
    return {k: _fill_template(v, args) for k, v in d.items()}


def execute_tool_call(
    tool_name: str,
    json_spec: dict,
    execution_instruction: str,
    arguments: dict
) -> dict:
    """
    Generic HTTP tool executor.

    Expected json_spec shape:
    {
      "input_schema": { ... JSON Schema ... },
      "http": {
        "method": "GET" | "POST" | ...,
        "url_template": "https://.../{placeholder}...",
        "query_params": { "param": "{arg_name}" },   # optional
        "headers": { "Header": "{arg_name}" },       # optional
        "body": { ... } | "..."                      # optional
      }
    }
    """

    # 1. Validate arguments against input_schema (if present)
    input_schema = json_spec.get("input_schema")
    if input_schema and jsonschema is not None:
        jsonschema.validate(instance=arguments, schema=input_schema)

    http_cfg = json_spec.get("http", {})
    method = http_cfg.get("method", "GET").upper()
    url_template = http_cfg.get("url_template", "")

    # 2. Build URL
    url = _fill_template(url_template, arguments)

    # 3. Query params
    params = None
    if "query_params" in http_cfg:
        params = _resolve_dict_templates(http_cfg["query_params"], arguments)

    # 4. Headers
    headers = None
    if "headers" in http_cfg:
        headers = _resolve_dict_templates(http_cfg["headers"], arguments)

    # 5. Body (for POST/PUT/etc.)
    body = None
    if "body" in http_cfg:
        raw_body = http_cfg["body"]
        if isinstance(raw_body, str):
            body = _fill_template(raw_body, arguments)
        else:
            # dict/list -> JSON string, then fill templates if needed
            body_str = json.dumps(raw_body)
            body = _fill_template(body_str, arguments)

    # 6. Perform HTTP request
    resp = requests.request(
        method=method,
        url=url,
        params=params,
        headers=headers,
        data=body if isinstance(body, str) else json.dumps(body) if body else None,
        timeout=10
    )
    resp.raise_for_status()

    # 7. Parse response
    try:
        result = resp.json()
    except Exception:
        result = resp.text

    # 8. Return structured result
    return json.dumps({
        "tool_name": tool_name,
        "execution_instruction": execution_instruction,
        "result": result
    })
