"""Thin wrapper over the Meta Model API's chat endpoint (OpenAI-compatible, so the openai client library calls it).
Every call is checked to be a Meta model."""
import json
import re
import time

from . import config

_client = None
_down_until = 0.0   # after a failed call, later calls fail fast for a while instead of each waiting out a timeout
DOWN_FOR_S = 60.0


class ModelUnavailable(RuntimeError):
    """The Meta Model API can't answer (mock mode, timeout, error). Callers fall back to rules for that step."""


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            base_url=config.LLM_BASE_URL or None,
            api_key=config.LLM_API_KEY or "not-needed",
            timeout=CALL_TIMEOUT_S, max_retries=1,   # a stalled call falls back to rules instead of hanging a step
        )
    return _client


CALL_TIMEOUT_S = 90.0
CHECK_TIMEOUT_S = 15.0


def check_key() -> str:
    """Check the key once at startup with the cheapest call there is (listing models: no tokens), and set the mode.
    A missing or invalid key never crashes the app: it switches to mock mode and says why."""
    if config.FORCE_MOCK:
        config.MOCK_MODE, config.MODE_STATUS = True, "SIMGATE_MOCK_MODE=1. Running in mock mode."
        return config.MODE_STATUS
    if not config.LLM_API_KEY:
        config.MOCK_MODE, config.MODE_STATUS = True, "Key missing/invalid: LLM_API_KEY is not set. Running in mock mode."
        return config.MODE_STATUS
    try:
        listed = {model.id for model in client().with_options(timeout=CHECK_TIMEOUT_S, max_retries=0).models.list()}
        missing = [name for name in (config.PLANNER_MODEL, config.BUILDER_MODEL) if name not in listed]
        reason = f"the key can't use {', '.join(missing)}" if missing else None
    except Exception as exc:   # bad key, no network, wrong URL: all mean "no model", none crash the app
        status = getattr(exc, "status_code", None)
        reason = ("the Meta Model API rejected the key (401 Unauthorized)" if status == 401
                  else f"couldn't reach the Meta Model API ({type(exc).__name__}{f' {status}' if status else ''})")
    if reason:
        config.MOCK_MODE, config.MODE_STATUS = True, f"Key missing/invalid: {reason}. Running in mock mode."
    else:
        config.MOCK_MODE = False
        config.MODE_STATUS = f"Connected to Meta Model API ({config.PLANNER_MODEL}, {config.BUILDER_MODEL})"
    return config.MODE_STATUS


def sends_temperature() -> bool:
    """LLM_SEND_TEMPERATURE=false leaves temperature out for an endpoint that objects to it."""
    return config.LLM_SEND_TEMPERATURE is not False


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2) -> str:
    """One chat call. Raises ModelUnavailable (never an API error) when there's no model to answer: in mock mode
    without calling anything, or when the API times out or fails, which is recorded for the request's cost line."""
    global _down_until
    kwargs = {"model": model or config.PLANNER_MODEL, "messages": messages}
    if not config.DEV_ALLOW_OTHER_MODELS and not config.is_meta_model(kwargs["model"]):
        raise ValueError(f"Refusing to call {kwargs['model']!r}: SimGate only calls Meta models.")
    if config.MOCK_MODE:   # mock mode makes zero API calls; callers take their rule-based path on this error
        raise ModelUnavailable("mock mode")
    if sends_temperature():
        kwargs["temperature"] = temperature
    from . import cost

    if time.monotonic() < _down_until:
        reason = "the Meta Model API failed moments ago"
        cost.add_model_failure(reason)
        raise ModelUnavailable(reason)
    cost.add_model_call()
    try:
        resp = client().chat.completions.create(**kwargs)
    except Exception as exc:   # timeout, connection, 4xx/5xx: this step uses rules instead
        status = getattr(exc, "status_code", None)
        reason = f"{type(exc).__name__}{f' {status}' if status else ''}"
        _down_until = time.monotonic() + DOWN_FOR_S
        cost.add_model_failure(reason)
        raise ModelUnavailable(reason) from exc
    return resp.choices[0].message.content or ""


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ``` fences and prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model reply.")
        candidate = text[start : end + 1]
    return json.loads(candidate)


def extract_code(text: str, lang: str = "xml") -> str:
    fenced = re.search(rf"```(?:{lang})?\s*(.*?)```", text, re.S)
    return (fenced.group(1) if fenced else text).strip()


def chat_json(messages: list[dict], model: str, parse) -> tuple[object | None, list[dict]]:
    """Ask for JSON and parse it with ``parse`` (which raises ValueError/TypeError on a bad reply). One retry
    shows the model its own reply and the exact error. Returns (parsed or None, one record per call); None also when
    the model is unavailable, so every caller's rules fallback covers an outage too."""
    messages = list(messages)
    calls: list[dict] = []
    for attempt in range(2):
        try:
            reply = chat(messages, model)
        except ModelUnavailable as exc:   # the caller's "no valid reply" path is the rules
            calls.append({"attempt": attempt + 1, "valid": False, "error": f"model unavailable: {exc}"})
            return None, calls
        try:
            parsed = parse(extract_json(reply))
        except (ValueError, TypeError) as exc:
            calls.append({"attempt": attempt + 1, "valid": False, "error": str(exc)[:500]})
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"That reply failed validation with this exact error:\n{exc}\nReturn corrected JSON only."})
            continue
        calls.append({"attempt": attempt + 1, "valid": True})
        return parsed, calls
    return None, calls
