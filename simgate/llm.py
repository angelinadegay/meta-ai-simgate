"""Thin wrapper over any OpenAI-compatible chat endpoint."""
import json
import re

from . import config

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            base_url=config.LLM_BASE_URL or None,
            api_key=config.LLM_API_KEY or "not-needed",
        )
    return _client


def sends_temperature() -> bool:
    """Claude's OpenAI-compatible endpoint rejects temperature < 1 on current models, so leave it out there.
    LLM_SEND_TEMPERATURE=false turns it off for other providers that object to it."""
    if config.LLM_SEND_TEMPERATURE is not None:
        return config.LLM_SEND_TEMPERATURE
    return "anthropic.com" not in config.LLM_BASE_URL


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2) -> str:
    kwargs = {"model": model or config.PLANNER_MODEL, "messages": messages}
    if sends_temperature():
        kwargs["temperature"] = temperature
    from . import cost

    cost.add_model_call()
    resp = client().chat.completions.create(**kwargs)
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
    shows the model its own reply and the exact error. Returns (parsed or None, one record per call)."""
    messages = list(messages)
    calls: list[dict] = []
    for attempt in range(2):
        reply = chat(messages, model)
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
