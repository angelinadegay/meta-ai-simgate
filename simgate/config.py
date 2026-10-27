"""Configuration, read from environment variables or a .env file.

Any OpenAI-compatible endpoint works: OpenAI, OpenRouter, a local Ollama
server, and (from kickoff) the Meta Model API. If no API key is set, the
framework runs in MOCK mode with a simple keyword planner so you can test
the pipeline end to end without any model.
"""
import os
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# Model used for planning, editing and explaining (fast, structured output).
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
# Model used when the builder writes simulation code itself (BUILDER_MODE=llm).
BUILDER_MODEL = os.getenv("BUILDER_MODEL", PLANNER_MODEL)
# "template": spec -> deterministic simulation code (reliable default)
# "llm": the builder model writes/extends the simulation code, with a
#        compile-check-and-retry loop and template fallback.
BUILDER_MODE = os.getenv("BUILDER_MODE", "template")
BUILDER_MAX_RETRIES = int(os.getenv("BUILDER_MAX_RETRIES", "3"))

# Unset = automatic (omitted for Claude's endpoint, sent elsewhere); "true"/"false" forces it.
_temp = os.getenv("LLM_SEND_TEMPERATURE", "").strip().lower()
LLM_SEND_TEMPERATURE = None if _temp == "" else _temp in ("1", "true", "yes")

# Ollama and some local servers accept any key; treat a base URL alone as enough.
MOCK_MODE = not (LLM_API_KEY or LLM_BASE_URL)
