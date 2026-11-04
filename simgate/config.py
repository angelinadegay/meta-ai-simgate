"""Configuration, read from environment variables or a .env file.

Every model SimGate calls at runtime is a Meta model, served by the Meta Model API: Muse Spark plans and explains,
Muse Wonder proposes robot designs and fix strategies. Any other model name is rejected at startup
(SIMGATE_DEV_ALLOW_OTHER_MODELS=1 lifts that for local development only; it is never used for the submission).
If no valid API key is set (or SIMGATE_MOCK_MODE=1), SimGate runs in MOCK mode: rule-based fallbacks (code, not
models) plan, design and explain, with zero API calls.
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

META_BASE_URL = "https://api.meta.ai/v1"
META_MODEL_PREFIX = "muse-"   # Meta's Muse family: muse-spark-*, muse-wonder-*
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or META_BASE_URL
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# Muse Spark: planning, editing and explaining (fast, structured output).
PLANNER_MODEL = os.getenv("PLANNER_MODEL") or "muse-spark-1.3"
# Muse Wonder: robot designs and fix strategies, and simulation code when BUILDER_MODE=llm.
BUILDER_MODEL = os.getenv("BUILDER_MODEL") or "muse-wonder-1.0"
DEV_ALLOW_OTHER_MODELS = os.getenv("SIMGATE_DEV_ALLOW_OTHER_MODELS", "") == "1"


def is_meta_model(name: str) -> bool:
    return isinstance(name, str) and name.startswith(META_MODEL_PREFIX)


def check_models() -> None:
    """Refuse to start with a model that isn't Meta's (unless the dev-only override is set)."""
    if DEV_ALLOW_OTHER_MODELS:
        return
    others = [f"{key}={name!r}" for key, name in (("PLANNER_MODEL", PLANNER_MODEL), ("BUILDER_MODEL", BUILDER_MODEL))
              if not is_meta_model(name)]
    if others:
        raise RuntimeError(f"SimGate only runs Meta models ({META_MODEL_PREFIX}*): {', '.join(others)} isn't one. "
                           "Set SIMGATE_DEV_ALLOW_OTHER_MODELS=1 for local development only.")


check_models()
# "template": spec -> deterministic simulation code (reliable default)
# "llm": the builder model writes/extends the simulation code, with a
#        compile-check-and-retry loop and template fallback.
BUILDER_MODE = os.getenv("BUILDER_MODE", "template")
BUILDER_MAX_RETRIES = int(os.getenv("BUILDER_MAX_RETRIES", "3"))

# Unset = sent; "false" leaves `temperature` out of requests to an endpoint that rejects it.
_temp = os.getenv("LLM_SEND_TEMPERATURE", "").strip().lower()
LLM_SEND_TEMPERATURE = None if _temp == "" else _temp in ("1", "true", "yes")

# Mock mode: rule-based planner, designer and explanations, zero API calls. On with SIMGATE_MOCK_MODE=1, when no
# key is set, or (after startup) when the key check fails. MODE_STATUS is the one-line status shown at startup.
FORCE_MOCK = os.getenv("SIMGATE_MOCK_MODE", "0").strip().lower() in ("1", "true", "yes")
MOCK_MODE = FORCE_MOCK or not LLM_API_KEY
MODE_STATUS = ""


def mode_label() -> str:
    """The UI header's mode line."""
    return "Mock mode: rules, no model" if MOCK_MODE else f"Meta models: {PLANNER_MODEL} · {BUILDER_MODEL}"
