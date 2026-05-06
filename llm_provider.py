"""
llm_provider.py — Unified LLM provider that auto-selects Groq or Ollama
based on environment variables. Exposes the same chat() interface as
ollama_client.py so all phase modules work unchanged.

Priority:
  1. GROQ_API_KEY is set  → use Groq (cloud, no local install needed)
  2. OLLAMA_HOST is set   → use that Ollama instance
  3. Neither              → default to localhost Ollama (http://127.0.0.1:11434)
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
GROQ_MODEL   = os.getenv("GROQ_MODEL",  "")

# Determine active provider once at startup
ACTIVE_PROVIDER = "groq" if GROQ_API_KEY else "ollama"
ACTIVE_MODEL    = GROQ_MODEL if ACTIVE_PROVIDER == "groq" else OLLAMA_MODEL


def get_provider_info() -> dict:
    """Return current provider info for the /health endpoint."""
    return {
        "provider": ACTIVE_PROVIDER,
        "model":    ACTIVE_MODEL,
        "ollama_host": OLLAMA_HOST if ACTIVE_PROVIDER == "ollama" else None,
    }


def chat(phase: str, messages: list[dict], model: str = "", format: str = "", api_key: str = "") -> str:
    """
    Unified chat function. Routes to Groq or Ollama based on ACTIVE_PROVIDER.

    Args:
        phase:    Label for stats tracking (e.g. "phase1", "phase2").
        messages: List of role/content dicts.
        model:    Override model name (uses provider default if empty).
        format:   Output format hint ("json" or "").
        api_key:  Per-request Groq API key (overrides env var when provided).

    Returns:
        Assistant response text string.
    """
    key = api_key or GROQ_API_KEY
    if key:
        return _chat_groq(phase, messages, model or GROQ_MODEL, format, key)
    return _chat_ollama(phase, messages, model or OLLAMA_MODEL, format)


_groq_model_cache: dict[str, str] = {}

# Preferred small-fast models in order — first available wins
_GROQ_PREFERRED = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
]

def _resolve_groq_model(client) -> str:
    """Pick the first preferred model that is currently available on Groq."""
    try:
        available = {m.id for m in client.models.list().data}
        for m in _GROQ_PREFERRED:
            if m in available:
                return m
        # fallback: first model returned by the API
        return next(iter(available))
    except Exception:
        return "llama-3.1-8b-instant"


def _chat_groq(phase: str, messages: list[dict], model: str, format: str, api_key: str) -> str:
    """Send request to Groq API using openai-compatible client."""
    from groq import Groq
    from stats_tracker import tracker
    import time

    client = Groq(api_key=api_key)
    if not model:
        if api_key not in _groq_model_cache:
            _groq_model_cache[api_key] = _resolve_groq_model(client)
        model = _groq_model_cache[api_key]

    # If JSON format requested, append explicit instruction to system prompt
    if format == "json":
        for msg in messages:
            if msg["role"] == "system":
                msg = dict(msg)  # don't mutate original
                break
        messages = list(messages)
        messages[0] = dict(messages[0])
        if "json" not in messages[0].get("content", "").lower():
            messages[0]["content"] += "\n\nIMPORTANT: Reply with valid JSON only."

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,
        max_tokens=512,
        response_format={"type": "json_object"} if format == "json" else {"type": "text"},
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    content = response.choices[0].message.content or ""
    tracker.record(phase, latency_ms=latency_ms, estimated_tokens=len(content.split()) * 4 // 3)
    return content


def _chat_ollama(phase: str, messages: list[dict], model: str, format: str) -> str:
    """Send request to local Ollama server."""
    from ollama_client import chat as ollama_chat
    return ollama_chat(phase, messages, model=model, format=format)
