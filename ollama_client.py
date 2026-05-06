import time
import ollama
from sentence_transformers import SentenceTransformer
from config import OLLAMA_HOST, OLLAMA_MODEL, EMBEDDING_MODEL
from stats_tracker import tracker

_client = ollama.Client(host=OLLAMA_HOST)
_st_model = SentenceTransformer(EMBEDDING_MODEL)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5  # seconds; retry waits: 1.5s, 2.25s, 3.375s


def _ollama_chat(model: str, messages: list[dict], **kwargs):
    """Thin wrapper so tests can patch at this boundary."""
    return _client.chat(model=model, messages=messages, **kwargs)


def chat(phase: str, messages: list[dict], model: str = OLLAMA_MODEL, format: str = "") -> str:
    kwargs = {}
    if format:
        kwargs["format"] = format

    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.monotonic()
            response = _ollama_chat(model, messages, **kwargs)
            latency_ms = int((time.monotonic() - t0) * 1000)
            content = response.message.content
            estimated_tokens = len(content.split()) * 4 // 3  # rough approx
            tracker.record(phase, latency_ms=latency_ms, estimated_tokens=estimated_tokens)
            return content
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** attempt)

    raise last_exc


def embed(text: str) -> list[float]:
    """Return a normalized unit embedding vector using sentence-transformers."""
    vector = _st_model.encode(text, normalize_embeddings=True)
    return vector.tolist()
