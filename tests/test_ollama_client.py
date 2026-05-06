import time
from unittest.mock import patch, MagicMock, call
from ollama_client import chat, embed

def _make_response(content: str):
    r = MagicMock()
    r.message.content = content
    return r

def test_chat_returns_string():
    with patch("ollama_client._ollama_chat", return_value=_make_response("hello")):
        result = chat("phase1", [{"role": "user", "content": "hi"}])
    assert result == "hello"

def test_chat_retries_on_exception():
    responses = [Exception("timeout"), Exception("timeout"), _make_response("ok")]
    with patch("ollama_client._ollama_chat", side_effect=responses):
        with patch("ollama_client.time.sleep"):  # skip actual sleep in tests
            result = chat("phase1", [{"role": "user", "content": "hi"}])
    assert result == "ok"

def test_chat_raises_after_max_retries():
    with patch("ollama_client._ollama_chat", side_effect=Exception("always fails")):
        with patch("ollama_client.time.sleep"):
            try:
                chat("phase1", [{"role": "user", "content": "hi"}])
                assert False, "Should have raised"
            except Exception as e:
                assert "always fails" in str(e)

def test_chat_records_stats():
    from stats_tracker import tracker
    initial = tracker.total_calls
    with patch("ollama_client._ollama_chat", return_value=_make_response("hello")):
        chat("phase_test", [{"role": "user", "content": "hi"}])
    assert tracker.total_calls == initial + 1

def test_embed_returns_normalized_floats():
    with patch("ollama_client._st_model") as mock_model:
        import numpy as np
        mock_model.encode.return_value = np.array([0.6, 0.8], dtype="float32")
        result = embed("some text")
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)
