import json
from unittest.mock import patch
from phase2_content_engine import mock_searxng_search, run_content_engine
from schemas import BotPost

def test_crypto_returns_headline():
    result = mock_searxng_search.invoke({"query": "crypto bitcoin ETF"})
    assert len(result) > 10

def test_ai_returns_headline():
    result = mock_searxng_search.invoke({"query": "artificial intelligence model release"})
    assert len(result) > 10

def test_unknown_returns_fallback():
    result = mock_searxng_search.invoke({"query": "zzz_unknown_xyzzy_12345"})
    assert isinstance(result, str)
    assert len(result) > 0

def _fake_chat(phase, messages, model=None, format=""):
    if format == "json" or "JSON" in str(messages):
        return json.dumps({"score": 8, "feedback": "Great post."}) if phase == "phase2_critique" \
            else json.dumps({"bot_id": "bot_a", "topic": "AI", "post_content": "AI will eat jobs. Good."})
    return "ai model release latest development"

@patch("phase2_content_engine.chat", side_effect=_fake_chat)
@patch("phase2_content_engine.memory.get_recent_topics", return_value=[])
@patch("phase2_content_engine.memory.add_post")
def test_run_content_engine_returns_bot_post(mock_add, mock_topics, mock_chat):
    result = run_content_engine("bot_a")
    assert isinstance(result, BotPost)
    assert result.bot_id == "bot_a"
    assert len(result.post_content) <= 280

@patch("phase2_content_engine.chat", side_effect=_fake_chat)
@patch("phase2_content_engine.memory.get_recent_topics", return_value=["AI", "crypto"])
@patch("phase2_content_engine.memory.add_post")
def test_memory_topics_passed_to_llm(mock_add, mock_topics, mock_chat):
    run_content_engine("bot_b")
    # memory.get_recent_topics should have been called
    mock_topics.assert_called_once_with("bot_b", n=5)

def _low_score_then_high(phase, messages, model=None, format=""):
    """Returns score=4 twice then score=8 — triggers retry logic."""
    if not hasattr(_low_score_then_high, "_calls"):
        _low_score_then_high._calls = 0
    if phase == "phase2_critique":
        _low_score_then_high._calls += 1
        score = 4 if _low_score_then_high._calls <= 2 else 8
        return json.dumps({"score": score, "feedback": "Needs more punch."})
    return json.dumps({"bot_id": "bot_a", "topic": "AI", "post_content": "AI disrupts everything!"})

@patch("phase2_content_engine.chat", side_effect=_low_score_then_high)
@patch("phase2_content_engine.memory.get_recent_topics", return_value=[])
@patch("phase2_content_engine.memory.add_post")
def test_retry_loop_increments_retries(mock_add, mock_topics, mock_chat):
    _low_score_then_high._calls = 0
    result = run_content_engine("bot_a")
    assert isinstance(result, BotPost)
    assert result.retries >= 1
