import pytest
from pydantic import ValidationError
from schemas import RoutingResult, BotPost, PostScore, DefenseReply, LLMCallStats
from stats_tracker import StatsTracker

def test_routing_result_valid():
    r = RoutingResult(bot_id="bot_a", name="Tech Maximalist", similarity_score=0.42)
    assert r.bot_id == "bot_a"
    assert r.similarity_score == 0.42

def test_routing_result_rejects_bad_score():
    with pytest.raises(ValidationError):
        RoutingResult(bot_id="bot_a", name="Tech Maximalist", similarity_score=1.5)

def test_bot_post_enforces_280_char_limit():
    with pytest.raises(ValidationError):
        BotPost(bot_id="bot_a", topic="AI", post_content="x" * 281)

def test_bot_post_valid():
    p = BotPost(bot_id="bot_a", topic="AI", post_content="Short post.", retries=0)
    assert p.retries == 0

def test_post_score_valid():
    s = PostScore(score=8, feedback="Good post.")
    assert s.passed is True

def test_post_score_fail():
    s = PostScore(score=5, feedback="Too bland.")
    assert s.passed is False

def test_defense_reply_valid():
    d = DefenseReply(reply="Facts don't care.", injection_detected=True)
    assert d.injection_detected is True

def test_llm_call_stats():
    s = LLMCallStats(phase="phase2", latency_ms=320, estimated_tokens=210)
    assert s.phase == "phase2"

def test_stats_tracker_records_call():
    tracker = StatsTracker()
    tracker.record("phase1", latency_ms=150, estimated_tokens=80)
    assert tracker.total_calls == 1
    assert tracker.total_latency_ms == 150

def test_stats_tracker_summary():
    tracker = StatsTracker()
    tracker.record("phase1", latency_ms=100, estimated_tokens=50)
    tracker.record("phase2", latency_ms=200, estimated_tokens=100)
    summary = tracker.summary()
    assert summary["total_calls"] == 2
    assert summary["total_latency_ms"] == 300
    assert "phase1" in summary["by_phase"]
