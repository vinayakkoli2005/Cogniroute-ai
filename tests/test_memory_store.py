import pytest
import tempfile
import os
from memory_store import BotMemoryStore

@pytest.fixture
def tmp_store():
    import tempfile
    import shutil
    tmpdir = tempfile.mkdtemp()
    yield BotMemoryStore(persist_path=tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)

def test_add_and_retrieve_post(tmp_store):
    tmp_store.add_post("bot_a", "AI is taking over jobs.", topic="AI")
    topics = tmp_store.get_recent_topics("bot_a", n=5)
    assert "AI" in topics

def test_empty_store_returns_empty_list(tmp_store):
    topics = tmp_store.get_recent_topics("bot_a", n=5)
    assert topics == []

def test_multiple_bots_isolated(tmp_store):
    tmp_store.add_post("bot_a", "AI post", topic="AI")
    tmp_store.add_post("bot_b", "Finance post", topic="Finance")
    topics_a = tmp_store.get_recent_topics("bot_a", n=5)
    topics_b = tmp_store.get_recent_topics("bot_b", n=5)
    assert "AI" in topics_a
    assert "Finance" not in topics_a
    assert "Finance" in topics_b

def test_deduplication_no_exact_repeat(tmp_store):
    tmp_store.add_post("bot_a", "AI is taking over.", topic="AI")
    tmp_store.add_post("bot_a", "AI is taking over.", topic="AI")
    topics = tmp_store.get_recent_topics("bot_a", n=10)
    assert topics.count("AI") == 1  # deduped by topic name
