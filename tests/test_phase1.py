import numpy as np
from unittest.mock import patch
from phase1_router import build_persona_store, route_post_to_bots
from schemas import RoutingResult
from benchmark import BENCHMARK_POSTS, run_benchmark

def _fake_embed(text: str) -> list[float]:
    import hashlib
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(384).astype("float32")
    v /= np.linalg.norm(v)
    return v.tolist()

@patch("phase1_router.embed", side_effect=_fake_embed)
def test_build_returns_three_entries(mock_embed):
    store = build_persona_store()
    assert store["index"].ntotal == 3
    assert len(store["bot_ids"]) == 3

@patch("phase1_router.embed", side_effect=_fake_embed)
def test_route_returns_routing_result_list(mock_embed):
    store = build_persona_store()
    results = route_post_to_bots("OpenAI just released a new model.", store=store, threshold=0.0)
    assert isinstance(results, list)
    assert all(isinstance(r, RoutingResult) for r in results)

@patch("phase1_router.embed", side_effect=_fake_embed)
def test_route_sorted_descending(mock_embed):
    store = build_persona_store()
    results = route_post_to_bots("OpenAI released a new model.", store=store, threshold=0.0)
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True)

@patch("phase1_router.embed", side_effect=_fake_embed)
def test_route_high_threshold_empty(mock_embed):
    store = build_persona_store()
    results = route_post_to_bots("OpenAI released a new model.", store=store, threshold=0.9999)
    assert results == []

@patch("phase1_router.embed", side_effect=_fake_embed)
def test_benchmark_returns_results_for_all_posts(mock_embed):
    store = build_persona_store()
    with patch("benchmark.build_persona_store", return_value=store):
        with patch("benchmark.embed", side_effect=_fake_embed):
            results = run_benchmark(store=store, threshold=0.0)
    assert len(results) == len(BENCHMARK_POSTS)
    for r in results:
        assert "post" in r
        assert "expected" in r
        assert "matched" in r
