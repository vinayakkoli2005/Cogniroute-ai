"""
benchmark.py — Measures Phase 1 routing accuracy against 10 labeled posts.
Prints a per-bot precision/recall table and an ASCII confusion matrix.

Run: python benchmark.py
"""
from phase1_router import build_persona_store, route_post_to_bots
from schemas import RoutingResult
from config import SIMILARITY_THRESHOLD
from ollama_client import embed

# 10 labeled posts: each has expected bot IDs that should match
BENCHMARK_POSTS = [
    {"post": "OpenAI GPT-5 will make every software engineer obsolete by 2026.",
     "expected": ["bot_a", "bot_b"]},
    {"post": "Elon Musk's Neuralink just got FDA approval for human brain chips.",
     "expected": ["bot_a"]},
    {"post": "AI regulation in the EU is the biggest threat to innovation this decade.",
     "expected": ["bot_a", "bot_b"]},
    {"post": "S&P 500 hits all-time high; options traders are printing money.",
     "expected": ["bot_c"]},
    {"post": "Fed rate cut signals a bull run — time to go long on tech ETFs.",
     "expected": ["bot_c"]},
    {"post": "Quant hedge funds up 22% YTD — human fund managers are irrelevant.",
     "expected": ["bot_c"]},
    {"post": "Big Tech layoffs expose how surveillance capitalism exploits workers.",
     "expected": ["bot_b"]},
    {"post": "Meta's data collection is a human rights violation dressed as a product.",
     "expected": ["bot_b"]},
    {"post": "Bitcoin ETF approval sends crypto market cap past $3 trillion.",
     "expected": ["bot_a", "bot_c"]},
    {"post": "SpaceX Starship completes first commercial Mars cargo mission.",
     "expected": ["bot_a"]},
]


def run_benchmark(
    store: dict | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Run all benchmark posts and return result dicts."""
    if store is None:
        store = build_persona_store()
    results = []
    for item in BENCHMARK_POSTS:
        matched = route_post_to_bots(item["post"], threshold=threshold, store=store)
        results.append({
            "post": item["post"],
            "expected": item["expected"],
            "matched": [r.bot_id for r in matched],
            "scores": {r.bot_id: round(r.similarity_score, 3) for r in matched},
        })
    return results


def print_report(results: list[dict]) -> None:
    bot_ids = ["bot_a", "bot_b", "bot_c"]
    bot_names = {"bot_a": "Tech Maximalist", "bot_b": "Doomer/Skeptic", "bot_c": "Finance Bro"}

    print("\n" + "="*70)
    print("PHASE 1 ROUTING BENCHMARK")
    print("="*70)

    # Per-post routing table
    for r in results:
        status = "HIT" if set(r["expected"]) & set(r["matched"]) else "MISS"
        print(f"\n{status} Post: {r['post'][:65]}...")
        print(f"  Expected : {r['expected']}")
        print(f"  Matched  : {r['matched']}")
        print(f"  Scores   : {r['scores']}")

    # Per-bot recall table
    print("\n" + "-"*70)
    print(f"{'Bot':<20} {'Expected':>10} {'Matched':>10} {'Recall':>10}")
    print("-"*70)
    for bot_id in bot_ids:
        expected_posts = [r for r in results if bot_id in r["expected"]]
        matched_posts  = [r for r in expected_posts if bot_id in r["matched"]]
        recall = len(matched_posts) / len(expected_posts) if expected_posts else 0.0
        print(f"{bot_names[bot_id]:<20} {len(expected_posts):>10} {len(matched_posts):>10} {recall:>9.0%}")

    # ASCII confusion matrix (expected rows × matched cols)
    print("\n" + "-"*70)
    print("CONFUSION MATRIX  (rows=expected, cols=matched)")
    print(f"{'':>18}", end="")
    for b in bot_ids:
        print(f"{b:>10}", end="")
    print()
    for row_bot in bot_ids:
        print(f"{bot_names[row_bot]:<18}", end="")
        for col_bot in bot_ids:
            count = sum(
                1 for r in results
                if row_bot in r["expected"] and col_bot in r["matched"]
            )
            print(f"{count:>10}", end="")
        print()
    print("="*70 + "\n")


if __name__ == "__main__":
    store = build_persona_store()
    results = run_benchmark(store=store)
    print_report(results)
