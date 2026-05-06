import numpy as np
import faiss
from config import BOT_PERSONAS, SIMILARITY_THRESHOLD
from ollama_client import embed
from schemas import RoutingResult


def build_persona_store() -> dict:
    """
    Build an in-memory FAISS IndexFlatIP from bot persona descriptions.
    IndexFlatIP on normalized vectors = cosine similarity directly.
    Returns dict with keys 'index' (faiss.IndexFlatIP) and 'bot_ids' (list[str]).
    """
    bot_ids = list(BOT_PERSONAS.keys())
    vectors = [embed(BOT_PERSONAS[bid]["description"]) for bid in bot_ids]
    matrix = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return {"index": index, "bot_ids": bot_ids}


def route_post_to_bots(
    post_content: str,
    threshold: float = SIMILARITY_THRESHOLD,
    store: dict | None = None,
) -> list[RoutingResult]:
    """
    Embed the post and return bots with cosine similarity >= threshold.

    Args:
        post_content: The social media post text.
        threshold: Minimum cosine similarity. Default from config.
        store: Pre-built FAISS store. Built fresh if None.

    Returns:
        List of RoutingResult sorted by descending similarity.
    """
    if store is None:
        store = build_persona_store()

    query = np.array([embed(post_content)], dtype="float32")
    scores, indices = store["index"].search(query, k=len(store["bot_ids"]))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            bot_id = store["bot_ids"][idx]
            results.append(RoutingResult(
                bot_id=bot_id,
                name=BOT_PERSONAS[bot_id]["name"],
                similarity_score=float(score),
            ))

    return sorted(results, key=lambda r: r.similarity_score, reverse=True)


if __name__ == "__main__":
    store = build_persona_store()
    post = "OpenAI just released a new model that might replace junior developers."
    results = route_post_to_bots(post, store=store)
    print(f"\nPost: {post}")
    for r in results:
        print(f"  {r.name} ({r.bot_id}): {r.similarity_score:.4f}")
