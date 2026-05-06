import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import BOT_PERSONAS, OLLAMA_MODEL, CRITIQUE_PASS_SCORE, MAX_CRITIQUE_RETRIES
import llm_provider
from memory_store import memory
from schemas import BotPost, PostScore
from langchain_core.tools import tool

# per-request api_key injected by run_content_engine before graph execution
_request_api_key: str = ""

def chat(phase, messages, model="", format=""):
    return llm_provider.chat(phase, messages, model=model, format=format, api_key=_request_api_key)

_MOCK_HEADLINES = {
    "crypto":            "Bitcoin hits new all-time high amid regulatory ETF approvals — institutional demand surges.",
    "bitcoin":           "Bitcoin hits new all-time high amid regulatory ETF approvals — institutional demand surges.",
    "ai":                "OpenAI launches GPT-5 with autonomous agent capabilities; rivals scramble to respond.",
    "artificial intelligence": "OpenAI launches GPT-5 with autonomous agent capabilities; rivals scramble to respond.",
    "elon":              "Elon Musk's xAI raises $6B, targets Mars colony logistics AI by 2027.",
    "space":             "SpaceX Starship completes first commercial payload delivery to lunar orbit.",
    "stock":             "S&P 500 closes at record high as Fed signals two rate cuts before year-end.",
    "market":            "S&P 500 closes at record high as Fed signals two rate cuts before year-end.",
    "interest rate":     "Fed holds rates steady; analysts predict 25bps cut in Q3 2026.",
    "trading":           "Quant funds outperform benchmarks by 18% YTD using momentum-reversion hybrid.",
    "regulation":        "EU AI Act enforcement begins; 12 major tech firms face compliance audits.",
    "privacy":           "Meta fined €1.2B under GDPR for cross-border data transfers to US servers.",
    "climate":           "IPCC report: carbon capture technology 10 years behind schedule.",
    "ev":                "Tesla reports 40% battery degradation improvement in new 4680 cell design.",
    "electric vehicle":  "Tesla reports 40% battery degradation improvement in new 4680 cell design.",
    "layoff":            "Big Tech layoffs continue — 45,000 workers cut in Q1 2026 across FAANG.",
    "rate":              "Fed holds rates steady; analysts predict 25bps cut in Q3 2026.",
}

_FALLBACK = "Breaking: Global markets react to new geopolitical developments; analysts divided on impact."


@tool
def mock_searxng_search(query: str) -> str:
    """Search for recent news headlines relevant to the query. Returns one headline string."""
    q = query.lower()
    for keyword, headline in _MOCK_HEADLINES.items():
        if keyword in q:
            return headline
    return _FALLBACK


class ContentState(TypedDict):
    bot_id: str
    persona_description: str
    recent_topics: list[str]
    search_query: str
    search_results: str
    draft_content: str
    draft_topic: str
    critique: PostScore | None
    retries: int
    final_post: BotPost | None


# ── Node 1: Check Memory ──────────────────────────────────────────────────────

def check_memory_node(state: ContentState) -> dict:
    topics = memory.get_recent_topics(state["bot_id"], n=5)
    return {"recent_topics": topics}


# ── Node 2: Decide Search Query ───────────────────────────────────────────────

def decide_search_node(state: ContentState) -> dict:
    avoid = ", ".join(state["recent_topics"]) if state["recent_topics"] else "none"
    system = (
        f"You are a social media bot. Persona:\n{state['persona_description']}\n\n"
        f"Topics you have recently posted about (AVOID these): {avoid}\n\n"
        "Decide a NEW topic to post about today. "
        "Reply with ONLY a short search query (5–10 words, no punctuation)."
    )
    query = chat("phase2", [
        {"role": "system", "content": system},
        {"role": "user", "content": "What should I search for today?"},
    ]).strip().strip('"').strip("'")
    return {"search_query": query}


# ── Node 3: Web Search ────────────────────────────────────────────────────────

def web_search_node(state: ContentState) -> dict:
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    return {"search_results": results}


# ── Node 4: Draft Post ────────────────────────────────────────────────────────

def draft_post_node(state: ContentState) -> dict:
    feedback_line = ""
    if state["critique"] is not None and state["retries"] > 0:
        feedback_line = f"\nPrevious attempt was scored {state['critique'].score}/10. Feedback: {state['critique'].feedback}. Improve on this."

    system = (
        f"You are a social media bot. Persona:\n{state['persona_description']}\n\n"
        f"News context: {state['search_results']}{feedback_line}\n\n"
        "Write a highly opinionated post under 280 characters.\n"
        "Reply with ONLY this JSON (no markdown, no fences):\n"
        '{"bot_id": "<bot_id>", "topic": "<1-3 word topic>", "post_content": "<post under 280 chars>"}'
    )
    raw = chat("phase2", [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Draft a post. bot_id={state['bot_id']}"},
    ], format="json").strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return {
        "draft_content": data.get("post_content", "")[:280],
        "draft_topic": data.get("topic", state["search_query"][:20]),
    }


# ── Node 5: Self-Critique ─────────────────────────────────────────────────────

def self_critique_node(state: ContentState) -> dict:
    system = (
        f"You are a quality judge for social media posts.\n"
        f"Bot persona: {state['persona_description']}\n\n"
        f"Score this post 1–10 on: persona alignment, opinionatedness, length ≤ 280 chars, topic freshness.\n"
        f"Post: \"{state['draft_content']}\"\n\n"
        "Reply with ONLY this JSON:\n"
        '{"score": <int 1-10>, "feedback": "<one sentence>"}'
    )
    raw = chat("phase2_critique", [
        {"role": "system", "content": system},
        {"role": "user", "content": "Score this post."},
    ], format="json").strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    critique = PostScore(score=int(data.get("score", 7)), feedback=data.get("feedback", ""))
    return {"critique": critique}


# ── Node 6: Publish or Retry ──────────────────────────────────────────────────

def publish_or_retry_node(state: ContentState) -> dict:
    critique: PostScore = state["critique"]
    retries = state["retries"]

    if critique.passed or retries >= MAX_CRITIQUE_RETRIES:
        post = BotPost(
            bot_id=state["bot_id"],
            topic=state["draft_topic"],
            post_content=state["draft_content"][:280],
            retries=retries,
        )
        memory.add_post(state["bot_id"], post.post_content, topic=post.topic)
        return {"final_post": post}

    return {"retries": retries + 1}


def _route_after_critique(state: ContentState) -> str:
    critique: PostScore = state["critique"]
    if critique.passed or state["retries"] >= MAX_CRITIQUE_RETRIES:
        return END
    return "draft_post"


# ── Build Graph ───────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(ContentState)
    g.add_node("check_memory",      check_memory_node)
    g.add_node("decide_search",     decide_search_node)
    g.add_node("web_search",        web_search_node)
    g.add_node("draft_post",        draft_post_node)
    g.add_node("self_critique",     self_critique_node)
    g.add_node("publish_or_retry",  publish_or_retry_node)

    g.set_entry_point("check_memory")
    g.add_edge("check_memory",     "decide_search")
    g.add_edge("decide_search",    "web_search")
    g.add_edge("web_search",       "draft_post")
    g.add_edge("draft_post",       "self_critique")
    g.add_edge("self_critique",    "publish_or_retry")
    g.add_conditional_edges("publish_or_retry", _route_after_critique, {"draft_post": "draft_post", END: END})

    return g.compile()


_content_graph = _build_graph()


def run_content_engine(bot_id: str, api_key: str = "") -> BotPost:
    """
    Run the 6-node content engine for a bot. Returns a validated BotPost.
    Graph: check_memory → decide_search → web_search → draft_post → self_critique → publish_or_retry
    With retry loop back to draft_post if score < CRITIQUE_PASS_SCORE and retries < MAX_CRITIQUE_RETRIES.
    """
    global _request_api_key
    _request_api_key = api_key
    persona = BOT_PERSONAS[bot_id]
    initial: ContentState = {
        "bot_id": bot_id,
        "persona_description": persona["description"],
        "recent_topics": [],
        "search_query": "",
        "search_results": "",
        "draft_content": "",
        "draft_topic": "",
        "critique": None,
        "retries": 0,
        "final_post": None,
    }
    final = _content_graph.invoke(initial)
    return final["final_post"]


if __name__ == "__main__":
    for bot_id in ["bot_a", "bot_b", "bot_c"]:
        print(f"\n--- {bot_id} ---")
        result = run_content_engine(bot_id)
        print(json.dumps(result.model_dump(), indent=2))
