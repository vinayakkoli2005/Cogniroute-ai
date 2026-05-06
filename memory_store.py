import chromadb
from chromadb.config import Settings
from config import CHROMA_PATH


class BotMemoryStore:
    """
    Per-bot ChromaDB collections for storing past posts and retrieving recent topics.
    Each bot gets its own collection: 'bot_memory_{bot_id}'.
    """

    def __init__(self, persist_path: str = CHROMA_PATH):
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, bot_id: str) -> chromadb.Collection:
        if bot_id not in self._collections:
            self._collections[bot_id] = self._client.get_or_create_collection(
                name=f"bot_memory_{bot_id}",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[bot_id]

    def add_post(self, bot_id: str, post_content: str, topic: str) -> None:
        """Store a published post in the bot's memory collection."""
        col = self._get_collection(bot_id)
        # Use topic as document ID to deduplicate by topic
        doc_id = f"{bot_id}_{topic.lower().replace(' ', '_')}"
        existing = col.get(ids=[doc_id])
        if existing["ids"]:
            return  # already stored this topic
        col.add(
            documents=[post_content],
            metadatas=[{"topic": topic, "bot_id": bot_id}],
            ids=[doc_id],
        )

    def get_recent_topics(self, bot_id: str, n: int = 5) -> list[str]:
        """Return the n most recently stored topic names for a bot."""
        col = self._get_collection(bot_id)
        results = col.get(include=["metadatas"])
        topics = [m["topic"] for m in results["metadatas"]]
        return list(dict.fromkeys(topics))[:n]  # preserve order, dedupe


# Module-level singleton
memory = BotMemoryStore()
