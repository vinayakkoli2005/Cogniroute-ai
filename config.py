import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH  = os.getenv("CHROMA_PATH",  "./chroma_db")

# Cosine similarity threshold for persona routing.
# all-MiniLM-L6-v2 similarity range for related short texts: ~0.25–0.55.
# Lower this to 0.20 if no bots match; raise to 0.40 if all bots match everything.
SIMILARITY_THRESHOLD = 0.20

# Self-critique: minimum score (1–10) to publish a draft without retry
CRITIQUE_PASS_SCORE = 7
# Maximum self-critique retries before publishing best attempt
MAX_CRITIQUE_RETRIES = 3

BOT_PERSONAS = {
    "bot_a": {
        "id": "bot_a",
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. "
            "I am highly optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "id": "bot_b",
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
        ),
    },
    "bot_c": {
        "id": "bot_c",
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}
