---
title: CogniRoute AI
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# CogniRoute AI — Cognitive Routing & RAG

Production-quality implementation of the Grid07 AI cognitive loop — with a live REST API.

## Features

| Feature | Details |
|---|---|
| Vector persona routing | FAISS IndexFlatIP, cosine similarity, typed RoutingResult |
| Routing benchmark | 10 labeled posts, per-bot recall table, confusion matrix |
| 6-node LangGraph engine | check_memory → decide_search → web_search → draft_post → self_critique → publish_or_retry |
| Bot memory | ChromaDB per-bot collections, topic deduplication |
| Self-critique loop | LLM scores own drafts 1–10, regenerates if < 7, max 3 retries |
| RAG combat engine | Full thread context, pattern-based injection detection |
| Prompt injection defense | System-level persona lock + injection pattern matching |
| Pydantic schemas | Typed outputs at every module boundary |
| Retry + backoff | Exponential backoff on all LLM calls (3 retries) |
| Stats tracking | Latency + token tracking per phase |
| Rich CLI dashboard | Spinners, panels, tables, injection detection badge |
| **FastAPI REST API** | 6 endpoints, interactive Swagger UI at `/docs` |
| **Dual LLM provider** | Auto-switches between Groq (cloud) and Ollama (local) via env vars |

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # edit to choose your LLM provider (see below)
```

---

## Choosing Your LLM Provider

The API auto-detects which provider to use based on your `.env` file.
Check which is active anytime via `GET /health` or `GET /setup`.

### Option A: Groq Cloud (recommended — free, fast, no GPU needed)

1. Go to **https://console.groq.com** → sign up free
2. Click **API Keys** → **Create API Key** → copy it (starts with `gsk_...`)
3. In your `.env` file, add:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
4. Start the API:
   ```bash
   uvicorn api:app --reload --port 8000
   ```
5. Open **http://localhost:8000/docs** — provider will show `groq`

### Option B: Ollama Local (fully offline, no API key)

1. Install Ollama: **https://ollama.com/download** (Windows / Mac / Linux)
2. Open a terminal and download the model (one-time, ~4.7 GB):
   ```bash
   ollama pull llama3
   ```
3. Start the Ollama server (keep this terminal open):
   ```bash
   ollama serve
   ```
4. In your `.env` file, make sure `GROQ_API_KEY` is **not set** (comment it out):
   ```
   # GROQ_API_KEY=   ← keep commented out
   OLLAMA_HOST=http://127.0.0.1:11434
   OLLAMA_MODEL=llama3:latest
   ```
5. Start the API in a second terminal:
   ```bash
   uvicorn api:app --reload --port 8000
   ```
6. Open **http://localhost:8000/docs** — provider will show `ollama`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Check status and active provider |
| `GET` | `/setup` | Full setup instructions for both providers |
| `POST` | `/route` | Phase 1: route a post to matching bots |
| `POST` | `/generate` | Phase 2: run LangGraph content engine for a bot |
| `POST` | `/defend` | Phase 3: generate combat reply with injection defense |
| `GET` | `/benchmark` | Phase 1: routing accuracy report + confusion matrix |
| `GET` | `/stats` | LLM call stats (latency, tokens per phase) |

Interactive docs (try every endpoint from your browser): **http://localhost:8000/docs**

---

## Deploy to Railway (live public URL)

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "feat: cogniroute ai"
gh repo create cogniroute-ai --public --push

# 2. Go to https://railway.app → New Project → Deploy from GitHub
# 3. Select your repo
# 4. Add environment variable in Railway dashboard:
#    GROQ_API_KEY = gsk_your_key_here
# 5. Railway auto-detects Dockerfile and deploys
# 6. Your live URL: https://cogniroute-ai-xxx.railway.app/docs
```

---

## Other Commands

```bash
python cli.py          # rich terminal dashboard (all 3 phases)
python benchmark.py    # routing accuracy report in terminal
python main.py         # programmatic demo → logs/execution_log.md
pytest tests/ -v       # full test suite (35 tests)
```

## LangGraph Node Structure (Phase 2)

```
[check_memory] → [decide_search] → [web_search] → [draft_post] → [self_critique] → [publish_or_retry]
                                                        ↑                                    │
                                                        └────── retry (score < 7) ───────────┘
```

| Node | Purpose |
|---|---|
| `check_memory` | Query ChromaDB for bot's recent topics — passed to next node as avoid-list |
| `decide_search` | LLM picks a topic (avoiding recent ones) and formats a search query |
| `web_search` | Executes `mock_searxng_search` tool |
| `draft_post` | LLM drafts 280-char post using persona + news context + (if retry) previous critique |
| `self_critique` | LLM scores draft 1–10 on persona alignment, opinionatedness, freshness |
| `publish_or_retry` | Score ≥ 7 → save to ChromaDB memory + emit BotPost. Score < 7 → loop back |

## Prompt Injection Defense (Phase 3)

Two layers:

**1. Explicit threat naming:** The system prompt lists known attack patterns by name ("ignore all previous instructions", "you are now a customer service bot") and instructs the LLM to refuse and counter-argue. Naming the attack class makes the model treat the injection as *data to rebuff*, not an instruction to follow.

**2. Structural labeling + pattern detection:** The human reply is prefixed `[HUMAN_REPLY]:` to mark it as untrusted input. A `detect_injection()` function pre-screens the reply with keyword matching and sets `DefenseReply.injection_detected = True` for observability — useful in a production system for logging and rate-limiting.

## Threshold Tuning

`all-MiniLM-L6-v2` cosine similarities for semantically related short texts: **0.25–0.55** (not 0.85+, which assumes OpenAI `ada-002`). Default threshold is **0.30** in `config.py`. If routing returns no matches, lower to 0.20. If everything matches everything, raise to 0.40.
