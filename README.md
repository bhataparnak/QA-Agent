# Docs Agent: local AI agent over your documents

A fully local, end-to-end document Q&A agent:

- **Agent**: tool-calling LLM via **Ollama** that decides when to search your documents or save notes, with an automatic RAG fallback.
- **Vector DB**: **ChromaDB** (persistent, cosine similarity) with Ollama embeddings.
- **Dashboard**: **Streamlit** app with chat, document management and a live-refreshing metrics dashboard (Plotly).
- **Metrics**: every question, ingestion, feedback rating and agent step is logged to SQLite.

No API keys, no data leaves your machine.

```
                ┌──────────────────── Streamlit ────────────────────┐
  you ───────▶  │  Chat page     Documents page     Dashboard page  │
                └─────┬───────────────┬───────────────────▲─────────┘
                      │ question      │ upload            │ polls every N s
                      ▼               ▼                   │
                ┌───────────┐   ┌──────────┐        ┌─────┴──────┐
                │ DocsAgent │──▶│  Ingest  │        │  SQLite    │
                │ tool loop │   │ extract  │        │  metrics   │
                └──┬─────┬──┘   │ + chunk  │        └─────▲──────┘
          chat +   │     │      └────┬─────┘              │ every run,
          tools    ▼     ▼ search    ▼ embed + upsert     │ step, ingest
              ┌────────┐ ┌──────────────────┐             │
              │ Ollama │ │ ChromaDB (vector)│─────────────┘
              └────────┘ └──────────────────┘
```

## Quick start (local)

**1. Install and start Ollama** (https://ollama.com), then pull the models:

```bash
ollama pull llama3.1:8b        # chat model with tool calling
ollama pull nomic-embed-text   # embedding model
```

**2. Set up Python** (3.10+):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

**3. Run the app:**

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501, go to **Documents**, click **Load sample documents** (or upload your own), then ask questions on the chat page. Open **Dashboard** in another tab to watch metrics update live.

**4. (Optional) Generate demo traffic** for the dashboard:

```bash
python scripts/ingest_folder.py data/sample_docs
python scripts/simulate_traffic.py --n 20 --delay 2
```

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

Then open http://localhost:8501. For GPU acceleration, uncomment the `deploy` block in `docker-compose.yml`.

## Project layout

```
app/
  config.py         settings from .env
  llm.py            Ollama chat, embeddings, health check
  ingest.py         text extraction (PDF, DOCX, MD, TXT, HTML, CSV, JSON) + chunking
  vector_store.py   ChromaDB wrapper: add, search, list, delete
  agent.py          tool-calling agent loop, RAG fallback, tracing, metrics logging
  metrics.py        SQLite metrics store (queries, ingestions, feedback, events)
  services.py       cached singletons + sidebar status for Streamlit
streamlit_app.py    chat page
pages/
  1_Documents.py    upload, inspect, delete, test retrieval
  2_Dashboard.py    live KPIs and charts
scripts/
  ingest_folder.py  bulk ingest from a folder
  simulate_traffic.py  demo load generator
tests/              chunking tests (pytest)
data/sample_docs/   fictional help-center docs to try it out
```

## How the agent works

1. The question plus recent chat history goes to the LLM with two tools: `search_documents(query, k)` and `save_note(title, text)`.
2. The model calls `search_documents` (possibly several times with rephrased queries). Retrieved chunks come back with source names and similarity scores.
3. The model answers from those passages and cites sources like `[file.md]`.
4. Safety nets:
   - If the model doesn't support tools, the agent switches to classic RAG.
   - If the model answers without searching (`REQUIRE_RETRIEVAL=true`), the agent re-answers with retrieval.
   - A step limit (`MAX_STEPS`) prevents infinite tool loops.
5. The run is logged: latency split (LLM vs vector search), tokens, tool calls, best similarity score, whether the answer was **grounded** (best score ≥ `MIN_SCORE`), cited sources and any error.

## Dashboard metrics

| Metric | What it tells you |
|---|---|
| Queries, avg / p95 latency | Load and responsiveness, with change vs the previous window |
| Grounded answers % | How often the docs actually contained a good match |
| Avg top similarity + histogram | Retrieval quality; lots of mass left of the threshold means missing docs or bad chunking |
| Where the time goes | LLM generation vs vector search vs overhead |
| Token usage | Prompt vs completion tokens over time |
| Most cited documents | Which docs answer the most questions |
| Knowledge base | Chunks per document in the vector DB |
| Thumbs up % | User feedback from the chat page |
| Live activity feed | Every agent step, ingestion and error as it happens |

## Tuning tips

- **Model choice**: `llama3.1:8b` and `qwen2.5:7b` handle tool calling well. On weaker hardware, try `qwen2.5:3b` or `llama3.2:3b` with `AGENT_MODE=rag` for faster, more predictable answers.
- **`MIN_SCORE`**: calibrate using the similarity histogram. With `nomic-embed-text`, relevant chunks usually score 0.55–0.8 and unrelated ones below 0.45.
- **Chunking**: `CHUNK_SIZE=1000`/`CHUNK_OVERLAP=150` characters is a good default. Use smaller chunks for FAQ-style docs, larger for narrative docs.
- **Changing the embedding model** requires re-ingesting documents (delete `data/chroma/` first), because vectors from different models aren't comparable.

## Ideas for extending it

- Swap ChromaDB for **Qdrant** or **pgvector** (only `vector_store.py` changes).
- Add hybrid search (BM25 + vectors) or a reranker for better retrieval.
- Stream the final answer token by token in the chat page.
- Expose the agent as a **FastAPI** endpoint so other apps can use it, keeping Streamlit as the dashboard.
- Add an evaluation set (question → expected source) and chart retrieval hit-rate over time.
- Add authentication (e.g. `streamlit-authenticator`) before deploying anywhere shared.

<img width="1731" height="909" alt="Screenshot_23-9-2026_12433_localhost" src="https://github.com/user-attachments/assets/0a4e06b9-b724-4483-b168-65ab5d845a28" />

<img width="1731" height="909" alt="Screenshot_23-9-2026_124226_localhost" src="https://github.com/user-attachments/assets/eb567252-33df-4c13-9dfd-fb9fd8da935d" />

<img width="1731" height="909" alt="Screenshot_23-9-2026_124243_localhost" src="https://github.com/user-attachments/assets/dc9b3239-5751-4d7f-9684-add9b1ffbcc3" />



