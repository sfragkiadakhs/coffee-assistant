# Coffee Assistant

A coffee recommendation/Q&A RAG assistant built for the LLM Zoomcamp capstone project.

## Data

Coffee-related Wikipedia articles (Coffee, Espresso, Coffee roasting, Arabica coffee,
coffee-producing regions, etc.), ingested and chunked via a [dlt](https://dlthub.com/)
pipeline into a local DuckDB database.

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.

## Running

### Streamlit app

```bash
uv run streamlit run streamlit_app.py
```

Opens a browser UI with a question box. On startup it builds the knowledge base
(fetching/chunking the Wikipedia articles into DuckDB if it doesn't exist yet, then
the keyword index and vector embeddings), shown with a loading spinner; every
question after that hits the cached index directly.

### From Python

```python
from coffee_assistant.rag import rag

rag("What regions produce Arabica coffee?")
```

## Ground truth

`notebooks/ground-truth-generation.ipynb` generates 5 synthetic user questions per chunk with
`gpt-4o-mini`, keyed by `chunk_id`, saved to `data/ground-truth-retrieval.csv`
(504 chunks, 2520 questions) — used for the retrieval/LLM evaluation steps.

## Retrieval evaluation

`notebooks/retrieval-eval.ipynb` evaluates three retrieval strategies against the
ground-truth question set, using Hit Rate and MRR (top 10 results):

| Method | Hit Rate | MRR |
|---|---|---|
| Keyword (TF-IDF) | 0.556 | 0.338 |
| Vector (ONNX MiniLM embeddings) | 0.633 | 0.404 |
| Hybrid (RRF, k=60 default) | 0.751 | 0.391 |
| **Hybrid (RRF, k=1)** | **0.751** | **0.421** |

Hybrid search — merging keyword and vector rankings via Reciprocal Rank Fusion (RRF) —
wins on Hit Rate outright, and with a tuned `k=1` it also beats vector search on MRR
(the RRF literature default of `k=60` is tuned for rankings in the thousands; it
barely differentiates a top-10 candidate list — see the notebook for the full
explanation). **Hybrid search with `k=1` is used in production**, in
`coffee_assistant/retrieval.py`'s `hybrid_search()`, called from `rag.py`'s `search()`.

Vector embeddings come from a local ONNX MiniLM model (`coffee_assistant/embedder.py`,
adapted from the LLM Zoomcamp `02-vector-search` module) — no API calls, no PyTorch
dependency, and a small enough footprint to keep the eventual Docker image lightweight.

## Project structure

- `coffee_assistant/ingest.py` — fetches Wikipedia articles, chunks them, loads them
  into DuckDB via a dlt pipeline, and builds the keyword search index.
- `coffee_assistant/embedder.py` — local ONNX MiniLM embedder used for vector search
  (no API calls); `download_embedder.py` is the one-time script that pulls the model
  into `models/` (gitignored).
- `coffee_assistant/retrieval.py` — keyword, vector, and hybrid (RRF) search; this is
  the production search used by `rag.py`.
- `coffee_assistant/rag.py` — prompt building and the LLM call; `search()` delegates
  to `retrieval.hybrid_search()`.
- `streamlit_app.py` — Streamlit UI: a question box that calls `rag()` and displays
  the answer with token usage.
- `notebooks/ground-truth-generation.ipynb` — generates the ground-truth question set
  used for retrieval/LLM evaluation.
- `notebooks/retrieval-eval.ipynb` — evaluates keyword vs. vector vs. hybrid search
  (results in [Retrieval evaluation](#retrieval-evaluation) above).
- `data/` — source titles list and ground-truth CSV; the generated DuckDB file is
  gitignored and rebuilt on first run.

This README will be expanded with architecture, evaluation, and monitoring sections
as the project progresses.
