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

Opens a browser UI with a question box. The first question triggers building the
local knowledge base (fetches and chunks the Wikipedia articles, loads them into
DuckDB) if it doesn't exist yet, shown with a loading spinner; every question after
that hits the cached index directly.

### From Python

```python
from coffee_assistant.rag import rag

rag("What regions produce Arabica coffee?")
```

## Ground truth

`notebooks/eval.ipynb` generates 5 synthetic user questions per chunk with
`gpt-4o-mini`, keyed by `chunk_id`, saved to `data/ground-truth-retrieval.csv`
(504 chunks, 2520 questions) — used for the retrieval/LLM evaluation steps.

## Project structure

- `coffee_assistant/ingest.py` — fetches Wikipedia articles, chunks them, loads them
  into DuckDB via a dlt pipeline, and builds the search index.
- `coffee_assistant/rag.py` — retrieval, prompt building, and the LLM call.
- `streamlit_app.py` — Streamlit UI: a question box that calls `rag()` and displays
  the answer with token usage.
- `notebooks/eval.ipynb` — generates the ground-truth question set used for
  retrieval/LLM evaluation.
- `data/` — source titles list and ground-truth CSV; the generated DuckDB file is
  gitignored and rebuilt on first run.

This README will be expanded with architecture, evaluation, and monitoring sections
as the project progresses.
