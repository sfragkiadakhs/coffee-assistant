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

```python
from coffee_assistant.rag import rag

rag("What regions produce Arabica coffee?")
```

The first call builds the local knowledge base automatically (fetches and chunks the
Wikipedia articles, loads them into DuckDB) if it doesn't exist yet.

## Project structure

- `coffee_assistant/ingest.py` — fetches Wikipedia articles, chunks them, loads them
  into DuckDB via a dlt pipeline, and builds the search index.
- `coffee_assistant/rag.py` — retrieval, prompt building, and the LLM call.
- `notebooks/` — exploratory notebooks used during development.
- `data/` — source titles list; the generated DuckDB file is gitignored and rebuilt
  on first run.

This README will be expanded with architecture, evaluation, and monitoring sections
as the project progresses.