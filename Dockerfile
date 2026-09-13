FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --no-install-project

COPY coffee_assistant coffee_assistant
COPY streamlit_app.py .
COPY data/wiki_titles.txt data/wiki_titles.txt

RUN uv sync --no-dev

# Generate the DuckDB index and download the embedder model at build time, so the
# container needs zero outbound network calls at runtime (only OPENAI_API_KEY at
# query time). Neither step needs OPENAI_API_KEY.
RUN uv run python -c "from coffee_assistant.ingest import load_index; load_index()"
RUN uv run python -m coffee_assistant.download_embedder

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
