FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --no-install-project

COPY coffee_assistant coffee_assistant
COPY streamlit_app.py .

RUN uv sync --no-dev

# Copy the committed data snapshot instead of fetching Wikipedia at build time.
# The image then depends only on git and holds exactly the data the eval measured.
COPY data/coffee.duckdb data/coffee.duckdb
COPY data/embeddings.npz data/embeddings.npz

# Download the embedder model at build time, so the container makes no outbound
# network calls at runtime (only OPENAI_API_KEY at query time).
RUN uv run python -m coffee_assistant.download_embedder

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
