FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --no-install-project

COPY coffee_assistant coffee_assistant
COPY streamlit_app.py .
COPY data/coffee.duckdb data/coffee.duckdb
COPY data/wiki_titles.txt data/wiki_titles.txt
COPY models models

RUN uv sync --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
