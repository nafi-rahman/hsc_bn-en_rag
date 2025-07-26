# Multi-stage build: FastAPI & Streamlit share the same image
FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

# install Poetry
RUN pip install poetry==2.1.3
RUN poetry config virtualenvs.create false

# copy lockfile first for Docker layer cache
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# copy rest of the source
COPY . .

# expose ports
EXPOSE 8000 8501

# choose service at runtime
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]