#!/usr/bin/env bash
# --- add this line ---
ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
# ---------------------

# 0. Make sure we can execute
set -euo pipefail

# 1. Prefer a specific interpreter
PY="$(command -v python3 || command -v python)"
if [[ -z "$PY" ]]; then
    echo "Python not found" >&2
    exit 1
fi

# 2. Create venv with that interpreter
if [[ ! -d "$ENV_DIR" ]]; then
    echo "Creating virtual environment..."
    "$PY" -m venv "$ENV_DIR"
fi

# 3. Activate with error check
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"
if ! command -v pip >/dev/null 2>&1; then
    echo "venv activation failed or pip missing" >&2
    exit 1
fi

# 4. Upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Embed / ingest
python data/scripts/embed_and_upsert_qdrant.py

# 6. Start services in *this* window (or use & for background)
# (uvicorn rag.retrieval:app --host 0.0.0.0 --port 8000) &
# (streamlit run app.py --server.port 8501) &

# echo "Backend and frontend started. Check jobs with 'jobs', kill with 'kill %1', etc."