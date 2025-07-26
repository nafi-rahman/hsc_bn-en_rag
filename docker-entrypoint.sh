#!/usr/bin/env bash
set -e

case "$SERVICE" in
  fastapi)
    echo "Starting FastAPI..."
    uvicorn rag.retrieval:app --host 0.0.0.0 --port 8000
    ;;
  streamlit)
    echo "Starting Streamlit..."
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
    ;;
  embed)
    echo "Running embedding and Qdrant upload..."
    python scripts/embed_and_upsert_qdrant.py
    ;;
  *)
    echo "❌ Unknown SERVICE: $SERVICE"
    echo "Please set SERVICE=fastapi, streamlit or embed"
    exit 1
    ;;
esac
