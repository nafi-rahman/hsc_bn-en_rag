#!/usr/bin/env bash
set -e

case "$SERVICE" in
  fastapi)
    uvicorn rag.retrieval:app --host 0.0.0.0 --port 8000
    ;;
  streamlit)
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
    ;;
  *)
    echo "Set SERVICE=fastapi or SERVICE=streamlit"
    exit 1
    ;;
esac