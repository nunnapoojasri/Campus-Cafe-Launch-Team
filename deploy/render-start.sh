#!/usr/bin/env bash
set -euo pipefail

ollama serve &
ollama_pid=$!
streamlit_pid=""

shutdown() {
    if [[ -n "$streamlit_pid" ]]; then
        kill "$streamlit_pid" 2>/dev/null || true
    fi
    kill "$ollama_pid" 2>/dev/null || true
    wait "$ollama_pid" 2>/dev/null || true
}
trap shutdown EXIT

ready=false
for _ in $(seq 1 90); do
    if curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null; then
        ready=true
        break
    fi
    sleep 2
done
if [[ "$ready" != true ]]; then
    echo "Ollama did not become ready within 180 seconds." >&2
    exit 1
fi

ollama pull "${OLLAMA_MODEL:-llama3.2:3b}"

streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-10000}" \
    --server.headless true &
streamlit_pid=$!

set +e
wait -n "$ollama_pid" "$streamlit_pid"
status=$?
set -e
exit "$status"
