FROM ollama/ollama:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py brief_worker.py README.md final_brief.md reflection.md ./
COPY assets ./assets
COPY .streamlit/config.toml .streamlit/config.toml
COPY deploy/render-start.sh /usr/local/bin/render-start
RUN chmod +x /usr/local/bin/render-start

ENV PORT=10000 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/var/data/ollama/models \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODEL=llama3.2:3b \
    CREW_MODEL=ollama/llama3.2:3b \
    OUTPUT_PATH=/var/data/final_brief.md \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 10000
ENTRYPOINT ["/usr/local/bin/render-start"]
