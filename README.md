# Campus Café Launch Intelligence

A cafe-themed Streamlit app that coordinates up to five CrewAI agents to create a reviewed launch brief. It uses a **local Ollama language model**, so no OpenAI account, API key, or paid model endpoint is required.

## Local setup

Python 3.12 is recommended. Install [Ollama](https://ollama.com/download), start it, and download the default local model once:

```powershell
ollama pull llama3.2:3b
```

From PowerShell in the repository, create and activate a virtual environment, install dependencies, and copy the environment template:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` for that terminal session, then activate again. The `.env` file contains configuration only; do not put secrets in it or commit it.

On macOS/Linux, activate with `source .venv/bin/activate` after creating the environment with `python3.12 -m venv .venv`.

Run the tests and start the app:

```powershell
python -m pytest -q
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. The default backend is local Ollama (`CREW_MODEL=ollama/llama3.2:3b`, `OLLAMA_BASE_URL=http://localhost:11434`); no API key is required or read. Override those values in `.env` to use another model already available from an Ollama-compatible endpoint. System environment variables take precedence over `.env`.

The app writes generated reports to `final_brief.md` and `collaboration_reflection.md` in the repository by default. Set `OUTPUT_PATH` and `REFLECTION_PATH` to change those locations. A run is marked complete only after review passes and both writes succeed; a report-writing failure is shown as a failed run, not a success.

The dashboard has Overview, Market Research, Financial Planner, Agent Workflow, Final Brief, and Reflection sections. Local generation uses a background worker; the UI shows actual task outputs and failures rather than simulated completion.

## Render deployment

The existing `render.yaml` is intentionally preserved as a Free sample deployment. It does not run AI: an Ollama model installed on your PC is not available to a Render instance. No API key or secret is required for this sample mode.

Exact Blueprint commands:

```text
Build: pip install -r requirements-demo.txt
Start: streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
Health check: /_stcore/health
```

Configure this environment variable in the Render dashboard (also declared by the Blueprint):

| Variable | Value | Purpose |
|---|---|---|
| `APP_DEMO_MODE` | `true` | Shows the reviewed sample and disables local-model generation. |

Do not add `OLLAMA_BASE_URL` or `CREW_MODEL` to the free Render service expecting it to reach your computer. Live hosted generation requires a separately hosted, network-reachable model service; this repository's Render Blueprint does not configure one. The optional Dockerfile and `deploy/render-start.sh` are a different, resource-intensive model-hosting path and are not used by the free Blueprint.

## The agent team

One configured local Ollama model powers four primary specialists: **Market Researcher**, **Financial Analyst**, **Marketing Writer**, and **Skeptical Reviewer**. Research feeds finance; both feed the draft; the reviewer receives the draft and upstream evidence/calculations, records three corrections, and produces a corrected final brief. The reviewer checks sustainability claims as part of its review.

The campus/city and café concept entered in the sidebar are included in research and drafting prompts. Research, calculations, sustainability notes, draft, and critique are passed explicitly through task contexts. The final brief appears in the app, can be downloaded, and is saved as `final_brief.md`. See also the included [sample brief](./final_brief.md) and [collaboration reflection](./reflection.md).

## Settings and limitations

- Set the campus/city and café concept, then edit enrollment, operating days, average transaction, per-customer variable costs, and fixed costs in the sidebar. These are planning assumptions, not quotes or proof of demand.
- `CREW_MODEL` and `OLLAMA_BASE_URL` configure the local Ollama model. `OLLAMA_NUM_CTX` defaults to `2048` and is applied to native Ollama chat/tool requests to limit memory use; increase it only when the host has enough memory. No API credential is used by the current local backend.
- Web research uses public DuckDuckGo and Wikipedia endpoints, so network access is needed. Search errors and empty results are surfaced as research limitations; citations must match results actually returned by tools. Curated references are explicitly labeled and do not establish local demand.
- The supplied `final_brief.md` is an illustrative sample. Validate its assumptions, sources, local prices, staffing, and facilities costs before using it for a funding decision.
