# Campus Café Launch Team

A cafe-themed Streamlit app that coordinates four CrewAI agents to create a reviewed launch brief. It uses a **local Ollama language model**, so no OpenAI account, API key, or paid model endpoint is required.

## Get started

### 1. Install Ollama and download a local model

Install Ollama from [ollama.com/download](https://ollama.com/download), then in a terminal:

```bash
ollama pull llama3.2:3b
```

Keep Ollama running. If it does not start automatically, run `ollama serve` in a separate terminal. The model is downloaded once and then inference runs locally. The default model is `ollama/llama3.2:3b`.

### 2. Install Python dependencies

Python 3.10–3.13 is supported; Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Start the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. No API key environment variable is read or required.

Creating a brief on a CPU-only computer can take a few minutes. The page stays responsive while a separate worker prepares the brief, showing plain-language progress updates. Detailed agent and tool logs stay hidden unless an error needs troubleshooting.

## Deploy the no-cost demo to Render

`render.yaml` configures a free native Python web service. It displays the included reviewed sample brief, but **AI generation is intentionally disabled**: Render's free instance cannot run the local 3B Ollama model. No model API key or paid resource is needed for this demo. The deployment uses `requirements-demo.txt` so the free instance does not load the much larger CrewAI stack.

After pushing this repository to GitHub:

1. Open the [Render Blueprint dashboard](https://dashboard.render.com/blueprints).
2. Select **New Blueprint Instance** and connect `nunnapoojasri/Campus-Cafe-Launch-Team`.
3. Review that the service uses the **Free** plan, then create the Blueprint.
4. Open the service's `onrender.com` URL after its health check passes.

To use all four AI agents without an API key, run the app locally with Ollama as described above. The optional Dockerfile and `deploy/render-start.sh` describe a model-hosting setup for a paid server with enough memory and a persistent disk; the free Render Blueprint does not use them.

## The agent team

1. **Market Researcher** uses LangChain DuckDuckGo Search and Wikipedia tools to gather findings with source links and limitations.
2. **Financial Analyst** uses a custom LangChain calculator to show every monthly operating break-even step.
3. **Marketing Writer** combines the findings and calculations into a one-page decision brief.
4. **Skeptical Reviewer** identifies weaknesses and then produces the final version.

The five tasks run sequentially, passing research, calculations, draft, and critique explicitly through task contexts. The final brief appears in the app, can be downloaded, and is saved as `final_brief.md`. See also the included [sample brief](./final_brief.md) and [collaboration reflection](./reflection.md).

## Settings and limitations

- Edit enrollment, open days, average transaction, variable costs, and fixed costs in the sidebar. These values are planning assumptions, not quotes or proof of demand.
- Set `CREW_MODEL` if you have another model installed in Ollama, for example `ollama/qwen2.5:3b`. Set `OLLAMA_BASE_URL` if Ollama is not at `http://localhost:11434`.
- Web research uses public DuckDuckGo and Wikipedia endpoints, so network access is needed for live search. If those tools are unavailable, the researcher can use a small built-in set of cited reference notes, explicitly disclose the limitation, and avoid inventing local pricing or demand data.
- The supplied `final_brief.md` is an illustrative sample. Validate its assumptions, sources, local prices, staffing, and facilities costs before using it for a funding decision.
