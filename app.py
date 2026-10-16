"""Interactive Campus Café Launch Team brief generator."""

from __future__ import annotations

import ast
import atexit
import json
import operator
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Type
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", str(ROOT / "final_brief.md")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("CREW_MODEL", "ollama/llama3.2:3b")
DEMO_MODE = os.getenv("APP_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
_JOB_PROCESSES: dict[str, subprocess.Popen[Any]] = {}
_PROGRESS_STEPS = (
    "Finding useful café and student information",
    "Working out the budget",
    "Writing a clear proposal",
    "Reviewing the details",
    "Finishing your brief",
)


def stop_generation_processes() -> None:
    for process in _JOB_PROCESSES.values():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


atexit.register(stop_generation_processes)


def write_job_status(path: str | Path, **updates: Any) -> None:
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {}
    current.update(updates)
    current["updated_at"] = time.time()
    temporary_path = status_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(current), encoding="utf-8")
    temporary_path.replace(status_path)


def check_local_model() -> tuple[bool, str]:
    model_name = DEFAULT_MODEL.removeprefix("ollama/").lower()
    try:
        with urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2) as response:
            payload = json.loads(response.read())
        installed = [model.get("name", "").lower() for model in payload.get("models", [])]
        if any(
            name == model_name or name.split(":")[0] == model_name.split(":")[0]
            for name in installed
        ):
            return True, f"Local model ready: `{DEFAULT_MODEL}`"
        return (
            False,
            f"Ollama is running, but `{DEFAULT_MODEL}` is not installed. "
            f"Run `ollama pull {model_name}`.",
        )
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        return (
            False,
            "Local Ollama is not running. Install Ollama, start it, and pull the "
            f"model with `ollama pull {model_name}`. ({exc})",
        )


def calculator(expression: str) -> str:
    """Calculate basic arithmetic using numbers and +, -, *, /, and parentheses."""
    operators: dict[type[ast.operator], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Only basic arithmetic expressions are supported.")

    try:
        result = evaluate(ast.parse(expression, mode="eval"))
        if not (-1e15 < result < 1e15):
            return "Error: result is outside the supported range."
        formatted = f"{result:.8f}"
        return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
    except (OverflowError, SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
        return f"Error: {exc}"





def verified_reference_notes(topic: str) -> str:
    """Return built-in, cited campus café market facts when web search is unavailable."""
    return (
        "Curated reference notes (use as cited context, not local demand evidence):\n"
        "1. National Coffee Association, 2026 National Coffee Data Trends: its "
        "June 2026 report says 47% of U.S. adults had specialty coffee in the "
        "past day, compared with 42% for traditional coffee; 38% of past-day "
        "coffee drinkers had coffee out of home, the highest share since "
        "January 2020. These are U.S. adult category indicators, not college "
        "student spending or campus demand. Source: "
        "https://www.ncausa.org/Newsroom/Coffee-Consumption-Statistics\n"
        "2. The Hope Center's Student Basic Needs Survey is used by participating "
        "institutions to identify student needs and inform local action. The "
        "referenced page does not supply a café-spending estimate or prove demand "
        "at this campus. Source: "
        "https://hope.temple.edu/research/hope-center-basic-needs-survey\n"
        "3. IFIC's 2025 Food & Health Survey examines U.S. consumer food beliefs "
        "and purchasing behavior. Its referenced overview does not establish "
        "campus-specific plant-based demand or a typical café menu price. Source: "
        "https://ific.org/research/2025-food-health-survey/\n"
        "Pricing evidence: no independently verified local menu-price benchmark "
        "is included in this built-in reference pack. Do not invent a typical "
        "price. Label any example ticket/menu prices as planning assumptions and "
        "recommend checking nearby menus and surveying students.\n"
        f"Requested topic: {topic}"
    )


def get_task_output(task: Any) -> str:
    output = getattr(task, "output", None)
    raw = getattr(output, "raw", None)
    return str(raw if raw is not None else output or "")


def publish_task_progress(_: Any) -> None:
    """Write task milestones to disk so the UI can safely poll a worker process."""
    status_path = os.getenv("CAFE_JOB_STATUS_PATH")
    if not status_path:
        return
    try:
        current = json.loads(Path(status_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {"progress": 0}
    completed = min(int(current.get("progress", 0)) + 1, len(_PROGRESS_STEPS))
    write_job_status(status_path, state="running", progress=completed)


def run_brief_generation(
    job_id: str,
    assumptions: dict[str, Any],
    status_path: str,
) -> None:
    try:
        crew, tasks = create_crew(assumptions, task_callback=publish_task_progress)
        result = crew.kickoff()
        brief = get_task_output(tasks[-1]).strip() or str(result).strip()
        if not brief:
            raise RuntimeError("The reviewer returned an empty brief.")
        OUTPUT_PATH.write_text(brief + "\n", encoding="utf-8")
        write_job_status(
            status_path,
            state="complete",
            progress=len(_PROGRESS_STEPS),
            brief=brief,
            critique=get_task_output(tasks[3]),
            research=get_task_output(tasks[0]),
            finance=get_task_output(tasks[1]),
        )
    except Exception as exc:
        write_job_status(
            status_path,
            state="error",
            error=str(exc) or type(exc).__name__,
        )


def create_crew(
    assumptions: dict[str, Any],
    task_callback: Any = None,
) -> tuple[Any, list[Any]]:
    if DEMO_MODE:
        raise RuntimeError(
            "AI generation is disabled in the no-cost Render demo. "
            "Run locally with Ollama to use the multi-agent workflow."
        )

    from crewai import Agent, Crew, LLM, Process, Task
    from crewai.tools import BaseTool
    from langchain_core.tools import tool
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, TimeoutException
    from pydantic import BaseModel

    class ToolInput(BaseModel):
        query: str

    class LangChainToolAdapter(BaseTool):
        """Adapt a LangChain tool to CrewAI's tool interface."""

        name: str
        description: str
        args_schema: Type[BaseModel] = ToolInput
        lc_tool: Any

        def _run(self, query: str) -> str:
            return str(self.lc_tool.invoke(query))

    def wrap_langchain_tool(lc_tool: Any) -> LangChainToolAdapter:
        return LangChainToolAdapter(
            name=lc_tool.name,
            description=lc_tool.description,
            lc_tool=lc_tool,
        )

    @tool
    def web_search(query: str) -> str:
        """Find up to three current web results, with a short request timeout."""
        try:
            with DDGS(timeout=5) as search:
                results = search.text(query, max_results=3, backend="duckduckgo")
            if not results:
                return "No web results were returned. Use verified reference notes and disclose the evidence gap."
            return "\n".join(
                f"- {item.get('title', 'Untitled')}: {item.get('body', '')} "
                f"Source: {item.get('href', '')}"
                for item in results
            )
        except (DDGSException, TimeoutException) as exc:
            return (
                f"Live web search is unavailable: {exc}. Use verified reference "
                "notes and disclose the evidence gap; do not invent findings."
            )

    @tool
    def wikipedia_search(query: str) -> str:
        """Search Wikipedia for relevant background pages with a short timeout."""
        url = "https://en.wikipedia.org/w/api.php?" + urlencode(
            {
                "action": "opensearch",
                "search": query,
                "limit": 3,
                "namespace": 0,
                "format": "json",
            }
        )
        try:
            with urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
            return "\n".join(
                f"- {title}: {description} Source: {page_url}"
                for title, description, page_url in zip(data[1], data[2], data[3])
            ) or "No Wikipedia results were returned."
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            return (
                f"Wikipedia search is unavailable: {exc}. Use verified reference "
                "notes and disclose the evidence gap."
            )

    llm = LLM(
        model=DEFAULT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )
    web_search_tool = wrap_langchain_tool(web_search)
    wikipedia_tool = wrap_langchain_tool(wikipedia_search)
    reference_notes = wrap_langchain_tool(tool(verified_reference_notes))
    calculator_tool = wrap_langchain_tool(tool(calculator))

    researcher = Agent(
        role="Market Researcher",
        goal=(
            "Find current, relevant evidence on student food and drink habits, "
            "café pricing, and sustainability and plant-based trends. Cite sources."
        ),
        backstory=(
            "You are a careful campus-retail researcher. You distinguish national "
            "evidence from local demand, cite direct URLs, and never invent a statistic."
        ),
        tools=[web_search_tool, wikipedia_tool, reference_notes],
        llm=llm,
        verbose=False,
        max_iter=4,
    )
    analyst = Agent(
        role="Financial Analyst",
        goal="Produce an auditable monthly operating break-even estimate.",
        backstory=(
            "You are a skeptical accountant. Every input is either a stated "
            "planning assumption or a sourced fact, and every calculation uses "
            "the calculator tool with its arithmetic shown."
        ),
        tools=[calculator_tool],
        llm=llm,
        verbose=False,
        max_iter=6,
    )
    writer = Agent(
        role="Marketing Writer",
        goal="Write a clear, persuasive one-page decision brief for the Dean.",
        backstory=(
            "You turn complex proposals into concise, plain-language decisions. "
            "You never turn an assumption into a fact."
        ),
        llm=llm,
        verbose=False,
        max_iter=2,
    )
    reviewer = Agent(
        role="Skeptical Reviewer",
        goal=(
            "Find unsupported claims, missing risks, and unclear calculations, "
            "then ensure the final brief corrects them."
        ),
        backstory=(
            "You advise senior university leaders and challenge optimistic "
            "business cases. You preserve useful evidence while flagging limits."
        ),
        llm=llm,
        allow_delegation=True,
        verbose=False,
        max_iter=4,
    )

    assumption_text = "\n".join(
        [
            f"- Student population: {assumptions['students']:,} (scenario input; not a demand forecast).",
            f"- Open days per month: {assumptions['open_days']} (planning assumption).",
            f"- Average transaction: ${assumptions['average_ticket']:.2f} (planning assumption; menu prices require local validation).",
            f"- Variable costs: {assumptions['variable_cost_pct']:.1f}% of sales (planning assumption, including ingredients, packaging, and payment fees).",
            f"- Monthly fixed costs: ${assumptions['fixed_costs']:,.2f} (planning assumption; includes rent/occupancy, labor, equipment, and other operating costs).",
        ]
    )

    research_task = Task(
        description=(
            "Research current evidence for a proposed student café at a university "
            f"with {assumptions['students']:,} students. Use web search and Wikipedia "
            "tools to find trustworthy source pages. Cover (a) student food/drink "
            "spending or affordability, (b) typical café menu prices with geography "
            "and date if credible data exists, and (c) 2-3 current café/food trends, "
            "including plant-based choices and/or sustainability. Return 5-8 concise "
            "findings, each with the exact source title, URL, publication date where "
            "available, and what it does and does not establish. Do not invent "
            "prices or student-specific findings. If strong local or student-specific "
            "evidence is unavailable, use the verified_reference_notes tool, state "
            "the gap, and recommend how to collect it. If web search tools fail, "
            "report that limitation and use the curated source notes rather than "
            "ending the task or inventing results."
        ),
        expected_output="Evidence findings with source title, URL, date, and limitations.",
        agent=researcher,
    )
    finance_task = Task(
        description=(
            "Estimate monthly operating break-even using only the planning "
            "assumptions below. Do not add unprovided revenues or costs. Use the "
            "calculator tool for EVERY arithmetic operation and show the actual "
            "expressions and results. Compute contribution per transaction = average "
            "transaction × (1 − variable-cost rate), monthly transactions to break "
            "even = fixed costs ÷ contribution per transaction, transactions per "
            "open day = monthly transactions ÷ open days, and daily transactions as "
            "a percentage of enrollment = daily transactions ÷ student population × 100. "
            "Explicitly state this is not the share of unique students because repeat "
            "visits are possible. Clearly "
            "label every value as an assumption or a calculation, round only the "
            "displayed result, and state exclusions such as startup costs, taxes, "
            "seasonality, and debt service.\n\n"
            f"Planning assumptions:\n{assumption_text}"
        ),
        expected_output=(
            "A transparent assumption table and auditable monthly and daily "
            "break-even arithmetic, including exclusions."
        ),
        agent=analyst,
        context=[research_task],
    )
    draft_task = Task(
        description=(
            "Draft a decision-ready, one-page Markdown launch brief for the Dean. "
            "Use the research findings and financial analysis provided in context. "
            "Include an executive recommendation, evidence-led opportunity, concise "
            "pricing/trend implications, break-even arithmetic with assumptions, "
            "the most important risks and next steps, and source links. Keep it "
            "roughly 300-450 words. Mark estimates as assumptions and don't assert "
            "national data proves local demand."
        ),
        expected_output="A concise one-page Markdown draft with sources and arithmetic.",
        agent=writer,
        context=[research_task, finance_task],
    )
    critique_task = Task(
        description=(
            "Critically assess the draft against the research and calculations in "
            "context. Identify the three most material weaknesses, checking citation "
            "quality, student-specific relevance, price assumptions, arithmetic, "
            "risks, and whether sustainability/plant-based claims are supported. "
            "For each, state a specific correction. Do not produce the final brief "
            "in this task."
        ),
        expected_output="Three evidence-based weaknesses with concrete corrections.",
        agent=reviewer,
        context=[draft_task, research_task, finance_task],
    )
    final_task = Task(
        description=(
            "Create the final, corrected one-page Markdown launch brief for the "
            "Dean. Incorporate the critique and verify all statements against the "
            "research and calculations in context. Include sourced factual claims "
            "with URLs, explicitly labeled planning assumptions, the full break-even "
            "formula and arithmetic, and material risks/next steps. If evidence is "
            "weak, say so rather than overstating it. Output ONLY the final brief "
            "(no critique preamble or agent notes), in clear persuasive language, "
            "roughly 300-450 words. Every number must cite a source or be clearly "
            "labeled as an assumption or calculation; remove unsupported numbers."
        ),
        expected_output="The final, reviewed, one-page Markdown brief only.",
        agent=reviewer,
        context=[draft_task, critique_task, research_task, finance_task],
    )
    crew = Crew(
        agents=[researcher, analyst, writer, reviewer],
        tasks=[research_task, finance_task, draft_task, critique_task, final_task],
        process=Process.sequential,
        verbose=False,
        task_callback=task_callback,
    )
    return crew, [research_task, finance_task, draft_task, critique_task, final_task]


def render_assumptions() -> dict[str, Any]:
    st.sidebar.header("Planning assumptions")
    st.sidebar.caption(
        "These are editable scenario inputs—not vendor quotes or a demand forecast."
    )
    students = st.sidebar.number_input(
        "Enrolled students", min_value=1, value=8000, step=100
    )
    open_days = st.sidebar.number_input(
        "Open days per month", min_value=1, max_value=31, value=22, step=1
    )
    average_ticket = st.sidebar.number_input(
        "Average spend per transaction ($)",
        min_value=0.01,
        value=7.50,
        step=0.25,
        format="%.2f",
    )
    variable_cost_pct = st.sidebar.number_input(
        "Variable costs (% of sales)",
        min_value=0.0,
        max_value=99.0,
        value=35.0,
        step=1.0,
        format="%.1f",
    )
    st.sidebar.markdown("**Monthly fixed costs ($)**")
    occupancy = st.sidebar.number_input(
        "Rent / occupancy / utilities", min_value=0.0, value=4000.0, step=250.0
    )
    labor = st.sidebar.number_input(
        "Staffing", min_value=0.0, value=12500.0, step=500.0
    )
    equipment = st.sidebar.number_input(
        "Equipment lease / maintenance", min_value=0.0, value=1500.0, step=250.0
    )
    other = st.sidebar.number_input(
        "Other operating costs", min_value=0.0, value=2000.0, step=250.0
    )
    return {
        "students": int(students),
        "open_days": int(open_days),
        "average_ticket": float(average_ticket),
        "variable_cost_pct": float(variable_cost_pct),
        "fixed_costs": float(occupancy + labor + equipment + other),
    }


def main() -> None:
    st.set_page_config(
        page_title="Campus Café Launch Team",
        page_icon="☕",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 8% 3%, rgba(221, 73, 137, .16), transparent 27%),
                radial-gradient(ellipse at 90% 8%, rgba(96, 82, 195, .16), transparent 29%),
                radial-gradient(ellipse at 65% 95%, rgba(40, 157, 146, .11), transparent 30%),
                #180f1b;
        }
        [data-testid="stHeader"] {background: rgba(24, 15, 27, .9);}
        [data-testid="stSidebar"] {
            background:
                linear-gradient(160deg, rgba(179, 59, 117, .17), transparent 42%),
                linear-gradient(330deg, rgba(82, 81, 167, .16), transparent 45%),
                #211524;
            border-right: 1px solid #56344f;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {color: #F5DDEC;}
        [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {color: #E9DCEB;}
        h1, h2, h3 {color: #FFF2FA;}
        h1 {letter-spacing: -.035em;}
        h1::after {
            content: "";
            display: block;
            width: min(300px, 52vw);
            height: 4px;
            margin-top: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, #F06C9B, #F4A261, #E9C46A, #66B58B, #58A6C8, #9B7ED8);
            box-shadow: 0 0 18px rgba(221, 73, 137, .27);
        }
        [data-testid="stNumberInput"] input {
            background: #2D2032;
            color: #FFF5FC;
            border-color: #69435F;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(55, 35, 60, .96), rgba(37, 29, 52, .96));
            border: 1px solid #684564;
            border-radius: 14px;
            padding: 14px 18px;
        }
        [data-testid="stExpander"] {
            background: rgba(43, 29, 48, .9);
            border: 1px solid #5B3A57;
            border-radius: 12px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 2.8rem;
            border: 1px solid rgba(247, 224, 242, .28);
            background: linear-gradient(105deg, #A93870, #82448C 36%, #515FA7 68%, #287F83);
            color: #FFF8FD;
            box-shadow: 0 7px 22px rgba(125, 53, 119, .22);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #F3B8DA;
            color: #FFFFFF;
            filter: brightness(1.12);
        }
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #F06C9B, #F4A261, #E9C46A, #66B58B, #58A6C8, #9B7ED8);
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
            background-color: #2A1D30;
            border-color: #64445F;
        }
        [data-testid="stAlert"] p, [data-testid="stAlert"] span {color: #F7EDF6;}
        [data-testid="stSpinner"] p {color: #F7EDF6;}
        hr {border-color: #4B344C;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    assumptions = render_assumptions()
    st.title("☕ Campus Café")
    st.markdown(
        "### Launch Team"
    )
    st.write(
        "A warm, clear workspace for building an evidence-aware launch proposal. "
        "Four AI specialists research, calculate, write, and challenge the brief "
        "before it is saved."
    )
    st.caption(
        "When you generate a brief, your local computer may take a few minutes "
        "to complete the research and review. Keep this page open; we’ll show "
        "simple progress updates as the brief comes together."
    )

    with st.expander("How the agent team works", expanded=False):
        st.markdown(
            "- **Market Researcher** searches for student, pricing, and trend evidence.\n"
            "- **Financial Analyst** uses a calculator tool to show every break-even step.\n"
            "- **Marketing Writer** turns both handoffs into a one-page brief.\n"
            "- **Skeptical Reviewer** identifies weaknesses, then produces the final version.\n\n"
            "The research and financial outputs are passed explicitly to the next tasks. "
            "National evidence is not treated as proof of local demand. If public search "
            "is unavailable, the researcher has a small, cited local reference pack and "
            "is instructed to disclose that it was used."
        )

    model_ready, model_status = check_local_model() if not DEMO_MODE else (
        False,
        "No-cost Render demo: AI generation is disabled; the reviewed sample brief is shown below.",
    )
    if DEMO_MODE:
        st.info(model_status)
        sample_path = ROOT / "final_brief.md"
        if "brief" not in st.session_state and sample_path.is_file():
            st.session_state["brief"] = sample_path.read_text(encoding="utf-8")
    elif model_ready:
        st.success(model_status)
    else:
        st.warning(model_status)
        st.markdown(
            "This app uses a **local Ollama model**. Your prompts and brief are "
            "processed on this machine; no model API key is needed. Public web search "
            "requires an internet connection; a small cited reference pack is available "
            "if online search fails."
        )
    st.caption(
        f"Estimated fixed costs in this scenario: ${assumptions['fixed_costs']:,.2f}/month · "
        f"Average ticket: ${assumptions['average_ticket']:.2f} · "
        f"Open {assumptions['open_days']} days/month"
    )

    generation_running = st.session_state.get("generation_state") == "running"
    if st.button(
        "AI generation is disabled in this free demo"
        if DEMO_MODE
        else "Create my café launch brief",
        type="primary",
        disabled=DEMO_MODE or not model_ready or generation_running,
        use_container_width=True,
    ):
        try:
            job_id = uuid.uuid4().hex
            job_directory = Path(tempfile.gettempdir()) / "campus-cafe-launch-team"
            job_directory.mkdir(parents=True, exist_ok=True)
            status_path = job_directory / f"{job_id}.json"
            log_path = job_directory / f"{job_id}.log"
            write_job_status(status_path, state="running", progress=0)
            worker_environment = os.environ.copy()
            worker_environment["CAFE_JOB_STATUS_PATH"] = str(status_path)
            with log_path.open("w", encoding="utf-8") as worker_log:
                worker = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "brief_worker.py"),
                        job_id,
                        json.dumps(assumptions),
                        str(status_path),
                    ],
                    cwd=ROOT,
                    env=worker_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            _JOB_PROCESSES[job_id] = worker
            st.session_state["generation_job_id"] = job_id
            st.session_state["generation_status_path"] = str(status_path)
            st.session_state["generation_log_path"] = str(log_path)
            st.session_state["generation_state"] = "running"
            st.session_state["generation_progress"] = 0
            st.session_state["generation_started"] = time.time()
        except Exception as exc:
            st.error(
                "We couldn’t start the brief. Please refresh the page and try again."
            )
            with st.expander("More details (optional)"):
                st.caption(str(exc) or "No additional details were provided.")

    if st.session_state.get("generation_state") == "running":
        render_generation_status()
    elif st.session_state.get("generation_state") == "complete":
        st.success("Your reviewed café launch brief is ready below.")
    elif st.session_state.get("generation_state") == "error":
        st.error(
            "We couldn’t finish your brief. Please try again; if it keeps "
            "happening, check that the local assistant is running."
        )
        with st.expander("More details (optional)"):
            st.caption(st.session_state.get("generation_error", "No additional details."))

    if "brief" in st.session_state:
        st.divider()
        st.subheader("Final launch brief")
        st.markdown(st.session_state["brief"])
        st.download_button(
            "Download final_brief.md",
            data=st.session_state["brief"],
            file_name="final_brief.md",
            mime="text/markdown",
        )
        with st.expander("Review notes and agent handoffs"):
            st.markdown("**Skeptical review**")
            st.write(st.session_state.get("critique", "No critique output available."))
            st.markdown("**Market research**")
            st.write(st.session_state.get("research", "No research output available."))
            st.markdown("**Financial analysis**")
            st.write(st.session_state.get("finance", "No financial output available."))

    st.divider()
    st.caption(
        "Decision support only: validate local demand, supplier prices, labor, "
        "facilities, and university policies before committing funds."
    )


@st.fragment(run_every="1s")
def render_generation_status() -> None:
    job_id = st.session_state.get("generation_job_id")
    status_path = Path(st.session_state["generation_status_path"])
    try:
        job_status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        job_status = {}
    process = _JOB_PROCESSES.get(job_id)
    if process is not None and process.poll() is not None and job_status.get("state") == "running":
        log_path = Path(st.session_state.get("generation_log_path", ""))
        st.session_state["generation_state"] = "error"
        st.session_state["generation_error"] = (
            log_path.read_text(encoding="utf-8")[-1200:]
            if log_path.is_file()
            else "The brief-writing process stopped unexpectedly."
        )
        st.rerun()
        return
    state = job_status.get("state")
    if state == "complete":
        st.session_state["brief"] = job_status["brief"]
        st.session_state["critique"] = job_status.get("critique", "")
        st.session_state["research"] = job_status.get("research", "")
        st.session_state["finance"] = job_status.get("finance", "")
        st.session_state["generation_state"] = "complete"
        _JOB_PROCESSES.pop(job_id, None)
        st.rerun(scope="app")
        return
    if state == "error":
        st.session_state["generation_state"] = "error"
        st.session_state["generation_error"] = job_status.get(
            "error", "The brief could not be completed."
        )
        _JOB_PROCESSES.pop(job_id, None)
        st.rerun(scope="app")
        return
    progress = int(job_status.get("progress", 0))
    started = float(st.session_state.get("generation_started", time.time()))
    elapsed_seconds = max(0, int(time.time() - started))
    elapsed_minutes = elapsed_seconds // 60
    elapsed = (
        f"It has been working for {elapsed_minutes} "
        f"{'minute' if elapsed_minutes == 1 else 'minutes'}"
        if elapsed_minutes
        else "Just getting started"
    )
    if progress < len(_PROGRESS_STEPS):
        current_step = _PROGRESS_STEPS[progress]
        step_label = (
            "Getting started"
            if progress == 0
            else f"{progress} of {len(_PROGRESS_STEPS)} steps complete"
        )
        st.progress(progress / len(_PROGRESS_STEPS), text=step_label)
        st.info(
            f"**{current_step}.** Your café brief is being prepared. "
            f"{elapsed}. Local models can take a few minutes; "
            "you can keep this page open."
        )


if __name__ == "__main__":
    main()
