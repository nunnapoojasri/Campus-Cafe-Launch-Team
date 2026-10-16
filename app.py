"""Interactive Campus Café Launch Team brief generator."""

from __future__ import annotations

import ast
import atexit
import base64
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
_TASK_COMPLETED = 0
_TASK_OFFSET = 0
_TASK_TOTAL = 1
_PROGRESS_STEPS = (
    ("Market Researcher", "gathering campus and café evidence"),
    ("Financial Analyst", "calculating the monthly break-even point"),
    ("Sustainability Officer", "checking environmental claims"),
    ("Marketing Writer", "drafting the launch brief"),
    ("Skeptical Reviewer", "checking risks and evidence"),
    ("Skeptical Reviewer", "correcting and finalizing the brief"),
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


@st.cache_data(ttl=20, show_spinner=False)
def check_local_model(
    base_url: str = OLLAMA_BASE_URL,
    configured_model: str = DEFAULT_MODEL,
) -> tuple[bool, str]:
    model_name = configured_model.removeprefix("ollama/").lower()
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=2) as response:
            payload = json.loads(response.read())
        installed = [model.get("name", "").lower() for model in payload.get("models", [])]
        if any(
            name == model_name or name.split(":")[0] == model_name.split(":")[0]
            for name in installed
        ):
            return True, f"Local model ready: `{configured_model}`"
        return (
            False,
            f"Ollama is running, but `{configured_model}` is not installed. "
            f"Run `ollama pull {model_name}`.",
        )
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        return (
            False,
            "Local Ollama is not running. Install Ollama, start it, and pull the "
            f"model with `ollama pull {model_name}`. ({exc})",
        )


@st.cache_data(show_spinner=False)
def read_illustration(filename: str) -> str:
    """Read a bundled illustration; no external image service is used."""
    asset_path = ROOT / "assets" / filename
    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


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
    global _TASK_COMPLETED
    status_path = os.getenv("CAFE_JOB_STATUS_PATH")
    if not status_path:
        return
    _TASK_COMPLETED += 1
    completed = min(_TASK_OFFSET + _TASK_COMPLETED, _TASK_TOTAL)
    write_job_status(
        status_path,
        state="running",
        completed_tasks=completed,
        total_tasks=_TASK_TOTAL,
    )


def read_usage_metrics(crew: Any) -> dict[str, int]:
    usage = getattr(crew, "usage_metrics", None)
    if usage is None:
        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    return {
        "total_tokens": int(getattr(usage, "total_tokens", 0)),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0)),
    }


def make_calculator_tool(failure_mode: str) -> Any:
    from langchain_core.tools import tool

    calls = 0

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate basic arithmetic; optional failure modes are for the exercise lab."""
        nonlocal calls
        calls += 1
        if failure_mode == "fail_once" and calls == 1:
            return "Simulated tool failure: temporary calculator outage. Retry once."
        if failure_mode == "always_fail":
            return "Simulated tool failure: calculator unavailable for this run."
        return calculator(expression)

    return calculator_tool


def run_brief_generation(
    job_id: str,
    assumptions: dict[str, Any],
    status_path: str,
    options: dict[str, Any] | None = None,
    experiment: str = "brief",
) -> None:
    global _TASK_COMPLETED, _TASK_OFFSET, _TASK_TOTAL
    options = options or {}
    try:
        variants: list[tuple[str, dict[str, Any]]] = []
        if experiment == "compare_processes":
            variants = [
                ("Sequential", {**options, "process_mode": "sequential"}),
                ("Hierarchical", {**options, "process_mode": "hierarchical"}),
            ]
        elif experiment == "compare_reviewers":
            variants = [
                ("Skeptical reviewer", {**options, "reviewer_tone": "skeptical"}),
                ("Cheerleader reviewer", {**options, "reviewer_tone": "cheerleader"}),
            ]
        else:
            variants = [("Brief", options)]

        crews_and_tasks = [
            (
                name,
                *create_crew(
                    assumptions,
                    task_callback=publish_task_progress,
                    options=variant,
                ),
            )
            for name, variant in variants
        ]
        _TASK_COMPLETED = 0
        _TASK_OFFSET = 0
        _TASK_TOTAL = sum(len(tasks) for _, _, tasks in crews_and_tasks)
        write_job_status(
            status_path,
            state="running",
            completed_tasks=0,
            total_tasks=_TASK_TOTAL,
        )

        results = []
        for index, (name, crew, tasks) in enumerate(crews_and_tasks):
            _TASK_OFFSET = sum(len(item[2]) for item in crews_and_tasks[:index])
            _TASK_COMPLETED = 0
            started = time.monotonic()
            result = crew.kickoff()
            duration = time.monotonic() - started
            brief = get_task_output(tasks[-1]).strip() or str(result).strip()
            results.append(
                {
                    "name": name,
                    "brief": brief,
                    "duration_seconds": round(duration, 2),
                    "usage": read_usage_metrics(crew),
                    "task_count": len(tasks),
                }
            )

        if experiment == "brief":
            result = results[0]
            if not result["brief"]:
                raise RuntimeError("The reviewer returned an empty brief.")
            OUTPUT_PATH.write_text(result["brief"] + "\n", encoding="utf-8")
            first_tasks = crews_and_tasks[0][2]
            sustainability_task = next(
                (
                    task
                    for task in first_tasks
                    if getattr(getattr(task, "agent", None), "role", "")
                    == "Sustainability Officer"
                ),
                None,
            )
            write_job_status(
                status_path,
                state="complete",
                completed_tasks=_TASK_TOTAL,
                total_tasks=_TASK_TOTAL,
                brief=result["brief"],
                critique=get_task_output(first_tasks[-2]),
                research=get_task_output(first_tasks[0]),
                finance=get_task_output(first_tasks[1]),
                sustainability=(
                    get_task_output(sustainability_task)
                    if sustainability_task
                    else ""
                ),
                exercise_result=None,
            )
        else:
            write_job_status(
                status_path,
                state="complete",
                completed_tasks=_TASK_TOTAL,
                total_tasks=_TASK_TOTAL,
                exercise_result={
                    "kind": experiment,
                    "results": results,
                    "cost_note": (
                        "Both runs used the local Ollama model, so model API "
                        "charges are $0. This does not include electricity or "
                        "hardware costs. Token counts are reported when available."
                    ),
                },
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
    options: dict[str, Any] | None = None,
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
    options = options or {}
    process_mode = options.get("process_mode", "sequential")
    reviewer_tone = options.get("reviewer_tone", "skeptical")
    include_sustainability = bool(options.get("include_sustainability", True))
    share_research = bool(options.get("share_research", True))

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
    calculator_tool = wrap_langchain_tool(
        make_calculator_tool(options.get("calculator_failure", "none"))
    )

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
        max_iter=3,
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
            if reviewer_tone == "skeptical"
            else "You are an enthusiastic campus ambassador. You focus on the café's "
            "strengths and make its case sound optimistic, while still avoiding "
            "factual inventions; this tone may under-emphasize risks."
        ),
        llm=llm,
        allow_delegation=True,
        verbose=False,
        max_iter=4,
    )

    campus_context = assumptions.get("campus_context") or "campus location not specified"
    cafe_concept = assumptions.get("cafe_concept") or "café concept not specified"
    assumption_text = "\n".join(
        [
            f"- Campus / city: {campus_context} (user-provided scenario context).",
            f"- Café concept: {cafe_concept} (user-provided scenario description).",
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
            f"in {campus_context} with {assumptions['students']:,} students. The proposed "
            f"concept is: {cafe_concept}. Use web search and Wikipedia "
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
    calculator_failure = options.get("calculator_failure", "none")
    failure_instructions = (
        "The calculator is configured to fail once with a temporary error. Retry "
        "the exact failed expression one time, then show whether the retry recovered."
        if calculator_failure == "fail_once"
        else "The calculator is configured to keep failing. Do not guess or do "
        "mental arithmetic as a fallback; identify the unavailable calculation "
        "and explicitly state that break-even cannot be verified."
        if calculator_failure == "always_fail"
        else "If the calculator returns an error, report it clearly and do not "
        "guess or invent a calculated result."
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
            "seasonality, and debt service. "
            f"{failure_instructions}\n\n"
            f"Planning assumptions:\n{assumption_text}"
        ),
        expected_output=(
            "A transparent assumption table and auditable monthly and daily "
            "break-even arithmetic, including exclusions."
        ),
        agent=analyst,
        context=[research_task] if share_research else [],
    )
    sustainability_agent = None
    sustainability_task = None
    if include_sustainability:
        sustainability_agent = Agent(
            role="Sustainability Officer",
            goal="Check environmental and plant-based claims for evidence and limits.",
            backstory=(
                "You are a campus sustainability advisor. You distinguish sourced "
                "environmental facts from proposed actions and never claim a practice "
                "reduces impact without supporting evidence."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
        )
        sustainability_task = Task(
            description=(
                "Check the research findings for any environmental, sustainability, "
                "or plant-based claims. For each, classify it as (1) sourced fact, "
                "(2) proposed pilot action, or (3) unsupported claim to remove. Do "
                "not invent lifecycle or emissions benefits. Return only claims the "
                "writer can safely use, their citations, and the major evidence gaps."
            ),
            expected_output="A short verified-claims list, citations, and limitations.",
            agent=sustainability_agent,
            context=[research_task],
        )
    draft_task = Task(
        description=(
            "Draft a decision-ready, one-page Markdown launch brief for the Dean. "
            f"Make the recommendation specific to this scenario: {campus_context}; "
            f"concept: {cafe_concept}. "
            "Use the research findings and financial analysis provided in context. "
            "Include an executive recommendation, evidence-led opportunity, concise "
            "pricing/trend implications, break-even arithmetic with assumptions, "
            "the most important risks and next steps, and source links. Keep it "
            "roughly 300-450 words. Mark estimates as assumptions and don't assert "
            "national data proves local demand."
        ),
        expected_output="A concise one-page Markdown draft with sources and arithmetic.",
        agent=writer,
        context=[
            research_task,
            finance_task,
            *([sustainability_task] if sustainability_task else []),
        ],
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
        context=[
            draft_task,
            research_task,
            finance_task,
            *([sustainability_task] if sustainability_task else []),
        ],
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
        context=[
            draft_task,
            critique_task,
            research_task,
            finance_task,
            *([sustainability_task] if sustainability_task else []),
        ],
    )
    agents = [researcher, analyst, writer, reviewer]
    if sustainability_agent:
        agents.append(sustainability_agent)
    tasks = [research_task, finance_task]
    if sustainability_task:
        tasks.append(sustainability_task)
    tasks.extend([draft_task, critique_task, final_task])
    crew_config = {
        "agents": agents,
        "tasks": tasks,
        "process": (
            Process.hierarchical if process_mode == "hierarchical" else Process.sequential
        ),
        "verbose": False,
        "task_callback": task_callback,
    }
    if process_mode == "hierarchical":
        crew_config["manager_llm"] = llm
    crew = Crew(**crew_config)
    return crew, tasks


def render_assumptions() -> dict[str, Any]:
    st.sidebar.header("Planning assumptions")
    st.sidebar.caption(
        "These are editable scenario inputs—not vendor quotes or a demand forecast."
    )
    students = st.sidebar.number_input(
        "Enrolled students", min_value=1, value=8000, step=100
    )
    campus_context = st.sidebar.text_input(
        "Campus / city", value="Your campus", max_chars=120
    )
    cafe_concept = st.sidebar.text_area(
        "What is your café idea?",
        value="A welcoming campus café with affordable coffee, quick meals, and plant-based options.",
        max_chars=500,
        height=100,
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
        "campus_context": campus_context.strip(),
        "cafe_concept": cafe_concept.strip(),
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
                radial-gradient(ellipse at 8% 3%, rgba(235, 133, 176, .17), transparent 27%),
                radial-gradient(ellipse at 90% 8%, rgba(146, 128, 222, .16), transparent 29%),
                radial-gradient(ellipse at 65% 95%, rgba(93, 183, 162, .12), transparent 30%),
                #fff9fc;
        }
        [data-testid="stHeader"] {background: rgba(255, 249, 252, .94);}
        [data-testid="stSidebar"] {
            background:
                linear-gradient(160deg, rgba(225, 137, 177, .2), transparent 42%),
                linear-gradient(330deg, rgba(139, 137, 210, .16), transparent 45%),
                #f7eff5;
            border-right: 1px solid #ead8e8;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {color: #53334c;}
        [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {color: #493b4e;}
        h1, h2, h3 {color: #39243e;}
        h1 {letter-spacing: -.035em;}
        h1::after {
            content: "";
            display: block;
            width: min(300px, 52vw);
            height: 4px;
            margin-top: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, #F06C9B, #F4A261, #E9C46A, #66B58B, #58A6C8, #9B7ED8);
            box-shadow: 0 0 16px rgba(190, 87, 151, .2);
        }
        [data-testid="stNumberInput"] input {
            background: #fffdfd;
            color: #35283A;
            border-color: #d8bfd5;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(255, 253, 255, .98), rgba(247, 239, 250, .98));
            border: 1px solid #e5d2e7;
            border-radius: 14px;
            padding: 14px 18px;
        }
        [data-testid="stExpander"] {
            background: rgba(255, 253, 255, .88);
            border: 1px solid #e5d2e7;
            border-radius: 12px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 2.8rem;
            border: 1px solid rgba(106, 48, 104, .25);
            background: linear-gradient(105deg, #B63870, #84458C 36%, #535FA2 68%, #267A78);
            color: #FFFFFF;
            box-shadow: 0 7px 18px rgba(125, 53, 119, .16);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #C96A9E;
            color: #FFFFFF;
            filter: brightness(1.12);
        }
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #F06C9B, #F4A261, #E9C46A, #66B58B, #58A6C8, #9B7ED8);
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
            background-color: #F7EFF7;
            border-color: #DFCADE;
        }
        [data-testid="stAlert"] p, [data-testid="stAlert"] span {color: #49354B;}
        [data-testid="stSpinner"] p {color: #49354B;}
        hr {border-color: #E5D5E5;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    assumptions = render_assumptions()
    hero_text, hero_picture = st.columns([1.55, 1], vertical_alignment="center")
    with hero_text:
        st.title("☕ Campus Café")
        st.markdown("### Launch Team")
        st.write(
            "Turn your café idea into a clear, evidence-aware launch brief. "
            "Set a few planning numbers, then let the team do the research, "
            "budgeting, writing, and review."
        )
    with hero_picture:
        st.markdown(
            f'<img src="{read_illustration("cafe-hero.svg")}" '
            'alt="Illustration of a welcoming campus café with coffee, tulips, '
            'and a pastry" style="width:100%;max-height:230px;object-fit:contain;">',
            unsafe_allow_html=True,
        )

    st.subheader("Your AI team")
    st.caption(
        f"Five specialists collaborate using one local Ollama model ({DEFAULT_MODEL}). "
        "Research and calculations are handed forward into the draft and review."
    )
    agent_columns = st.columns(5)
    agents = [
        ("Market Researcher", "Finds evidence and cites its limits."),
        ("Financial Analyst", "Calculates an auditable break-even."),
        ("Sustainability Officer", "Checks environmental claims."),
        ("Marketing Writer", "Builds the decision-ready brief."),
        ("Skeptical Reviewer", "Challenges gaps and corrects the draft."),
    ]
    for column, (role, description) in zip(agent_columns, agents):
        with column:
            with st.container(border=True):
                st.markdown(f"**{role}**")
                st.caption(description)

    model_ready, model_status = check_local_model(
        OLLAMA_BASE_URL, DEFAULT_MODEL
    ) if not DEMO_MODE else (
        False,
        "Hosted sample mode: the reviewed example is shown here. Run the app locally with Ollama to generate a brief.",
    )
    generation_running = st.session_state.get("generation_state") == "running"

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

    clicked_normal = st.button(
        "AI generation is disabled in this free demo"
        if DEMO_MODE
        else "Create my café launch brief",
        type="primary",
        disabled=DEMO_MODE or not model_ready or generation_running,
        use_container_width=True,
    )
    if clicked_normal:
        try:
            job_id = uuid.uuid4().hex
            job_directory = Path(tempfile.gettempdir()) / "campus-cafe-launch-team"
            job_directory.mkdir(parents=True, exist_ok=True)
            status_path = job_directory / f"{job_id}.json"
            log_path = job_directory / f"{job_id}.log"
            write_job_status(
                status_path,
                state="running",
                completed_tasks=0,
                total_tasks=1,
                experiment=experiment,
            )
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
                        json.dumps({}),
                        "brief",
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
            st.session_state["generation_experiment"] = experiment
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
            if st.session_state.get("sustainability"):
                st.markdown("**Sustainability check**")
                st.write(st.session_state["sustainability"])

    st.divider()
    st.caption(
        "Decision support only: validate local demand, supplier prices, labor, "
        "facilities, and university policies before committing funds."
    )


@st.fragment(run_every="2s")
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
        if job_status.get("brief"):
            st.session_state["brief"] = job_status["brief"]
            st.session_state["critique"] = job_status.get("critique", "")
            st.session_state["research"] = job_status.get("research", "")
            st.session_state["finance"] = job_status.get("finance", "")
            st.session_state["sustainability"] = job_status.get("sustainability", "")
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
    completed = int(job_status.get("completed_tasks", 0))
    total = max(1, int(job_status.get("total_tasks", 1)))
    progress = min(1.0, completed / total)
    stage_index = min(len(_PROGRESS_STEPS) - 1, completed)
    started = float(st.session_state.get("generation_started", time.time()))
    elapsed_seconds = max(0, int(time.time() - started))
    elapsed_minutes = elapsed_seconds // 60
    elapsed = (
        f"It has been working for {elapsed_minutes} "
        f"{'minute' if elapsed_minutes == 1 else 'minutes'}"
        if elapsed_minutes
        else "Just getting started"
    )
    current_agent, current_step = _PROGRESS_STEPS[stage_index]
    st.progress(
        progress,
        text=(
            f"{completed} of {int(job_status.get('total_tasks', 1))} team steps complete"
            if completed
            else "Getting started"
        ),
    )
    st.info(
        f"**{current_agent} is {current_step}.** Your café brief is being prepared. "
        f"{elapsed}. Local models can take a few minutes; "
        "you can keep this page open."
    )


if __name__ == "__main__":
    main()
