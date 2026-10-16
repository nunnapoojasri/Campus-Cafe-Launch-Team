"""Interactive Campus Café Launch Team brief generator."""

from __future__ import annotations

import ast
import atexit
import base64
import json
import math
import operator
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Type
from urllib.parse import urlencode, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", str(ROOT / "final_brief.md")))
REFLECTION_PATH = Path(
    os.getenv("REFLECTION_PATH", str(ROOT / "collaboration_reflection.md"))
)
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
    ("Marketing Writer", "drafting the launch brief"),
    ("Skeptical Reviewer", "correcting and finalizing the brief"),
)


class ResearchFinding(BaseModel):
    title: str = Field(min_length=2)
    url: str = Field(min_length=8)
    published_date: str | None = None
    source_method: Literal["duckduckgo", "wikipedia", "curated_reference"]
    source_excerpt: str = Field(min_length=10)
    finding: str = Field(min_length=10)
    evidence_scope: Literal["campus", "national", "industry", "background"]
    supports: str = Field(min_length=5)
    limitations: str = Field(min_length=5)

    @field_validator("url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Research sources must use a valid HTTP(S) URL.")
        return value


class ResearchReport(BaseModel):
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=8)
    campus_specific_evidence_available: bool
    limitations: list[str] = Field(min_length=1)


class ReviewIssue(BaseModel):
    weakness: str = Field(min_length=15)
    correction: str = Field(min_length=15)


class ReviewCritique(BaseModel):
    issues: list[ReviewIssue] = Field(min_length=3, max_length=3)


class ReviewedBrief(BaseModel):
    corrections_applied: list[str] = Field(min_length=3, max_length=3)
    final_brief: str = Field(min_length=100, max_length=6000)


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
    if not isinstance(expression, str) or len(expression) > 256:
        return "Error: expression must be text no longer than 256 characters."
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
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Only basic arithmetic expressions are supported.")

    try:
        parsed = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(parsed)) > 64:
            return "Error: expression is too complex."
        result = evaluate(parsed)
        if not math.isfinite(result) or not (-1e15 < result < 1e15):
            return "Error: result is outside the supported range."
        formatted = f"{result:.8f}"
        return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
    except (OverflowError, RecursionError, SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
        return f"Error: {exc}"





def verified_reference_notes(topic: str) -> str:
    """Return audited offline source records without implying live search."""
    results = [
        {
            "title": "NCA 2026 National Coffee Data Trends Specialty Coffee Report",
            "url": "https://www.ncausa.org/Newsroom/Coffee-Consumption-Statistics",
            "published_date": "2026-06-02",
            "snippet": (
                "NCA reports 47% of American adults had specialty coffee in the "
                "past day; 38% of past-day coffee drinkers had coffee out of home, "
                "the highest share since January 2020. U.S. adult category context only."
            ),
        },
        {
            "title": "The Hope Center Student Basic Needs Survey",
            "url": "https://hope.temple.edu/research/hope-center-basic-needs-survey",
            "published_date": None,
            "snippet": (
                "A survey used to document participating institutions' student "
                "basic needs. The page does not provide café spending estimates "
                "or establish demand at this campus."
            ),
        },
        {
            "title": "IFIC 2025 Food & Health Survey",
            "url": "https://ific.org/research/2025-food-health-survey/",
            "published_date": "2026-01-13",
            "snippet": (
                "A U.S. consumer food and health survey. It does not establish "
                "campus-specific plant-based demand or local café prices."
            ),
        },
    ]
    return json.dumps(
        {
            "status": "curated",
            "topic": topic,
            "results": results,
            "limitations": [
                "These are audited offline references, not live search results.",
                "No campus-specific student spending, willingness-to-pay, or competing-menu evidence is supplied.",
            ],
        }
    )


def get_task_output(task: Any) -> str:
    output = getattr(task, "output", None)
    raw = getattr(output, "raw", None)
    return str(raw if raw is not None else output or "")


def get_structured_task_output(task: Any, schema: Type[BaseModel]) -> BaseModel:
    output = getattr(task, "output", None)
    structured = getattr(output, "pydantic", None)
    if isinstance(structured, schema):
        return structured
    if isinstance(structured, BaseModel):
        return schema.model_validate(structured.model_dump())
    if isinstance(structured, dict):
        return schema.model_validate(structured)
    raw = getattr(output, "raw", None)
    if isinstance(raw, str) and raw.strip():
        return schema.model_validate_json(raw)
    raise ValueError(f"CrewAI did not return validated {schema.__name__} output.")


def validate_research_provenance(
    report: ResearchReport,
    source_records: list[dict[str, Any]],
    tool_observations: list[str] | None = None,
) -> None:
    by_url: dict[str, list[dict[str, Any]]] = {}
    for record in source_records:
        by_url.setdefault(str(record.get("url", "")), []).append(record)
    for finding in report.findings:
        matches = by_url.get(finding.url, [])
        if not matches:
            raise ValueError(f"Research cited a URL not returned by a tool: {finding.url}")
        matching_source = next(
            (
                record
                for record in matches
                if " ".join(str(record.get("title", "")).split()).casefold()
                == " ".join(finding.title.split()).casefold()
                and record.get("published_date") == finding.published_date
                and record.get("source_method") == finding.source_method
                and " ".join(str(record.get("snippet", "")).split()).casefold()
                == " ".join(finding.source_excerpt.split()).casefold()
            ),
            None,
        )
        if matching_source is None:
            raise ValueError(
                f"Research title, date, or source type did not match tool output for {finding.url}."
            )
    has_campus_finding = any(finding.evidence_scope == "campus" for finding in report.findings)
    if report.campus_specific_evidence_available != has_campus_finding:
        raise ValueError("Research report misclassified campus-specific evidence availability.")
    if not has_campus_finding and not any(
        "campus" in limitation.casefold() or "local" in limitation.casefold()
        for limitation in report.limitations
    ):
        raise ValueError("Research limitations must disclose the lack of campus-specific evidence.")
    disclosed_limitations = " ".join(report.limitations).casefold()
    for observation in tool_observations or []:
        if observation.casefold() not in disclosed_limitations:
            raise ValueError(f"Research limitations must disclose tool observation: {observation}")


def render_research_report(report: ResearchReport) -> str:
    sections = []
    for finding in report.findings:
        date_label = finding.published_date or "Publication date unavailable"
        sections.append(
            f"### {finding.title}\n"
            f"{finding.finding}\n\n"
            f"> Source excerpt: {finding.source_excerpt}\n\n"
            f"- Evidence scope: {finding.evidence_scope}\n"
            f"- Date: {date_label}\n"
            f"- What it supports: {finding.supports}\n"
            f"- Limitations: {finding.limitations}\n"
            f"- Source type: {finding.source_method}\n"
            f"- Source: {finding.url}"
        )
    if not sections:
        sections.append("No source-backed findings were returned by the research tools.")
    limitations = "\n".join(f"- {item}" for item in report.limitations)
    return "## Research findings\n\n" + "\n\n".join(sections) + "\n\n## Limitations\n" + limitations


def render_research_sources(report: ResearchReport) -> str:
    if not report.findings:
        return "## Sources\n\nNo verified sources were returned by the research tools."
    entries = []
    for finding in report.findings:
        date_label = finding.published_date or "date unavailable"
        entries.append(
            f"- [{finding.title}]({finding.url}) ({date_label}; "
            f"{finding.evidence_scope}; {finding.source_method})"
        )
    return "## Verified sources\n\n" + "\n".join(entries)


def render_financial_audit(assumptions: dict[str, Any], plan: dict[str, Any]) -> str:
    rows = [
        "## Financial assumptions and audited break-even",
        "All inputs below are planning assumptions, not quotes or a demand forecast.",
    ]
    breakdown = assumptions.get("fixed_cost_breakdown")
    if isinstance(breakdown, dict):
        rows.extend(f"- {name}: ${float(value):,.2f} (assumption)" for name, value in breakdown.items())
    rows.extend(
        [
            f"- Monthly fixed costs: ${plan['fixed_costs']:,.2f} (assumption)",
            f"- Average transaction: ${plan['average_ticket']:,.2f} (assumption)",
            (
                f"- Variable cost per customer: ${plan['variable_cost']:,.2f} (assumption)"
                if "variable_cost_per_customer" in assumptions
                else f"- Variable cost rate: {plan['variable_cost_pct']:.2f}% of sales (assumption)"
            ),
            (
                f"- Variable cost rate: {plan['variable_cost_pct']:.2f}% of sales (derived using calculator)"
                if "variable_cost_per_customer" in assumptions
                else f"- Variable cost per customer: ${plan['variable_cost']:,.2f} (derived using calculator)"
            ),
            f"- Operating days per month: {plan['open_days']} (assumption)",
            f"- Contribution per customer: ${plan['average_ticket']:,.2f} - ${plan['variable_cost']:,.2f} = ${plan['contribution']:,.2f}",
            f"- Exact monthly break-even volume: ${plan['fixed_costs']:,.2f} / ${plan['contribution']:,.2f} = {plan['exact_monthly_customers']:,.8f} customers; minimum whole-customer target = **{plan['minimum_monthly_customers']:,}**.",
            f"- Exact daily volume: {plan['exact_monthly_customers']:,.8f} / {plan['open_days']} = {plan['exact_daily_customers']:,.8f}; minimum target = **{plan['minimum_daily_customers']:,} customers per open day**.",
            f"- Gross sales at exact break-even: {plan['exact_monthly_customers']:,.8f} x ${plan['average_ticket']:,.2f} = **${plan['exact_break_even_revenue']:,.2f}**. This is revenue, not profit.",
            f"- Gross sales at the rounded monthly target: {plan['minimum_monthly_customers']:,} x ${plan['average_ticket']:,.2f} = ${plan['minimum_target_revenue']:,.2f}.",
            "- Excludes start-up costs, financing, taxes, seasonality, and any costs not listed above.",
        ]
    )
    return "\n".join(rows)


def build_collaboration_reflection(
    research: ResearchReport,
    critique: ReviewCritique,
) -> str:
    evidence_summary = (
        f"The researcher returned {len(research.findings)} findings tied to observed sources."
        if research.findings
        else "The research tools returned no source-backed findings."
    )
    issues = "\n".join(
        f"{index}. {issue.weakness} Correction: {issue.correction}"
        for index, issue in enumerate(critique.issues, start=1)
    )
    limitations = "\n".join(f"- {item}" for item in research.limitations)
    return (
        "# Collaboration Reflection\n\n"
        "The Market Researcher gathered source-backed evidence and stated its scope; "
        "the Financial Analyst used the calculator-backed assumptions to produce an "
        "auditable break-even estimate; and the Marketing Writer turned those inputs "
        "into a concise recommendation. The Skeptical Reviewer examined the draft "
        "against both upstream outputs and recorded three specific corrections.\n\n"
        f"{evidence_summary} The review findings and corrections were:\n\n{issues}\n\n"
        "## Remaining limitations\n\n"
        f"{limitations}\n\n"
        "The collaboration improves traceability by keeping evidence, calculations, "
        "drafting, and review as separate stages. It does not independently validate "
        "the sources or make planning assumptions reliable; local demand, prices, "
        "staffing, and facilities costs still require verification."
    )


def write_report_atomically(path: str | Path, content: str) -> None:
    report_path = Path(path)
    temporary_path = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        temporary_path.replace(report_path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError(f"Could not safely write report to {report_path}: {exc}") from exc


def render_review_critique(critique: ReviewCritique) -> str:
    return "\n".join(
        f"{index}. **Weakness:** {issue.weakness}\n   **Correction:** {issue.correction}"
        for index, issue in enumerate(critique.issues, start=1)
    )


def validate_reviewed_brief(
    critique: ReviewCritique,
    reviewed: ReviewedBrief,
    verified_source_urls: set[str],
) -> None:
    normalize = lambda value: " ".join(value.split()).casefold()
    expected = [normalize(issue.correction) for issue in critique.issues]
    applied = [normalize(item) for item in reviewed.corrections_applied]
    if applied != expected:
        raise ValueError("The final reviewer output did not carry all three critique corrections forward exactly.")
    word_count = len(reviewed.final_brief.split())
    if not 250 <= word_count <= 450:
        raise ValueError(f"The reviewed brief must be 250-450 words; received {word_count}.")
    if re.search(r"\$\s*\d", reviewed.final_brief):
        raise ValueError("The reviewed narrative contains currency values outside the audited financial ledger.")
    brief_without_urls = re.sub(r"https?://[^\s<>()]+", "", reviewed.final_brief)
    if re.search(r"\d", brief_without_urls):
        raise ValueError("The reviewed narrative must not introduce unaudited numerical assumptions or results.")
    unverified_urls = set(source_urls(reviewed.final_brief)) - verified_source_urls
    if unverified_urls:
        raise ValueError(
            "The final brief cited sources that were not returned by research tools: "
            + ", ".join(sorted(unverified_urls))
        )


def publish_task_progress(task_output: Any) -> None:
    """Write task milestones to disk so the UI can safely poll a worker process."""
    global _TASK_COMPLETED
    status_path = os.getenv("CAFE_JOB_STATUS_PATH")
    if not status_path:
        return
    _TASK_COMPLETED += 1
    completed = min(_TASK_OFFSET + _TASK_COMPLETED, _TASK_TOTAL)
    try:
        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    task_outputs = status.get("task_outputs", [])
    if not isinstance(task_outputs, list):
        task_outputs = []
    task_outputs.append(
        {
            "agent": str(getattr(task_output, "agent", "") or ""),
            "task": str(getattr(task_output, "name", "") or ""),
            "output": str(
                getattr(task_output, "raw", None)
                or getattr(task_output, "summary", None)
                or ""
            ),
        }
    )
    write_job_status(
        status_path,
        state="running",
        completed_tasks=completed,
        total_tasks=_TASK_TOTAL,
        task_outputs=task_outputs,
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
            first_tasks = crews_and_tasks[0][2]
            research_task, finance_task = first_tasks[0], first_tasks[1]
            critique_task, final_task = first_tasks[-2], first_tasks[-1]
            research_report = get_structured_task_output(research_task, ResearchReport)
            critique = get_structured_task_output(critique_task, ReviewCritique)
            reviewed = get_structured_task_output(final_task, ReviewedBrief)
            source_records = [
                record
                for research_tool in getattr(research_task.agent, "tools", [])
                for record in getattr(research_tool, "source_records", [])
            ]
            tool_observations = [
                observation
                for research_tool in getattr(research_task.agent, "tools", [])
                for observation in getattr(research_tool, "tool_observations", [])
            ]
            validate_research_provenance(
                research_report,
                source_records,
                tool_observations=tool_observations,
            )
            verified_source_urls = {str(record["url"]) for record in source_records}
            validate_reviewed_brief(critique, reviewed, verified_source_urls)
            calculator_calls = sum(
                int(getattr(tool, "invocation_count", 0))
                for tool in getattr(finance_task.agent, "tools", [])
                if getattr(tool, "name", "") == "calculator_tool"
            )
            if calculator_calls < 1:
                raise RuntimeError("The Financial Analyst did not invoke the calculator tool.")
            financial_plan = calculate_financial_plan(assumptions)
            if not financial_plan["valid"]:
                raise ValueError("Financial audit failed: " + "; ".join(financial_plan["errors"]))
            research_text = render_research_report(research_report)
            finance_text = get_task_output(finance_task).strip()
            if not finance_text:
                raise ValueError("The Financial Analyst returned no task output.")
            critique_text = render_review_critique(critique)
            correction_section = "## Reviewer corrections incorporated\n\n" + "\n".join(
                f"- {correction}" for correction in reviewed.corrections_applied
            )
            limitations_section = "## Evidence limitations\n\n" + "\n".join(
                f"- {limitation}" for limitation in research_report.limitations
            )
            final_report = "\n\n".join(
                [
                    reviewed.final_brief.strip(),
                    correction_section,
                    render_research_sources(research_report),
                    render_financial_audit(assumptions, financial_plan),
                    limitations_section,
                ]
            )
            reflection = build_collaboration_reflection(research_report, critique)
            write_report_atomically(OUTPUT_PATH, final_report)
            write_report_atomically(REFLECTION_PATH, reflection)
            sustainability_task = next(
                (
                    task
                    for task in first_tasks
                    if getattr(getattr(task, "agent", None), "role", "")
                    == "Sustainability Officer"
                ),
                None,
            )
            try:
                prior_status = json.loads(Path(status_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior_status = {}
            task_outputs = prior_status.get("task_outputs", [])
            for item in task_outputs:
                if item.get("agent", "").casefold() == "market researcher":
                    item["output"] = research_text
                elif item.get("agent", "").casefold() == "skeptical reviewer":
                    item["output"] = critique_text
            write_job_status(
                status_path,
                state="complete",
                completed_tasks=_TASK_TOTAL,
                total_tasks=_TASK_TOTAL,
                brief=final_report,
                draft=get_task_output(first_tasks[-3]),
                critique=critique_text,
                research=research_text,
                finance=finance_text,
                sustainability=(
                    get_task_output(sustainability_task)
                    if sustainability_task
                    else ""
                ),
                task_outputs=task_outputs,
                reflection=reflection,
                financial_audit=financial_plan,
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
    options = options or {}
    process_mode = options.get("process_mode", "sequential")
    reviewer_tone = options.get("reviewer_tone", "skeptical")
    include_sustainability = bool(options.get("include_sustainability", False))

    class ToolInput(BaseModel):
        query: str

    class LangChainToolAdapter(BaseTool):
        """Adapt a LangChain tool to CrewAI's tool interface."""

        name: str
        description: str
        args_schema: Type[BaseModel] = ToolInput
        lc_tool: Any
        source_records: list[dict[str, Any]] = Field(default_factory=list)
        tool_observations: list[str] = Field(default_factory=list)
        invocation_count: int = 0

        def _run(self, query: str) -> str:
            self.invocation_count += 1
            result = str(self.lc_tool.invoke(query))
            if self.name in {"web_search", "wikipedia_search", "verified_reference_notes"}:
                try:
                    payload = json.loads(result)
                except json.JSONDecodeError:
                    return json.dumps({
                        "status": "error",
                        "source": self.name,
                        "message": "Research tool returned invalid structured data.",
                        "results": [],
                    })
                source_method = {
                    "web_search": "duckduckgo",
                    "wikipedia_search": "wikipedia",
                    "verified_reference_notes": "curated_reference",
                }[self.name]
                source_name = str(payload.get("source") or self.name)
                if payload.get("status") == "error":
                    self.tool_observations.append(
                        f"{source_name}: {payload.get('message', 'tool failed')}"
                    )
                elif payload.get("status") == "empty":
                    self.tool_observations.append(f"{source_name}: no results returned")
                self.tool_observations.extend(
                    str(limitation)
                    for limitation in payload.get("limitations", [])
                    if str(limitation).strip()
                )
                self.source_records.extend(
                    {**record, "source_method": source_method}
                    for record in payload.get("results", [])
                    if isinstance(record, dict) and record.get("url")
                )
            return result

    def wrap_langchain_tool(lc_tool: Any) -> LangChainToolAdapter:
        return LangChainToolAdapter(
            name=lc_tool.name,
            description=lc_tool.description,
            lc_tool=lc_tool,
        )

    @tool
    def web_search(query: str) -> str:
        """Find current web results and return their actual titles, dates, URLs, and snippets."""
        try:
            with DDGS(timeout=5) as search:
                results = search.text(query, max_results=3, backend="duckduckgo")
            if not results:
                return json.dumps({"status": "empty", "source": "DuckDuckGo", "results": []})
            records = [
                {
                    "title": str(item.get("title") or "Untitled"),
                    "url": str(item.get("href") or ""),
                    "published_date": item.get("date") or item.get("published"),
                    "snippet": str(item.get("body") or ""),
                }
                for item in results
                if urlsplit(str(item.get("href") or "")).scheme in {"http", "https"}
                and urlsplit(str(item.get("href") or "")).netloc
            ]
            return json.dumps({"status": "ok" if records else "empty", "source": "DuckDuckGo", "results": records})
        except Exception as exc:
            return json.dumps({
                "status": "error",
                "source": "DuckDuckGo",
                "message": f"{type(exc).__name__}: {exc}",
                "results": [],
            })

    @tool
    def wikipedia_search(query: str) -> str:
        """Search Wikipedia and return actual result titles, URLs, and descriptions."""
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
            records = [
                {
                    "title": title,
                    "url": page_url,
                    "published_date": None,
                    "snippet": description,
                }
                for title, description, page_url in zip(data[1], data[2], data[3])
                if urlsplit(page_url).scheme in {"http", "https"} and urlsplit(page_url).netloc
            ]
            return json.dumps({"status": "ok" if records else "empty", "source": "Wikipedia", "results": records})
        except Exception as exc:
            return json.dumps({
                "status": "error",
                "source": "Wikipedia",
                "message": f"{type(exc).__name__}: {exc}",
                "results": [],
            })

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
            (
                f"- Variable cost per customer: ${assumptions['variable_cost_per_customer']:.3f} (planning assumption)."
                if "variable_cost_per_customer" in assumptions
                else f"- Variable costs: {assumptions['variable_cost_pct']:.1f}% of sales (planning assumption, including ingredients, packaging, and payment fees)."
            ),
            f"- Monthly fixed costs: ${assumptions['fixed_costs']:,.2f} (planning assumption; includes rent/occupancy, labor, equipment, and other operating costs).",
        ]
    )

    research_task = Task(
        description=(
            "Research current evidence for a proposed student café at a university "
            f"in {campus_context} with {assumptions['students']:,} students. The proposed "
            f"concept is: {cafe_concept}. Use web search and Wikipedia "
            "tools to find actual sources about student food/drink spending, café "
            "prices, and current food, sustainability, or plant-based trends. Return "
            "only findings grounded in tool results. Use exact returned titles, URLs, "
            "and dates; use null when a date is unavailable. Label evidence scope and "
            "what the source does and does not support. Do not infer campus demand from "
            "national or industry data. If campus-specific spending evidence is absent, "
            "set campus_specific_evidence_available to false and explicitly state the "
            "gap and recommended local validation. Search/provider errors and rate "
            "limits must appear in limitations. Do not invent sources, facts, or dates."
        ),
        expected_output="Structured evidence findings validated against actual tool results, plus limitations.",
        agent=researcher,
        output_pydantic=ResearchReport,
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
    financial_plan = calculate_financial_plan(assumptions)
    if not financial_plan["valid"]:
        raise ValueError("Invalid financial assumptions: " + "; ".join(financial_plan["errors"]))
    financial_audit = render_financial_audit(assumptions, financial_plan)
    finance_task = Task(
        description=(
            "Use your calculator tool to independently check every expression and "
            "result in the audited financial summary. Do not change any supplied "
            "assumption or calculated value. Distinguish fixed costs, variable costs, "
            "contribution, break-even volume, and gross revenue; gross revenue is not "
            "profit. Flag any tool discrepancy rather than silently fixing it.\n\n"
            f"Calculator-verified financial summary:\n{financial_audit}\n\n"
            f"Scenario context:\n{assumption_text}"
        ),
        expected_output=(
            "Calculator-verified assumptions and break-even arithmetic, with gross "
            "revenue distinguished from profit and limitations stated."
        ),
        agent=analyst,
        context=[research_task],
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
            "Return exactly three meaningful weakness/correction pairs. Ground each "
            "issue in the supplied draft, evidence, or assumptions. Do not silently "
            "change any financial assumption."
        ),
        expected_output="Exactly three evidence-based weaknesses, each with a concrete correction.",
        agent=reviewer,
        context=[
            draft_task,
            research_task,
            finance_task,
            *([sustainability_task] if sustainability_task else []),
        ],
        output_pydantic=ReviewCritique,
    )
    final_task = Task(
        description=(
            "Create a concise 250-450 word final brief for the Dean. Incorporate all "
            "three critique corrections. Return corrections_applied as the exact "
            "three correction strings copied in the same order. Cite only URLs present "
            "in the research output. Explicitly disclose missing campus-specific "
            "spending evidence. Do not include currency amounts or calculated customer "
            "counts; the application appends the validated financial ledger. Do not "
            "change assumptions or add unsupported claims."
        ),
        expected_output="A concise final brief and three applied corrections matching the validated output schema.",
        agent=reviewer,
        context=[
            draft_task,
            critique_task,
            research_task,
            finance_task,
            *([sustainability_task] if sustainability_task else []),
        ],
        output_pydantic=ReviewedBrief,
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
        "process": Process.sequential,
        "verbose": False,
        "task_callback": task_callback,
    }
    crew = Crew(**crew_config)
    return crew, tasks


def render_assumptions() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### Launch scenario")
        students = st.number_input(
            "University population", min_value=1, value=8000, step=100, key="scenario_students"
        )
        campus_context = st.text_input(
            "Campus / city", value="Your campus", max_chars=120, key="scenario_campus"
        )
        cafe_concept = st.text_area(
            "Café concept",
            value="A welcoming campus café with affordable coffee, quick meals, and plant-based options.",
            max_chars=500,
            height=90,
            key="scenario_concept",
        )
        with st.expander("Advanced financial assumptions", expanded=False):
            st.caption("Planning assumptions only. Validate costs and local prices before making a funding decision.")
            open_days = st.number_input(
                "Operating days per month", min_value=0, max_value=31, value=22, step=1,
                key="finance_open_days",
            )
            average_ticket = st.number_input(
                "Average transaction ($)", min_value=0.0, value=7.50, step=0.25,
                format="%.2f", key="finance_average_ticket",
            )
            variable_cost_per_customer = st.number_input(
                "Variable cost per customer ($)", min_value=0.0,
                value=2.625, step=0.25, format="%.3f",
                key="finance_variable_cost_per_customer",
            )
            occupancy = st.number_input(
                "Rent and occupancy ($ / month)", min_value=0.0, value=4000.0,
                step=250.0, key="finance_occupancy",
            )
            labor = st.number_input(
                "Staffing ($ / month)", min_value=0.0, value=12500.0,
                step=500.0, key="finance_staffing",
            )
            equipment = st.number_input(
                "Equipment lease and maintenance ($ / month)", min_value=0.0,
                value=1500.0, step=250.0, key="finance_equipment",
            )
            other = st.number_input(
                "Utilities and other ($ / month)", min_value=0.0,
                value=2000.0, step=250.0, key="finance_other",
            )
        st.caption("All figures are scenario assumptions, not quotes or demand forecasts.")
    fixed_cost_result = str(
        make_calculator_tool("none").invoke(
            {"expression": f"{occupancy} + {labor} + {equipment} + {other}"}
        )
    )
    try:
        fixed_costs = float(fixed_cost_result)
    except (TypeError, ValueError, OverflowError):
        fixed_costs = float("nan")
    return {
        "students": int(students),
        "campus_context": campus_context.strip(),
        "cafe_concept": cafe_concept.strip(),
        "open_days": int(open_days),
        "average_ticket": float(average_ticket),
        "variable_cost_per_customer": float(variable_cost_per_customer),
        "fixed_costs": fixed_costs,
        "fixed_cost_breakdown": {
            "Rent and occupancy": float(occupancy),
            "Staffing": float(labor),
            "Equipment lease and maintenance": float(equipment),
            "Utilities and other": float(other),
        },
    }


def calculate_financial_plan(assumptions: dict[str, Any]) -> dict[str, Any]:
    """Calculate an assumption-based plan using the actual LangChain calculator tool."""
    if not isinstance(assumptions, dict):
        return {"valid": False, "errors": ["Financial assumptions must be a mapping."]}
    variable_key = (
        "variable_cost_per_customer"
        if "variable_cost_per_customer" in assumptions
        else "variable_cost_pct"
    )
    breakdown = assumptions.get("fixed_cost_breakdown")
    required = ("students", "average_ticket", variable_key, "open_days")
    if "fixed_costs" not in assumptions and not breakdown:
        required += ("fixed_costs",)
    missing = [name for name in required if name not in assumptions]
    if missing:
        return {"valid": False, "errors": [f"Missing financial inputs: {', '.join(missing)}."]}

    calculator_tool = make_calculator_tool("none")

    def calculate(expression: str) -> float:
        result = str(calculator_tool.invoke({"expression": expression}))
        if result.startswith("Error:"):
            raise ValueError(result)
        number = float(result)
        if not math.isfinite(number):
            raise ValueError("Calculator returned a non-finite result.")
        return number

    values: dict[str, float] = {}
    errors = []
    input_names = list(required)
    if "fixed_costs" in assumptions and "fixed_costs" not in input_names:
        input_names.append("fixed_costs")
    if "variable_cost_pct" in assumptions and "variable_cost_pct" not in input_names:
        input_names.append("variable_cost_pct")
    for name in input_names:
        value = assumptions[name]
        if isinstance(value, bool):
            errors.append(f"{name.replace('_', ' ').capitalize()} must be numeric.")
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{name.replace('_', ' ').capitalize()} must be numeric.")
            continue
        if not math.isfinite(numeric_value) or abs(numeric_value) > 1e15:
            errors.append(f"{name.replace('_', ' ').capitalize()} must be finite and within the supported range.")
            continue
        values[name] = numeric_value

    if errors:
        return {"valid": False, "errors": errors}

    average_ticket = values["average_ticket"]
    variable_cost_pct = values.get("variable_cost_pct")
    variable_cost_input = values[variable_key]
    fixed_costs = values.get("fixed_costs")
    raw_open_days = assumptions["open_days"]
    students = values["students"]
    if average_ticket <= 0:
        errors.append("Average transaction must be greater than zero.")
    if students <= 0 or not students.is_integer():
        errors.append("Student population must be a positive whole number.")
    if variable_key == "variable_cost_per_customer" and variable_cost_input < 0:
        errors.append("Variable cost per customer cannot be negative.")
    if variable_key == "variable_cost_pct" and not 0 <= variable_cost_input <= 100:
        errors.append("Variable costs must be between 0% and 100% of sales.")
    if isinstance(raw_open_days, bool) or not values["open_days"].is_integer():
        errors.append("Operating days per month must be a whole number.")
    open_days = int(values["open_days"])
    if open_days <= 0 or open_days > 31:
        errors.append("Operating days per month must be between 1 and 31.")
    if fixed_costs is not None and fixed_costs < 0:
        errors.append("Monthly fixed costs cannot be negative.")

    if breakdown is not None:
        if not isinstance(breakdown, dict) or not breakdown:
            errors.append("Fixed-cost breakdown must contain at least one cost.")
        else:
            breakdown_values = []
            for name, value in breakdown.items():
                if isinstance(value, bool):
                    errors.append(f"Fixed cost '{name}' must be numeric.")
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"Fixed cost '{name}' must be numeric.")
                    continue
                if not math.isfinite(numeric_value) or not 0 <= numeric_value <= 1e15:
                    errors.append(f"Fixed cost '{name}' must be finite, non-negative, and within the supported range.")
                else:
                    breakdown_values.append(numeric_value)
            if not errors:
                try:
                    calculated_fixed_costs = calculate(
                        " + ".join(str(value) for value in breakdown_values)
                    )
                    if fixed_costs is not None and not math.isclose(
                        calculated_fixed_costs, fixed_costs, rel_tol=1e-9, abs_tol=1e-7
                    ):
                        errors.append("Fixed-cost total does not match the displayed cost breakdown.")
                    fixed_costs = calculated_fixed_costs
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    errors.append(f"Fixed costs could not be calculated: {exc}")

    if errors:
        return {"valid": False, "errors": errors}

    try:
        if variable_key == "variable_cost_per_customer":
            variable_cost = variable_cost_input
            variable_cost_pct = calculate(f"{variable_cost} / {average_ticket} * 100")
            if "variable_cost_pct" in values:
                pct_based_cost = calculate(f"{average_ticket} * {values['variable_cost_pct']} / 100")
                if not math.isclose(variable_cost, pct_based_cost, rel_tol=1e-8, abs_tol=1e-7):
                    return {"valid": False, "errors": ["Variable cost amount conflicts with the supplied percentage."]}
        else:
            variable_cost_pct = variable_cost_input
            variable_cost = calculate(f"{average_ticket} * {variable_cost_pct} / 100")
        contribution = calculate(f"{average_ticket} - {variable_cost}")
        if contribution <= 0:
            return {
                "valid": False,
                "errors": ["Contribution per transaction must be positive to calculate break-even."],
            }
        exact_monthly_customers = calculate(f"{fixed_costs} / {contribution}")
        minimum_monthly_customers = math.ceil(exact_monthly_customers)
        exact_daily_customers = calculate(f"{exact_monthly_customers} / {open_days}")
        minimum_daily_customers = math.ceil(exact_daily_customers)
        exact_break_even_revenue = calculate(f"{exact_monthly_customers} * {average_ticket}")
        minimum_target_revenue = calculate(f"{minimum_monthly_customers} * {average_ticket}")
        chart_limit = max(20, math.ceil(calculate(f"{max(exact_monthly_customers, 1)} * 1.4")))
        chart_counts = sorted(
            {math.ceil(calculate(f"{chart_limit} * {point} / 20")) for point in range(21)}
            | {minimum_monthly_customers}
        )
        chart_revenue = [calculate(f"{count} * {average_ticket}") for count in chart_counts]
        chart_costs = [
            calculate(f"{fixed_costs} + ({count} * {variable_cost})")
            for count in chart_counts
        ]
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        return {"valid": False, "errors": [f"Financial calculation failed: {exc}"]}

    return {
        "valid": True,
        "students": int(students),
        "average_ticket": average_ticket,
        "variable_cost_pct": variable_cost_pct,
        "variable_cost": variable_cost,
        "contribution": contribution,
        "fixed_costs": fixed_costs,
        "open_days": open_days,
        "exact_monthly_customers": exact_monthly_customers,
        "minimum_monthly_customers": minimum_monthly_customers,
        "exact_daily_customers": exact_daily_customers,
        "minimum_daily_customers": minimum_daily_customers,
        "exact_break_even_revenue": exact_break_even_revenue,
        "minimum_target_revenue": minimum_target_revenue,
        "chart": {
            "Monthly transactions": chart_counts,
            "Gross sales revenue": chart_revenue,
            "Fixed plus variable costs": chart_costs,
        },
    }


def current_job_status() -> dict[str, Any]:
    status_path = st.session_state.get("generation_status_path")
    if not status_path:
        return {}
    try:
        return json.loads(Path(status_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def live_agent_output(role: str) -> str:
    status = current_job_status()
    outputs = status.get("task_outputs", st.session_state.get("generation_task_outputs", []))
    return "\n\n".join(
        output.get("output", "")
        for output in outputs
        if output.get("agent", "").casefold() == role.casefold() and output.get("output")
    )


def source_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s<>\])\"']+", text)
    return list(dict.fromkeys(url.rstrip(".,;:") for url in urls))


def research_finding_blocks(text: str) -> tuple[str, list[str]]:
    preamble: list[str] = []
    findings: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line):
            if current:
                findings.append(current)
            current = [line]
        elif current:
            current.append(line)
        elif line.strip():
            preamble.append(line)
    if current:
        findings.append(current)
    if not findings and text.strip():
        findings = [[text.strip()]]
    return "\n".join(preamble).strip(), ["\n".join(block).strip() for block in findings]


def render_agent_workflow() -> None:
    status = current_job_status()
    session_state = st.session_state.get("generation_state", "idle")
    run_state = (
        session_state
        if session_state in {"complete", "error"}
        else status.get("state", session_state)
    )
    completed = int(status.get("completed_tasks", 0))
    total = int(status.get("total_tasks", 0))
    initialized = total > 1
    outputs = status.get("task_outputs", st.session_state.get("generation_task_outputs", []))
    agent_specs = [
        ("Market Researcher", "Finds current evidence and cites its limits.", 0, 1),
        ("Financial Analyst", "Audits the break-even assumptions and calculations.", 1, 2),
        ("Marketing Writer", "Turns evidence and analysis into a Dean-ready draft.", 2, 3),
        ("Skeptical Reviewer", "Challenges evidence, assumptions, and the draft.", 3, 5),
    ]
    if run_state == "idle" or not status and run_state not in {"complete", "error"}:
        st.info("Agent results will appear here after you generate a launch brief.")
    elif run_state == "running" and not initialized:
        st.info("The worker is starting. Agent execution will appear here when CrewAI initializes the task list.")
    for offset in range(0, len(agent_specs), 3):
        columns = st.columns(3)
        for column, (role, purpose, first_task, final_task) in zip(
            columns, agent_specs[offset : offset + 3]
        ):
            if run_state == "complete" or completed >= final_task:
                agent_state = "Completed"
            elif run_state == "error" and initialized and completed >= first_task:
                agent_state = "Failed"
            elif run_state == "running" and initialized and completed >= first_task:
                agent_state = "Running"
            else:
                agent_state = "Pending"
            with column:
                with st.container(border=True):
                    st.markdown(f"**{role}**")
                    st.caption(purpose)
                    st.markdown(
                        f'<span class="status-pill status-{agent_state.lower()}">{agent_state}</span>',
                        unsafe_allow_html=True,
                    )
                    agent_results = [
                        output for output in outputs
                        if output.get("agent", "").casefold() == role.casefold()
                    ]
                    for result in agent_results:
                        if result.get("output"):
                            with st.expander(result.get("task") or "Completed task output"):
                                st.markdown(result["output"])
    if run_state == "error":
        st.error(st.session_state.get("generation_error") or status.get("error") or "The workflow failed. Retry from Overview.")
    elif run_state == "complete":
        st.success("All workflow tasks completed. Review the evidence and final brief in their sections.")


def render_research_page() -> None:
    st.subheader("Market Research")
    research = st.session_state.get("research", "") or live_agent_output("Market Researcher")
    if not research:
        st.info("No research results are available yet. Generate a brief to run live web and Wikipedia research.")
        return
    preamble, findings = research_finding_blocks(research)
    if preamble:
        st.markdown(preamble)
    links_found = 0
    for finding in findings:
        with st.container(border=True):
            st.markdown(finding)
            urls = source_urls(finding)
            if urls:
                source_columns = st.columns(min(3, len(urls)))
                for url in urls:
                    links_found += 1
                    with source_columns[(links_found - 1) % len(source_columns)]:
                        st.link_button(f"Open cited source {links_found}", url, use_container_width=True)
    if not links_found:
        st.warning("The research output did not include a direct source URL. Treat its claims as unverified.")


def render_financial_page(assumptions: dict[str, Any]) -> None:
    st.subheader("Financial Planner")
    st.caption("Live values are calculated from the assumptions in the sidebar using the app's safe arithmetic parser. They are estimates, not forecasts.")
    plan = calculate_financial_plan(assumptions)
    if not plan["valid"]:
        for error in plan["errors"]:
            st.error(error)
        return
    metrics = st.columns(4)
    metrics[0].metric("Contribution / transaction", f"${plan['contribution']:,.2f}")
    metrics[1].metric("Break-even / month", f"{plan['exact_monthly_customers']:,.2f} customers")
    metrics[2].metric("Minimum / open day", f"{plan['minimum_daily_customers']:,} transactions")
    metrics[3].metric("Break-even gross sales", f"${plan['exact_break_even_revenue']:,.2f} / month")
    breakdown = assumptions["fixed_cost_breakdown"]
    rows = "\n".join(
        f"| {name} | ${value:,.2f} | Planning assumption |"
        for name, value in breakdown.items()
    )
    st.markdown(
        "| Monthly fixed-cost assumption | Amount | Type |\n|---|---:|---|\n"
        + rows
        + f"\n| **Total fixed costs** | **${plan['fixed_costs']:,.2f}** | **Planning assumption** |"
    )
    st.markdown(
        f"**Auditable calculation:** ${plan['average_ticket']:,.2f} average transaction "
        f"− ${plan['variable_cost']:,.2f} variable cost = **${plan['contribution']:,.2f} contribution**. "
        f"${plan['fixed_costs']:,.2f} fixed costs ÷ ${plan['contribution']:,.2f} contribution "
        f"= {plan['exact_monthly_customers']:,.2f} customers per month; minimum whole-customer target "
        f"is **{plan['minimum_monthly_customers']:,}**. Daily exact target is "
        f"{plan['exact_daily_customers']:,.2f}, rounded up to **{plan['minimum_daily_customers']:,}**. "
        f"Minimum-target gross sales are ${plan['minimum_target_revenue']:,.2f}; gross sales are not profit."
    )
    st.line_chart(
        plan["chart"],
        x="Monthly transactions",
        y=["Gross sales revenue", "Fixed plus variable costs"],
        x_label="Monthly customer transactions",
        y_label="USD",
    )
    st.caption("The chart compares gross sales with fixed plus variable costs. Values depend entirely on the editable assumptions; all inputs are unvalidated planning assumptions.")
    finance_output = st.session_state.get("finance", "") or live_agent_output("Financial Analyst")
    if finance_output:
        with st.expander("Financial Analyst task output"):
            st.markdown(finance_output)


def render_final_brief_page() -> None:
    st.subheader("Final Brief")
    brief = st.session_state.get("brief", "")
    if not brief:
        draft = st.session_state.get("draft", "") or live_agent_output("Marketing Writer")
        if draft:
            st.warning("The writer's draft is available, but the reviewer has not completed the final brief.")
            with st.container(border=True):
                st.markdown(draft)
        st.info("No final brief is available yet. Generate a launch brief to create the reviewed report.")
        return
    if st.session_state.get("brief_source") == "sample":
        st.warning("This is the included sample brief, not a report generated for the current scenario.")
    else:
        completed_at = st.session_state.get("generation_completed_at")
        started_at = st.session_state.get("generation_started")
        if completed_at:
            generated_label = time.strftime("%Y-%m-%d %H:%M", time.localtime(completed_at))
            st.caption(f"Generated {generated_label} · Run {st.session_state.get('generation_job_id', 'unknown')}")
        elif started_at:
            st.caption(f"Run {st.session_state.get('generation_job_id', 'unknown')}")
    with st.container(border=True):
        st.markdown(brief)
    action_columns = st.columns([1, 1, 3])
    with action_columns[0]:
        st.download_button(
            "Download Markdown", brief, file_name="final_brief.md",
            mime="text/markdown", use_container_width=True,
        )
    with action_columns[1]:
                with st.expander("Copy-ready Markdown"):
                        st.code(brief, language="markdown")
    critique = st.session_state.get("critique", "")
    if critique:
        with st.expander("Skeptical reviewer critique", expanded=True):
            st.markdown(critique)
    else:
        st.info("Reviewer critique will be shown separately when the workflow completes.")


def render_reflection_page() -> None:
    st.subheader("Collaboration Reflection")
    generated_reflection = REFLECTION_PATH
    project_reflection = ROOT / "reflection.md"
    if generated_reflection.is_file():
        st.caption("Reflection artifact from the backend workflow.")
        reflection_path = generated_reflection
    elif project_reflection.is_file():
        st.caption("Included project reflection; it is not generated for the current run.")
        reflection_path = project_reflection
    else:
        st.info("No collaboration reflection is available yet.")
        return
    reflection = reflection_path.read_text(encoding="utf-8")
    with st.container(border=True):
        st.markdown(reflection)
    st.download_button(
        "Download reflection", reflection,
        file_name=reflection_path.name, mime="text/markdown",
    )


def main() -> None:
    st.set_page_config(
        page_title="Campus Café Launch Intelligence",
        page_icon="assets/cafe-hero.svg",
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
        :root {
            --forest: #1F4839;
            --forest-deep: #17372D;
            --ivory: #F7F6EF;
            --sage: #E5EBE2;
            --sage-deep: #718678;
            --coffee: #654B3C;
            --copper: #A96039;
            --ink: #24352D;
            --muted: #5D6B62;
            --line: #D6DED4;
        }
        .block-container {max-width: 1380px !important; padding-top: 1.4rem !important;}
        [data-testid="stAppViewContainer"] {background: var(--ivory) !important; color: var(--ink);}
        [data-testid="stHeader"] {background: var(--ivory) !important;}
        [data-testid="stSidebar"] {background: var(--sage) !important; border-right: 1px solid var(--line);}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {color: var(--forest-deep) !important;}
        [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {color: var(--ink);}
        h1, h2, h3 {color: var(--forest-deep) !important; font-family: Georgia, 'Times New Roman', serif; letter-spacing: 0 !important;}
        h1::after {display: none !important;}
        [data-testid="stMetric"] {background: #FFFEFA !important; border: 1px solid var(--line); border-radius: 6px; box-shadow: 0 2px 8px rgba(26, 53, 41, .05);}
        [data-testid="stMetricLabel"] {color: var(--muted) !important;}
        [data-testid="stMetricValue"] {color: var(--forest) !important;}
        .stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a {
            background: var(--forest) !important; color: #FFFFFF !important; border: 1px solid var(--forest-deep) !important;
            border-radius: 5px !important; box-shadow: none !important; font-weight: 650; min-height: 2.65rem;
        }
        .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stLinkButton"] a:hover {background: var(--forest-deep) !important;}
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {outline: 3px solid var(--copper) !important; outline-offset: 2px;}
        [data-testid="stVerticalBlockBorderWrapper"] {background: #FFFEFA; border-color: var(--line) !important; border-radius: 6px !important; box-shadow: 0 2px 8px rgba(26, 53, 41, .04);}
        [data-testid="stRadio"] > div[role="radiogroup"] {gap: .4rem; flex-wrap: wrap;}
        [data-testid="stRadio"] label {background: #FFFEFA; border: 1px solid var(--line); border-radius: 5px; padding: .35rem .65rem;}
        [data-testid="stRadio"] label:has(input:checked) {background: var(--forest) !important; border-color: var(--forest) !important; color: #FFFFFF !important;}
        [data-testid="stRadio"] label:has(input:checked) p {color: #FFFFFF !important;}
        [data-testid="stProgress"] > div > div {background: var(--forest) !important;}
        [data-testid="stAlert"] {border-radius: 5px;}
        .status-pill {display: inline-block; padding: .2rem .55rem; border-radius: 999px; font-size: .78rem; font-weight: 700; border: 1px solid currentColor;}
        .status-pending {color: #59665E; background: #EFF1ED;}
        .status-running {color: #245C47; background: #E2EEE5;}
        .status-completed {color: #1E5C3C; background: #DCEBDF;}
        .status-failed {color: #8A3E2C; background: #F6E5DF;}
        @media (max-width: 760px) {
            .block-container {padding-left: 1rem !important; padding-right: 1rem !important;}
            [data-testid="stHorizontalBlock"] {flex-wrap: wrap !important; gap: .65rem !important;}
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {min-width: min(100%, 230px) !important; flex: 1 1 230px !important;}
        }
        [data-testid="stExpander"] {background: #FFFEFA !important; border-color: var(--line) !important; border-radius: 6px !important;}
        [data-testid="stAlert"] {background: #EDF2EA !important; border-color: var(--line) !important; color: var(--ink) !important;}
        [data-testid="stAlert"] p, [data-testid="stAlert"] span {color: var(--ink) !important;}
        [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {background: #FFFEFA; color: var(--ink); border-color: var(--line);}
        </style>
        """,
        unsafe_allow_html=True,
    )
    assumptions = render_assumptions()
    model_ready, model_status = check_local_model(
        OLLAMA_BASE_URL, DEFAULT_MODEL
    ) if not DEMO_MODE else (
        False,
        "Hosted sample mode: generation is disabled; the included brief is shown as an example.",
    )
    if DEMO_MODE and "brief" not in st.session_state:
        sample_path = ROOT / "final_brief.md"
        if sample_path.is_file():
            st.session_state["brief"] = sample_path.read_text(encoding="utf-8")
            st.session_state["brief_source"] = "sample"
    st.title("Campus Café Launch Intelligence")
    st.caption(
        "An evidence-aware planning workspace for university café decisions. "
        "Generated briefs are AI-assisted estimates, not demand forecasts or funding guarantees."
    )
    if DEMO_MODE:
        st.info(model_status)
    elif model_ready:
        st.success(model_status)
    else:
        st.warning(model_status)
    active_view = st.radio(
        "Workspace sections",
        ("Overview", "Market Research", "Financial Planner", "Agent Workflow", "Final Brief", "Reflection"),
        horizontal=True,
        label_visibility="collapsed",
        key="workspace_section",
    )
    generation_running = st.session_state.get("generation_state") == "running"
    if active_view != "Overview":
        if generation_running:
            render_generation_status()
        elif st.session_state.get("generation_state") == "error":
            st.error("The previous launch brief run failed. Retry from Overview after reviewing the details.")
            with st.expander("Run error details"):
                st.caption(st.session_state.get("generation_error", "No additional details were provided."))
        if active_view == "Market Research":
            render_research_page()
        elif active_view == "Financial Planner":
            render_financial_page(assumptions)
        elif active_view == "Agent Workflow":
            st.subheader("Agent Workflow")
            render_agent_workflow()
        elif active_view == "Final Brief":
            render_final_brief_page()
        else:
            render_reflection_page()
        return

    st.subheader("Overview")
    st.write("Set the campus scenario in the sidebar, then generate a concise, reviewed proposal for the Dean.")
    current_plan = calculate_financial_plan(assumptions)
    summary_columns = st.columns(4)
    summary_columns[0].metric("University population", f"{assumptions['students']:,}")
    summary_columns[1].metric("Fixed costs / month", f"${assumptions['fixed_costs']:,.0f}")
    summary_columns[2].metric(
        "Minimum break-even / day",
        f"{current_plan['minimum_daily_customers']:,}" if current_plan["valid"] else "Needs review",
    )
    run_state = st.session_state.get("generation_state", "Pending").title()
    summary_columns[3].metric("Latest run", run_state)
    if not current_plan["valid"]:
        for error in current_plan["errors"]:
            st.warning(error)
    st.markdown("**Workflow**  Market evidence → Financial analysis → Draft → Skeptical review → Final brief")
    st.caption("The workflow also includes a Sustainability Officer to check environmental and plant-based claims.")
    st.caption(
        f"Estimated fixed costs in this scenario: ${assumptions['fixed_costs']:,.2f}/month · "
        f"Average ticket: ${assumptions['average_ticket']:.2f} · "
        f"Open {assumptions['open_days']} days/month"
    )

    clicked_normal = st.button(
        "Generate Launch Brief (local Ollama only)"
        if DEMO_MODE
        else "Generate Launch Brief",
        type="primary",
        disabled=DEMO_MODE or not model_ready or generation_running,
        use_container_width=True,
    )
    if clicked_normal:
        try:
            for key in (
                "brief", "draft", "critique", "research", "finance",
                "sustainability", "generation_task_outputs", "generation_error",
            ):
                st.session_state.pop(key, None)
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
            st.session_state["brief_source"] = "generated"
            st.session_state["draft"] = job_status.get("draft", "")
            st.session_state["critique"] = job_status.get("critique", "")
            st.session_state["research"] = job_status.get("research", "")
            st.session_state["finance"] = job_status.get("finance", "")
            st.session_state["sustainability"] = job_status.get("sustainability", "")
            st.session_state["generation_task_outputs"] = job_status.get("task_outputs", [])
            st.session_state["generation_completed_at"] = job_status.get("updated_at")
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
    actual_total = int(job_status.get("total_tasks", 1))
    total = max(1, actual_total)
    if actual_total <= 1:
        st.progress(0, text="Starting agent workflow")
        st.info("The worker is starting CrewAI. Agent progress will appear as real task results arrive.")
        return
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
