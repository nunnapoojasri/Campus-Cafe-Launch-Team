import json
import math
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import pytest
import crewai
import ddgs

import app


SOURCE = {
    "title": "NCA Specialty Coffee Report",
    "url": "https://www.ncausa.org/Newsroom/Coffee-Consumption-Statistics",
    "published_date": "2026-06-02",
    "snippet": "U.S. adult specialty coffee category data.",
    "source_method": "duckduckgo",
}

BASE_ASSUMPTIONS = {
    "students": 8000,
    "open_days": 22,
    "average_ticket": 7.50,
    "variable_cost_per_customer": 2.625,
    "fixed_costs": 20000.0,
    "fixed_cost_breakdown": {
        "Rent and occupancy": 4000.0,
        "Staffing": 12500.0,
        "Equipment lease and maintenance": 1500.0,
        "Utilities and other": 2000.0,
    },
}

CORRECTIONS = [
    "Limit national coffee statistics to broad category context and disclose that they do not establish campus demand.",
    "Label all menu prices, costs, and transaction inputs as assumptions until verified with local quotes or surveys.",
    "Separate gross sales revenue from contribution and operating profit, and state excluded costs.",
]


def research_report(*, campus: bool = False) -> app.ResearchReport:
    finding = app.ResearchFinding(
        title=SOURCE["title"],
        url=SOURCE["url"],
        published_date=SOURCE["published_date"],
        source_method="duckduckgo",
        source_excerpt=SOURCE["snippet"],
        finding="NCA reports a national adult specialty-coffee consumption measure.",
        evidence_scope="campus" if campus else "national",
        supports="Broad U.S. adult category interest only.",
        limitations="It is not student-specific and does not establish local campus demand.",
    )
    return app.ResearchReport(
        findings=[finding],
        campus_specific_evidence_available=campus,
        limitations=["No campus-specific or local student spending evidence was returned."],
    )


def reviewer_critique() -> app.ReviewCritique:
    return app.ReviewCritique(
        issues=[
            app.ReviewIssue(weakness=f"Risk {index} is material and unsupported in the current draft.", correction=correction)
            for index, correction in enumerate(CORRECTIONS, start=1)
        ]
    )


def build_mocked_crew(monkeypatch, ddgs_type=None):
    agent_factory = lambda **kwargs: SimpleNamespace(**kwargs)
    task_factory = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setattr(crewai, "LLM", lambda **kwargs: object())
    monkeypatch.setattr(crewai, "Agent", agent_factory)
    monkeypatch.setattr(crewai, "Task", task_factory)
    monkeypatch.setattr(crewai, "Crew", lambda **kwargs: SimpleNamespace(**kwargs))
    if ddgs_type is not None:
        monkeypatch.setattr(ddgs, "DDGS", ddgs_type)
    return app.create_crew(BASE_ASSUMPTIONS)


def valid_final_text() -> str:
    return " ".join(
        [
            "The university should authorize a limited validation phase before committing operating funds.",
            "The proposal is an AI-assisted planning estimate, not a forecast or a guarantee of profitability.",
            "The national coffee source provides broad adult category context and cannot establish student interest at this campus.",
            "No campus-specific spending, willingness-to-pay, or competing-menu evidence was returned by the research tools.",
            "The university should survey students, test menu prices, and observe candidate locations before approval.",
            "All financial inputs are planning assumptions and should be replaced with facilities, staffing, equipment, and supplier quotes.",
            "The financial planner calculates contribution after variable cost and compares it with fixed operating costs.",
            "Break-even sales are gross revenue at the modeled transaction volume; they are not profit or cash available to the university.",
            "The model excludes startup investment, financing, taxes, seasonal variation, and costs not entered by the user.",
            "The reviewer checked the draft against the evidence and assumptions and required explicit limitations.",
            "Any reported local demand should be based on a campus survey or measured pilot rather than national market data.",
            "Management should return with verified quotes, a downside case, and observed student purchase behavior.",
            "The recommended next step is a small, time-limited validation exercise with a clear spending cap.",
            "No final launch approval should be inferred from this illustrative calculation or from the existence of a polished brief.",
            "The university should document data collection methods and revisit the assumptions when new evidence becomes available.",
            "The pilot should include a clear stop condition, a small authorized budget, and a review date agreed by the Dean and Finance Office.",
        ]
    )


def test_calculator_tool_verifies_every_financial_expression(monkeypatch):
    expressions = []
    real_factory = app.make_calculator_tool

    class RecordingTool:
        def __init__(self, failure_mode):
            self.tool = real_factory(failure_mode)

        def invoke(self, arguments):
            expressions.append(arguments["expression"])
            return self.tool.invoke(arguments)

    monkeypatch.setattr(app, "make_calculator_tool", RecordingTool)
    plan = app.calculate_financial_plan(BASE_ASSUMPTIONS)

    assert plan["valid"] is True
    assert plan["fixed_costs"] == 20000
    assert plan["variable_cost"] == 2.625
    assert plan["contribution"] == 4.875
    assert plan["exact_monthly_customers"] == pytest.approx(20000 / 4.875)
    assert plan["minimum_monthly_customers"] == 4103
    assert plan["exact_daily_customers"] == pytest.approx((20000 / 4.875) / 22)
    assert plan["minimum_daily_customers"] == 187
    assert plan["exact_break_even_revenue"] == pytest.approx((20000 / 4.875) * 7.5)
    assert plan["minimum_target_revenue"] == 4103 * 7.5
    assert "4000.0 + 12500.0 + 1500.0 + 2000.0" in expressions
    assert "7.5 - 2.625" in expressions
    assert "20000.0 / 4.875" in expressions
    assert "7.5" in " ".join(expressions)
    assert len(expressions) > 20


def test_legacy_percentage_input_matches_per_customer_cost():
    assumptions = {
        **BASE_ASSUMPTIONS,
        "variable_cost_pct": 35,
    }
    assumptions.pop("variable_cost_per_customer")
    plan = app.calculate_financial_plan(assumptions)
    assert plan["valid"] is True
    assert plan["variable_cost"] == 2.625
    assert plan["contribution"] == 4.875
    assert plan["minimum_monthly_customers"] == 4103


def test_percentage_and_per_customer_inputs_cannot_conflict():
    plan = app.calculate_financial_plan({**BASE_ASSUMPTIONS, "variable_cost_pct": 50})
    assert plan["valid"] is False
    assert "conflicts" in plan["errors"][0]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"variable_cost_per_customer": 7.50}, "Contribution"),
        ({"variable_cost_per_customer": 8.00}, "Contribution"),
        ({"open_days": 0}, "Operating days"),
        ({"open_days": 32}, "Operating days"),
        ({"open_days": 1.5}, "whole number"),
        ({"average_ticket": 0}, "Average transaction"),
        ({"students": 0}, "Student population"),
        ({"fixed_costs": -1}, "cannot be negative"),
        ({"fixed_costs": 1e300}, "supported range"),
        ({"average_ticket": math.nan}, "finite"),
        ({"variable_cost_per_customer": math.inf}, "finite"),
    ],
)
def test_invalid_financial_inputs_are_rejected(updates, message):
    plan = app.calculate_financial_plan({**BASE_ASSUMPTIONS, **updates})
    assert plan["valid"] is False
    assert any(message.casefold() in error.casefold() for error in plan["errors"])


@pytest.mark.parametrize("missing", ["students", "open_days", "average_ticket", "fixed_costs"])
def test_missing_financial_inputs_return_validation_errors(missing):
    assumptions = dict(BASE_ASSUMPTIONS)
    assumptions.pop(missing)
    if missing == "fixed_costs":
        assumptions.pop("fixed_cost_breakdown")
    plan = app.calculate_financial_plan(assumptions)
    assert plan["valid"] is False
    assert any(missing in error for error in plan["errors"])


def test_fixed_cost_breakdown_must_reconcile():
    assumptions = {**BASE_ASSUMPTIONS, "fixed_costs": 21000}
    plan = app.calculate_financial_plan(assumptions)
    assert plan["valid"] is False
    assert any("does not match" in error for error in plan["errors"])


def test_customer_targets_use_ceiling_and_exact_target_does_not_round_up_again():
    assumptions = {
        "students": 8000,
        "open_days": 4,
        "average_ticket": 7.50,
        "variable_cost_per_customer": 2.625,
        "fixed_costs": 97.5,
    }
    plan = app.calculate_financial_plan(assumptions)
    assert plan["valid"] is True
    assert plan["exact_monthly_customers"] == 20
    assert plan["minimum_monthly_customers"] == 20
    assert plan["exact_daily_customers"] == 5
    assert plan["minimum_daily_customers"] == 5
    assert plan["exact_break_even_revenue"] == 150


def test_financial_report_labels_assumptions_and_revenue_not_profit():
    plan = app.calculate_financial_plan(BASE_ASSUMPTIONS)
    report = app.render_financial_audit(BASE_ASSUMPTIONS, plan)
    assert "(assumption)" in report
    assert "Gross sales" in report
    assert "revenue, not profit" in report
    assert "4,103" in report
    assert "187 customers per open day" in report


def test_research_source_must_match_observed_tool_result():
    report = research_report()
    app.validate_research_provenance(report, [SOURCE])
    fabricated = report.model_copy(update={
        "findings": [report.findings[0].model_copy(update={"url": "https://fake.example/source"})]
    })
    with pytest.raises(ValueError, match="not returned by a tool"):
        app.validate_research_provenance(fabricated, [SOURCE])


def test_research_without_local_evidence_requires_limitations():
    report = research_report()
    no_local_limit = report.model_copy(update={"limitations": ["Broad category evidence only."]})
    with pytest.raises(ValueError, match="lack of campus-specific"):
        app.validate_research_provenance(no_local_limit, [SOURCE])


def test_research_renders_missing_date_and_evidence_scope():
    report = research_report().model_copy(update={
        "findings": [research_report().findings[0].model_copy(update={"published_date": None})]
    })
    text = app.render_research_report(report)
    assert "Publication date unavailable" in text
    assert "Evidence scope: national" in text
    assert "## Limitations" in text


def test_real_crew_graph_uses_four_agents_sequentially_with_explicit_context(monkeypatch):
    crew, tasks = build_mocked_crew(monkeypatch)
    assert [agent.role for agent in crew.agents] == [
        "Market Researcher",
        "Financial Analyst",
        "Marketing Writer",
        "Skeptical Reviewer",
    ]
    assert crew.process == crewai.Process.sequential
    assert len(tasks) == 5
    research, finance, draft, critique, final = tasks
    assert finance.context == [research]
    assert research in draft.context and finance in draft.context
    assert draft in critique.context and research in critique.context and finance in critique.context
    assert draft in final.context and critique in final.context
    assert research in final.context and finance in final.context
    assert research.output_pydantic is app.ResearchReport
    assert critique.output_pydantic is app.ReviewCritique
    assert final.output_pydantic is app.ReviewedBrief


def test_duckduckgo_tool_records_actual_results_and_source_date(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results, backend):
            return [
                {
                    "title": SOURCE["title"],
                    "href": SOURCE["url"],
                    "date": SOURCE["published_date"],
                    "body": SOURCE["snippet"],
                }
            ]

    crew, _ = build_mocked_crew(monkeypatch, FakeDDGS)
    web_search = crew.agents[0].tools[0]
    payload = json.loads(web_search.run(query="campus coffee"))
    assert payload["status"] == "ok"
    assert payload["results"][0]["url"] == SOURCE["url"]
    assert payload["results"][0]["published_date"] == SOURCE["published_date"]
    assert web_search.source_records[0]["source_method"] == "duckduckgo"
    assert web_search.invocation_count == 1


def test_search_rate_limit_is_reported_without_fabricated_source(monkeypatch):
    class RateLimitedDDGS:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, *args, **kwargs):
            raise RuntimeError("HTTP 429 rate limit")

    crew, _ = build_mocked_crew(monkeypatch, RateLimitedDDGS)
    web_search = crew.agents[0].tools[0]
    payload = json.loads(web_search.run(query="campus coffee"))
    assert payload["status"] == "error"
    assert "429" in payload["message"]
    assert payload["results"] == []
    assert web_search.source_records == []
    assert web_search.tool_observations == ["DuckDuckGo: RuntimeError: HTTP 429 rate limit"]
    report = app.ResearchReport(
        findings=[],
        campus_specific_evidence_available=False,
        limitations=["No campus-specific or local spending evidence was returned."],
    )
    with pytest.raises(ValueError, match="tool observation"):
        app.validate_research_provenance(report, [], web_search.tool_observations)
    report.limitations.append(web_search.tool_observations[0])
    app.validate_research_provenance(report, [], web_search.tool_observations)


def test_wikipedia_failure_is_reported_without_fabricated_citation(monkeypatch):
    crew, _ = build_mocked_crew(monkeypatch)
    monkeypatch.setattr(app, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")))
    wikipedia_search = crew.agents[0].tools[1]
    payload = json.loads(wikipedia_search.run(query="campus cafe"))
    assert payload["status"] == "error"
    assert "offline" in payload["message"]
    assert payload["results"] == []
    assert wikipedia_search.source_records == []
    assert wikipedia_search.tool_observations[0].startswith("Wikipedia: URLError")


def test_review_requires_three_corrections_and_rejects_unverified_sources():
    critique = reviewer_critique()
    reviewed = app.ReviewedBrief(
        corrections_applied=CORRECTIONS,
        final_brief=valid_final_text(),
    )
    app.validate_reviewed_brief(critique, reviewed, {SOURCE["url"]})

    with pytest.raises(ValueError, match="did not carry all three"):
        app.validate_reviewed_brief(
            critique,
            reviewed.model_copy(update={"corrections_applied": CORRECTIONS[:2] + ["Different correction"]}),
            {SOURCE["url"]},
        )
    with pytest.raises(ValueError, match="currency values"):
        app.validate_reviewed_brief(
            critique,
            reviewed.model_copy(update={"final_brief": valid_final_text() + " Budget: $200."}),
            {SOURCE["url"]},
        )
    with pytest.raises(ValueError, match="unaudited numerical"):
        app.validate_reviewed_brief(
            critique,
            reviewed.model_copy(update={"final_brief": valid_final_text() + " Target: 250 transactions daily."}),
            {SOURCE["url"]},
        )
    with pytest.raises(ValueError, match="not returned by research tools"):
        app.validate_reviewed_brief(
            critique,
            reviewed.model_copy(update={"final_brief": valid_final_text() + " https://fake.example/claim"}),
            {SOURCE["url"]},
        )


def test_atomic_report_writer_and_unwritable_path(tmp_path):
    output = tmp_path / "nested" / "final_brief.md"
    app.write_report_atomically(output, "Report text")
    assert output.read_text(encoding="utf-8") == "Report text\n"
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocker", encoding="utf-8")
    with pytest.raises(OSError, match="Could not safely write report"):
        app.write_report_atomically(blocker / "report.md", "must not succeed")


def test_worker_publishes_audited_brief_and_reflection(tmp_path, monkeypatch):
    source = {**SOURCE}
    finding = research_report().findings[0]
    research = app.ResearchReport(
        findings=[finding],
        campus_specific_evidence_available=False,
        limitations=["No campus-specific or local spending evidence was returned."],
    )
    critique = reviewer_critique()
    reviewed = app.ReviewedBrief(
        corrections_applied=CORRECTIONS,
        final_brief=valid_final_text() + f" Source: {SOURCE['url']}",
    )
    researcher_agent = SimpleNamespace(
        role="Market Researcher",
        tools=[SimpleNamespace(name="web_search", source_records=[source], invocation_count=1)],
    )
    finance_agent = SimpleNamespace(
        role="Financial Analyst",
        tools=[SimpleNamespace(name="calculator_tool", invocation_count=1)],
    )

    def task(agent, raw, pydantic=None):
        return SimpleNamespace(
            agent=agent,
            output=SimpleNamespace(raw=raw, pydantic=pydantic),
        )

    tasks = [
        task(researcher_agent, research.model_dump_json(), research),
        task(finance_agent, "Audited finance task used calculator outputs."),
        task(SimpleNamespace(role="Marketing Writer", tools=[]), "Draft output"),
        task(SimpleNamespace(role="Skeptical Reviewer", tools=[]), critique.model_dump_json(), critique),
        task(SimpleNamespace(role="Skeptical Reviewer", tools=[]), reviewed.model_dump_json(), reviewed),
    ]
    fake_crew = SimpleNamespace(
        kickoff=lambda: "crew completed",
        usage_metrics=None,
    )
    monkeypatch.setattr(app, "create_crew", lambda *args, **kwargs: (fake_crew, tasks))
    monkeypatch.setattr(app, "OUTPUT_PATH", tmp_path / "final_brief.md")
    monkeypatch.setattr(app, "REFLECTION_PATH", tmp_path / "collaboration_reflection.md")
    status_path = tmp_path / "job.json"

    app.run_brief_generation("run-1", BASE_ASSUMPTIONS, str(status_path))

    status = json.loads(status_path.read_text(encoding="utf-8"))
    final_brief = (tmp_path / "final_brief.md").read_text(encoding="utf-8")
    reflection = (tmp_path / "collaboration_reflection.md").read_text(encoding="utf-8")
    assert status["state"] == "complete"
    assert "## Financial assumptions and audited break-even" in final_brief
    assert "revenue, not profit" in final_brief
    assert "## Reviewer corrections incorporated" in final_brief
    assert f"[{SOURCE['title']}]({SOURCE['url']})" in final_brief
    assert "No campus-specific or local spending evidence" in reflection
    assert len(critique.issues) == 3
    assert status["financial_audit"]["minimum_monthly_customers"] == 4103


def test_native_ollama_llm_uses_bounded_context_and_returns_crewai_tool_calls(monkeypatch):
    observed = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "calculator_tool",
                                    "arguments": {"expression": "6 * 7"},
                                }
                            }
                        ]
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        observed["payload"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr(app, "urlopen", fake_urlopen)
    llm = app.make_ollama_llm("ollama/llama3.2:3b", "http://ollama:11434/")
    result = llm.call(
        [{"role": "user", "content": "Calculate 6 * 7."}],
        tools=[{"type": "function", "function": {"name": "calculator_tool"}}],
    )

    assert observed["url"] == "http://ollama:11434/api/chat"
    assert observed["payload"]["model"] == "llama3.2:3b"
    assert observed["payload"]["options"]["num_ctx"] == app.OLLAMA_NUM_CTX
    assert observed["payload"]["tools"][0]["function"]["name"] == "calculator_tool"
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "calculator_tool"
    assert result[0]["function"]["arguments"] == {"expression": "6 * 7"}


def test_native_ollama_llm_converts_openai_tool_history(monkeypatch):
    captured_payload = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"message":{"content":"done"}}'

    def fake_urlopen(request, timeout):
        captured_payload.update(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(app, "urlopen", fake_urlopen)
    llm = app.make_ollama_llm("ollama/llama3.2:3b", "http://ollama:11434")
    result = llm.call(
        [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "type": "function", "function": {
                    "name": "calculator_tool", "arguments": "{\"expression\":\"6 * 7\"}"
                }}
            ]},
            {"role": "tool", "name": "calculator_tool", "tool_call_id": "call-1", "content": "42"},
        ]
    )

    assert result == "done"
    assert captured_payload["messages"][0]["tool_calls"][0]["function"]["arguments"] == {
        "expression": "6 * 7"
    }
    assert captured_payload["messages"][1]["tool_name"] == "calculator_tool"
    assert "tool_call_id" not in captured_payload["messages"][1]


def test_native_ollama_llm_surfaces_http_errors(monkeypatch):
    error = HTTPError(
        "http://ollama:11434/api/chat",
        503,
        "model unavailable",
        {},
        BytesIO(b"model runner unavailable"),
    )
    monkeypatch.setattr(app, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    llm = app.make_ollama_llm("ollama/llama3.2:3b", "http://ollama:11434")
    with pytest.raises(RuntimeError, match="Ollama returned HTTP 503: model runner unavailable"):
        llm.call("Return a short answer.")


@pytest.mark.parametrize("value", ["0", "511", "131073", "not-an-integer"])
def test_ollama_context_config_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_NUM_CTX", value)
    with pytest.raises(ValueError, match="OLLAMA_NUM_CTX"):
        app.read_ollama_context_size()
