import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import crewai

import app
import brief_worker


ASSUMPTIONS = {
    "students": 8000,
    "open_days": 22,
    "average_ticket": 7.5,
    "variable_cost_pct": 35,
    "fixed_costs": 20000,
}


class CalculatorFailureTests(unittest.TestCase):
    def test_calculator_works_normally(self):
        result = app.make_calculator_tool("none").invoke(
            {"expression": "(120 * 4.5 - 300) * 30"}
        )
        self.assertEqual(result, "7200")

    def test_temporary_failure_can_recover_on_retry(self):
        calculator_tool = app.make_calculator_tool("fail_once")
        first_result = calculator_tool.invoke({"expression": "10 / 2"})
        retry_result = calculator_tool.invoke({"expression": "10 / 2"})
        self.assertIn("temporary calculator outage", first_result)
        self.assertEqual(retry_result, "5")

    def test_persistent_failure_never_returns_a_calculation(self):
        calculator_tool = app.make_calculator_tool("always_fail")
        for _ in range(2):
            result = calculator_tool.invoke({"expression": "10 / 2"})
            self.assertIn("calculator unavailable", result)


class CrewExerciseConfigurationTests(unittest.TestCase):
    def build_crew(self, options):
        fake_crew = MagicMock(name="crew")
        with (
            patch.object(app, "DEMO_MODE", False),
            patch.object(crewai, "LLM", return_value=MagicMock(name="llm")),
            patch.object(
                crewai,
                "Agent",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ) as agent_factory,
            patch.object(
                crewai,
                "Task",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ) as task_factory,
            patch.object(crewai, "Crew", return_value=fake_crew) as crew_factory,
        ):
            crew, tasks = app.create_crew(ASSUMPTIONS, options=options)
        return crew, tasks, agent_factory, task_factory, crew_factory

    def test_research_handoff_is_mandatory_and_sustainability_is_optional(self):
        _, tasks, agents, task_factory, _ = self.build_crew(
            {"share_research": False, "include_sustainability": True}
        )
        self.assertEqual(agents.call_count, 5)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(task_factory.call_args_list[1].kwargs["context"], [tasks[0]])
        self.assertEqual(
            tasks[2].agent.role,
            "Sustainability Officer",
        )

    def test_sequential_mode_and_no_sustainability_are_configured(self):
        _, tasks, agents, task_factory, crew_factory = self.build_crew(
            {"share_research": True, "include_sustainability": False}
        )
        crew_config = crew_factory.call_args.kwargs
        self.assertEqual(crew_config["process"], crewai.Process.sequential)
        self.assertNotIn("manager_llm", crew_config)
        self.assertEqual(agents.call_count, 4)
        self.assertEqual(len(tasks), 5)
        self.assertEqual(task_factory.call_args_list[1].kwargs["context"], [tasks[0]])

    def test_sequential_mode_is_enforced(self):
        _, _, _, _, crew_factory = self.build_crew({"process_mode": "hierarchical"})
        crew_config = crew_factory.call_args.kwargs
        self.assertEqual(crew_config["process"], crewai.Process.sequential)
        self.assertNotIn("manager_llm", crew_config)

    def test_reviewer_tone_switch_changes_reviewer_backstory(self):
        _, _, agents, _, _ = self.build_crew({"reviewer_tone": "cheerleader"})
        reviewer = agents.call_args_list[3].kwargs
        self.assertIn("enthusiastic campus ambassador", reviewer["backstory"])

    def test_comparisons_run_both_variants_and_report_results(self):
        for experiment, expected_values in (
            (
                "compare_processes",
                ["sequential", "hierarchical"],
            ),
            (
                "compare_reviewers",
                ["skeptical", "cheerleader"],
            ),
        ):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as directory:
                status_path = os.path.join(directory, "status.json")
                options_seen = []
                crews = []

                def create_fake_crew(assumptions, task_callback=None, options=None):
                    options_seen.append(options)
                    crew = SimpleNamespace(
                        usage_metrics=SimpleNamespace(
                            total_tokens=12,
                            prompt_tokens=8,
                            completion_tokens=4,
                        ),
                        kickoff=MagicMock(),
                    )
                    tasks = [
                        SimpleNamespace(
                            output=SimpleNamespace(raw=f"{options_seen[-1]} result")
                        )
                    ]
                    crews.append(crew)
                    return crew, tasks

                with patch.object(app, "create_crew", side_effect=create_fake_crew):
                    app.run_brief_generation(
                        "job",
                        ASSUMPTIONS,
                        status_path,
                        options={"include_sustainability": False},
                        experiment=experiment,
                    )

                with open(status_path, encoding="utf-8") as status_file:
                    status = json.load(status_file)
                self.assertEqual(status["state"], "complete")
                self.assertEqual(len(status["exercise_result"]["results"]), 2)
                self.assertEqual(
                    [
                        next(
                            value
                            for key, value in options.items()
                            if key in {"process_mode", "reviewer_tone"}
                        )
                        for options in options_seen
                    ],
                    expected_values,
                )


class WorkerHandoffTests(unittest.TestCase):
    def test_worker_forwards_exercise_options_and_mode(self):
        arguments = [
            "brief_worker.py",
            "job-id",
            json.dumps(ASSUMPTIONS),
            "/tmp/status.json",
            json.dumps({"share_research": False}),
            "compare_reviewers",
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(brief_worker, "run_brief_generation") as run_generation,
        ):
            brief_worker.main()
        run_generation.assert_called_once_with(
            "job-id",
            ASSUMPTIONS,
            "/tmp/status.json",
            options={"share_research": False},
            experiment="compare_reviewers",
        )


if __name__ == "__main__":
    unittest.main()
