"""Real parallel dispatch, honest budgets and structured sandbox submission."""
import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from deepagents.backends import StateBackend
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr

import agents
import research
import workflow
from test_research import ArtifactBackend, artifacts
from validate_report import inspect_files, remember_sources


@tool
def task(description: str, subagent_type: str) -> str:
    """Test-only delegation schema; never perform research."""
    raise AssertionError("unit test must not execute this tool")


def planned_request(count=1):
    return ModelRequest(
        model=Mock(), tools=[task], runtime=Mock(),
        state={"run_model_call_count": count, "thread_model_call_count": count},
        messages=[HumanMessage(content="Full topic and supplied current date"),
                  AIMessage(content="", tool_calls=[{
                      "name": "write_todos", "id": "todo", "args": {}}]),
                  ToolMessage(content="Todos updated", tool_call_id="todo")],
    )


def task_reply():
    return ModelResponse(result=[AIMessage(content="", tool_calls=[{
        "name": "task", "args": {"subagent_type": "researcher",
                                  "description": "Investigate this assigned sub-question"},
        "id": "provider-reuses-id",
    }], usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})])


class PlanningTests(unittest.TestCase):
    def test_extra_provider_proposals_are_not_dispatched_or_counted(self):
        response = task_reply()
        first = response.result[0].tool_calls[0]
        response.result[0].tool_calls.extend([
            {**first, "id": "extra-proposal"},
            {**first, "id": "wrong-role", "args": {"subagent_type": "general-purpose",
                                                       "description": "unused"}},
        ])
        result = workflow.ResearchPlanningMiddleware(run_limit=10).wrap_model_call(
            planned_request(), Mock(return_value=response))
        calls = result.result[0].tool_calls
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c["id"].startswith("provider-reuses-id") for c in calls))
        self.assertEqual(research.summarize(result.result, 1, "fixture")["subagent_calls"], 3)
        self.assertEqual(research.summarize(result.result, 1, "fixture")["tokens"],
                         {"input": 21, "output": 9})

    def test_sync_and_async_collect_actual_calls_and_count_every_model_request(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                middleware = workflow.ResearchPlanningMiddleware(run_limit=5)
                requests = []
                def handler(request):
                    requests.append(request)
                    return task_reply()
                async def ahandler(request):
                    return handler(request)
                if asynchronous:
                    response = asyncio.run(middleware.awrap_model_call(planned_request(), ahandler))
                else:
                    response = middleware.wrap_model_call(planned_request(), handler)
                self.assertEqual(len(requests), 3)
                message = response.result[0]
                self.assertEqual(len(message.tool_calls), 3)
                self.assertEqual(len({c["id"] for c in message.tool_calls}), 3)
                for index, call in enumerate(message.tool_calls, 1):
                    self.assertIn(f"/{index:02d}-", call["args"]["description"])
                    self.assertIn("Full topic and supplied current date", call["args"]["description"])
                    self.assertIn("Investigate this assigned sub-question", call["args"]["description"])
                for request in requests:
                    self.assertEqual(request.tool_choice, "task")
                    # Previously collected calls are never replayed without their results.
                    self.assertEqual(len(request.messages), 4)
                self.assertEqual(middleware.after_model({
                    "messages": [message], "run_model_call_count": 1,
                    "thread_model_call_count": 1}, None), {
                        "run_model_call_count": 4, "thread_model_call_count": 4})
                summary = research.summarize([message], 1, "offline")
                self.assertEqual(summary["tokens"], {"input": 21, "output": 9})
                self.assertEqual(summary["subagent_calls"], 3)

    def test_budget_shortfall_and_invalid_planning_never_invent_tasks(self):
        middleware = workflow.ResearchPlanningMiddleware(run_limit=3)
        handler = Mock()
        with self.assertRaisesRegex(RuntimeError, "not enough model budget"):
            middleware.wrap_model_call(planned_request(), handler)
        handler.assert_not_called()
        middleware = workflow.ResearchPlanningMiddleware(run_limit=10)
        handler = Mock(side_effect=[task_reply(), ModelResponse(result=[AIMessage(content="Done")])])
        with self.assertRaisesRegex(RuntimeError, "did not produce one valid researcher task"):
            middleware.wrap_model_call(planned_request(), handler)
        self.assertEqual(handler.call_count, 2)

    def test_provider_usage_fallback_is_preserved(self):
        response = task_reply()
        response.result[0].usage_metadata = None
        response.result[0].response_metadata = {"token_usage": {
            "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
        result = workflow.ResearchPlanningMiddleware(run_limit=10).wrap_model_call(
            planned_request(), Mock(return_value=response))
        self.assertEqual(research.summarize(result.result, 0, "offline")["tokens"],
                         {"input": 6, "output": 3})

    def test_real_deepagents_task_tool_runs_three_researchers_concurrently(self):
        class SingleCallModel(BaseChatModel):
            _barrier: threading.Barrier = PrivateAttr(default_factory=lambda: threading.Barrier(3))
            _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
            _researchers: int = PrivateAttr(default=0)

            @property
            def _llm_type(self):
                return "offline-single-tool-call-provider"

            def bind_tools(self, tools, **kwargs):
                return self.bind(**kwargs)

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                system = str(messages[0].content)
                choice = kwargs.get("tool_choice")
                if "You are an evidence-gathering researcher" in system:
                    with self._lock:
                        self._researchers += 1
                    # Sequential task execution breaks this barrier and fails the test.
                    self._barrier.wait(timeout=5)
                    message = AIMessage(content="Offline researcher completed its assigned scope")
                elif choice == "write_todos":
                    message = AIMessage(content="", tool_calls=[{
                        "name": "write_todos", "id": "todo", "args": {"todos": [
                            {"content": "Research independent questions", "status": "in_progress"},
                            {"content": "Synthesize and verify", "status": "pending"}]}}])
                elif choice == "task":
                    message = task_reply().result[0]
                else:
                    message = AIMessage(content="Research tasks returned; ready for synthesis")
                if not message.usage_metadata:
                    message.usage_metadata = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
                return ChatResult(generations=[ChatGeneration(message=message)])

        model = SingleCallModel()
        with patch("httpx.Client.send", side_effect=AssertionError("unexpected network")):
            graph = agents.build_lead_agent(StateBackend(), model)
            result = graph.invoke({"messages": [HumanMessage(content="Offline topic with current date")]},
                                  config={"recursion_limit": 1000},
                                  output_keys=["messages", "run_model_call_count", "thread_model_call_count"])
        self.assertEqual(model._researchers, 3)
        self.assertEqual(result["run_model_call_count"], 5)
        self.assertEqual(result["thread_model_call_count"], 5)
        actual_calls = [c for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls
                        if c["name"] == "task"]
        self.assertEqual(len(actual_calls), 3)
        completed_ids = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
        self.assertTrue(all(c["id"] in completed_ids for c in actual_calls))
        self.assertEqual(research.summarize(result["messages"], 0, "offline")["tokens"],
                         {"input": 25, "output": 11})


class FileErrorTests(unittest.TestCase):
    def request(self, name="write_file", path="/tmp/work/research/notes/review.md"):
        return SimpleNamespace(tool_call={"name": name, "id": "file-call",
                                          "args": {"file_path": path}})

    def test_directory_is_rejected_before_backend_upload(self):
        middleware = workflow.SandboxFileErrorsMiddleware()
        for path in ("/tmp/work/research/notes", "/tmp/work/report/", "/"):
            handler = Mock()
            result = middleware.wrap_tool_call(self.request(path=path), handler)
            self.assertEqual(result.status, "error")
            self.assertIn("filename, not a directory", result.content)
            handler.assert_not_called()

    def test_file_backend_exception_is_redacted_and_can_be_corrected(self):
        middleware = workflow.SandboxFileErrorsMiddleware()
        with patch.dict("os.environ", {"DAYTONA_API_KEY": "private-test-credential"}):
            result = middleware.wrap_tool_call(self.request(), Mock(side_effect=RuntimeError(
                "400 is a directory private-test-credential")))
        self.assertEqual(result.tool_call_id, "file-call")
        self.assertEqual(result.status, "error")
        self.assertNotIn("private-test-credential", result.content)
        self.assertIn("400 is a directory", result.content)
        success = ToolMessage(content="written", tool_call_id="file-call")
        self.assertIs(middleware.wrap_tool_call(self.request(), Mock(return_value=success)), success)

    def test_async_file_exception_and_other_tool_failures(self):
        middleware = workflow.SandboxFileErrorsMiddleware()
        async def fail(request): raise RuntimeError("backend unavailable")
        result = asyncio.run(middleware.awrap_tool_call(self.request(), fail))
        self.assertEqual(result.status, "error")
        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            asyncio.run(middleware.awrap_tool_call(self.request(name="execute"), fail))
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            middleware.wrap_tool_call(self.request(name="task"), Mock(side_effect=RuntimeError("budget exceeded")))


class SubmissionTests(unittest.TestCase):
    def payload(self):
        return {"title": "Evidence fixture", "tldr": ["Finding [1].", "Comparison [2].", "Limit [3]."],
                "background": "Definition [1].", "themes": [
                    {"heading": "Dynamics", "body": "Approach [1]."},
                    {"heading": "Video", "body": "Approach [2]."},
                    {"heading": "Evaluation", "body": "Approach [3]."}],
                "trends_and_open_problems": "Interpretation with evidence limits [1][2][3].",
                "sources": json.loads(artifacts()[research.SOURCES_PATH])}

    def test_structured_submission_finalizes_and_checks_real_artifacts_in_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class QualityBackend(ArtifactBackend):
                def execute(self, command):
                    if not command.startswith(f"python3 {research.QUALITY_VALIDATOR_PATH}"):
                        return super().execute(command)
                    self.commands.append(command)
                    catalog = root / "candidates.json"
                    if command.endswith("--remember-only"):
                        remember_sources(json.loads(self.files[research.SOURCES_PATH]), catalog)
                        return SimpleNamespace(exit_code=0, output="remembered")
                    report, sources = root / "report.md", root / "sources.json"
                    report.write_bytes(self.files[research.REPORT_PATH])
                    sources.write_bytes(self.files[research.SOURCES_PATH])
                    result = inspect_files(report, sources, catalog, root / "notes")
                    return SimpleNamespace(exit_code=0 if result["ok"] else 1, output=json.dumps(result))
            backend = QualityBackend({})
            submit = workflow.submission_tool(backend)
            def upload(target, files):
                self.assertIs(target, backend)
                self.assertEqual(set(files), {research.REPORT_PATH, research.SOURCES_PATH})
                backend.files.update(files)
            with patch.object(workflow, "upload", side_effect=upload), patch.object(
                    workflow, "download", side_effect=backend.download):
                outcome = json.loads(submit.invoke(self.payload()))
                self.assertTrue(outcome["ok"], outcome)
                self.assertEqual(len(outcome["checks"]), 4)
                report = backend.files[research.REPORT_PATH].decode()
                sources = json.loads(backend.files[research.SOURCES_PATH])
                self.assertEqual(research.output_problems(report, sources, 3), [])
                self.assertIn("## Trends and open problems", report)
                self.assertIn("## References", report)
                # Correct metadata alone cannot pass after unused sources are dropped.
                missing = self.payload()
                missing["tldr"][2] = "Limit [1]."
                missing["themes"][2]["body"] = "Approach [1]."
                missing["trends_and_open_problems"] = "Interpretation [1][2]."
                failed = json.loads(submit.invoke(missing))
                self.assertFalse(failed["ok"])
                self.assertIn("at least three source families", failed["checks"][-1]["output"])

    def test_invalid_structure_or_provenance_is_rejected_before_upload(self):
        for change in ("themes", "background", "provenance"):
            payload = self.payload()
            if change == "themes":
                payload["themes"] = payload["themes"][:1]
            elif change == "background":
                payload["background"] = " "
            else:
                payload["sources"][0]["id"] = "incorrect"
            with self.subTest(change=change), patch.object(workflow, "upload") as upload:
                result = json.loads(workflow.submission_tool(Mock()).invoke(payload))
                self.assertFalse(result["ok"])
                self.assertFalse(result["saved"])
                upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
