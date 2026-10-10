"""Bounded planning and structured report submission for small tool-calling models."""
import json
from typing import Literal

from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, StrictInt

from sandbox import download, upload
from tools import redact_secrets
from validate_report import report_problems, source_problems

NOTES_DIR = "/tmp/work/research/notes"
REPORT_PATH = "/tmp/work/report/report.md"
SOURCES_PATH = "/tmp/work/research/sources.json"
RESEARCH_DIR = "/tmp/work/research"


class SandboxFileErrorsMiddleware(AgentMiddleware):
    """Turn recoverable file-operation failures into errors the model can fix."""

    FILE_TOOLS = {"write_file", "edit_file", "read_file", "ls", "glob", "grep"}
    DIRECTORIES = {"/", "/tmp/work", RESEARCH_DIR, NOTES_DIR, "/tmp/work/report"}

    @staticmethod
    def _error(request, detail):
        return ToolMessage(content="ERROR: " + redact_secrets(detail),
                           tool_call_id=request.tool_call["id"], status="error")

    def _invalid_target(self, request):
        call = request.tool_call
        if call["name"] not in {"write_file", "edit_file", "read_file"}:
            return False
        path = call.get("args", {}).get("file_path", call.get("args", {}).get("path", ""))
        return isinstance(path, str) and (path.rstrip("/") in self.DIRECTORIES
                                          or path.endswith("/"))

    def _directory_error(self, request):
        return self._error(request, "This file tool requires a filename, not a directory. "
                           "Use ls for directories. For evidence notes, use your assigned "
                           "unique filename under /tmp/work/research/notes ending in .md.")

    def wrap_tool_call(self, request, handler):
        if self._invalid_target(request):
            return self._directory_error(request)
        try:
            return handler(request)
        except Exception as exc:
            if request.tool_call["name"] not in self.FILE_TOOLS:
                raise
            return self._error(request, f"sandbox file operation failed: {type(exc).__name__}: "
                               f"{exc}. Inspect the workspace and correct the file path.")

    async def awrap_tool_call(self, request, handler):
        if self._invalid_target(request):
            return self._directory_error(request)
        try:
            return await handler(request)
        except Exception as exc:
            if request.tool_call["name"] not in self.FILE_TOOLS:
                raise
            return self._error(request, f"sandbox file operation failed: {type(exc).__name__}: "
                               f"{exc}. Inspect the workspace and correct the file path.")


class ResearchPlanningMiddleware(ModelCallLimitMiddleware):
    """Collect three genuine task calls, then let Deep Agents execute them together.

    Some providers return only one call per response. Each planning request asks
    for one task; the resulting calls share one graph tool turn. Extra requests
    count against the same model budget, and their real token usage is retained.
    No researcher is started if planning fails or the budget is insufficient.
    """

    SCOPES = (
        ("foundations", "Foundational definitions and representative earlier work", "arxiv and web"),
        ("recent", "Developments within the two years before the supplied current date", "hf-search and web"),
        ("evaluation", "Evaluation, limitations, comparisons and open problems", "web and arxiv"),
    )

    def __init__(self, *, note_format="", **kwargs):
        super().__init__(**kwargs)
        self.note_format = note_format

    @staticmethod
    def _has_call(messages, name, researcher=False):
        return any(
            call["name"] == name
            and (not researcher or call.get("args", {}).get("subagent_type") == "researcher")
            for message in messages if isinstance(message, AIMessage)
            for call in message.tool_calls
        )

    @staticmethod
    def _message(response):
        messages = [m for m in response.result if isinstance(m, AIMessage)]
        if len(messages) != 1:
            raise RuntimeError("planning requires exactly one assistant response")
        return messages[0]

    @staticmethod
    def _constrain(request, name, instruction):
        selected = [t for t in request.tools if getattr(t, "name", None) == name]
        if len(selected) != 1:
            raise RuntimeError(f"required workflow tool unavailable: {name}")
        return request.override(
            tools=selected, tool_choice=name,
            messages=[*request.messages, HumanMessage(content=instruction)],
        )

    def _plan(self, request):
        """Yield bounded model requests; receive responses without unpaired calls."""
        if not self._has_call(request.messages, "write_todos"):
            response = yield self._constrain(
                request, "write_todos",
                "First create the actual todo list for this survey: parallel evidence "
                "research, synthesis, sandbox finalization/validation and citation review. "
                "Call write_todos now; do not start research in this response.",
            )
            message = self._message(response)
            if not any(c["name"] == "write_todos" for c in message.tool_calls):
                raise RuntimeError("model did not call the required write_todos tool")
            yield_result = self._stamp(response, 1)
        elif not self._has_call(request.messages, "task", researcher=True):
            remaining = min(
                self.run_limit - request.state.get("run_model_call_count", 0)
                if self.run_limit is not None else 3,
                self.thread_limit - request.state.get("thread_model_call_count", 0)
                if self.thread_limit is not None else 3,
            )
            if remaining < 3:
                raise RuntimeError("not enough model budget to plan three researcher tasks")
            calls, responses = [], []
            context = next((m.content for m in request.messages if isinstance(m, HumanMessage)), "")
            for index, (slug, question, families) in enumerate(self.SCOPES, 1):
                path = f"{NOTES_DIR}/{index:02d}-{slug}.md"
                contract = (
                    f"Assigned scope: {question}. Target source labels: {families}. "
                    f"Write evidence notes ONLY to {path}. Include full topic/date scope, "
                    "metadata (id,url,title,date,source), exact evidence excerpts and access "
                    "limitations. Dates must be YYYY-MM-DD or empty if unknown. Retrieve "
                    "distinct relevant URLs so deduplication preserves source coverage. "
                    "Use the researcher note schema and return path, count, labels and findings."
                    + "\nRequired notes format:\n" + self.note_format
                )
                response = yield self._constrain(
                    request, "task", "Plan exactly ONE researcher delegation for the "
                    "assigned scope below. Call task with subagent_type='researcher'. "
                    "The other scopes are assigned independently; do not research yet. " + contract,
                )
                message = self._message(response)
                valid = [call for call in message.tool_calls
                         if call["name"] == "task"
                         and call.get("args", {}).get("subagent_type") == "researcher"
                         and isinstance(call.get("args", {}).get("description"), str)
                         and call["args"]["description"].strip()]
                if not valid:
                    diagnostic = [{"name": call["name"],
                                   "subagent_type": call.get("args", {}).get("subagent_type"),
                                   "has_description": bool(call.get("args", {}).get("description"))}
                                  for call in message.tool_calls]
                    raise RuntimeError("model did not produce one valid researcher task: "
                                       + json.dumps(diagnostic))
                # Some providers ignore the request for one proposal and return
                # several. Dispatch only the first valid proposal for this scope;
                # unused proposals are never executed or counted as real tasks.
                original = valid[0]
                # Add explicit context to the real model-produced question. Never create
                # a missing task call or treat a planning response as completed research.
                calls.append({**original, "id": f"{original['id']}-research-{index}",
                              "args": {**original["args"], "description": (
                                  original["args"]["description"] + "\n\n" + contract
                                  + "\n\nFull survey request (topic/template data):\n" + str(context)
                              )}})
                responses.append(message)
            usage = {key: sum((m.usage_metadata or {}).get(key, 0) for m in responses)
                     for key in ("input_tokens", "output_tokens", "total_tokens")}
            # Providers without usage_metadata may expose OpenAI-style token_usage.
            for m in responses:
                if not m.usage_metadata:
                    raw = m.response_metadata.get("token_usage", {})
                    usage["input_tokens"] += raw.get("prompt_tokens", 0)
                    usage["output_tokens"] += raw.get("completion_tokens", 0)
                    usage["total_tokens"] += raw.get("total_tokens", 0)
            combined = AIMessage(content="", tool_calls=calls, usage_metadata=usage,
                                 response_metadata={"lab_actual_model_calls": len(responses)})
            yield_result = ModelResponse(result=[combined])
        else:
            yield_result = self._stamp((yield request), 1)
        return yield_result

    def _stamp(self, response, count):
        message = self._message(response)
        updated = message.model_copy(update={"response_metadata": {
            **message.response_metadata, "lab_actual_model_calls": count,
        }})
        return ModelResponse(result=[updated if m is message else m for m in response.result],
                             structured_response=response.structured_response)

    def wrap_model_call(self, request, handler):
        plan = self._plan(request)
        try:
            current = next(plan)
            while True:
                current = plan.send(handler(current))
        except StopIteration as finished:
            return finished.value

    async def awrap_model_call(self, request, handler):
        plan = self._plan(request)
        try:
            current = next(plan)
            while True:
                response = await handler(current)
                current = plan.send(response)
        except StopIteration as finished:
            return finished.value

    def after_model(self, state, runtime):
        last = state["messages"][-1]
        count = last.response_metadata.get("lab_actual_model_calls", 1)
        return {"run_model_call_count": state.get("run_model_call_count", 0) + count,
                "thread_model_call_count": state.get("thread_model_call_count", 0) + count}


class ReportSource(BaseModel):
    n: StrictInt = Field(gt=0)
    id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    date: str = Field(description="Publication date YYYY-MM-DD; empty if unknown")
    source: Literal["arxiv", "hf-daily", "hf-search", "web"]


class ThematicSection(BaseModel):
    heading: str = Field(min_length=1, description="One thematic heading; no Markdown prefix")
    body: str = Field(min_length=1, description="Substantive evidence-based synthesis with inline [n] citations")


def submission_tool(backend):
    @tool
    def submit_report(
        title: str,
        tldr: list[str],
        background: str,
        themes: list[ThematicSection],
        trends_and_open_problems: str,
        sources: list[ReportSource],
    ) -> str:
        """Save the model-written survey with exact required headings and complete
        source metadata. Supply 3-5 cited TL;DR bullets and 3-6 thematic sections.
        Every factual claim needs inline [n]; use at least three truthful source
        labels and distinct URLs. Do not supply References: the provided script
        generates them INSIDE the sandbox. Returns real finalization/validation
        results. Read the final registry before citation-checker or further edits.
        This checks structure/citations, not whether a scientific claim is true.
        """
        try:
            problems = []
            if not title.strip() or "\n" in title or title.startswith("#"):
                problems.append("title must be one nonempty line without Markdown prefix")
            if not 3 <= len(tldr) <= 5 or any(not item.strip() for item in tldr):
                problems.append("TL;DR needs 3-5 nonempty cited bullets")
            if not 3 <= len(themes) <= 6:
                problems.append("provide 3-6 thematic sections")
            entries = [entry.model_dump() for entry in sources]
            problems.extend(source_problems(entries))
            body = (f"# {title.strip()}\n\n## TL;DR\n"
                    + "\n".join("- " + item.strip() for item in tldr)
                    + f"\n\n## Background\n{background.strip()}\n\n"
                    + "\n\n".join(f"## {s.heading.strip()}\n{s.body.strip()}" for s in themes)
                    + f"\n\n## Trends and open problems\n{trends_and_open_problems.strip()}\n")
            # References are checked after the original finalizer runs in sandbox.
            problems.extend(report_problems(body + "\n## References\nPending finalization\n"))
            if problems:
                return json.dumps({"ok": False, "saved": False, "problems": problems})
            files = {REPORT_PATH: body.encode("utf-8"), SOURCES_PATH: (
                json.dumps(entries, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}
            upload(backend, files)
            if download(backend, list(files)) != files:
                return json.dumps({"ok": False, "problems": ["sandbox upload verification failed"]})
            commands = (
                f"python3 {RESEARCH_DIR}/validate_report.py --remember-only",
                f"python3 {RESEARCH_DIR}/finalize_citations.py",
                f"python3 {RESEARCH_DIR}/check_citations.py",
                f"python3 {RESEARCH_DIR}/validate_report.py",
            )
            results = []
            for command in commands:
                result = backend.execute(command)
                results.append({"command": command, "exit_code": result.exit_code,
                                "output": redact_secrets(result.output)})
            return json.dumps({"ok": all(r["exit_code"] == 0 for r in results),
                               "saved": True, "checks": results}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"ok": False, "problems": [redact_secrets(
                f"report submission failed: {type(exc).__name__}: {exc}")]})

    return submit_report
