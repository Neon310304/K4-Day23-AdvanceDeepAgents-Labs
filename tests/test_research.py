"""Output integrity and the complete runner lifecycle, without paid services."""
import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from deepagents.backends import StateBackend
from finalize_citations import finalize

import agents
import research
from validate_report import output_problems, remember_sources, coverage_details


def artifacts():
    sources = [
        {"n": 1, "id": "1803.10122", "url": "https://arxiv.org/abs/1803.10122",
         "title": "World Models", "date": "2018-03-27", "source": "arxiv"},
        {"n": 2, "id": "2402.08268", "url": "https://huggingface.co/papers/2402.08268",
         "title": "Long-context world model", "date": "2024-02-13", "source": "hf-search"},
        {"n": 3, "id": "project", "url": "https://example.org/project",
         "title": "Project evidence fixture", "date": "", "source": "web"},
    ]
    body = (
        "# Survey\n\n## TL;DR\n- Finding [1][2][3].\n\n"
        "## Background\nFoundation [1].\n\n## Dynamics\nApproach [1].\n\n"
        "## Video\nApproach [2].\n\n## Evaluation\nApproach [3].\n\n"
        "## Trends and open problems\nUncertainty [1][2][3].\n\n## References\n"
    )
    body += "\n".join(f"[{s['n']}] {s['title']}. {s['source']}. {s['url']} ({s['date'] or 'n.d.'})"
                      for s in sources) + "\n"
    return {research.REPORT_PATH: body.encode(), research.SOURCES_PATH: json.dumps(sources, indent=4).encode()}


def messages():
    return [AIMessage(content="delegating", tool_calls=[
        {"name": "task", "args": {"subagent_type": "researcher", "description": "question"}, "id": str(n)}
        for n in range(3)
    ], usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25})]


class ArtifactBackend:
    """Offline artifact backend executing the real finalizer/validator functions."""

    def __init__(self, files):
        self.files = dict(files)
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        report = self.files[research.REPORT_PATH].decode("utf-8")
        sources = json.loads(self.files[research.SOURCES_PATH])
        if command == f"python3 {research.FINALIZER_PATH}":
            report, sources, problems = finalize(report, sources)
            if not problems:
                self.files[research.REPORT_PATH] = report.encode()
                self.files[research.SOURCES_PATH] = json.dumps(sources).encode()
                return SimpleNamespace(exit_code=0, output="FINALIZED")
        elif command == f"python3 {research.VALIDATOR_PATH}":
            problems = research.check(report, sources)
            if not problems:
                return SimpleNamespace(exit_code=0, output=f"OK: {len(sources)} sources, all citations resolve")
        else:
            raise AssertionError(f"unexpected command: {command}")
        return SimpleNamespace(exit_code=1, output="; ".join(problems))

    def download(self, backend, paths):
        assert backend is self
        return {path: self.files.get(path) for path in paths}


class ProgressTests(unittest.TestCase):
    def test_progress_logs_never_include_tool_arguments_or_model_content(self):
        secret = "sk-" + "s" * 30
        log = research.ProgressLog()
        output = io.StringIO()
        with redirect_stdout(output):
            log.on_tool_start({"name": "web_fetch"}, secret)
            log.on_tool_start({"name": "execute"}, secret)
            for _ in range(10):
                log.on_llm_end(SimpleNamespace(generations=[[secret]]))
        self.assertNotIn(secret, output.getvalue())
        self.assertIn("Tool: web_fetch", output.getvalue())
        self.assertIn("Completed model requests: 10", output.getvalue())
        with patch.dict(research.os.environ, {"LAB_PROGRESS": "1"}):
            self.assertIsInstance(research._agent_config(1000)["callbacks"][0], research.ProgressLog)


class RunnerTests(unittest.TestCase):
    def test_review_revision_retains_measured_prior_work_without_double_count(self):
        prior = research.summarize(messages(), 10, "original-model")
        new_messages = [AIMessage(content="checked", tool_calls=[
            {"name": "task", "args": {"subagent_type": "citation-checker"}, "id": "review"},
            {"name": "execute", "args": {}, "id": "write"},
        ], usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9})]
        data = artifacts()
        with tempfile.TemporaryDirectory() as directory, patch.object(research, "download", return_value=data):
            path = research.save_outputs(Mock(), "world", new_messages, 3, "review-model",
                                         directory, prior_meta=prior)
            meta = json.loads(path.with_suffix(".meta.json").read_bytes())
            self.assertEqual(path.read_bytes(), data[research.REPORT_PATH])
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), data[research.SOURCES_PATH])
            self.assertEqual(meta["model"], "original-model")
            self.assertEqual(meta["subagent_calls"], 4)
            self.assertEqual(meta["tool_calls"], {"execute": 1, "task": 4})
            self.assertEqual(meta["tokens"], {"input": 27, "output": 7})
            self.assertEqual(meta["elapsed_s"], 13)
            self.assertEqual(meta["revisions"][0]["model"], "review-model")
            self.assertEqual(prior["subagent_calls"], 3)
            second = research._include_prior_run(research.summarize([], 1, "second"), meta)
            self.assertEqual(second["subagent_calls"], 4)
            self.assertEqual(len(second["revisions"]), 2)

    def test_review_refuses_incomplete_report_before_using_paid_services(self):
        with patch.object(research, "_complete_existing", return_value=False), patch.object(research, "make_model") as make, redirect_stderr(io.StringIO()):
            self.assertEqual(research.repair_existing("missing", ["incorrect units"]), 1)
        make.assert_not_called()

    def test_unchanged_review_stops_after_bounded_followups_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = artifacts()
            (root / "world.md").write_bytes(data[research.REPORT_PATH])
            (root / "world.sources.json").write_bytes(data[research.SOURCES_PATH])
            (root / "world.meta.json").write_text(json.dumps(research.summarize(messages(), 1, "original")))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            uploaded, cleanup = {}, []
            backend, agent = Mock(), Mock()
            backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
            agent.invoke.return_value = {"messages": [AIMessage(content="Claimed complete without edits")]}
            @contextmanager
            def sandbox():
                try: yield backend
                finally: cleanup.append(True)
            with patch.object(research, "REPORTS", root), patch.object(research, "_complete_existing", return_value=True), patch.object(research, "make_model", return_value=SimpleNamespace(model_name="review")), patch.object(research, "open_sandbox", sandbox), patch.object(research, "upload", side_effect=lambda _, files: uploaded.update(files)), patch.object(research, "download", side_effect=lambda _, paths: {p: uploaded.get(p) for p in paths}), patch.object(research, "build_repair_agent", return_value=agent), patch.object(research, "_finish_outputs") as finish, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(research.repair_existing("world", ["correct the claim"]), 1)
            self.assertEqual(agent.invoke.call_count, research.MAX_REPAIR_ATTEMPTS + 1)
            self.assertIn("last pass returned without changing", agent.invoke.call_args.args[0]["messages"][0]["content"])
            self.assertEqual(cleanup, [True])
            finish.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_review_preserves_original_artifacts_and_cleans_up_on_model_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = artifacts()
            (root / "world.md").write_bytes(data[research.REPORT_PATH])
            (root / "world.sources.json").write_bytes(data[research.SOURCES_PATH])
            meta = research.summarize(messages(), 1, "original")
            (root / "world.meta.json").write_text(json.dumps(meta))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            lifecycle, uploaded = [], {}
            backend = Mock()
            backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
            @contextmanager
            def sandbox():
                try:
                    yield backend
                finally:
                    lifecycle.append("cleanup")
            def upload(_, files): uploaded.update(files)
            agent = Mock()
            agent.invoke.side_effect = RuntimeError("quota exhausted")
            with patch.object(research, "REPORTS", root), patch.object(research, "_complete_existing", return_value=True), patch.object(research, "make_model", return_value=SimpleNamespace(model_name="review")), patch.object(research, "open_sandbox", sandbox), patch.object(research, "upload", upload), patch.object(research, "download", side_effect=lambda _, paths: {p: uploaded.get(p) for p in paths}), patch.object(research, "build_repair_agent", return_value=agent), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(research.repair_existing("world", ["incorrect units"]), 1)
            self.assertEqual(lifecycle, ["cleanup"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_slug_paths_empty_unicode_and_length(self):
        for topic in ("../../x", r"..\..\x", "C:/folder/file", ".", " ", "a" * 200, "chủ đề nghiên cứu"):
            with self.subTest(topic=topic):
                slug = research.slugify(topic)
                self.assertTrue(slug)
                self.assertLessEqual(len(slug), 60)
                self.assertNotIn("/", slug)
                self.assertNotIn("\\", slug)
                self.assertNotIn(".", slug)
        self.assertEqual(research.slugify("survey about world model"), "survey-about-world-model")
        self.assertEqual(research.slugify(""), "topic")

    def test_prompt_contains_template_date_and_workspace_contract(self):
        prompt = research.build_prompt("world models")
        for value in ("TL;DR", "Background", "Trends and open problems", research.REPORT_PATH,
                      "Asia/Bangkok", "world models"):
            self.assertIn(value, prompt)

    def test_summarize_objects_dicts_and_provider_fallback_without_double_count(self):
        lead = messages() + [
            {"tool_calls": [{"name": "execute", "args": {}}],
             "usage_metadata": {"input_tokens": 3, "output_tokens": 2},
             "response_metadata": {"token_usage": {"prompt_tokens": 999}}},
            {"response_metadata": {"token_usage": {"prompt_tokens": 7, "completion_tokens": 4}}},
        ]
        meta = research.summarize(lead, 12.34, "model")
        self.assertEqual(meta["subagent_calls"], 3)
        self.assertEqual(meta["tool_calls"], {"execute": 1, "task": 3})
        self.assertEqual(meta["tokens"], {"input": 30, "output": 11})
        self.assertEqual(meta["elapsed_s"], 12.3)

    def test_save_preserves_exact_sandbox_bytes_and_records_meta(self):
        data = artifacts()
        with tempfile.TemporaryDirectory() as directory, patch.object(research, "download", return_value=data):
            path = research.save_outputs(Mock(), "world models", messages(), 1.23, "model", directory)
            self.assertEqual(path.read_bytes(), data[research.REPORT_PATH])
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), data[research.SOURCES_PATH])
            meta = json.loads(path.with_suffix(".meta.json").read_bytes())
            self.assertEqual(meta["n_sources"], 3)
            self.assertEqual(meta["source_families"], ["arxiv", "hf-search", "web"])
            self.assertEqual(meta["subagent_calls"], 3)

    def test_invalid_downloads_write_nothing(self):
        good = artifacts()
        invalid = [
            {}, {**good, research.REPORT_PATH: None}, {**good, research.REPORT_PATH: b" "},
            {**good, research.REPORT_PATH: b"\xff"}, {**good, research.SOURCES_PATH: b"broken"},
            {**good, research.SOURCES_PATH: b"{}"}, {**good, research.SOURCES_PATH: b"[]"},
            {**good, research.REPORT_PATH: good[research.REPORT_PATH].replace(b"[1][2][3]", b"[99]")},
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-reports"
            for data in invalid:
                with self.subTest(data=data), patch.object(research, "download", return_value=data):
                    with self.assertRaises(RuntimeError):
                        research.save_outputs(Mock(), "world models", messages(), 1, "model", target)
                    self.assertFalse(target.exists())

    def test_failed_run_does_not_overwrite_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world-models.md"
            path.write_bytes(b"existing valid report")
            with patch.object(research, "download", return_value={}):
                with self.assertRaises(RuntimeError):
                    research.save_outputs(Mock(), "world models", messages(), 1, "model", directory)
            self.assertEqual(path.read_bytes(), b"existing valid report")

    def test_insufficient_delegation_and_wrong_provenance_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(research, "download", return_value=artifacts()):
                with self.assertRaisesRegex(RuntimeError, "three real subagent"):
                    research.save_outputs(Mock(), "world", [], 1, "model", directory)
            data = artifacts()
            sources = json.loads(data[research.SOURCES_PATH])
            sources[0]["source"] = "hf-search"
            data[research.SOURCES_PATH] = json.dumps(sources).encode()
            with patch.object(research, "download", return_value=data):
                with self.assertRaisesRegex(RuntimeError, "Hugging Face URL"):
                    research.save_outputs(Mock(), "world", messages(), 1, "model", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_publish_rolls_back_partial_io_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.md", Path(directory) / "a.json"
            first.write_bytes(b"previous")
            actual_replace = research.os.replace
            calls = [0]
            def replace(source, destination):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("simulated disk failure")
                actual_replace(source, destination)
            with patch.object(research.os, "replace", side_effect=replace):
                with self.assertRaises(RuntimeError):
                    research._publish_files({first: b"new", second: b"new sources"})
            self.assertEqual(first.read_bytes(), b"previous")
            self.assertFalse(second.exists())
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["a.md"])

    def test_main_checks_scripts_invokes_with_limits_and_validates_in_sandbox(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: 3 sources, all citations resolve")
        lifecycle = []
        @contextmanager
        def sandbox():
            lifecycle.append("open")
            try:
                yield backend
            finally:
                lifecycle.append("cleanup")
        agent = Mock()
        agent.invoke.return_value = {"messages": messages()}
        scripts = {research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
                   research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes(),
                   research.QUALITY_VALIDATOR_PATH: research.QUALITY_VALIDATOR_SOURCE.read_bytes()}
        with patch.object(research, "make_model", return_value=SimpleNamespace(model_name="fixture")), patch.object(research, "open_sandbox", sandbox), patch.object(research, "upload") as upload, patch.object(research, "download", return_value=scripts), patch.object(research, "build_lead_agent", return_value=agent), patch.object(research, "save_outputs", return_value=Path("fixture.md")) as save, redirect_stdout(io.StringIO()):
            self.assertEqual(research.main("world"), 0)
        self.assertEqual(lifecycle, ["open", "cleanup"])
        upload.assert_called_once_with(backend, scripts)
        self.assertEqual(agent.invoke.call_args.kwargs["config"]["recursion_limit"], 1000)
        commands = [call.args[0] for call in backend.execute.call_args_list]
        self.assertIn(f"python3 {research.FINALIZER_PATH}", commands)
        self.assertIn(f"python3 {research.VALIDATOR_PATH}", commands)
        save.assert_called_once()

    def test_main_cleans_sandbox_on_agent_failure_and_never_saves(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
        lifecycle = []
        @contextmanager
        def sandbox():
            try:
                yield backend
            finally:
                lifecycle.append("cleanup")
        agent = Mock()
        agent.invoke.side_effect = RuntimeError("model unavailable")
        scripts = {research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
                   research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes(),
                   research.QUALITY_VALIDATOR_PATH: research.QUALITY_VALIDATOR_SOURCE.read_bytes()}
        with patch.object(research, "make_model", return_value=SimpleNamespace(model_name="fixture")), patch.object(research, "open_sandbox", sandbox), patch.object(research, "upload"), patch.object(research, "download", return_value=scripts), patch.object(research, "build_lead_agent", return_value=agent), patch.object(research, "save_outputs") as save, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(research.main("world"), 1)
        self.assertEqual(lifecycle, ["cleanup"])
        save.assert_not_called()

    def test_main_rejects_scripts_modified_during_agent_run(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
        @contextmanager
        def sandbox():
            yield backend
        scripts = {research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
                   research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes(),
                   research.QUALITY_VALIDATOR_PATH: research.QUALITY_VALIDATOR_SOURCE.read_bytes()}
        modified = {**scripts, research.VALIDATOR_PATH: b"print('OK: forged')"}
        agent = Mock()
        agent.invoke.return_value = {"messages": messages()}
        with patch.object(research, "make_model", return_value=SimpleNamespace(model_name="fixture")), patch.object(research, "open_sandbox", sandbox), patch.object(research, "upload"), patch.object(research, "download", side_effect=[scripts, modified]), patch.object(research, "build_lead_agent", return_value=agent), patch.object(research, "save_outputs") as save, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(research.main("world"), 1)
        save.assert_not_called()
        self.assertEqual(backend.execute.call_count, 1)

    def test_empty_topic_fails_without_model(self):
        with patch.object(research, "make_model") as model, redirect_stderr(io.StringIO()):
            self.assertEqual(research.main(""), 2)
        model.assert_not_called()

    def test_resume_does_not_skip_corrupt_reports(self):
        data = artifacts()
        with tempfile.TemporaryDirectory() as directory, patch.object(research, "download", return_value=data):
            research.save_outputs(Mock(), "world", messages(), 1, "fixture", directory)
            with patch.object(research, "REPORTS", Path(directory)):
                self.assertTrue(research._complete_existing("world"))
                (Path(directory) / "world.sources.json").write_text("{}", encoding="utf-8")
                self.assertFalse(research._complete_existing("world"))

    def test_malformed_metadata_and_source_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(research, "REPORTS", Path(directory)):
            data = artifacts()
            for field, value in (("source", []), ("date", "2025-02-30"), ("date", None)):
                entries = json.loads(data[research.SOURCES_PATH])
                entries[0][field] = value
                changed = {**data, research.SOURCES_PATH: json.dumps(entries).encode()}
                with self.subTest(field=field, value=value), patch.object(research, "download", return_value=changed):
                    with self.assertRaises(RuntimeError):
                        research.save_outputs(Mock(), "world", messages(), 1, "fixture", directory)
            with patch.object(research, "download", return_value=data):
                research.save_outputs(Mock(), "world", messages(), 1, "fixture", directory)
            (Path(directory) / "world.meta.json").write_text("42", encoding="utf-8")
            self.assertFalse(research._complete_existing("world"))

    def test_report_structure_requires_exact_nonempty_sections(self):
        report = artifacts()[research.REPORT_PATH].decode()
        missing = report.replace("## Trends and open problems\nUncertainty [1][2][3].\n\n", "")
        self.assertIn("missing report section: Trends and open problems", research._report_problems(missing))
        renamed = report.replace("## Trends and open problems", "## Trends and Open Problems")
        self.assertIn("missing report section: Trends and open problems", research._report_problems(renamed))
        empty = report.replace("Uncertainty [1][2][3].", "<!-- placeholder -->")
        self.assertIn("empty report section: Trends and open problems", research._report_problems(empty))
        self.assertEqual(research._report_problems(report.replace("\n", "\r\n")), [])

    def test_missing_trends_is_repaired_before_sandbox_cleanup_and_save(self):
        good = artifacts()
        bad = {**good, research.REPORT_PATH: good[research.REPORT_PATH].replace(
            b"## Trends and open problems\nUncertainty [1][2][3].\n\n", b"")}
        scripts = {research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
                   research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes()}
        backend = ArtifactBackend({**bad, **scripts})
        repair_message = AIMessage(content="Repaired existing notes",
                                  usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
        def repair(payload, *, config):
            self.assertEqual(config["recursion_limit"], research.REPAIR_RECURSION_LIMIT)
            prompt = payload["messages"][0]["content"]
            self.assertIn("missing report section: Trends and open problems", prompt)
            self.assertIn("SAME sandbox", prompt)
            # The stand-in model edits the actual backend artifact, as a repair agent would.
            backend.files[research.REPORT_PATH] = backend.files[research.REPORT_PATH].replace(
                b"## References", b"## Trends and open problems\nUncertainty [1][2][3].\n\n## References")
            return {"messages": [repair_message]}
        repair_agent = Mock()
        repair_agent.invoke.side_effect = repair
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(research, "download", side_effect=backend.download), patch.object(research, "build_repair_agent", return_value=repair_agent) as build, redirect_stdout(output):
            actual_save = research.save_outputs
            with patch.object(research, "save_outputs", side_effect=lambda *args, **kwargs: actual_save(*args, reports_dir=directory, **kwargs)):
                path = research._finish_outputs(backend, "world", messages(), research.time.monotonic(),
                                                "fixture", scripts, Mock())
            self.assertEqual(path.read_bytes(), backend.files[research.REPORT_PATH])
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), backend.files[research.SOURCES_PATH])
            meta = json.loads(path.with_suffix(".meta.json").read_bytes())
            self.assertEqual(meta["tokens"], {"input": 27, "output": 8})
            self.assertEqual(meta["subagent_calls"], 3)
            self.assertEqual(meta["n_sources"], 3)
        build.assert_called_once()
        repair_agent.invoke.assert_called_once()
        self.assertEqual(len(backend.commands), 4)
        self.assertIn("Repairing report (1/2)", output.getvalue())
        self.assertIn("OK: 3 sources", output.getvalue())

    def test_repairs_stop_after_bounded_attempts_and_write_nothing(self):
        good = artifacts()
        scripts = {research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
                   research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes()}
        report = good[research.REPORT_PATH].replace(
            b"## Trends and open problems\nUncertainty [1][2][3].\n\n", b"")
        backend = ArtifactBackend({**good, **scripts, research.REPORT_PATH: report})
        repair_agent = Mock()
        repair_agent.invoke.return_value = {"messages": []}
        with tempfile.TemporaryDirectory() as directory, patch.object(research, "download", side_effect=backend.download), patch.object(research, "build_repair_agent", return_value=repair_agent), redirect_stdout(io.StringIO()):
            actual_save = research.save_outputs
            with patch.object(research, "save_outputs", side_effect=lambda *args, **kwargs: actual_save(*args, reports_dir=directory, **kwargs)):
                with self.assertRaisesRegex(research.InvalidOutputsError, "after 2 repair attempts"):
                    research._finish_outputs(backend, "world", messages(), research.time.monotonic(),
                                             "fixture", scripts, Mock())
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(repair_agent.invoke.call_count, research.MAX_REPAIR_ATTEMPTS)

    def test_citation_failure_is_repaired_without_loosening_validator(self):
        scripts = {research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
                   research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes()}
        data = artifacts()
        data[research.REPORT_PATH] = data[research.REPORT_PATH].replace(b"Foundation [1].", b"Foundation [99].")
        backend = ArtifactBackend({**data, **scripts})
        repair_agent = Mock()
        def repair(payload, *, config):
            self.assertIn("missing from sources.json", payload["messages"][0]["content"])
            backend.files[research.REPORT_PATH] = backend.files[research.REPORT_PATH].replace(b"[99]", b"[1]")
            return {"messages": []}
        repair_agent.invoke.side_effect = repair
        with tempfile.TemporaryDirectory() as directory, patch.object(research, "download", side_effect=backend.download), patch.object(research, "build_repair_agent", return_value=repair_agent), redirect_stdout(io.StringIO()):
            actual_save = research.save_outputs
            with patch.object(research, "save_outputs", side_effect=lambda *args, **kwargs: actual_save(*args, reports_dir=directory, **kwargs)):
                path = research._finish_outputs(backend, "world", messages(), research.time.monotonic(),
                                                "fixture", scripts, Mock())
            self.assertNotIn("[99]", path.read_text())
        repair_agent.invoke.assert_called_once()

    def test_disk_failure_never_triggers_llm_repair(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: valid")
        scripts = {"validator": b"original"}
        with patch.object(research, "download", return_value=scripts), patch.object(research, "save_outputs", side_effect=RuntimeError("disk full")), patch.object(research, "build_repair_agent") as repair:
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                research._finish_outputs(backend, "world", messages(), research.time.monotonic(), "fixture", scripts, Mock())
        repair.assert_not_called()


class AgentConfigurationTests(unittest.TestCase):
    def test_repair_graph_reaches_model_budget_without_recursion_error(self):
        class RepeatingValidationModel(BaseChatModel):
            """Exercise the real graph/middleware without API calls."""

            calls: int = 0
            batch_size: int = 1
            finish_after: int = 0

            @property
            def _llm_type(self):
                return "offline-validation-loop"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.calls += 1
                if self.calls == self.finish_after:
                    message = AIMessage(content="Repair complete")
                else:
                    message = AIMessage(content="", tool_calls=[
                        {"name": "validate_report", "args": {}, "id": f"{self.calls}-{n}"}
                        for n in range(self.batch_size)
                    ])
                return ChatResult(generations=[ChatGeneration(message=message)])

        for batch_size, finish_after in ((1, 0), (3, 0), (1, agents.REPAIR_MODEL_LIMIT)):
            with self.subTest(batch_size=batch_size, finish_after=finish_after):
                backend = StateBackend()
                model = RepeatingValidationModel(batch_size=batch_size, finish_after=finish_after)
                with patch.object(backend, "execute", create=True, return_value=SimpleNamespace(
                    exit_code=1, output='{"ok": false, "problems": ["missing section"]}'
                )) as execute, patch("httpx.Client.send", side_effect=AssertionError("unexpected network")):
                    graph = agents.build_repair_agent(backend, model)
                    result = graph.invoke(
                        {"messages": [{"role": "user", "content": "Repair existing artifacts"}]},
                        config={"recursion_limit": research.REPAIR_RECURSION_LIMIT},
                    )
                self.assertEqual(model.calls, agents.REPAIR_MODEL_LIMIT)
                expected_tools = min(
                    (model.calls - bool(finish_after)) * batch_size, agents.REPAIR_TOOL_LIMIT
                )
                self.assertEqual(execute.call_count, expected_tools)
                last = result["messages"][-1]
                if finish_after:
                    self.assertEqual(last.content, "Repair complete")
                else:
                    self.assertIn("Model call limits exceeded", last.content)
                self.assertFalse(last.tool_calls)

    def test_subagent_limits_are_fresh_and_cover_every_role(self):
        first, second = agents.build_subagents(), agents.build_subagents()
        self.assertEqual([s["name"] for s in first], ["researcher", "citation-checker"])
        for left, right in zip(first, second):
            self.assertEqual(len(left["middleware"]), 3)
            self.assertIsInstance(left["middleware"][-1], agents.SandboxFileErrorsMiddleware)
            for first_limit, second_limit in zip(left["middleware"], right["middleware"]):
                self.assertIsNot(first_limit, second_limit)
        with patch.object(agents, "create_deep_agent") as create:
            agents.build_lead_agent(Mock(), Mock())
        kwargs = create.call_args.kwargs
        self.assertEqual(len(kwargs["middleware"]), 4)
        for spec in kwargs["subagents"]:
            self.assertEqual(len(spec["middleware"]), 3)
            self.assertIsInstance(spec["middleware"][-1], agents.SandboxFileErrorsMiddleware)
        self.assertIn("general-purpose", [spec["name"] for spec in kwargs["subagents"]])

    def test_actual_deepagents_graph_compiles_without_a_network_call(self):
        # Compilation uses the pinned package, exercising real middleware wiring.
        model = ChatOpenAI(model="test-model", api_key="test-only")
        with patch("httpx.Client.send", side_effect=AssertionError("unexpected network")):
            graph = agents.build_lead_agent(StateBackend(), model)
        self.assertTrue(callable(graph.invoke))

    def test_repair_has_smaller_explicit_budget_and_existing_evidence_prompt(self):
        with patch.object(agents, "create_deep_agent") as create:
            agents.build_repair_agent(Mock(), Mock())
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["system_prompt"], agents.REPAIR_PROMPT)
        self.assertLess(agents.REPAIR_MODEL_LIMIT, agents.LEAD_MODEL_LIMIT)
        self.assertLess(agents.REPAIR_TOOL_LIMIT, agents.LEAD_TOOL_LIMIT)
        for spec in kwargs["subagents"]:
            self.assertEqual(len(spec["middleware"]), 3)
            self.assertIsInstance(spec["middleware"][-1], agents.SandboxFileErrorsMiddleware)
        self.assertIn("Do not restart", agents.REPAIR_PROMPT)

    def test_lead_and_repair_get_full_validation_tool_and_repair_source_tools(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=1, output='{"ok": false, "problems": ["missing family"]}')
        with patch.object(agents, "create_deep_agent") as create:
            agents.build_lead_agent(backend, Mock())
        tools = create.call_args.kwargs["tools"]
        self.assertEqual([tool.name for tool in tools], ["validate_report", "submit_report"])
        self.assertIn('"ok": false', tools[0].invoke({}))
        with patch.object(agents, "create_deep_agent") as create:
            agents.build_repair_agent(backend, Mock())
        tool_names = {tool.name for tool in create.call_args.kwargs["tools"]}
        self.assertTrue({"validate_report", "arxiv_search", "hf_search_papers", "hf_daily_papers", "web_search", "web_fetch"}.issubset(tool_names))


if __name__ == "__main__":
    unittest.main()
