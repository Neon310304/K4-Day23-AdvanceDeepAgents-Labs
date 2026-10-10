"""Validate source coverage, collect all defects and preserve uncited candidates."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
import io

from finalize_citations import finalize
from validate_report import (
    coverage_details, inspect_files, output_problems, remember_sources,
)

from test_research import artifacts, messages, ArtifactBackend
import research


class CoverageTests(unittest.TestCase):
    def test_full_pipeline_restores_dropped_family_and_missing_heading_in_one_repair(self):
        good = artifacts()
        broken = good[research.REPORT_PATH].decode().replace("[2]", "[1]").replace(
            "## Trends and open problems\nUncertainty [1][1][3].\n\n", "")
        scripts = {
            research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
            research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes(),
            research.QUALITY_VALIDATOR_PATH: research.QUALITY_VALIDATOR_SOURCE.read_bytes(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes = root / "notes"
            notes.mkdir()
            (notes / "02-hf.md").write_text("Evidence fixture: Long-context world model, hf-search.", encoding="utf-8")
            class QualityBackend(ArtifactBackend):
                def execute(self, command):
                    prefix = f"python3 {research.QUALITY_VALIDATOR_PATH}"
                    if not command.startswith(prefix):
                        return super().execute(command)
                    self.commands.append(command)
                    sources = json.loads(self.files[research.SOURCES_PATH])
                    catalog = root / "candidates.json"
                    if command.endswith("--remember-only"):
                        remember_sources(sources, catalog)
                        return SimpleNamespace(exit_code=0, output="remembered")
                    report_path, sources_path = root / "draft.md", root / "sources.json"
                    report_path.write_bytes(self.files[research.REPORT_PATH])
                    sources_path.write_bytes(self.files[research.SOURCES_PATH])
                    result = inspect_files(report_path, sources_path, catalog, notes)
                    return SimpleNamespace(exit_code=0 if result["ok"] else 1, output=json.dumps(result))
            backend = QualityBackend({**good, **scripts, research.REPORT_PATH: broken.encode()})
            repair_agent = Mock()
            def repair(payload, *, config):
                prompt = payload["messages"][0]["content"]
                self.assertIn("missing report section: Trends and open problems", prompt)
                self.assertIn("at least three source families", prompt)
                self.assertIn("missing options", prompt)
                self.assertIn("candidate_sources", prompt)
                self.assertIn("02-hf.md", prompt)
                candidates = json.loads((root / "candidates.json").read_text(encoding="utf-8"))
                hf = next(entry for entry in candidates if entry["source"] == "hf-search")
                current_sources = json.loads(backend.files[research.SOURCES_PATH])
                n = max(entry["n"] for entry in current_sources) + 1
                current_sources.append({**hf, "n": n})
                backend.files[research.SOURCES_PATH] = json.dumps(current_sources).encode()
                report = backend.files[research.REPORT_PATH].decode()
                report = report.replace("## References", f"## Trends and open problems\nEvidence from recovered HF note [{n}].\n\n## References")
                backend.files[research.REPORT_PATH] = report.encode()
                return {"messages": []}
            repair_agent.invoke.side_effect = repair
            output = io.StringIO()
            with patch.object(research, "download", side_effect=backend.download), patch.object(research, "build_repair_agent", return_value=repair_agent), redirect_stdout(output):
                actual_save = research.save_outputs
                with patch.object(research, "save_outputs", side_effect=lambda *args, **kwargs: actual_save(*args, reports_dir=root / "reports", **kwargs)):
                    path = research._finish_outputs(backend, "world", messages(), research.time.monotonic(), "fixture", scripts, Mock())
            meta = json.loads(path.with_suffix(".meta.json").read_bytes())
            self.assertEqual(meta["source_families"], ["arxiv", "hf-search", "web"])
            self.assertEqual(meta["n_sources"], 3)
            self.assertEqual(path.read_bytes(), backend.files[research.REPORT_PATH])
            repair_agent.invoke.assert_called_once()
            self.assertIn("Repairing report (1/2)", output.getvalue())
            self.assertNotIn("Repairing report (2/2)", output.getvalue())

    def test_heading_and_coverage_errors_are_returned_together(self):
        data = artifacts()
        sources = json.loads(data[research.SOURCES_PATH])
        # Truthfully label the original arxiv result as a web retrieval in this fixture;
        # both non-HF URLs remain web and there are only two distinct families.
        sources[0]["source"] = "web"
        report = data[research.REPORT_PATH].decode().replace(
            "## Trends and open problems\nUncertainty [1][2][3].\n\n", "")
        problems = output_problems(report, sources, subagent_calls=3)
        self.assertIn("missing report section: Trends and open problems", problems)
        coverage = next(problem for problem in problems if "at least three source families" in problem)
        self.assertIn("hf-search", coverage)
        self.assertIn("web", coverage)
        self.assertIn("arxiv", coverage)
        self.assertIn("need 1 additional", coverage)

    def test_save_error_lists_both_defects_before_attempting_any_writes(self):
        from unittest.mock import patch, Mock
        data = artifacts()
        sources = json.loads(data[research.SOURCES_PATH])
        sources[0]["source"] = "web"
        data[research.SOURCES_PATH] = json.dumps(sources).encode()
        data[research.REPORT_PATH] = data[research.REPORT_PATH].replace(
            b"## Trends and open problems\nUncertainty [1][2][3].\n\n", b"")
        with tempfile.TemporaryDirectory() as directory, patch.object(research, "download", return_value=data):
            with self.assertRaises(research.InvalidOutputsError) as context:
                research.save_outputs(Mock(), "world", messages(), 0, "fixture", directory)
            problems = context.exception.problems
            self.assertTrue(any("missing report section" in p for p in problems))
            self.assertTrue(any("at least three source families" in p for p in problems))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_snapshot_retains_hf_candidate_after_finalizer_drops_uncited_source(self):
        data = artifacts()
        sources = json.loads(data[research.SOURCES_PATH])
        report = data[research.REPORT_PATH].decode().replace("[2]", "[1]")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "candidates.json"
            remember_sources(sources, catalog)
            final_report, remaining, errors = finalize(report, sources)
            self.assertEqual(errors, [])
            self.assertNotIn("hf-search", coverage_details(remaining)["present"])
            report_path, sources_path = root / "report.md", root / "sources.json"
            report_path.write_text(final_report, encoding="utf-8")
            sources_path.write_text(json.dumps(remaining), encoding="utf-8")
            notes = root / "notes"
            notes.mkdir()
            note = notes / "01-foundations.md"
            note.write_text("Evidence fixture", encoding="utf-8")
            result = inspect_files(report_path, sources_path, catalog, notes)
            self.assertFalse(result["ok"])
            self.assertEqual(result["coverage"]["present"], ["arxiv", "web"])
            self.assertIn("hf-search", result["candidate_coverage"]["present"])
            hf_candidate = next(s for s in result["candidate_sources"] if s["source"] == "hf-search")
            self.assertEqual(hf_candidate["url"], "https://huggingface.co/papers/2402.08268")
            self.assertNotIn("n", hf_candidate)
            self.assertEqual(result["notes"], [str(note)])
            # Repeated snapshots are stable and do not multiply stale registry numbers.
            result_again = inspect_files(report_path, sources_path, catalog, notes)
            self.assertEqual(result_again["candidate_sources"], result["candidate_sources"])

    def test_wrong_family_urls_are_rejected_despite_three_labels(self):
        data = artifacts()
        sources = json.loads(data[research.SOURCES_PATH])
        sources[0]["url"] = "https://example.org/not-arxiv"
        self.assertEqual(len(coverage_details(sources)["present"]), 3)
        self.assertTrue(any("arxiv URL" in p for p in output_problems(data[research.REPORT_PATH].decode(), sources)))

    def test_additional_registry_entries_do_not_count_as_used_evidence(self):
        data = artifacts()
        sources = json.loads(data[research.SOURCES_PATH])
        # HF metadata is present but its only citation is removed.
        report = data[research.REPORT_PATH].decode().replace("[2]", "[1]")
        problems = output_problems(report, sources)
        self.assertIn("source [2] never cited", problems)
        finalized, remaining, errors = finalize(report, sources)
        self.assertEqual(errors, [])
        self.assertTrue(any("at least three source families" in p for p in output_problems(finalized, remaining)))

    def test_inspector_cli_works_with_standard_library_and_exits_nonzero_on_defects(self):
        data = artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path, sources_path = root / "report.md", root / "sources.json"
            report_path.write_bytes(data[research.REPORT_PATH])
            sources_path.write_bytes(data[research.SOURCES_PATH])
            command = [
                sys.executable, str(research.QUALITY_VALIDATOR_SOURCE),
                "--report", str(report_path), "--sources", str(sources_path),
                "--candidates", str(root / "candidates.json"), "--notes", str(root / "notes"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertTrue(result["ok"])
            self.assertEqual(result["coverage"]["additional_families_needed"], 0)
            report_path.write_text("# Empty", encoding="utf-8")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertFalse(json.loads(failed.stdout)["ok"])

    def test_candidate_snapshot_ignores_malformed_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            result = remember_sources([None, {"url": [], "source": []}], Path(directory) / "catalog.json")
            self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
