"""Verify the upload bundle and executable notebook cells without Colab or API keys."""
import io
import json
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.package_colab import ROOT, SOURCE_FILES, package


class ColabTests(unittest.TestCase):
    def test_bundle_is_complete_and_excludes_secrets_and_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = package(Path(directory) / "bundle.zip")
            with zipfile.ZipFile(destination) as archive:
                names = set(archive.namelist())
                self.assertTrue(set(SOURCE_FILES).issubset(names))
                self.assertTrue(any(name.startswith("tests/") for name in names))
                self.assertNotIn(".env", names)
                self.assertFalse(any(".venv/" in name or name.startswith("reports/") for name in names))
                for name in SOURCE_FILES:
                    self.assertEqual(archive.read(name), (ROOT / name).read_bytes(), name)

    def test_notebook_code_cells_compile(self):
        notebook = json.loads((ROOT / "Deep_Research_Colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"colab_cell_{number}", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_github_install_handles_existing_zip_project_and_preserves_reports(self):
        notebook = json.loads((ROOT / "Deep_Research_Colab.ipynb").read_text(encoding="utf-8"))
        installer = "".join(notebook["cells"][1]["source"])
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "existing"
            (project / "reports").mkdir(parents=True)
            (project / "reports" / "survey.md").write_bytes(b"verified report")
            (project / ".env").write_bytes(b"private local configuration")
            installer = installer.replace('PROJECT = Path("/content/deep-research-agent")',
                                          f"PROJECT = Path({str(project)!r})")
            def run(command, **kwargs):
                if command[:2] == ["git", "clone"]:
                    checkout = Path(command[-1])
                    (checkout / "reports").mkdir(parents=True)
                    (checkout / ".git").mkdir()
                    (checkout / ".env").write_bytes(b"must not copy")
                    (checkout / "reports" / "survey.md").write_bytes(b"older report")
                    (checkout / "requirements.txt").write_bytes(b"")
                    (checkout / "README.md").write_bytes(b"public source")
            with patch("subprocess.run", side_effect=run) as command, patch("sys.stdout", io.StringIO()):
                exec(installer, {})
            self.assertEqual(command.call_count, 2)
            self.assertEqual((project / "reports" / "survey.md").read_bytes(), b"verified report")
            self.assertEqual((project / ".env").read_bytes(), b"private local configuration")
            self.assertEqual((project / "README.md").read_bytes(), b"public source")
            self.assertFalse((project / ".git").exists())

    def test_secret_configuration_runs_without_echoing_credentials(self):
        notebook = json.loads((ROOT / "Deep_Research_Colab.ipynb").read_text(encoding="utf-8"))
        config = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and "def configured" in "".join(cell["source"]))
        class SecretNotFoundError(Exception):
            pass
        class NotebookAccessError(Exception):
            pass
        secrets = {"LAB_MODEL": "test-gpt-model", "LAB_BASE_URL": "https://router.example/v1",
                   "LAB_API_KEY": "router-credential", "DAYTONA_API_KEY": "sandbox-credential"}
        def get(name):
            if name not in secrets:
                raise SecretNotFoundError(name)
            return secrets[name]
        userdata = types.SimpleNamespace(get=get, SecretNotFoundError=SecretNotFoundError,
                                         NotebookAccessError=NotebookAccessError)
        colab = types.ModuleType("google.colab")
        colab.userdata = userdata
        for dedicated, override in ((False, ""), (True, ""),
                                    (False, "ag/gemini-3.1-pro-high"),
                                    (True, "ag/gemini-3.1-pro-high")):
            if dedicated:
                secrets["DAYTONA_LAB_API_KEY"] = "dedicated-sandbox-credential"
            namespace, output = {}, io.StringIO()
            stale = {"OPENAI_ENDPOINT": "https://old.example/v1", "OPENAI_KEY": "old-credential",
                     "LAB_BASE_URL": "https://old.example/v1", "LAB_API_KEY": "old-credential"}
            with self.subTest(dedicated=dedicated), patch.dict("sys.modules", {"google.colab": colab}), patch.dict("os.environ", stale, clear=True), patch("sys.stdout", output), patch("builtins.input", side_effect=AssertionError("unexpected prompt")), patch("getpass.getpass", side_effect=AssertionError("unexpected prompt")):
                selected_config = config.replace(
                    'MODEL_OVERRIDE = "ag/gemini-3.1-pro-high"',
                    f'MODEL_OVERRIDE = {override!r}')
                exec(selected_config, namespace)
                environment = namespace["os"].environ
                self.assertEqual(environment["SANDBOX"], "daytona")
                self.assertNotIn("OPENAI_ENDPOINT", environment)
                self.assertNotIn("OPENAI_KEY", environment)
                self.assertEqual(environment["LAB_MODEL"], override or secrets["LAB_MODEL"])
                self.assertEqual(environment["LAB_BASE_URL"], secrets["LAB_BASE_URL"])
                self.assertEqual(environment["LAB_API_KEY"], secrets["LAB_API_KEY"])
                expected = secrets.get("DAYTONA_LAB_API_KEY", secrets["DAYTONA_API_KEY"])
                self.assertEqual(environment["DAYTONA_API_KEY"], expected)
                from model import make_model
                model = make_model()
                self.assertEqual(model.model_name, override or secrets["LAB_MODEL"])
                self.assertEqual(model.openai_api_base, secrets["LAB_BASE_URL"])
                self.assertEqual(model.openai_api_key.get_secret_value(), secrets["LAB_API_KEY"])
            for name, value in secrets.items():
                if "KEY" in name:
                    self.assertNotIn(value, output.getvalue())
        del secrets["LAB_API_KEY"]
        with patch.dict("sys.modules", {"google.colab": colab}), patch.dict("os.environ", {}, clear=True), patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
            with self.assertRaisesRegex(RuntimeError, "Missing.*LAB_API_KEY"):
                exec(config, {})


if __name__ == "__main__":
    unittest.main()
