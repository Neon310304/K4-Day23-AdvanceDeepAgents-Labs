"""Package only public project source for upload into Google Colab."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = (
    "README.md", "GUIDE.md", "RUBRIC.md", "REPORT_TEMPLATE.md", "topics.md",
    "requirements.txt", ".env.example", ".gitignore", ".gitattributes", "model.py", "sandbox.py",
    "self_check.py", "finalize_citations.py", "check_citations.py", "tools.py",
    "agents.py", "research.py", "validate_report.py", "workflow.py", "Deep_Research_Colab.ipynb",
)


def package(destination):
    destination = Path(destination).resolve()
    files = [ROOT / name for name in SOURCE_FILES]
    files += sorted((ROOT / "tests").glob("test_*.py"))
    files += sorted((ROOT / "scripts").glob("*.py"))
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("Missing source files: " + ", ".join(missing))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "colab_bundle.zip")
    args = parser.parse_args()
    print(package(args.output))
