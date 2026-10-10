"""Standard-library report checks and source-candidate snapshots for the sandbox."""
import argparse
import json
import re
import sys
from pathlib import Path

from check_citations import REPORT, SOURCES, check

SOURCE_FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
CANDIDATES = "/tmp/work/research/source_candidates.json"
NOTES = "/tmp/work/research/notes"


def source_problems(sources):
    from datetime import datetime

    problems = []
    if not isinstance(sources, list) or not sources:
        return ["sources.json must be a nonempty list"]
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source {index} must be an object")
            continue
        if any(key not in source for key in ("n", "id", "url", "title", "date", "source")):
            problems.append(f"source {index} is missing required metadata")
        family, paper_id, url = source.get("source"), source.get("id"), source.get("url")
        if not isinstance(family, str) or family not in SOURCE_FAMILIES:
            problems.append(f"source {index} has an unknown source family")
            family = ""
        if not isinstance(source.get("title"), str) or not source["title"].strip():
            problems.append(f"source {index} has no title")
        if not isinstance(paper_id, str) or not paper_id.strip():
            problems.append(f"source {index} has no identifier")
        date = source.get("date")
        if not isinstance(date, str):
            problems.append(f"source {index}: date must be YYYY-MM-DD or empty")
        elif date:
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                    raise ValueError("invalid date format")
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                problems.append(f"source {index}: invalid publication date")
        if family == "arxiv" and url != f"https://arxiv.org/abs/{paper_id}":
            problems.append(f"source {index}: arxiv URL does not match its id")
        if family in {"hf-daily", "hf-search"} and url != f"https://huggingface.co/papers/{paper_id}":
            problems.append(f"source {index}: Hugging Face URL does not match its id")
        if family in {"arxiv", "hf-daily", "hf-search"} and re.search(r"v\d+$", str(paper_id)):
            problems.append(f"source {index}: paper id contains a version suffix")
    return problems


def report_problems(report):
    report = report.replace("\r\n", "\n")
    matches = list(re.finditer(r"(?m)^##[ \t]+(.+?)[ \t]*$", report))
    headings = [match.group(1) for match in matches]
    required = {"TL;DR", "Background", "Trends and open problems", "References"}
    problems = [f"missing report section: {name}" for name in sorted(required - set(headings))]
    themes = [name for name in headings if name not in required]
    if not 3 <= len(themes) <= 6:
        problems.append("report must have 3 to 6 thematic sections")
    if not re.search(r"(?m)^#[ \t]+\S", report):
        problems.append("report has no title")
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        content = re.sub(r"<!--.*?-->", "", report[match.end():end], flags=re.DOTALL)
        if not re.search(r"\w", content):
            problems.append(f"empty report section: {match.group(1)}")
    return problems


def coverage_details(sources):
    counts = {}
    for entry in sources if isinstance(sources, list) else []:
        family = entry.get("source") if isinstance(entry, dict) else None
        if isinstance(family, str) and family in SOURCE_FAMILIES:
            counts[family] = counts.get(family, 0) + 1
    present = sorted(counts)
    return {
        "required_count": 3, "present": present, "counts": counts,
        "missing": sorted(SOURCE_FAMILIES - set(present)),
        "additional_families_needed": max(0, 3 - len(present)),
    }


def output_problems(report, sources, subagent_calls=None):
    """Collect every defect at once instead of masking coverage with heading errors."""
    problems = source_problems(sources) + check(report, sources) + report_problems(report)
    coverage = coverage_details(sources)
    if coverage["additional_families_needed"]:
        problems.append(
            "final report must cite at least three source families; "
            f"present={coverage['present']}, counts={coverage['counts']}, "
            f"missing options={coverage['missing']}, "
            f"need {coverage['additional_families_needed']} additional family/families. "
            "Reuse supported notes/candidates or retrieve a distinct source from a missing "
            "family, integrate its supported findings with citations, then finalize again. "
            "Adding an uncited registry entry or relabelling an existing source does not fix coverage."
        )
    if subagent_calls is not None and subagent_calls < 3:
        problems.append("lead must delegate at least three real subagent tasks")
    return problems


def remember_sources(sources, path):
    """Retain candidate provenance before the finalizer drops unused entries."""
    path = Path(path)
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = []
    candidates, seen = [], set()
    for entry in (previous if isinstance(previous, list) else []) + (sources if isinstance(sources, list) else []):
        if not isinstance(entry, dict):
            continue
        url, family = entry.get("url"), entry.get("source")
        if (not isinstance(url, str) or not url.startswith(("http://", "https://"))
                or not isinstance(family, str) or family not in SOURCE_FAMILIES):
            continue
        key = (url, family)
        if key not in seen:
            seen.add(key)
            # Old citation numbers become stale after finalization; never reuse them.
            candidates.append({key: value for key, value in entry.items() if key != "n"})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return candidates


def inspect_files(report_path=REPORT, sources_path=SOURCES, candidates_path=CANDIDATES, notes_dir=NOTES):
    problems = []
    try:
        report = Path(report_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        report = ""
        problems.append(f"cannot read report: {exc}")
    try:
        sources = json.loads(Path(sources_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sources = []
        problems.append(f"cannot read sources.json: {exc}")
    candidates = remember_sources(sources, candidates_path)
    problems.extend(output_problems(report, sources))
    notes_path = Path(notes_dir)
    # Search only the agreed notes directory, never the sandbox root.
    note_paths = [str(path) for path in sorted(notes_path.glob("*.md")) if path.is_file()]
    return {
        "ok": not problems, "problems": problems,
        "coverage": coverage_details(sources),
        "source_candidates_path": str(candidates_path),
        "candidate_coverage": coverage_details(candidates),
        "candidate_sources": candidates[:30],
        "notes": note_paths[:30],
        "note": "Candidate metadata is not claim evidence. Read its notes and cite supported findings. Never reuse stale citation numbers.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--sources", default=SOURCES)
    parser.add_argument("--candidates", default=CANDIDATES)
    parser.add_argument("--notes", default=NOTES)
    parser.add_argument("--remember-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.remember_only:
            try:
                sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                sources = []
            candidates = remember_sources(sources, args.candidates)
            print(json.dumps({"candidate_coverage": coverage_details(candidates),
                              "source_candidates_path": args.candidates}))
            return 0
        result = inspect_files(args.report, args.sources, args.candidates, args.notes)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ok"] else 1
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "problems": [f"cannot inspect artifacts: {exc}"]}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
