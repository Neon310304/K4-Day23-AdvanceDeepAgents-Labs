"""Validate citations INSIDE the sandbox using only the standard library.

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter
from urllib.parse import urlsplit

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*\r?$")
_CITATION = re.compile(r"(?<![\\!])\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\s*\()")
_REFERENCE = re.compile(r"^\[(\d+)\](?:\s+|$)")
_URL = re.compile(r"https?://[^\s<>\"`]+")


def _without_code(text):
    """Mask code so examples do not supply citations or headings."""
    text = re.sub(r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$", "", text)
    text = re.sub(r"(?m)^(?: {4}|\t).*", "", text)
    return re.sub(r"(`+).*?\1", "", text, flags=re.DOTALL)


def _numbers(group):
    numbers = set()
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            first, last = map(int, span.groups())
            if not 0 <= last - first <= 200:
                raise ValueError(f"invalid citation range [{part}]")
            numbers.update(range(first, last + 1))
        else:
            numbers.add(int(part))
    return numbers


def _urls(text):
    urls = []
    for match in _URL.finditer(text):
        url = match.group().rstrip(".,;")
        # Strip Markdown delimiters while retaining balanced parentheses in URLs.
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        urls.append(url)
    return urls


def check(report_text, sources):
    """Return structural/consistency problems; an empty list means OK.

    Grouped citations and bounded ranges are supported. Code examples and
    Markdown citation links do not count; each source needs one reference URL.
    This checks consistency, not whether a scientific claim is supported.
    """
    problems = []
    if not isinstance(report_text, str):
        return ["report must be text"]
    if not isinstance(sources, list):
        return ["sources.json must be a JSON list of objects"]
    if not sources:
        problems.append("no sources in sources.json")

    by_n, seen_urls = {}, set()
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} must be an object")
            continue
        n, url = source.get("n"), source.get("url")
        if type(n) is not int or n < 1:
            problems.append(f"source entry {index}: n must be a positive integer")
        elif n in by_n:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_n[n] = source
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source entry {index}: url must start with http:// or https://")
            continue
        try:
            valid_url = bool(urlsplit(url).hostname) and not re.search(r"\s", url)
        except ValueError:
            valid_url = False
        if not valid_url:
            problems.append(f"source entry {index}: invalid URL")
        if url in seen_urls:
            problems.append(f"duplicate source URL: {url}")
        seen_urls.add(url)

    visible = _without_code(report_text)
    headings = list(_HEADING.finditer(visible))
    if not headings:
        problems.append("missing ## References heading")
        body, references = visible, ""
    else:
        if len(headings) != 1:
            problems.append("report must contain exactly one ## References heading")
        body, references = visible[:headings[0].start()], visible[headings[0].end():]
        if re.search(r"(?m)^#{1,2}\s+", references):
            problems.append("## References must be the last section")

    # Reference-style Markdown links such as [1][paper] are links, not citations.
    body = re.sub(r"\[\d+\]\[(?!\d+(?:\]|\s*[,–-]))[^\]]*\]", "", body)
    cited = set()
    for match in _CITATION.finditer(body):
        try:
            cited.update(_numbers(match.group(1)))
        except ValueError as exc:
            problems.append(str(exc))
    for n in sorted(cited - by_n.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_n.keys() - cited):
        problems.append(f"source [{n}] never cited")

    reference_counts = Counter()
    for line in references.splitlines():
        if not line.strip():
            continue
        match = _REFERENCE.match(line)
        if not match:
            problems.append("reference line must start with [n]: " + line[:100])
            continue
        n = int(match.group(1))
        reference_counts[n] += 1
        if n not in by_n:
            problems.append(f"reference [{n}] missing from sources.json")
        urls = _urls(line)
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL (found {len(urls)})")
        elif n in by_n and urls[0] != by_n[n].get("url"):
            problems.append(f"reference [{n}] URL does not match sources.json")
    for n in sorted(by_n):
        if reference_counts[n] != 1:
            problems.append(f"source [{n}] needs exactly one reference line (found {reference_counts[n]})")
    for n, count in sorted(reference_counts.items()):
        if count > 1 and n not in by_n:
            problems.append(f"duplicate reference number [{n}]")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
