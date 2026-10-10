"""Citation validation against malformed reports, not just happy-path examples."""
import unittest

from check_citations import check
from finalize_citations import finalize


def sources(count=3):
    return [{"n": n, "url": f"https://example.org/paper/{n}", "title": f"Paper {n}",
             "id": str(n), "date": "2025-01-01", "source": "web"}
            for n in range(1, count + 1)]


def report(body="Evidence [1][2][3].", entries=None):
    entries = sources() if entries is None else entries
    return "# Survey\n\n" + body + "\n\n## References\n" + "\n".join(
        f"[{s['n']}] {s['title']}. web. {s['url']} (2025-01-01)" for s in entries
    ) + "\n"


class CitationTests(unittest.TestCase):
    def test_valid_separate_grouped_and_range_citations(self):
        for body in ("Evidence [1][2][3].", "Evidence [1, 2, 3].", "Evidence [1-3].",
                     "Evidence [1–2][3]."):
            with self.subTest(body=body):
                self.assertEqual(check(report(body), sources()), [])

    def test_reject_each_consistency_failure(self):
        valid = report()
        cases = {
            "missing heading": valid.replace("## References", "## Bibliography"),
            "unknown citation": valid.replace("Evidence", "Unknown [99]. Evidence"),
            "unused source": valid.replace("Evidence [1][2][3]", "Evidence [1][2]"),
            "missing reference": valid[:valid.index("\n[3]")],
            "duplicate reference": valid + "[1] Paper. https://example.org/paper/1\n",
            "unknown reference": valid + "[99] Paper. https://example.org/99\n",
            "wrong URL": valid.replace("web. https://example.org/paper/1", "web. https://wrong.org/1"),
            "bundled URLs": valid.replace("Paper 1.", "Paper 1. https://another.org/source"),
            "no reference URL": valid.replace("https://example.org/paper/1", "no-url"),
            "extra references section": valid + "\n## References\n",
            "section after references": valid + "\n## Appendix\n",
            "unbounded range": valid.replace("Evidence [1][2][3]", "Evidence [1-999999]"),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                self.assertTrue(check(text, sources()), name)

    def test_malformed_source_data_never_crashes(self):
        for invalid in (None, {}, [], [None], ["text"], [{"n": True, "url": "x"}],
                        [{"n": [], "url": []}], [{"n": 1, "url": "https://"}]):
            with self.subTest(sources=invalid):
                self.assertTrue(check(report(), invalid))

    def test_duplicate_source_numbers_and_urls(self):
        for key in ("n", "url"):
            entries = sources()
            entries[1][key] = entries[0][key]
            self.assertTrue(check(report(), entries))

    def test_code_and_markdown_links_do_not_supply_citations(self):
        entries = sources(1)
        ignored = (
            "`[1]`", "```python\nx = '[1]'\n```",
            "~~~text\n[1]\n~~~", "    print('[1]')",
            "[1](https://example.org)", "[1][paper]", "![1](image.png)", r"\[1]",
        )
        for body in ignored:
            with self.subTest(body=body):
                problems = check(report(body, entries), entries)
                self.assertIn("source [1] never cited", problems)

    def test_reference_numbers_cannot_self_cite(self):
        self.assertIn("source [1] never cited", check(report("No evidence.", sources(1)), sources(1)))

    def test_url_with_balanced_parentheses_is_preserved(self):
        entries = sources(1)
        entries[0]["url"] = "https://example.org/Paper_(survey)"
        self.assertEqual(check(report("Claim [1]", entries), entries), [])

    def test_finalizer_roundtrip_and_idempotence(self):
        entries = sources(4)
        entries[3]["url"] = entries[0]["url"]
        body = "# Survey\nFinding [3, 4]. Comparison [1-2]."
        finalized, registry, problems = finalize(body, entries)
        self.assertEqual(problems, [])
        self.assertEqual(len(registry), 3)
        self.assertEqual(check(finalized, registry), [])
        self.assertEqual(finalize(finalized, registry), (finalized, registry, []))


if __name__ == "__main__":
    unittest.main()
