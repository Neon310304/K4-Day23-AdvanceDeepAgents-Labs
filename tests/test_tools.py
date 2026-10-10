"""Exercise retry timing, payloads, provenance and credential-safe failures offline."""
import json
import unittest
from unittest.mock import Mock, patch

import httpx

import tools


def response(body=None, status=200, headers=None, text=None):
    kwargs = {"text": text} if text is not None else {"json": body}
    return httpx.Response(status, headers=headers,
                          request=httpx.Request("GET", "https://example.org"), **kwargs)


def mcp(text="Evidence", metadata=None, is_error=False):
    return response({"jsonrpc": "2.0", "id": 1, "result": {
        "content": [{"type": "text", "text": text}],
        "_meta": metadata or {}, "isError": is_error,
    }})


class RetryTests(unittest.TestCase):
    def test_backoff_jitter_cap_and_no_sleep_after_last_attempt(self):
        fn = Mock(side_effect=tools.RetryableError("temporary"))
        with patch.object(tools.time, "sleep") as sleep, patch.object(tools.random, "uniform", return_value=.25):
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(fn, attempts=4, base=2, cap=5)
        self.assertEqual(fn.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2.25, 4.25, 5])

    def test_retry_after_wins_and_is_capped(self):
        fn = Mock(side_effect=[tools.RetryableError("wait", 100), "ok"])
        with patch.object(tools.time, "sleep") as sleep, patch.object(tools.random, "uniform") as jitter:
            self.assertEqual(tools.with_retry(fn, cap=30), "ok")
        sleep.assert_called_once_with(30)
        jitter.assert_not_called()

    def test_programming_errors_are_not_retried(self):
        fn = Mock(side_effect=ValueError("bug"))
        with patch.object(tools.time, "sleep") as sleep:
            with self.assertRaises(ValueError):
                tools.with_retry(fn)
        self.assertEqual(fn.call_count, 1)
        sleep.assert_not_called()

    def test_final_attempt_can_succeed(self):
        fn = Mock(side_effect=[tools.RetryableError("wait"), 42])
        with patch.object(tools.time, "sleep"):
            self.assertEqual(tools.with_retry(fn, attempts=2), 42)

    def test_invalid_attempts_fail_before_call(self):
        fn = Mock()
        for attempts in (0, -1, True):
            with self.assertRaises(ValueError):
                tools.with_retry(fn, attempts=attempts)
        fn.assert_not_called()

    def test_http_statuses_and_transport_errors(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status), patch.object(tools.httpx, "request", return_value=response(status=status, headers={"Retry-After": "3"})):
                with self.assertRaises(tools.RetryableError) as context:
                    tools._request("GET", "https://example.org")
                self.assertEqual(context.exception.retry_after, 3)
        with patch.object(tools.httpx, "request", side_effect=httpx.ConnectTimeout("timeout")):
            with self.assertRaises(tools.RetryableError):
                tools._request("GET", "https://example.org")
        with patch.object(tools.httpx, "request", return_value=response(status=401)):
            with self.assertRaises(httpx.HTTPStatusError):
                tools._request("GET", "https://example.org")


class SourceToolTests(unittest.TestCase):
    def test_arxiv_sanitizes_query_and_canonicalizes_versioned_id(self):
        xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>http://arxiv.org/abs/1803.10122v4</id>
          <title>World\n Models</title><published>2018-03-27T00:00:00Z</published>
          <summary>Multiple   spaces\nand lines.</summary></entry></feed>"""
        with patch.object(tools, "_request", return_value=response(text=xml)) as request:
            records = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND all:model OR'}))
        self.assertEqual(records[0]["id"], "1803.10122")
        self.assertEqual(records[0]["url"], "https://arxiv.org/abs/1803.10122")
        self.assertEqual(records[0]["title"], "World Models")
        self.assertEqual(request.call_args.kwargs["params"]["search_query"], "all:world AND all:model")
        self.assertTrue(request.call_args.args[1].startswith("https://"))

    def test_empty_queries_do_not_call_network(self):
        with patch.object(tools, "_request") as request:
            self.assertEqual(tools.arxiv_search.invoke({"query": '" : AND OR'}), "NO RESULTS")
            self.assertEqual(tools.hf_search_papers.invoke({"query": " "}), "NO RESULTS")
            self.assertEqual(tools.web_search.invoke({"query": " "}), "NO RESULTS")
        request.assert_not_called()

    def test_arxiv_throttle_applies_between_attempts(self):
        clock = [100.]
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(tools, "_ARXIV_LAST_CALL", None), patch.object(tools.time, "monotonic", side_effect=lambda: clock[0]), patch.object(tools.time, "sleep", side_effect=sleep) as waiting:
            tools._arxiv_wait()
            tools._arxiv_wait()
            tools._arxiv_wait()
        self.assertEqual([c.args[0] for c in waiting.call_args_list], [3., 3.])

    def test_hf_mapping_search_preference_filter_and_sort(self):
        items = [{"paper": {"id": "1", "title": "World model", "summary": "Original",
                            "ai_summary": "AI overview", "upvotes": 2, "githubStars": 7,
                            "githubRepo": "https://github.com/example/paper",
                            "publishedAt": "2025-01-02T00:00:00Z"}},
                 {"paper": {"id": "2", "title": "World simulation", "summary": "Other", "upvotes": 9}},
                 {"paper": {"title": "Missing id"}},
                 {"paper": {"id": "3", "title": "Unrelated", "summary": "None", "upvotes": 20}}]
        with patch.object(tools, "_request", return_value=response(items)) as request:
            daily = json.loads(tools.hf_daily_papers.invoke({"keyword": "WORLD", "date": "2025-01-02"}))
            search = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
        self.assertEqual([entry["id"] for entry in daily], ["2", "1"])
        self.assertEqual(daily[1]["published"], "2025-01-02")
        self.assertEqual(search[0]["summary"], "AI overview")
        self.assertEqual(search[0]["stars"], 7)
        self.assertEqual(search[0]["url"], "https://huggingface.co/papers/1")
        self.assertEqual(request.call_count, 2)

    def test_hf_empty_and_invalid_dates(self):
        with patch.object(tools, "_request", return_value=response([])):
            self.assertEqual(tools.hf_daily_papers.invoke({}), "NO RESULTS")
            self.assertEqual(tools.hf_search_papers.invoke({"query": "none"}), "NO RESULTS")
        with patch.object(tools, "_request") as request:
            self.assertTrue(tools.hf_daily_papers.invoke({"date": "2025-02-30"}).startswith("ERROR:"))
        request.assert_not_called()

    def test_exa_json_sse_multiline_and_nontext_content(self):
        data = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Evidence"}, {"type": "image", "data": "x"}]}}
        for reply in (response(data), response(text="event: message\ndata: " + json.dumps(data) + "\n\n"),
                      response(text='data: {"jsonrpc":"2.0","id":1,\ndata: "result":{"content":[]}}\n\n')):
            with self.subTest(reply=reply.text):
                self.assertIn("result", tools._mcp_message(reply))
        with patch.object(tools, "_request", return_value=response(data)):
            self.assertEqual(tools.web_search.invoke({"query": "survey"}), "Evidence")

    def test_exa_http_200_rate_limit_is_retried(self):
        replies = [mcp("Free tier", {"exa": {"isRateLimited": True, "retryAfter": 2}}), mcp()]
        with patch.object(tools, "_request", side_effect=replies) as request, patch.object(tools.time, "sleep") as sleep:
            self.assertEqual(tools.web_search.invoke({"query": "survey"}), "Evidence")
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_exa_text_rate_limit_is_not_research_evidence(self):
        with patch.object(tools, "_request", side_effect=[mcp("You have hit the Exa MCP rate limit."), mcp("real evidence")]), patch.object(tools.time, "sleep"):
            self.assertEqual(tools.web_search.invoke({"query": "survey"}), "real evidence")

    def test_exa_payloads_include_objective_array_and_full_fetch_limit(self):
        with patch.object(tools, "_request", return_value=mcp("x" * 13000)) as request:
            tools.web_search.invoke({"query": "survey", "num_results": 99})
            args = request.call_args.kwargs["json"]["params"]["arguments"]
            self.assertTrue(args["objective"])
            self.assertEqual(args["numResults"], 10)
            fetched = tools.web_fetch.invoke({"url": "https://example.org"})
            args = request.call_args.kwargs["json"]["params"]["arguments"]
            self.assertEqual(args, {"urls": ["https://example.org"], "maxCharacters": 12000})
            self.assertEqual(len(fetched), 12000)

    def test_jsonrpc_error_and_secret_redaction(self):
        secret = "private credential/with+symbols"
        with patch.dict(tools.os.environ, {"EXA_API_KEY": secret}):
            with patch.object(tools, "_request", return_value=response({"error": {"message": secret, "code": -1}})):
                result = tools.web_search.invoke({"query": "survey"})
            self.assertTrue(result.startswith("ERROR:"))
            self.assertNotIn(secret, result)
            self.assertIn("[REDACTED]", result)
            for variant in (tools.quote(secret, safe=""), tools.quote_plus(secret)):
                self.assertNotIn(variant, tools.redact_secrets("URL?exaApiKey=" + variant))

    def test_every_tool_converts_failures_to_error_strings(self):
        cases = [(tools.arxiv_search, {"query": "world"}),
                 (tools.hf_daily_papers, {}), (tools.hf_search_papers, {"query": "world"}),
                 (tools.web_search, {"query": "world"}), (tools.web_fetch, {"url": "https://example.org"})]
        for fn, args in cases:
            with self.subTest(name=fn.name), patch.object(tools, "_request", side_effect=ValueError("bad payload")):
                self.assertTrue(fn.invoke(args).startswith("ERROR:"))

    def test_invalid_fetch_url_does_not_call_network(self):
        with patch.object(tools, "_request") as request:
            self.assertTrue(tools.web_fetch.invoke({"url": "file:///etc/passwd"}).startswith("ERROR:"))
        request.assert_not_called()

    def test_exa_empty_results_are_not_evidence(self):
        for text in ("", "No results found."):
            with self.subTest(text=text), patch.object(tools, "_request", return_value=mcp(text)):
                self.assertEqual(tools.web_search.invoke({"query": "absent"}), "NO RESULTS")

    def test_direct_primary_fetch_retries_strips_scripts_and_redacts(self):
        secret = "private-test-host-credential"
        page = response(text='<html><title>Paper</title><script>hidden instruction</script>'
                        '<style>hidden style</style><p>Actual &amp; evidence ' + secret + '</p></html>',
                        headers={"content-type": "text/html; charset=utf-8"})
        with patch.dict(tools.os.environ, {"LAB_DIRECT_PRIMARY_FETCH": "1", "LAB_API_KEY": secret}), \
                patch.object(tools, "_request", side_effect=[tools.RetryableError("timeout"), page]) as request, \
                patch.object(tools.time, "sleep"):
            result = tools.web_fetch.invoke({"url": "https://arxiv.org/abs/2006.11239"})
        self.assertIn("Retrieved directly on host", result)
        self.assertIn("Actual & evidence", result)
        self.assertNotIn("hidden instruction", result)
        self.assertNotIn("hidden style", result)
        self.assertNotIn(secret, result)
        self.assertIn("[REDACTED]", result)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.args, ("GET", "https://arxiv.org/abs/2006.11239"))
        self.assertEqual(request.call_args.kwargs, {})

    def test_direct_fetch_is_opt_in_and_other_urls_still_use_exa(self):
        for enabled, url in (("0", "https://arxiv.org/abs/2006.11239"),
                             ("1", "https://example.org/paper"),
                             ("1", "https://arxiv.org.evil.example/abs/2006.11239")):
            with self.subTest(enabled=enabled, url=url), \
                    patch.dict(tools.os.environ, {"LAB_DIRECT_PRIMARY_FETCH": enabled}), \
                    patch.object(tools, "_request", return_value=mcp()) as request:
                self.assertEqual(tools.web_fetch.invoke({"url": url}), "Evidence")
                self.assertEqual(request.call_args.args[0], "POST")
        with patch.dict(tools.os.environ, {"LAB_DIRECT_PRIMARY_FETCH": "1"}), \
                patch.object(tools, "_request", side_effect=[
                    response(text="PDF", headers={"content-type": "application/pdf"}), mcp()
                ]) as request:
            self.assertEqual(tools.web_fetch.invoke({"url": "https://aclanthology.org/2020.acl-main.195.pdf"}), "Evidence")
            self.assertEqual([call.args[0] for call in request.call_args_list], ["GET", "POST"])

    def test_direct_primary_empty_and_failure_keep_tool_contract(self):
        with patch.dict(tools.os.environ, {"LAB_DIRECT_PRIMARY_FETCH": "1"}), \
                patch.object(tools, "_request", return_value=response(text="<script>only script</script>",
                             headers={"content-type": "text/html"})):
            self.assertEqual(tools.web_fetch.invoke({"url": "https://huggingface.co/papers/2006.11239"}), "NO RESULTS")
        with patch.dict(tools.os.environ, {"LAB_DIRECT_PRIMARY_FETCH": "1", "LAB_API_KEY": "host-private-secret"}), \
                patch.object(tools, "_request", side_effect=ValueError("host-private-secret")):
            result = tools.web_fetch.invoke({"url": "https://arxiv.org/abs/2006.11239"})
            self.assertTrue(result.startswith("ERROR:"))
            self.assertNotIn("host-private-secret", result)


if __name__ == "__main__":
    unittest.main()
