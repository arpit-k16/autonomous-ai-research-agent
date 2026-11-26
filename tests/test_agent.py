"""Offline tests — no API key or internet needed.   pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.crew import slugify  # noqa: E402
from agent.tools import CalculatorTool, DocReaderTool, WebSearchTool, safe_eval  # noqa: E402


def test_calculator_does_cagr():
    # 18 -> 52 over 2 years
    assert round(safe_eval("(52/18)**(1/2) - 1"), 4) == 0.6997
    assert "= 1500000" in CalculatorTool()._run("1,500,000")


@pytest.mark.parametrize("expr", ["__import__('os').system('dir')", "open('x')", "a + 1"])
def test_calculator_rejects_code(expr):
    assert "Could not evaluate" in CalculatorTool()._run(expr)


def test_doc_reader_lists_and_reads(tmp_path):
    (tmp_path / "notes.md").write_text("EV sales grew 40%", encoding="utf-8")
    tool = DocReaderTool(workspace=tmp_path)
    assert "notes.md" in tool._run("list")
    assert "EV sales grew 40%" in tool._run("notes.md")
    assert "not found" in tool._run("missing.md")


def test_doc_reader_blocks_path_traversal(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    tool = DocReaderTool(workspace=tmp_path / "ws")
    assert "Access denied" in tool._run("../secret.txt")


def test_bundled_workspace_doc_is_readable():
    out = DocReaderTool(workspace=ROOT / "workspace")._run("ev_market_internal_notes.md")
    assert "two-wheeler" in out


def test_web_search_formats_results(monkeypatch):
    fake = [{"title": "EV report", "href": "https://example.com/ev", "body": "Sales up 30%"}]
    monkeypatch.setattr(WebSearchTool, "_search", lambda self, q: fake)
    out = WebSearchTool()._run("ev trends")
    assert "[1] EV report" in out and "https://example.com/ev" in out


def test_web_search_handles_errors(monkeypatch):
    def boom(self, q):
        raise RuntimeError("network down")
    monkeypatch.setattr(WebSearchTool, "_search", boom)
    assert "Search failed" in WebSearchTool()._run("x")


def test_slugify():
    assert slugify("Find top EV market trends for 2027!") == "find-top-ev-market-trends-for-2027"


def test_crew_wiring(monkeypatch):
    """Build the crew with a dummy key — checks agents/tasks/tools are wired; makes no API call."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy")
    from agent.crew import build_crew

    crew = build_crew()
    assert [a.role for a in crew.agents] == ["Strategist", "Research Analyst", "Report Writer"]
    assert len(crew.tasks) == 3
    tool_names = {t.name for t in crew.agents[1].tools}
    assert tool_names == {"web_search", "read_webpage", "doc_reader", "calculator"}
    assert crew.tasks[2].context == [crew.tasks[0], crew.tasks[1]]


def test_tools_record_sources(monkeypatch, tmp_path):
    from agent import tools as T

    T.reset_seen_sources()
    monkeypatch.setattr(T.WebSearchTool, "_search",
                        lambda self, q: [{"title": "t", "href": "https://www.example.com/ev-report/", "body": "b"}])
    T.WebSearchTool()._run("ev")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    T.DocReaderTool(workspace=tmp_path)._run("notes.md")
    assert "example.com/ev-report" in T.SEEN_SOURCES and "notes.md" in T.SEEN_SOURCES


def test_verify_sources_flags_unseen_citations():
    from agent.crew import normalize_citations, verify_sources

    report = normalize_citations(
        "Sales doubled 【1】. Policy changed [2]. Internal plan [3].\n"
        "## Sources\n1. https://www.example.com/ev-report\n2. https://made-up.example.org/policy\n3. notes.md")
    assert "[1]" in report
    verified, unverified = verify_sources(report, {"example.com/ev-report", "notes.md"})
    assert verified == ["https://www.example.com/ev-report", "notes.md"]
    assert unverified == ["https://made-up.example.org/policy"]


def test_grounding_guardrail_retries_then_flags():
    from types import SimpleNamespace

    from agent import tools as T
    from agent.crew import make_grounding_guardrail

    T.reset_seen_sources()
    for u in ["https://a.example.com/x", "https://b.example.com/y", "https://c.example.com/z"]:
        T.record_source(u)
    events = []
    g = make_grounding_guardrail("research", 1, lambda k, d: events.append(d["result"]))

    good = SimpleNamespace(raw="- fact (https://a.example.com/x)\n- fact (https://b.example.com/y)\n- fact (https://c.example.com/z)")
    bad = SimpleNamespace(raw="- invented fact (https://made-up.example.org/q)")

    ok, out = g(bad)
    assert ok is False and "GROUNDING CHECK FAILED" in out      # attempt 1 -> retry with feedback
    ok, _ = g(bad)
    assert ok is True                                           # attempt 2 (last) -> accepted but flagged
    assert events == ["retry", "flagged"]  # GUARDRAIL_RETRIES = 1

    g2 = make_grounding_guardrail("report", 1, lambda k, d: events.append(d["result"]))
    ok, out = g2(good)
    assert ok is True and out == good.raw and events[-1] == "passed"


def test_refusal_gets_plain_feedback():
    from types import SimpleNamespace

    from agent import tools as T
    from agent.crew import make_grounding_guardrail

    T.reset_seen_sources()
    g = make_grounding_guardrail("research", 1, lambda k, d: None)
    ok, feedback = g(SimpleNamespace(raw="I'm sorry, but I can't comply with that."))
    assert ok is False and "no content" in feedback and "ignore" not in feedback.lower()


def test_force_final_answer_prompt_is_neutral():
    from agent.crew import SAFE_FINAL_ANSWER

    assert "ignore" not in SAFE_FINAL_ANSWER.lower()


def test_error_classification():
    from agent.crew import _is_retryable_rate_limit, _is_transient_model_glitch

    class RateLimitError(Exception):
        pass

    tpm = RateLimitError('{"error":{"message":"Rate limit reached ... tokens per minute (TPM)","code":"rate_limit_exceeded"}}')
    tpd = RateLimitError('{"error":{"message":"Rate limit reached ... tokens per day (TPD)","code":"rate_limit_exceeded"}}')
    parse = Exception('{"error":{"message":"Parsing failed. The model generated output...","code":"output_parse_failed"}}')
    assert _is_retryable_rate_limit(tpm) and not _is_retryable_rate_limit(tpd)
    assert not _is_retryable_rate_limit(parse)          # "generated" must not look like a rate limit
    assert _is_transient_model_glitch(parse) and not _is_transient_model_glitch(tpm)


def test_fetch_html_stops_at_size_cap(monkeypatch):
    import requests

    from agent.tools import fetch_html

    class FakeResp:
        encoding = "utf-8"
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            while True:                       # an endless, trickling server
                yield b"a" * chunk_size

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    html = fetch_html("https://example.com", deadline=5, max_bytes=50_000)
    assert 50_000 <= len(html) < 50_000 + 16_384


def test_timeouts_are_retried():
    from agent.crew import _is_transient_model_glitch

    class Timeout(Exception):
        pass

    assert _is_transient_model_glitch(Timeout("Request timed out after 90s"))


def test_daily_quota_is_non_retryable_for_crewai():
    from agent.crew import ServiceQuotaExceededException, _is_daily_quota

    class RateLimitError(Exception):
        pass

    tpd = RateLimitError('{"error":{"message":"Rate limit reached ... tokens per day (TPD) ...","code":"rate_limit_exceeded"}}')
    tpm = RateLimitError('{"error":{"message":"Rate limit reached ... tokens per minute (TPM) ...","code":"rate_limit_exceeded"}}')
    assert _is_daily_quota(tpd) and not _is_daily_quota(tpm)
    # CrewAI's retry wrapper never retries errors whose class name normalises to this code
    assert ServiceQuotaExceededException.__name__.replace("_", "").lower() == "servicequotaexceededexception"


def test_check_figures_flags_invented_numbers_and_broken_citations():
    from agent.crew import check_figures

    evidence = ("EV sales grew 40 per cent to 1,500 units. FAME outlay Rs 10,000 crore. "
                "Charging stations: 2,500 (2023). Growth of 35-40%.")
    report = ("# EV report\nSales grew 40% [1] to 1500 units [1]. Budget ₹10 000 cr [2]. "
              "Tata holds 25 % share [3]. Growth 35–40 % in 2027 [1]. There are 2,500 stations [9].\n"
              "## Sources\n1. https://a.example.com/x\n2. https://b.example.com/y\n3. https://c.example.com/z\n")
    r = check_figures(report, evidence)
    unsupported = [f for f, _ in r["unsupported"]]
    assert unsupported == ["25 %"]                 # the invented market share
    assert "40%" in r["supported"] and "1500 units" in r["supported"] and "2,500 stations" in r["supported"]
    assert any("35" in f and "40" in f for f in r["supported"])   # range: both ends checked
    assert r["dangling"] == [9]                    # [9] points to nothing
    assert not any("2027" in f for f in r["supported"] + unsupported)  # years ignored


def test_tools_record_evidence(monkeypatch, tmp_path):
    from agent import tools as T

    T.reset_seen_sources()
    monkeypatch.setattr(T.WebSearchTool, "_search",
                        lambda self, q: [{"title": "EV", "href": "https://x.example.com/a", "body": "sales up 37%"}])
    T.WebSearchTool()._run("ev")
    T.CalculatorTool()._run("100*1.37")
    ev = T.evidence_text()
    assert "sales up 37%" in ev and "137" in ev


def test_read_webpage_parses_pdf(monkeypatch):
    from agent import tools as T

    def fake_fetch(url, *a, **k):
        from io import BytesIO

        from reportlab.pdfgen import canvas  # available in most dev setups; skip otherwise
        buf = BytesIO(); c = canvas.Canvas(buf); c.drawString(72, 720, "EV stations reached 12,146 in 2025"); c.save()
        return buf.getvalue(), "application/pdf"

    import pytest
    pytest.importorskip("reportlab")
    monkeypatch.setattr(T, "fetch_bytes", fake_fetch)
    out = T.ReadWebpageTool()._run("https://example.gov.in/report.pdf")
    assert "12,146" in out


def test_figures_need_matching_units():
    from agent.crew import check_figures

    evidence = "There were 1 charging hub per 10 EVs. The FAME-II outlay was Rs 11,500 crore. Share rose 12-18%."
    report = ("A national target of 1 million stations [1]. FAME-II: ₹11,500 cr [1]. "
              "Share 12‑18 % [1].\n## Sources\n1. https://a.example.com/x\n")
    r = check_figures(report, evidence)
    assert [f for f, _ in r["unsupported"]] == ["1 million"]   # a stray "1" doesn't support "1 million"
    assert any("11,500" in f for f in r["supported"])          # Rs/₹ and cr/crore normalised
    assert any("12" in f and "18" in f for f in r["supported"])  # range with unit on both sides


def test_pdf_name_inside_url_is_not_a_separate_source():
    from agent.crew import verify_sources

    report = "Sources:\n1. https://theicct.org/wp-content/uploads/2024/06/ID-169-FAME-opps_report_final.pdf\n"
    verified, unverified = verify_sources(report, {"theicct.org/wp-content/uploads/2024/06/id-169-fame-opps_report_final.pdf"})
    assert unverified == [] and len(verified) == 1


def test_scrub_removes_only_claims_with_unverifiable_figures():
    from agent.crew import check_figures, scrub_unsupported_figures

    evidence = "The Indian EV market is projected to grow at over 40% CAGR till 2027. 29,277 public chargers."
    text = ("## Findings\n"
            "- The PLI scheme offers up to 15 % subsidy on capex. (source: https://a.example.com/x)\n"
            "- India had 29,277 public chargers, Karnataka leading with 8,500. Growth is fast. (https://b.example.com)\n"
            "- The market grows at over 40% CAGR till 2027 [1].\n"
            "- Rs. 500 crore was allotted. Policy support continues.\n\n"
            "## Sources\n1. https://a.example.com/x")
    clean, removed = scrub_unsupported_figures(text, evidence)
    assert "15 %" not in clean and "PLI" not in clean                     # whole line dropped (source tag only left)
    assert "8,500" not in clean and "Growth is fast." in clean           # only the bad sentence goes
    assert "40% CAGR" in clean and "Policy support continues." in clean and "500 crore" not in clean
    assert clean.rstrip().endswith("1. https://a.example.com/x")         # Sources section kept intact
    assert len(removed) == 3 and not check_figures(clean, evidence)["unsupported"]


def test_refusal_on_retry_never_replaces_the_real_report():
    from types import SimpleNamespace

    from agent import tools as T
    from agent.crew import REMOVED_FIGURE_CLAIMS, make_grounding_guardrail

    T.reset_seen_sources()
    REMOVED_FIGURE_CLAIMS.clear()
    urls = ["https://a.example.com/x", "https://b.example.com/y", "https://c.example.com/z"]
    for u in urls:
        T.record_source(u)
    T.record_evidence("Sales grew 40% in 2025.")
    events = []
    g = make_grounding_guardrail("report", 1, lambda k, d: events.append(d))
    real = SimpleNamespace(raw="Sales grew 40% [1]. Margins hit 73 % [2]. Demand rises [3]. Extra [4].\n\n## Sources\n"
                               + "\n".join(f"{i}. {u}" for i, u in enumerate(urls, 1)))
    ok, _ = g(real)                                    # dangling [4] -> retry
    assert ok is False and events[-1]["result"] == "retry"
    ok, out = g(SimpleNamespace(raw="I'm sorry, but I don't have the necessary source material."))
    assert ok is True and events[-1]["result"] == "flagged"
    assert "Sales grew 40%" in out and "sorry" not in out  # the first real attempt is kept...
    assert "73 %" not in out and REMOVED_FIGURE_CLAIMS == ["Margins hit 73 % [2]."]  # ...and fact-checked


def test_scrub_drops_table_rows_whole_and_empty_tables():
    from agent.crew import scrub_unsupported_figures

    evidence = "Mahindra holds 20% share."
    text = ("### Segments\n| Segment | CAGR | Driver |\n|---|---|---|\n| 2W | 70.5 % | Cheap. Popular. | [4] |\n\nNext.\n\n"
            "### OEMs\n| OEM | Share |\n|---|---|\n| Tata | ~25 % |\n| Mahindra | ~20 % |")
    clean, removed = scrub_unsupported_figures(text, evidence)
    assert "| Segment |" not in clean and "[4] |" not in clean      # emptied table removed, no row fragments
    assert "| Mahindra | ~20 % |" in clean and "Tata" not in clean and "| OEM | Share |" in clean
    assert len(removed) == 2


def test_run_result_fields_are_not_shifted():
    import dataclasses

    from agent.crew import RunResult

    names = [f.name for f in dataclasses.fields(RunResult)]
    assert {"figures_removed", "dangling_citations"} <= set(names)


def test_figure_must_come_from_the_source_it_cites():
    from agent.crew import Evidence, check_figures

    ev = Evidence("Mahindra holds 20% share. Tata: 1.5 lakh incentive.",
                  {"a.example.com/x": "Mahindra holds 20% share.", "b.example.com/y": "Tata sells cars.",
                   "notes.md": "Ola grew 35%."})
    report = ("- Mahindra has ~20% share [1].\n- Tata has ~20% share [2].\n- Ola grew 35% (notes.md).\n\n"
              "## Sources\n1. https://a.example.com/x\n2. https://www.b.example.com/y/")
    res = check_figures(report, ev)
    bad = [c for _, c in res["unsupported"]]
    assert len(bad) == 1 and "Tata" in bad[0]          # the 20% exists, but not in the source Tata cites
    assert "35%" in res["supported"]


def test_scrub_tidies_leftovers():
    from agent.crew import scrub_unsupported_figures

    text = ("## Policy\n- Subsidy is ₹1.5 L for scooters. [1]\n- Demand is rising. [1]\n\n"
            "## Segments\n- 2W grow 70.5 % a year [1].\n\n## Next steps\n1. Track policy.\n"
            "2. Hire 300 staff.\n3. Watch prices.\n\n## Sources\n1. https://a.example.com/x")
    clean, removed = scrub_unsupported_figures(text, "nothing numeric here")
    assert "- [1]" not in clean and "Demand is rising. [1]" in clean   # no citation-only stubs left
    assert "## Segments" not in clean                                   # emptied section removed
    assert "1. Track policy.\n2. Watch prices." in clean                # list renumbered
    assert len(removed) == 3


def test_sources_without_url_are_flagged_and_their_claims_removed():
    from agent.crew import Evidence, check_figures, scrub_unsupported_figures, verify_sources

    report = ("## Findings\n- GDP grows 7 % [1].\n- Local fabs open soon. | [3] |\n\n## Segments\n"
              "*Sources: internal market research – see [2].*\n\n## Sources\n1. https://a.example.com/x\n"
              "2. Internal market research reports – proprietary data.\n3. https://c.example.com/z")
    ev = Evidence("GDP grows 7 %.", {"a.example.com/x": "GDP grows 7 %.", "c.example.com/z": "fabs"})
    _, unverified = verify_sources(report, {"a.example.com/x", "c.example.com/z"})
    assert any("proprietary" in u for u in unverified)
    assert check_figures(report, ev)["dangling"] == [2]
    clean, removed = scrub_unsupported_figures(report, ev)
    assert "proprietary" not in clean and "internal market research" not in clean and "## Segments" not in clean
    assert "- Local fabs open soon. [2]" in clean and "2. https://c.example.com/z" in clean   # renumbered
    assert "| [" not in clean
