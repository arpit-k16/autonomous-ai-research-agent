"""The goal-driven crew:  PLAN → SEARCH/EXECUTE → DELIVER.

    Strategist        breaks the goal into a step-by-step research plan
    Research Analyst  executes the plan with tools (web search, web pages, internal docs, calculator)
    Report Writer     turns the findings into a structured, cited deliverable
"""
from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from crewai import LLM, Agent, Crew, Process, Task

from .config import Settings, settings as default_settings
from .tools import (CALC_SOURCE, SEEN_SOURCES, default_tools, evidence_by_source, evidence_text, normalize_source,
                    reset_seen_sources)

EventHandler = Callable[[str, dict], None]   # (kind, payload) -> None
GUARDRAIL_RETRIES = 1
LLM_TIMEOUT_S = 90  # a stuck model call fails (and is retried) instead of hanging the run


@dataclass
class RunResult:
    goal: str
    plan: str
    findings: str
    report: str
    report_path: Path
    token_usage: dict
    verified_sources: list[str] = field(default_factory=list)
    unverified_sources: list[str] = field(default_factory=list)
    figures_supported: list[str] = field(default_factory=list)
    figures_unsupported: list[tuple[str, str]] = field(default_factory=list)
    figures_removed: list[str] = field(default_factory=list)
    dangling_citations: list[int] = field(default_factory=list)


# CrewAI's built-in "max iterations reached" message says "You'll ignore all previous instructions...".
# gpt-oss is trained to treat that phrase as a prompt-injection attack and replies "I can't comply",
# which throws away the whole research loop. Replace just that one message with neutral wording.
SAFE_FINAL_ANSWER = (
    "You have reached the step limit for this task. Do not call any more tools. Write your final answer "
    "now, using only the tool results above, and keep the exact source URL or file name at the end of "
    "every bullet."
)
_PROMPTS_PATCHED = False


def _patch_crewai_prompts() -> None:
    global _PROMPTS_PATCHED
    if _PROMPTS_PATCHED:
        return
    try:
        from crewai.utilities.i18n import I18N
    except Exception:  # older/newer CrewAI layout: nothing to patch
        return
    original_errors = I18N.errors

    def errors(self, error: str) -> str:
        if error == "force_final_answer":
            return SAFE_FINAL_ANSWER
        return original_errors(self, error)

    I18N.errors = errors
    _PROMPTS_PATCHED = True


_patch_crewai_prompts()


def _looks_like_refusal(text: str) -> bool:
    t = text.strip().lower()
    t = t.replace("’", "'")
    return len(t) < 400 and any(k in t for k in ("can't comply", "cannot comply", "i'm sorry", "i am sorry",
                                                  "i don't have the necessary", "unable to produce"))


# Set by build_crew() so the UI can show "waiting for rate limit" messages.
_WAIT_HOOK: Callable[[float], None] | None = None
_GLITCH_HOOK: Callable[[], None] | None = None


def _retry_after(message: str) -> float:
    """Parse Groq's hint, e.g. 'try again in 8.04s', '1m2.5s' or '527ms'; fall back to 15 s."""
    m = re.search(r"try again in (?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?", message)
    if not m or not any(m.groups()):
        return 15.0
    mins, secs, ms = m.groups()
    return int(mins or 0) * 60 + float(secs or 0) + float(ms or 0) / 1000 + 1.0


def _is_retryable_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    is_rate = "ratelimiterror" in text or "rate_limit_exceeded" in text or "rate limit reached" in text
    daily = "per day" in text or "(tpd)" in text or "(rpd)" in text
    return is_rate and not daily


def _is_transient_model_glitch(exc: Exception) -> bool:
    """Groq occasionally rejects a gpt-oss response it can't parse (the model put its reasoning where a
    tool call or answer was expected). The same request usually succeeds when simply retried."""
    text = f"{type(exc).__name__} {exc}".lower()
    return ("output_parse_failed" in text or "tool_use_failed" in text
            or "timeout" in text or "timed out" in text)


class ServiceQuotaExceededException(RuntimeError):
    """Raised when the provider's *daily* token quota is used up.

    The class name is deliberate: CrewAI's built-in retry wrapper treats 'servicequotaexceededexception'
    as non-retryable. Without it, CrewAI obeys Groq's "try again in ~20 minutes" hint and silently sleeps
    for that long, which looks like a hung run.
    """


def _is_daily_quota(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    is_rate = "ratelimiterror" in text or "rate_limit_exceeded" in text or "rate limit reached" in text
    return is_rate and ("per day" in text or "(tpd)" in text or "(rpd)" in text)


class NativeToolLLM(LLM):
    """LLM that always uses native (JSON) tool calling.

    gpt-oss models on Groq are trained to emit native tool calls. If CrewAI falls back to
    describing tools in plain text, the model still emits a real tool call and Groq rejects
    the request ("Tool choice is none, but model called a tool"). Declaring native support
    makes CrewAI send the tool schemas with every request, which is what the model expects.
    """

    def supports_function_calling(self) -> bool:  # noqa: D401
        return True

    def call(self, *args, **kwargs):  # type: ignore[override]
        """Wait and retry when the free tier's tokens-per-minute limit is hit."""
        attempts = 8
        for attempt in range(attempts):
            try:
                return super().call(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if _is_daily_quota(exc):
                    raise ServiceQuotaExceededException(
                        f"DAILY_QUOTA_EXHAUSTED: Groq's free daily token allowance for {self.model} is used up "
                        f"(it resets within 24 h). Set CHAT_MODEL=openai/gpt-oss-20b in .env, which has a "
                        f"separate allowance, or try again tomorrow."
                    ) from exc
                if attempt == attempts - 1:
                    raise
                if _is_retryable_rate_limit(exc):
                    wait = min(_retry_after(str(exc)), 65.0)
                    if _WAIT_HOOK:
                        _WAIT_HOOK(wait)
                elif _is_transient_model_glitch(exc):
                    wait = 2.0
                    if _GLITCH_HOOK:
                        _GLITCH_HOOK()
                else:
                    raise
                time.sleep(wait)


def build_llm(s: Settings) -> LLM:
    # Free tiers have tight tokens-per-minute limits; let LiteLLM back off and retry on HTTP 429
    # instead of failing the whole multi-agent run.
    try:
        import litellm

        litellm.num_retries = 0  # retries are handled (visibly) by NativeToolLLM.call
        litellm.drop_params = True  # ignore params a provider doesn't support instead of erroring
    except Exception:
        pass
    llm_cls = NativeToolLLM if s.llm_provider == "groq" else LLM
    extra = {"reasoning_effort": s.reasoning_effort} if "gpt-oss" in s.chat_model else {}
    return llm_cls(model=s.litellm_model, temperature=s.temperature, max_tokens=s.max_tokens,
                   timeout=LLM_TIMEOUT_S, **extra)


def build_crew(s: Settings = default_settings, on_event: EventHandler | None = None,
               llm: LLM | None = None) -> Crew:
    emit = on_event or (lambda kind, payload: None)
    global _WAIT_HOOK, _GLITCH_HOOK
    _WAIT_HOOK = lambda seconds: emit("wait", {"seconds": seconds})  # noqa: E731
    _GLITCH_HOOK = lambda: emit("glitch", {})  # noqa: E731
    llm = llm or build_llm(s)
    native_tools = bool(getattr(llm, "supports_function_calling", lambda: False)())
    tools = default_tools()

    strategist = Agent(
        role="Strategist",
        goal="Turn the user's goal into a short, concrete research plan that can actually be executed",
        backstory=("You are a senior consultant who scopes projects. You break fuzzy goals into 3-5 "
                   "specific questions, and you say which source (web search or internal docs) answers each."),
        llm=llm, allow_delegation=False, max_iter=2, verbose=True,
    )
    researcher = Agent(
        role="Research Analyst",
        goal="Execute the research plan and gather accurate, sourced facts and numbers",
        backstory=("You are a meticulous analyst. You check the internal workspace documents first, then "
                   "search the web. You ONLY report facts that appear in tool results from this session, "
                   "each followed by the exact URL or file it came from. Your own memory is not a source: "
                   "if a tool did not return it, write 'Not found in sources'. Policies and figures change "
                   "often, so prefer the most recent sources and note their date. Use the calculator for maths."),
        tools=list(tools.values()),
        llm=llm, allow_delegation=False, max_iter=s.max_iter, verbose=True,
    )
    writer = Agent(
        role="Report Writer",
        goal="Deliver a clear, decision-ready report that fully answers the user's goal",
        backstory=("You write crisp executive reports. You use ONLY the analyst's findings, never your own "
                   "knowledge. Every claim carries a numbered citation like [1] that maps to a URL or file in "
                   "the Sources list. Claims without a source are dropped. You flag gaps and conflicting data."),
        llm=llm, allow_delegation=False, max_iter=2, verbose=True,
    )

    plan_task = Task(
        description=("Today is {today}. The user's goal is:\n\n\"{goal}\"\n\n"
                     "Write a research plan: 3-5 numbered questions that together achieve the goal. "
                     "For each, name the best source (internal docs via doc_reader, or web_search) and "
                     "what a good answer looks like. Finish with the outline of the final deliverable."),
        expected_output="A numbered research plan (3-5 questions with sources) plus a deliverable outline.",
        agent=strategist,
    )
    research_task = Task(
        description=("Execute the research plan for the goal \"{goal}\". First call doc_reader with "
                     "filename='list' to see internal files and read any relevant ones. Then use web_search "
                     "(and read_webpage for detail) to answer each question. Use calculator for growth rates "
                     "or percentages. Collect facts as bullet points, each followed by its source URL or file."),
        expected_output="Bullet-point findings grouped by plan question, every bullet ending with (source).",
        agent=researcher,
        context=[plan_task],
        guardrail=make_grounding_guardrail("research", GUARDRAIL_RETRIES, emit),
        guardrail_max_retries=GUARDRAIL_RETRIES,
    )
    report_task = Task(
        description=("Write the final deliverable for the goal \"{goal}\" using ONLY the research findings. "
                     "Format in Markdown:\n# <title>\n## Executive summary (3-5 bullets)\n"
                     "## Key findings (sections following the plan, with numbers and [source] tags)\n"
                     "## Opportunities & risks\n## Recommended next steps\n## Sources (numbered list of the exact URLs/files from the findings)\n"
                     "Cite with plain square brackets like [1]. Use only facts from the findings and drop any claim "
                     "without a source. If evidence is thin or conflicting for something, say so explicitly."),
        expected_output="A complete Markdown report with executive summary, findings, risks, next steps and sources.",
        agent=writer,
        context=[plan_task, research_task],
        guardrail=make_grounding_guardrail("report", GUARDRAIL_RETRIES, emit),
        guardrail_max_retries=GUARDRAIL_RETRIES,
    )

    def step_callback(step) -> None:
        # Text-mode (ReAct) tool use only; native tool calls are reported via the event bus in run().
        tool = getattr(step, "tool", None)
        if tool and not native_tools:
            emit("tool", {"tool": tool, "input": str(getattr(step, "tool_input", ""))[:200]})

    def task_callback(output) -> None:
        emit("task_done", {"agent": str(getattr(output, "agent", "")),
                           "summary": str(getattr(output, "summary", "") or "")[:200]})

    return Crew(
        agents=[strategist, researcher, writer],
        tasks=[plan_task, research_task, report_task],
        process=Process.sequential,
        max_rpm=s.max_rpm,
        step_callback=step_callback,
        task_callback=task_callback,
        verbose=True,
    )


def _short_args(args) -> str:
    if isinstance(args, dict):
        args = ", ".join(f"{v}" for v in args.values())
    return str(args)[:200]


URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+")
FILE_RE = re.compile(r"\b[\w.-]+\.(?:md|pdf|docx|txt|csv)\b", re.I)


def normalize_citations(text: str) -> str:
    """Some models write 【1】 instead of [1]."""
    return text.replace("\u3010", "[").replace("\u3011", "]")


def verify_sources(report: str, seen: set[str]) -> tuple[list[str], list[str]]:
    """Split the URLs/files cited in the report into ones the tools really saw and ones they didn't."""
    urls = [u.rstrip(".,;:") for u in URL_RE.findall(report)]
    cited = urls + FILE_RE.findall(URL_RE.sub(" ", report))   # file names only outside URLs
    verified, unverified = [], []
    unverified += [f"[{n}] {t}" for n, t in _source_entries(report)[3].items()]   # a source with no URL/file
    for c in dict.fromkeys(cited):  # de-duplicate, keep order
        n = normalize_source(c)
        ok = n in seen or any(n.startswith(x) or x.startswith(n) for x in seen if len(x) > 12)
        (verified if ok else unverified).append(c)
    return verified, unverified


# ------------------------------------------------------------------ figure fact-check
_NUM = r"\d{1,3}(?:[,    ]\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = (r"%|percent\b|per\s?cent\b|crore\b|cr\b|lakh\b|lakhs\b|lac\b|l\b|million\b|mn\b|billion\b|bn\b|trillion\b|"
         r"gwh\b|gw\b|mwh\b|mw\b|kwh\b|kw\b|km\b|units\b|vehicles\b|stations\b|chargers\b|cars\b")
_FIGURE_RE = re.compile(rf"(?P<cur>₹|rs\.?|inr|us\$|\$|usd)?\s?(?P<num>{_NUM})\s?(?P<unit>{_UNIT})?", re.I)
_RANGE_LEFT_RE = re.compile(rf"(?P<num>{_NUM})\s?(?:[-–‑—]|to)\s?$", re.I)
_ANY_NUM_RE = re.compile(_NUM)


def _canon(num: str) -> str | None:
    try:
        x = float(re.sub(r"[,    ]", "", num))
    except ValueError:
        return None
    return str(int(x)) if x.is_integer() else f"{x:.4f}".rstrip("0").rstrip(".")


_UNIT_CLASS = [
    ("pct", r"%|percent|per\s?cent"), ("crore", r"crore|cr"), ("lakh", r"lakhs?|lac|l"), ("million", r"million|mn"),
    ("billion", r"billion|bn"), ("trillion", r"trillion"), ("gwh", r"gwh"), ("gw", r"gw"), ("mwh", r"mwh"),
    ("mw", r"mw"), ("kwh", r"kwh"), ("kw", r"kw"), ("km", r"km"),
]
_STRICT_UNITS = {c for c, _ in _UNIT_CLASS}   # for these the unit must match too ("1 million", "40 %")


def _unit_class(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.lower().strip()
    for cls, pat in _UNIT_CLASS:
        if re.fullmatch(pat, u):
            return cls
    return "count"  # units / vehicles / stations ...: wording varies, so only the number must match


def _figures(text: str) -> list[tuple[str, list[str], str | None, int, int]]:
    """Find figures: (display, [canonical numbers], unit class, start, end). Ranges give two numbers."""
    out = []
    for m in _FIGURE_RE.finditer(text):
        num, cur, unit = m.group("num"), m.group("cur"), m.group("unit")
        canon = _canon(num)
        if canon is None:
            continue
        value = float(canon)
        is_year = 1900 <= value <= 2100 and canon.isdigit() and not (cur or unit)
        precise = ("." in num) or ("," in num) or value >= 100
        if is_year or not (cur or unit or precise):
            continue
        cls = _unit_class(unit)
        nums, start = [canon], m.start()
        left = _RANGE_LEFT_RE.search(text[max(0, m.start() - 12):m.start()])
        if left and unit:                                       # "35–40 %" -> 35 and 40, both in %
            nums.insert(0, _canon(left.group("num")))
            start = max(0, m.start() - 12) + left.start()
        out.append((text[start:m.end()].strip(), [n for n in nums if n], cls, start, m.end()))
    return out


class Evidence:
    """What the tools returned, globally and per source, with parsed numbers cached for fast checks."""

    def __init__(self, text: str, by_source: dict[str, str] | None = None):
        self.text, self.by_source = text, by_source or {}
        self._cache: dict[tuple, tuple[set, set]] = {}

    def parsed(self, keys: tuple = ()) -> tuple[set, set]:
        if keys not in self._cache:
            ev = self.text if not keys else "\n".join(self.by_source.get(k, "") for k in keys)
            nums = {c for c in (_canon(n) for n in _ANY_NUM_RE.findall(ev)) if c}
            pairs = {(n, cls) for _, ns, cls, _, _ in _figures(ev) for n in ns if cls}
            self._cache[keys] = (nums, pairs)
        return self._cache[keys]

    def match(self, source: str) -> str | None:
        """Map a cited URL/file to the key its evidence was stored under (tolerates trailing paths)."""
        n = normalize_source(source)
        if n in self.by_source:
            return n
        hits = [k for k in self.by_source if len(k) > 12 and (n.startswith(k) or k.startswith(n))]
        return max(hits, key=len) if hits else None


def _as_evidence(evidence) -> Evidence:
    return evidence if isinstance(evidence, Evidence) else Evidence(evidence)


_SRC_ENTRY_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s*(.*)$")


def _source_entries(report: str) -> tuple[str, str, dict[int, str], dict[int, str]]:
    """(body, sources section incl. heading, {n: URL/file}, {n: entry text with no URL/file})."""
    m = re.search(r"(?im)^\s{0,3}#{1,6}\s*sources\b.*$", report)
    body, sources = (report[:m.start()], report[m.start():]) if m else (report, "")
    numbered: dict[int, str] = {}
    unbacked: dict[int, str] = {}
    for line in sources.splitlines():
        lm = _SRC_ENTRY_RE.match(line)
        if lm:
            n, rest = int(lm.group(1) or lm.group(2)), lm.group(3)
            ref = (URL_RE.findall(rest) or FILE_RE.findall(URL_RE.sub(" ", rest)) or [None])[0]
            if ref:
                numbered[n] = ref.rstrip(".,;:")
            elif rest.strip():
                unbacked[n] = rest.strip()   # "Internal market research (proprietary)" -> nothing to check
    return body, sources, numbered, unbacked


def _source_list(report: str) -> tuple[str, str, dict[int, str]]:
    """(body, sources section incl. heading, {citation number: URL/file})."""
    body, sources, numbered, _ = _source_entries(report)
    return body, sources, numbered


def _evidence_keys(fragment: str, numbered: dict[int, str], ev: Evidence) -> tuple:
    """Which sources' evidence a fragment must be checked against.

    Cited sources (by [n] or inline URL/file) -> only their evidence (+ calculator results).
    No citation -> the evidence of everything listed in Sources. A cited source we hold no text
    for (or no per-source data at all) -> all evidence, i.e. the lenient run-wide check.
    """
    if not ev.by_source:
        return ()
    calc = {CALC_SOURCE} & ev.by_source.keys()
    refs = [numbered[int(n)] for n in re.findall(r"\[(\d+)\]", fragment) if int(n) in numbered]
    refs += [u.rstrip(".,;:") for u in URL_RE.findall(fragment)]
    refs += FILE_RE.findall(URL_RE.sub(" ", fragment))
    keys = [ev.match(r) for r in refs]
    if refs and all(keys):
        return tuple(sorted(set(keys) | calc))
    listed = [k for k in (ev.match(r) for r in numbered.values()) if k]
    if not refs and listed:
        return tuple(sorted(set(listed) | calc))
    return ()


def _unsupported(fragment: str, nums: set, pairs: set) -> tuple[list[str], list[tuple[str, str]]]:
    scan = URL_RE.sub(" ", fragment)
    scan = re.sub(r"\[\d+\]", " ", scan)                       # citation markers
    scan = re.sub(r"(?m)^\s*(?:\d+[.)]|[-*])\s+", " ", scan)     # list numbering
    ok, bad = [], []
    for figure, ns, cls, start, end in _figures(scan):
        good = all((n, cls) in pairs for n in ns) if cls in _STRICT_UNITS else all(n in nums for n in ns)
        if good:
            ok.append(figure)
        else:
            bad.append((figure, " ".join(scan[max(0, start - 60):end + 40].split())))
    return ok, bad


def check_figures(report: str, evidence) -> dict:
    """Every figure with a unit/currency (or a large/precise number) must appear in the evidence.

    `evidence` is the run's tool text, or an Evidence object that also knows which source each piece
    came from: then a figure must appear in the source cited on the same line ([n] or inline URL),
    not just anywhere in the run, so a real "20%" from one page can't vouch for a claim citing another.
    For percentages and magnitudes (crore, lakh, million, billion, kW, GW, km...) the number must
    appear WITH the same unit, so "1 million" isn't "supported" by any stray "1".
    Formats are normalised (1,500 == 1500; 40% == 40 per cent; Rs == ₹; cr == crore; L == lakh).
    Years, citation markers and list numbering are ignored.
    Returns {"supported": [...], "unsupported": [(figure, context), ...], "dangling": [n, ...]}.
    """
    ev = _as_evidence(evidence)
    report = normalize_citations(report)
    body, sources, numbered = _source_list(report)
    supported, unsupported, seen = [], [], set()
    for line in body.splitlines():
        if not line.strip():
            continue
        keys = _evidence_keys(line, numbered, ev)
        ok, bad = _unsupported(line, *ev.parsed(keys))
        for f in ok:                                   # the same figure may be fine for one source, not another
            if (f, keys) not in seen:
                seen.add((f, keys))
                supported.append(f)
        for f, ctx in bad:
            if (f, keys) not in seen:
                seen.add((f, keys))
                unsupported.append((f, ctx))

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    if numbered:                       # a citation must point at a listed source that has a URL/file
        dangling = sorted(c for c in cited if c not in numbered)
    else:
        n_sources = len(set(URL_RE.findall(report)))
        dangling = sorted(c for c in cited if c < 1 or c > n_sources) if cited else []
    return {"supported": supported, "unsupported": unsupported, "dangling": dangling}



# Sentence boundary that ignores "Rs. 500", "vs. X", "e.g. Y" and decimals.
_SENT_SPLIT_RE = re.compile(r"(?<!\bRs\.)(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bNo\.)(?<=[.!?])\s+(?=[A-Z(\[*_\"'₹$])")
_SOURCE_ONLY_RE = re.compile(r"^[\s\-*•\d.)]*(?:\(?\s*(?:source|sources)\s*:?\s*)?(?:\S*(?:https?://|\.md|\.pdf|\.txt|\.csv)\S*\s*\)?[\s,;.]*)*$", re.I)
# Removed figures leave dangling bridges such as "with" or ", while" behind; a sentence that loses
# its figure is dropped whole, so a claim is never silently changed into a different one.


_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _drop_empty_tables(lines: list[str]) -> list[str]:
    """Remove a table's header and separator when every body row was scrubbed."""
    out, i = [], 0
    while i < len(lines):
        is_row = lines[i].lstrip().startswith("|")
        if is_row and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
            nxt = lines[i + 2] if i + 2 < len(lines) else ""
            if not nxt.lstrip().startswith("|"):
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return out


_BARE_CITE_RE = re.compile(r"^[\s\-*•]*(?:\d+[.)])?\s*(?:\[\d+\][\s,;.]*)+$")


def _tidy(lines: list[str]) -> list[str]:
    """After scrubbing: drop citation-only bullets, emptied tables and sections, renumber lists."""
    lines = [ln if ln.lstrip().startswith("|") else re.sub(r"\s*\|\s*((?:\[\d+\]\s*)+)\|\s*$", r" \1", ln).rstrip()
             for ln in lines]
    lines = [ln for ln in lines if not _BARE_CITE_RE.match(ln)]
    lines = _drop_empty_tables(lines)
    out: list[str] = []
    for i, ln in enumerate(lines):          # a heading followed only by a same/higher-level heading is empty
        hm = re.match(r"^\s{0,3}(#{1,6})\s", ln)
        if hm:
            nxt = next((x for x in lines[i + 1:] if x.strip()), None)
            nm = re.match(r"^\s{0,3}(#{1,6})\s", nxt) if nxt else None
            if nxt is None or (nm and len(nm.group(1)) <= len(hm.group(1))):
                continue
        out.append(ln)
    n, prev_indent = 0, None
    for i, ln in enumerate(out):            # 1. 2. 4. -> 1. 2. 3.
        m = re.match(r"^(\s*)(\d+)([.)])(\s+)", ln)
        if m:
            n = n + 1 if prev_indent == m.group(1) else 1
            prev_indent = m.group(1)
            out[i] = f"{m.group(1)}{n}{m.group(3)}{m.group(4)}{ln[m.end():]}"
        elif ln.strip():
            prev_indent = None
    return out


def scrub_unsupported_figures(text: str, evidence) -> tuple[str, list[str]]:
    """Deterministically drop every sentence whose figures are not in the evidence of its cited source.

    The model's own rewrite loop is expensive and sometimes answers a correction with a refusal,
    so the last line of defence is code: a claim with an unverifiable number is removed, never
    "fixed". Table rows go whole; headings, the Sources section and figure-free text are kept.
    Returns (clean_text, removed_sentences).
    """
    ev = _as_evidence(evidence)
    body, sources, numbered, unbacked = _source_entries(normalize_citations(text))
    removed: list[str] = []
    out_lines: list[str] = []
    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        cites = {int(n) for n in re.findall(r"\[(\d+)\]", line)}
        if numbered and cites and not (cites & numbered.keys()) and not URL_RE.search(line):
            removed.append(" ".join(line.split()) + "  (cited source has no URL or file)")
            continue                               # rests only on sources nobody can open
        nums_pairs = ev.parsed(_evidence_keys(line, numbered, ev))   # a line's citations cover its sentences
        if not _unsupported(line, *nums_pairs)[1]:
            out_lines.append(line)
            continue
        if line.lstrip().startswith("|"):          # a table row is one claim: drop it whole, never split it
            removed.append(" ".join(line.split()))
            continue
        indent = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*)", line).group(1)
        kept = []
        for sent in _SENT_SPLIT_RE.split(line[len(indent):]):
            if _unsupported(sent, *nums_pairs)[1]:
                removed.append(" ".join(sent.split()))
            else:
                kept.append(sent)
        rest = " ".join(kept).strip()
        if rest and not _SOURCE_ONLY_RE.match(rest):
            out_lines.append(indent + rest)
    clean = "\n".join(_tidy(out_lines))
    clean = re.sub(r"\n{3,}", "\n\n", clean).rstrip()
    if sources:
        clean, sources = _renumber_sources(clean, sources, numbered)
        clean += "\n\n" + sources.lstrip("\n")
    return clean, removed


def _renumber_sources(body: str, sources: str, numbered: dict[int, str]) -> tuple[str, str]:
    """Keep only sources with a URL/file, number them 1..n and rewrite the [n] markers to match."""
    if not numbered:
        return body, sources
    mapping = {old: new for new, old in enumerate(sorted(numbered), 1)}
    out = []
    for line in sources.splitlines():
        lm = _SRC_ENTRY_RE.match(line)
        if lm:
            old = int(lm.group(1) or lm.group(2))
            if old not in mapping:
                continue                           # unbacked entry: dropped
            line = f"{mapping[old]}. {lm.group(3)}"
        out.append(line)
    body = re.sub(r"\[(\d+)\]", lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping
                  else "", body)
    return body, "\n".join(out)



def current_evidence() -> "Evidence":
    return Evidence(evidence_text(), evidence_by_source())


MIN_VERIFIED_SOURCES = 3
REMOVED_FIGURE_CLAIMS: list[str] = []   # filled by the guardrails during a run


def make_grounding_guardrail(stage: str, max_retries: int, emit: EventHandler):
    """CrewAI task guardrail: the output must cite sources the tools really retrieved in this run.

    On failure the task is sent back to the agent with concrete feedback. On the final attempt the
    best attempt so far is accepted but flagged, because CrewAI would otherwise abort the whole run
    (and a retry that answers with a refusal must never replace a real report).
    Whatever is accepted then goes through scrub_unsupported_figures(): figures are fact-checked by
    code, not by asking the model again, which is cheaper and cannot be talked around.
    (No return annotation on purpose: CrewAI validates it, and postponed annotations are strings.)
    """
    state = {"attempts": 0, "best": None, "best_score": -1}

    def accept(text: str, info: dict, result: str):
        """Final, deterministic fact-check: sentences with figures not found in any tool result are removed."""
        clean, removed = scrub_unsupported_figures(text, current_evidence())
        if removed:
            REMOVED_FIGURE_CLAIMS.extend(removed)
        emit("guardrail", {**info, "result": result, "removed": len(removed)})
        return True, clean

    def guardrail(output):
        state["attempts"] += 1
        text = normalize_citations(getattr(output, "raw", None) or str(output))
        refusal = _looks_like_refusal(text)
        verified, unverified = verify_sources(text, set(SEEN_SOURCES))
        total = len(verified) + len(unverified)
        figs = check_figures(text, current_evidence())
        bad_figs = figs["unsupported"]
        dangling = figs["dangling"] if stage == "report" else []
        if stage == "research":
            sources_ok = len(verified) >= MIN_VERIFIED_SOURCES and len(verified) >= 0.7 * total
        else:
            sources_ok = len(verified) >= MIN_VERIFIED_SOURCES and not unverified
        ok = sources_ok and not dangling and not refusal
        n_figs = len(figs["supported"]) + len(bad_figs)
        info = {"stage": stage, "verified": len(verified), "total": total, "attempt": state["attempts"],
                "figures_ok": len(figs["supported"]), "figures_total": n_figs, "dangling": len(dangling)}
        # Remember the best real attempt, so a weaker retry (or a refusal) can never replace it.
        score = -1 if refusal else len(verified) - len(unverified)
        if score > state["best_score"]:
            state.update(best=text, best_score=score, best_info=info, best_attempt=state["attempts"])
        if ok:
            return accept(text, info, "passed")
        if state["attempts"] > max_retries:  # out of retries: keep the best work, but flag it loudly
            if state["best"] is not None and state.get("best_attempt") != state["attempts"]:
                text, info = state["best"], {**state["best_info"], "attempt": state["attempts"]}
            return accept(text, info, "flagged")
        emit("guardrail", {**info, "result": "retry"})
        if sources_ok and not refusal:
            feedback = "Sources are fine."
        elif refusal:
            feedback = ("Your previous reply contained no content. Do not apologise or refuse: write it now from the "
                        "material you already have, keep only claims backed by a source, and cite each one.")
        elif stage == "research":
            feedback = (
                f"GROUNDING CHECK FAILED: only {len(verified)} of the {total} sources you cited were actually "
                f"returned by your tools in this session (need at least {MIN_VERIFIED_SOURCES}, and at least 70% "
                f"of all cited sources). "
                + (f"Not retrieved by any tool: {unverified[:5]}. " if unverified else "You cited no URLs or files. ")
                + "Use web_search (and read_webpage for detail) to find real sources, and end every bullet with "
                "the exact URL copied from the tool results. Never cite sources from memory."
            )
        else:
            feedback = (
                f"GROUNDING CHECK FAILED: the report must only cite URLs/files that appear in the research findings "
                f"and were retrieved by tools (need at least {MIN_VERIFIED_SOURCES}). "
                + (f"Remove these sources and every claim that depends on them: {unverified[:5]}. " if unverified else "")
                + "Copy the exact URLs from the findings into the numbered Sources list."
            )
        if bad_figs:
            listed = "; ".join(f"'{f}' (in: \"{c}\")" for f, c in bad_figs[:8])
            feedback += (f" Also note: these figures do not appear in any tool result and will be removed automatically "
                         f"unless you replace them with figures from the tool results: {listed}.")
        if dangling:
            feedback += (f" CITATION CHECK FAILED: citations {['[%d]' % d for d in dangling]} do not match any entry "
                         f"in the numbered Sources list. Renumber the citations or add the missing source.")
        return False, feedback

    return guardrail


def slugify(text: str, max_len: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:max_len] or "report"


def run(goal: str, s: Settings = default_settings, on_event: EventHandler | None = None) -> RunResult:
    goal = goal.strip()
    if len(goal) < 10:
        raise ValueError("Please describe the goal in a full sentence (at least 10 characters).")

    emit = on_event or (lambda kind, payload: None)
    reset_seen_sources()
    REMOVED_FIGURE_CLAIMS.clear()
    crew = build_crew(s, emit)
    inputs = {"goal": goal, "today": dt.date.today().isoformat()}
    try:
        from crewai.events import crewai_event_bus
        from crewai.events.types.tool_usage_events import ToolUsageErrorEvent, ToolUsageStartedEvent
    except Exception:  # older CrewAI without the event bus
        result = crew.kickoff(inputs=inputs)
    else:
        with crewai_event_bus.scoped_handlers():
            @crewai_event_bus.on(ToolUsageStartedEvent)
            def _tool_started(source, event):
                emit("tool", {"tool": event.tool_name, "input": _short_args(event.tool_args),
                              "agent": event.agent_role or ""})

            @crewai_event_bus.on(ToolUsageErrorEvent)
            def _tool_error(source, event):
                emit("tool_error", {"tool": event.tool_name, "error": str(getattr(event, "error", ""))[:200]})

            result = crew.kickoff(inputs=inputs)
    outputs = [normalize_citations(t.output.raw) if t.output else "" for t in crew.tasks]
    report = normalize_citations(result.raw)
    verified, unverified = verify_sources(report, set(SEEN_SOURCES))
    figs = check_figures(report, current_evidence())

    s.output_dir.mkdir(parents=True, exist_ok=True)
    path = s.output_dir / f"{dt.datetime.now():%Y%m%d-%H%M}-{slugify(goal)}.md"
    path.write_text(report, encoding="utf-8")

    usage = getattr(result, "token_usage", None)
    usage = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
    return RunResult(goal, outputs[0], outputs[1], report, path, usage, verified, unverified,
                     figures_supported=figs["supported"], figures_unsupported=figs["unsupported"],
                     dangling_citations=figs["dangling"], figures_removed=list(REMOVED_FIGURE_CLAIMS))
