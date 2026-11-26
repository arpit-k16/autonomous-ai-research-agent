"""Streamlit UI for the autonomous agent.   Run:  streamlit run app.py"""
from __future__ import annotations

import faulthandler
import os
import sys

import streamlit as st

faulthandler.enable(file=sys.stderr, all_threads=True)  # a native crash prints every thread's Python stack to the logs

try:  # Streamlit Cloud secrets -> env vars (before settings are read)
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from agent import run  # noqa: E402
from agent.config import settings  # noqa: E402

EXAMPLES = [
    "Find top EV market trends in India for 2027",
    "Compare the 3 most popular open-source vector databases for a startup",
    "Summarise the key AI regulations a SaaS company in the EU must follow",
]
STAGES = {"Strategist": "📋 Plan", "Research Analyst": "🔎 Search & Execute", "Report Writer": "📦 Deliver"}

st.set_page_config(page_title="Autonomous AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 Autonomous AI Agent")
st.caption("Give it a goal. It plans, searches the web and your docs, executes, and delivers a sourced report.")

with st.sidebar:
    st.subheader("How it works")
    st.markdown("1. **Plan** — Strategist breaks the goal into questions\n"
                "2. **Search** — Analyst uses `web_search`, `read_webpage`, `doc_reader`\n"
                "3. **Execute** — verifies numbers with `calculator`\n"
                "4. **Deliver** — Writer produces a cited report")
    st.divider()
    st.subheader("📁 Workspace docs")
    st.caption("Files added here are shared with everyone using this server, so don't upload private documents to a public demo.")
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    for up in st.file_uploader("Add internal docs for doc_reader", accept_multiple_files=True,
                               type=["pdf", "docx", "md", "txt", "csv"]) or []:
        (settings.workspace_dir / os.path.basename(up.name)).write_bytes(up.getbuffer())
    for p in sorted(settings.workspace_dir.glob("*")):
        st.write("•", p.name)
    st.divider()
    st.caption(f"Model: `{settings.litellm_model}` · max {settings.max_rpm} req/min")

key = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}.get(settings.llm_provider)
if key and not os.getenv(key):
    st.error(f"Add `{key}` to `.env` (local) or the app's Secrets (Streamlit Cloud).")
    st.stop()

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.goal = ex

goal = st.text_area("🎯 Goal", key="goal", height=80,
                    placeholder="e.g. Find top EV market trends in India for 2027")


@st.cache_resource
def _run_lock():
    """One run at a time per server: the fact-check evidence is process-wide, and the free API quota is shared."""
    import threading
    return threading.Lock()


if st.button("🚀 Run agent", type="primary", disabled=not goal.strip()):
    lock = _run_lock()
    if not lock.acquire(blocking=False):
        st.warning("Another run is in progress on this server. Please try again in a few minutes.")
        st.stop()
    import queue
    import threading
    import time

    status = st.status("Agent is working…", expanded=True)
    events: "queue.Queue[tuple[str, dict]]" = queue.Queue()
    box: dict = {}

    def worker() -> None:  # runs CrewAI off the Streamlit thread; events go through the queue
        try:
            box["result"] = run(goal, on_event=lambda kind, data: (
                print(f"[agent] {kind}: {str(data)[:160]}", flush=True),   # progress in the server logs
                events.put((kind, data))))
        except Exception as exc:  # surfaced in the UI below
            box["error"] = exc
        finally:  # released by the worker, so a page reload mid-run can't leave the lock held
            lock.release()

    # Diagnostics: if a run stalls, every thread's stack is written to outputs/stall_trace.log every 45 s.
    import faulthandler
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    trace_file = open(settings.output_dir / "stall_trace.log", "w", encoding="utf-8")
    faulthandler.dump_traceback_later(45, repeat=True, file=trace_file)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    started, n_tools = time.time(), 0
    while th.is_alive() or not events.empty():
        try:
            kind, data = events.get(timeout=0.5)
        except queue.Empty:
            status.update(label=f"Agent is working… {int(time.time() - started)} s · {n_tools} tool calls")
            continue
        if kind == "tool":
            n_tools += 1
            status.write(f"🔧 **{data['tool']}** → `{data['input']}`")
        elif kind == "guardrail":
            label = {"research": "research findings", "report": "final report"}[data["stage"]]
            score = (f"{data['verified']}/{data['total']} cited sources retrieved by tools · "
                     f"{data.get('figures_ok', 0)}/{data.get('figures_total', 0)} figures found in sources"
                     + (f" · {data['dangling']} broken citation(s)" if data.get("dangling") else ""))
            scrub = (f" · 🧹 removed {data['removed']} claim(s) whose figures no source contains"
                     if data.get("removed") else "")
            if data["result"] == "passed":
                status.write(f"🛡️ Grounding check on {label} **passed** ({score}){scrub}")
            elif data["result"] == "retry":
                status.write(f"🛡️ Grounding check on {label} **failed** ({score}) → sent back to the agent with feedback")
            else:
                status.write(f"🛡️ Grounding check on {label} still failing after retries ({score}) → "
                             f"kept the best attempt, flagged below{scrub}")
        elif kind == "glitch":
            status.write("🔁 Model call glitch (unparseable output or timeout), retrying…")
        elif kind == "wait":
            status.write(f"⏳ Waiting {data['seconds']:.0f} s for Groq's free-tier rate limit…")
        elif kind == "tool_error":
            status.write(f"⚠️ {data['tool']} failed: {data['error']}")
        elif kind == "task_done":
            stage = STAGES.get(data["agent"].strip(), data["agent"])
            status.write(f"✅ **{stage}** complete")

    faulthandler.cancel_dump_traceback_later()
    trace_file.close()

    if "error" in box:
        raw = str(box["error"])
        low = raw.lower()
        is_rate = "rate_limit_exceeded" in low or "ratelimiterror" in low or "rate limit reached" in low
        if "daily_quota_exhausted" in low or (is_rate and ("per day" in low or "(tpd)" in low or "(rpd)" in low)):
            msg = ("Groq's free **daily** limit for this model is used up. It resets within 24 h. "
                   "Meanwhile, set `CHAT_MODEL=openai/gpt-oss-20b` (a separate daily quota) in `.env` and restart.")
        elif is_rate:
            msg = "Free-tier per-minute rate limit hit even after retries. Wait a minute and try again, or lower MAX_RPM."
        else:
            msg = "The run failed."
        status.update(label="Failed", state="error")
        st.error(msg)
        with st.expander("Error details"):
            st.code(raw[:3000])
        st.stop()

    status.update(label=f"Done ✅ in {int(time.time() - started)} s · {n_tools} tool calls", state="complete", expanded=False)
    st.session_state.result = box["result"]

if result := st.session_state.get("result"):
    tab_report, tab_plan, tab_findings = st.tabs(["📦 Report", "📋 Plan", "🔎 Raw findings"])
    with tab_report:
        v, u = result.verified_sources, result.unverified_sources
        if v or u:
            total = len(v) + len(u)
            (st.success if not u else st.warning)(
                f"🔍 Source check: **{len(v)}/{total}** cited sources were actually opened or returned by the "
                f"agent's tools during this run." + ("" if not u else " Unverified sources are listed below the report."))
        fs, fu, dc = result.figures_supported, result.figures_unsupported, result.dangling_citations
        rm = getattr(result, "figures_removed", [])
        if fs or fu or rm:
            (st.success if not fu and not dc else st.warning)(
                f"📊 Figures check: **{len(fs)}/{len(fs) + len(fu)}** figures in the report appear in the text the "
                f"tools retrieved." + (f" {len(dc)} citation(s) point to no source." if dc else "")
                + (f" {len(rm)} claim(s) with unverifiable figures were removed by the fact-checker." if rm else ""))
        st.markdown(result.report)
        if fu or dc:
            with st.expander(f"⚠️ {len(fu)} figure(s) not found in any retrieved source"):
                for fig, ctx in fu:
                    st.markdown(f"- **{fig}** — _…{ctx}…_")
                if dc:
                    st.markdown(f"- Broken citations: {', '.join(f'[{d}]' for d in dc)}")
        if rm:
            with st.expander(f"🧹 {len(rm)} claim(s) removed: their figures appear in no retrieved source"):
                for c in rm:
                    st.markdown(f"- ~~{c}~~")
        if u:
            with st.expander(f"⚠️ {len(u)} unverified source(s): cited, but never seen by the tools"):
                for x in u:
                    st.write("•", x)
        st.download_button("⬇️ Download report (.md)", result.report, file_name=result.report_path.name)
    with tab_plan:
        st.markdown(result.plan)
    with tab_findings:
        st.markdown(result.findings)
    if result.token_usage:
        tok = result.token_usage.get("total_tokens")
        st.caption(f"Tokens used: {tok:,}" if isinstance(tok, int) else "Tokens used: ?") 
        st.caption(f"Saved to `{result.report_path.name}`")
