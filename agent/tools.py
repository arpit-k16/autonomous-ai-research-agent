"""Tools the agents can call.

    web_search   -> DuckDuckGo search (free, no API key)
    read_webpage -> fetch a URL and return its readable text
    doc_reader   -> read a PDF / DOCX / MD / TXT / CSV from the workspace folder
    calculator   -> safe arithmetic (growth rates, CAGR, percentages)

Every tool returns plain text, truncated so it fits in a free-tier context window.
"""
from __future__ import annotations

import ast
import operator
import re
import threading
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .config import settings

USER_AGENT = "Mozilla/5.0 (compatible; PortfolioResearchAgent/1.0)"
MAX_PAGE_BYTES = 2_000_000   # stop reading huge pages
PAGE_DEADLINE_S = 20.0       # total time budget per page, not just per chunk

# Every URL / file the tools actually returned during a run. Used afterwards to verify that the
# sources cited in the final report were really seen by the agent (and not recalled from memory).
_SEEN_LOCK = threading.Lock()
SEEN_SOURCES: set[str] = set()
# Every piece of text the tools handed to the model during a run (search results, pages, documents,
# calculator results). Figures in the final report are checked against this.
EVIDENCE: list[str] = []
# The same evidence keyed by the source it came from (normalised URL or file name), so a figure can be
# checked against the source cited next to it, not just against anything seen in the run.
EVIDENCE_BY_SOURCE: dict[str, list[str]] = {}
CALC_SOURCE = "__calculator__"   # calculator results back any citation: they are derived numbers


def reset_seen_sources() -> None:
    with _SEEN_LOCK:
        SEEN_SOURCES.clear()
        EVIDENCE.clear()
        EVIDENCE_BY_SOURCE.clear()


def record_evidence(text: str, source: str | None = None, parts: dict[str, str] | None = None) -> str:
    """Store tool output as evidence; `source` / `parts` say which source each piece came from."""
    if text:
        with _SEEN_LOCK:
            EVIDENCE.append(text)
            for src, piece in ({source: text} if source else (parts or {})).items():
                if src and piece:
                    key = src if src == CALC_SOURCE else normalize_source(src)
                    EVIDENCE_BY_SOURCE.setdefault(key, []).append(piece)
    return text


def evidence_by_source() -> dict[str, str]:
    with _SEEN_LOCK:
        return {k: "\n".join(v) for k, v in EVIDENCE_BY_SOURCE.items()}


def evidence_text() -> str:
    with _SEEN_LOCK:
        return "\n".join(EVIDENCE)


def record_source(src: str) -> None:
    if src:
        with _SEEN_LOCK:
            SEEN_SOURCES.add(normalize_source(src))


def normalize_source(src: str) -> str:
    src = src.strip().strip("<>()[]\"'.,;").lower()
    src = re.sub(r"^https?://(www\.)?", "", src)
    return src.rstrip("/")


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) <= limit else text[:limit] + f"\n…[truncated, {len(text) - limit} more chars]"


# ---------------------------------------------------------------- web_search
class SearchInput(BaseModel):
    query: str = Field(..., description="What to search the web for. Be specific, include the year if relevant.")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = ("Search the web. Returns titles, URLs and snippets of the top results. "
                        "Use read_webpage on a URL when you need more detail than the snippet.")
    args_schema: Type[BaseModel] = SearchInput
    max_results: int = settings.search_results

    def _search(self, query: str) -> list[dict]:
        from ddgs import DDGS

        return list(DDGS().text(query, max_results=self.max_results))

    def _run(self, query: str) -> str:
        try:
            results = self._search(query)
        except Exception as exc:
            return f"Search failed ({exc}). Try a different query."
        if not results:
            return "No results found. Try a broader query."
        for r in results:
            record_source(r.get("href", ""))
        lines = [f"[{i}] {r.get('title', '')}\n    URL: {r.get('href', '')}\n    {r.get('body', '')}"
                 for i, r in enumerate(results, 1)]
        parts = {r.get("href", ""): f"{r.get('title', '')}\n{r.get('body', '')}" for r in results}
        return record_evidence("\n".join(lines), parts=parts)


# ---------------------------------------------------------------- read_webpage
class WebpageInput(BaseModel):
    url: str = Field(..., description="Full http(s) URL to read")


def fetch_bytes(url: str, deadline: float = PAGE_DEADLINE_S, max_bytes: int = MAX_PAGE_BYTES) -> tuple[bytes, str]:
    """Download a URL with a hard total-time and size budget. Returns (body, content_type).

    requests' own timeout only limits the gap between bytes, so a server that trickles data can
    hold a call for minutes. Stream the body and stop at the deadline or size cap instead.
    """
    import time

    import requests

    start = time.monotonic()
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(5, 10), stream=True) as resp:
        resp.raise_for_status()
        chunks, size = [], 0
        for chunk in resp.iter_content(chunk_size=16_384):
            chunks.append(chunk)
            size += len(chunk)
            if size >= max_bytes or time.monotonic() - start > deadline:
                break
        ctype = str(getattr(resp, "headers", {}).get("Content-Type", "")).lower()
        encoding = resp.encoding or "utf-8"
    body = b"".join(chunks)
    return body, ctype if ctype else f"text/html; charset={encoding}"


def fetch_html(url: str, deadline: float = PAGE_DEADLINE_S, max_bytes: int = MAX_PAGE_BYTES) -> str:
    body, ctype = fetch_bytes(url, deadline, max_bytes)
    m = re.search(r"charset=([\w-]+)", ctype)
    return body.decode(m.group(1) if m else "utf-8", errors="ignore")


def pdf_to_text(data: bytes, max_pages: int = 20) -> str:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages[:max_pages], 1):
        pages.append(f"[page {i}] {page.extract_text() or ''}")
    return "\n".join(pages)


class ReadWebpageTool(BaseTool):
    name: str = "read_webpage"
    description: str = "Fetch a web page and return its main readable text (scripts, menus and footers removed)."
    args_schema: Type[BaseModel] = WebpageInput
    max_chars: int = settings.page_chars

    def _run(self, url: str) -> str:
        from bs4 import BeautifulSoup

        if not url.startswith(("http://", "https://")):
            return "Please pass a full URL starting with http:// or https://"
        try:
            body, ctype = fetch_bytes(url)
        except Exception as exc:
            return f"Could not fetch {url}: {exc}"
        record_source(url)
        if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf") or body[:5] == b"%PDF-":
            try:
                text = pdf_to_text(body)
            except Exception as exc:  # truncated or scanned PDFs
                return f"Could not read the PDF at {url}: {exc}"
            return record_evidence(_truncate(f"# PDF: {url}\nSource: {url}\n\n{text}", self.max_chars), source=url)
        m = re.search(r"charset=([\w-]+)", ctype)
        html = body.decode(m.group(1) if m else "utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else url
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if len(line.strip()) > 30)
        return record_evidence(_truncate(f"# {title}\nSource: {url}\n\n{text}", self.max_chars), source=url)


# ---------------------------------------------------------------- doc_reader
class DocInput(BaseModel):
    filename: str = Field(..., description="File name inside the workspace, e.g. 'ev_notes.md'. "
                                           "Use 'list' to see available files.")


class DocReaderTool(BaseTool):
    name: str = "doc_reader"
    description: str = ("Read an internal document from the workspace folder (PDF, DOCX, MD, TXT, CSV). "
                        "Call with filename='list' first to see which files exist.")
    args_schema: Type[BaseModel] = DocInput
    workspace: Path = settings.workspace_dir
    max_chars: int = settings.page_chars

    def _files(self) -> list[Path]:
        exts = {".pdf", ".docx", ".md", ".txt", ".csv"}
        return sorted(p for p in self.workspace.rglob("*") if p.is_file() and p.suffix.lower() in exts)

    def _run(self, filename: str) -> str:
        if not self.workspace.exists():
            return "The workspace folder is empty."
        if filename.strip().lower() in {"list", "ls", "*", ""}:
            files = self._files()
            return "Available files:\n" + "\n".join(f"- {p.relative_to(self.workspace)}" for p in files) \
                if files else "The workspace folder is empty."

        root = self.workspace.resolve()
        path = (root / filename).resolve()
        if root not in path.parents:  # block ../../ path traversal
            return "Access denied: only files inside the workspace can be read."
        if not path.exists():
            return f"'{filename}' not found. Call doc_reader with filename='list' to see available files."

        record_source(path.name)
        ext = path.suffix.lower()
        if ext == ".pdf":
            from pypdf import PdfReader

            text = "\n\n".join(f"[page {i}]\n{p.extract_text() or ''}"
                               for i, p in enumerate(PdfReader(str(path)).pages, 1))
        elif ext == ".docx":
            import docx2txt

            text = docx2txt.process(str(path))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        return record_evidence(_truncate(f"# {path.name}\n\n{text}", self.max_chars), source=path.name)


# ---------------------------------------------------------------- calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic only — no names, calls or attributes (so no code execution)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("Only numbers and + - * / ** % ( ) are allowed")
    return ev(ast.parse(expr.replace("^", "**").replace(",", ""), mode="eval"))


class CalcInput(BaseModel):
    expression: str = Field(..., description="Arithmetic expression, e.g. '(450/120)**(1/5) - 1' for a 5-year CAGR")


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Do exact arithmetic: growth rates, CAGR, percentages, unit conversions. Never do math in your head."
    args_schema: Type[BaseModel] = CalcInput

    def _run(self, expression: str) -> str:
        try:
            return record_evidence(f"{expression} = {round(safe_eval(expression), 6)}", source=CALC_SOURCE)
        except Exception as exc:
            return f"Could not evaluate '{expression}': {exc}"


def default_tools() -> dict[str, BaseTool]:
    return {"web_search": WebSearchTool(), "read_webpage": ReadWebpageTool(),
            "doc_reader": DocReaderTool(), "calculator": CalculatorTool()}
