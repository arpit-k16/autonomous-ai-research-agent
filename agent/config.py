"""Settings from environment / .env (or Streamlit secrets)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)  # the project .env wins over system-wide variables

# CrewAI extras we don't want: anonymous telemetry, and the interactive "view your traces? [y/N]"
# prompt that newer versions show after a run (it would block a web app waiting for input).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

DEFAULT_MODEL = {"groq": "openai/gpt-oss-120b", "openai": "gpt-4o-mini"}


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower())
    chat_model: str = field(default_factory=lambda: _env(
        "CHAT_MODEL", DEFAULT_MODEL.get(_env("LLM_PROVIDER", "groq").lower(), "")))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.2")))
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "2048")))
    max_retries: int = field(default_factory=lambda: int(_env("MAX_RETRIES", "6")))
    # gpt-oss "thinks" before answering; low effort cuts tokens a lot with little quality loss here
    reasoning_effort: str = field(default_factory=lambda: _env("REASONING_EFFORT", "low"))

    # Keep the free Groq tier happy: cap requests/minute and agent loop length
    max_rpm: int = field(default_factory=lambda: int(_env("MAX_RPM", "10")))
    max_iter: int = field(default_factory=lambda: int(_env("MAX_ITER", "10")))
    search_results: int = field(default_factory=lambda: int(_env("SEARCH_RESULTS", "4")))
    page_chars: int = field(default_factory=lambda: int(_env("PAGE_CHARS", "2000")))

    # The doc_reader tool can only read files inside this folder
    workspace_dir: Path = field(default_factory=lambda: Path(_env("WORKSPACE_DIR", str(ROOT / "workspace"))))
    output_dir: Path = field(default_factory=lambda: Path(_env("OUTPUT_DIR", str(ROOT / "outputs"))))

    @property
    def litellm_model(self) -> str:
        """CrewAI routes models through LiteLLM-style ids, e.g. 'groq/openai/gpt-oss-120b'."""
        return f"{self.llm_provider}/{self.chat_model}"


settings = Settings()
