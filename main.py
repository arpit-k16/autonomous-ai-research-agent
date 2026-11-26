"""Run the agent from the terminal.

    python main.py "Find top EV market trends for 2027"
    python main.py            # prompts for a goal
"""
from __future__ import annotations

import os
import sys

from agent import run
from agent.config import settings

ICONS = {"tool": "🔧", "thought": "💭", "task_done": "✅"}


def on_event(kind: str, data: dict) -> None:
    if kind == "tool":
        print(f"  🔧 {data['tool']}({data['input']})")
    elif kind == "task_done":
        print(f"\n✅ {data['agent']} finished\n")


def main() -> None:
    key = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}.get(settings.llm_provider)
    if key and not os.getenv(key):
        sys.exit(f"Set {key} in .env first (see .env.example).")

    goal = " ".join(sys.argv[1:]) or input("🎯 What should the agent achieve? ")
    print(f"\n🤖 Working on: {goal}\n   Model: {settings.litellm_model}\n")
    result = run(goal, on_event=on_event)

    print("\n" + "=" * 80 + "\n")
    print(result.report)
    print("\n" + "=" * 80)
    print(f"📄 Saved to {result.report_path}")
    if result.token_usage:
        print(f"🔢 Tokens used: {result.token_usage.get('total_tokens', '?')}")


if __name__ == "__main__":
    main()
