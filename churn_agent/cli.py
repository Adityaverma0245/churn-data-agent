"""Command line: python -m churn_agent.cli "Which factors drive churn?" """
from __future__ import annotations

import argparse
import json

import pandas as pd

from .agent import Agent
from .llm import AnthropicLLM, RuleBasedPlanner
from .memory import SQLiteMemory
from .tools import TARGET, DataTools, build_registry


def build_agent(data: str, llm_name: str = "rule", db: str = ":memory:", max_steps: int = 6) -> Agent:
    df = pd.read_csv(data)
    registry = build_registry(DataTools(df))
    cols = [c for c in df.columns if c != TARGET]
    llm = AnthropicLLM() if llm_name == "anthropic" else RuleBasedPlanner(cols)
    return Agent(llm, registry, SQLiteMemory(db), max_steps=max_steps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Churn data agent")
    ap.add_argument("question")
    ap.add_argument("--data", default="data/sample_churn.csv")
    ap.add_argument("--llm", choices=["rule", "anthropic"], default="rule")
    ap.add_argument("--db", default=":memory:", help="SQLite file for persistent memory")
    ap.add_argument("--trace", action="store_true", help="print the step trace")
    a = ap.parse_args()
    res = build_agent(a.data, a.llm, a.db).run(a.question)
    print(f"[{res.status} in {res.steps} step(s)] {res.answer}")
    if a.trace:
        print(json.dumps(res.trace, indent=2))


if __name__ == "__main__":
    main()
