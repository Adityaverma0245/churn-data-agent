"""LLM backends.

The agent only needs `complete(system, messages) -> str`, so any model can be
plugged in. Two backends are included:

- AnthropicLLM: calls the Anthropic Messages API (needs `pip install anthropic`
  and ANTHROPIC_API_KEY).
- RuleBasedPlanner: an OFFLINE stand-in with simple keyword rules. It is NOT
  intelligent. It exists so the loop, tools, memory and eval harness can be
  tested and demoed without an API key or network.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict]) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 600):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install the SDK first: pip install anthropic") from e
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict]) -> str:  # pragma: no cover
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class RuleBasedPlanner:
    """Keyword planner: maps a question to one tool call, then reports the observation."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def complete(self, system: str, messages: list[dict]) -> str:
        last = messages[-1]["content"]
        if last.startswith("OBSERVATION:"):
            obs = last[len("OBSERVATION:"):].strip()
            if obs.startswith("ERROR"):
                return json.dumps({"final": f"I could not complete that: {obs}"})
            return json.dumps({"final": f"Result: {obs}"})

        q = messages[0]["content"].split("QUESTION:", 1)[-1].strip().lower()
        col = next((c for c in self.columns if c.lower() in q), None)

        if re.search(r"\bpredict|probability|likely to churn\b", q):
            pairs = dict(re.findall(r"(\w+)=([\w\-\.]+)", messages[0]["content"].split("QUESTION:", 1)[-1]))
            cust = {k: (float(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else v) for k, v in pairs.items()}
            return json.dumps({"thought": "predict for the given customer", "action": "predict_churn", "args": {"customer": cust}})
        if re.search(r"\bdriver|important|factor|why\b", q):
            return json.dumps({"thought": "need feature importances", "action": "top_churn_drivers", "args": {"n": 5}})
        if col and re.search(r"\bby\b|rate|compare|per\b", q):
            return json.dumps({"thought": f"group churn by {col}", "action": "churn_rate_by", "args": {"column": col}})
        if re.search(r"\bdescribe|overview|how many|columns|rows|summary\b", q):
            return json.dumps({"thought": "dataset overview", "action": "describe_dataset", "args": {}})
        return json.dumps({"final": "I can't answer that with the available tools."})
