"""The agent loop: think -> act (tool call) -> observe -> repeat -> final answer.

Protocol: each model turn must be ONE JSON object, either
  {"thought": "...", "action": "<tool>", "args": {...}}   or
  {"final": "<answer>"}

Guardrails: step limit, invalid-JSON recovery, repeated-call (loop) detection,
validated tool calls, and a full trace of every step.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .memory import SQLiteMemory, WorkingMemory
from .tools import Tool, call_tool, describe_tools

SYSTEM = """You are a data analyst agent for a customer-churn table.
Use the tools to answer. Never state numbers you did not get from a tool result.
Reply with exactly ONE JSON object per turn and nothing else:
  {{"thought": "short reason", "action": "<tool name>", "args": {{...}}}}
or, when you have the answer:
  {{"final": "<answer grounded in tool results>"}}
If no tool can answer the question, say so in a final answer.

Tools:
{tools}
"""


@dataclass
class RunResult:
    answer: str
    status: str  # ok | max_steps | error
    steps: int
    trace: list[dict] = field(default_factory=list)
    invalid_actions: int = 0
    tools_called: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class Agent:
    def __init__(self, llm, registry: dict[str, Tool], memory: SQLiteMemory | None = None, max_steps: int = 6):
        self.llm, self.registry, self.memory, self.max_steps = llm, registry, memory, max_steps
        self.system = SYSTEM.format(tools=describe_tools(registry))

    def run(self, question: str) -> RunResult:
        wm = WorkingMemory()
        context = ""
        if self.memory:
            prior = self.memory.recall(question)
            if prior:
                context = "PRIOR RUNS (may help, verify with tools):\n" + "\n".join(
                    f"- Q: {p['question']} | A: {p['answer'][:200]}" for p in prior) + "\n"
        wm.add("user", f"{context}QUESTION: {question}")

        res = RunResult(answer="", status="max_steps", steps=0)
        seen: set[str] = set()

        for step in range(1, self.max_steps + 1):
            res.steps = step
            try:
                raw = self.llm.complete(self.system, wm.view())
            except Exception as e:
                res.status, res.answer = "error", f"LLM call failed: {type(e).__name__}: {e}"
                break
            wm.add("assistant", raw)
            msg = _extract_json(raw)

            if msg is None or ("final" not in msg and "action" not in msg):
                res.invalid_actions += 1
                res.trace.append({"step": step, "type": "invalid", "raw": raw[:300]})
                wm.add("user", 'OBSERVATION: ERROR: reply with one JSON object: {"action":..., "args":...} or {"final":...}')
                continue

            if "final" in msg:
                res.answer, res.status = str(msg["final"]), "ok"
                res.trace.append({"step": step, "type": "final"})
                break

            name, args = msg.get("action"), msg.get("args", {})
            key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            if key in seen:
                res.invalid_actions += 1
                res.trace.append({"step": step, "type": "repeat", "action": name})
                wm.add("user", "OBSERVATION: ERROR: you already made this exact call. Use the earlier result or give a final answer.")
                continue
            seen.add(key)

            obs = call_tool(self.registry, str(name), args)
            res.tools_called.append(str(name))
            if obs.startswith("ERROR"):
                res.invalid_actions += 1
            res.trace.append({"step": step, "type": "tool", "action": name, "args": args, "observation": obs[:300]})
            wm.add("user", f"OBSERVATION: {obs}")

        if res.status == "max_steps":
            res.answer = "Stopped: step limit reached without a final answer."
        if self.memory:
            self.memory.save_run(question, res.answer, res.status, res.steps, res.trace)
        return res
