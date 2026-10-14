"""Evaluation harness.

For every case it runs the agent and scores:
  task_success   : status is ok AND the answer contains every expected string
  tool_correct   : the expected tool was called (or NO tool, for out-of-scope cases)
  invalid_rate   : share of steps that were malformed, repeated or errored
  avg_steps      : average loop iterations
Success = task_success. A case FAILS if either check fails; failures are listed
with the reason so you can fix the prompt, tools or validation, then re-run.

Usage: python evals/run_evals.py [--llm rule|anthropic] [--data data/sample_churn.csv]
Writes evals/results.json and evals/results.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churn_agent.cli import build_agent  # noqa: E402


def score_case(case: dict, res) -> dict:
    answer_ok = res.status == "ok" and all(s.lower() in res.answer.lower() for s in case["expect_contains"])
    exp = case["expect_tool"]
    tool_ok = (exp in res.tools_called) if exp else (len(res.tools_called) == 0)
    reasons = []
    if res.status != "ok":
        reasons.append(f"status={res.status}")
    if not answer_ok and res.status == "ok":
        reasons.append("answer missing expected content")
    if not tool_ok:
        reasons.append(f"tools called {res.tools_called}, expected {exp}")
    return {"id": case["id"], "success": answer_ok and tool_ok, "answer_ok": answer_ok, "tool_ok": tool_ok,
            "steps": res.steps, "invalid_actions": res.invalid_actions, "reasons": reasons,
            "answer": res.answer[:200]}


def run(llm: str, data: str) -> dict:
    cases = json.loads((ROOT / "evals" / "cases.json").read_text())
    rows = []
    for c in cases:
        agent = build_agent(data, llm)  # fresh agent and memory per case: no leakage between cases
        rows.append(score_case(c, agent.run(c["question"])))
    n = len(rows)
    total_steps = sum(r["steps"] for r in rows)
    summary = {
        "cases": n,
        "task_success_rate": round(sum(r["success"] for r in rows) / n, 3),
        "tool_selection_accuracy": round(sum(r["tool_ok"] for r in rows) / n, 3),
        "avg_steps": round(total_steps / n, 2),
        "invalid_action_rate": round(sum(r["invalid_actions"] for r in rows) / max(total_steps, 1), 3),
    }
    return {"llm": llm, "summary": summary, "cases": rows}


def to_markdown(out: dict) -> str:
    s = out["summary"]
    lines = [f"# Eval results ({out['llm']})", "",
             f"- Task success rate: **{s['task_success_rate']}**",
             f"- Tool selection accuracy: **{s['tool_selection_accuracy']}**",
             f"- Average steps: **{s['avg_steps']}**",
             f"- Invalid action rate: **{s['invalid_action_rate']}**", "",
             "| case | success | steps | notes |", "|---|---|---|---|"]
    for r in out["cases"]:
        lines.append(f"| {r['id']} | {'PASS' if r['success'] else 'FAIL'} | {r['steps']} | {'; '.join(r['reasons'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["rule", "anthropic"], default="rule")
    ap.add_argument("--data", default=str(ROOT / "data" / "sample_churn.csv"))
    a = ap.parse_args()
    out = run(a.llm, a.data)
    (ROOT / "evals" / "results.json").write_text(json.dumps(out, indent=2))
    md = to_markdown(out)
    (ROOT / "evals" / "results.md").write_text(md)
    print(md)
