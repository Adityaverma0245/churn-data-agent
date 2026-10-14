import json
import pathlib
import sys
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churn_agent import Agent, DataTools, SQLiteMemory, WorkingMemory, build_registry  # noqa: E402
from churn_agent.tools import call_tool  # noqa: E402

DF = pd.read_csv(ROOT / "data" / "sample_churn.csv")


class Scripted:
    """Fake LLM that replays a fixed list of replies."""
    def __init__(self, replies): self.replies, self.i = replies, 0
    def complete(self, system, messages):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r


def agent(replies, **kw):
    return Agent(Scripted(replies), build_registry(DataTools(DF)), **kw)


class ToolValidation(unittest.TestCase):
    def setUp(self): self.reg = build_registry(DataTools(DF))

    def test_unknown_tool(self): self.assertTrue(call_tool(self.reg, "nope", {}).startswith("ERROR"))
    def test_missing_arg(self): self.assertIn("missing", call_tool(self.reg, "churn_rate_by", {}))
    def test_extra_arg(self): self.assertIn("unexpected", call_tool(self.reg, "describe_dataset", {"x": 1}))
    def test_wrong_type(self): self.assertIn("must be", call_tool(self.reg, "churn_rate_by", {"column": 5}))
    def test_bad_column_is_observation_not_crash(self):
        self.assertTrue(call_tool(self.reg, "churn_rate_by", {"column": "Salary"}).startswith("ERROR"))
    def test_int_coercion(self): self.assertIn("top_features", call_tool(self.reg, "top_churn_drivers", {"n": "3"}))
    def test_predict_returns_probability(self):
        out = json.loads(call_tool(self.reg, "predict_churn", {"customer": {"tenure": 2}}))
        self.assertTrue(0 <= out["churn_probability"] <= 1)


class Loop(unittest.TestCase):
    def test_tool_then_final(self):
        r = agent(['{"action":"describe_dataset","args":{}}', '{"final":"2000 rows"}']).run("overview")
        self.assertEqual((r.status, r.steps, r.tools_called), ("ok", 2, ["describe_dataset"]))

    def test_recovers_from_invalid_json(self):
        r = agent(["not json at all", '{"final":"done"}']).run("q")
        self.assertEqual((r.status, r.invalid_actions), ("ok", 1))

    def test_repeat_call_detected(self):
        call = '{"action":"describe_dataset","args":{}}'
        r = agent([call, call, '{"final":"ok"}']).run("q")
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.tools_called.count("describe_dataset"), 1)
        self.assertGreaterEqual(r.invalid_actions, 1)

    def test_step_limit_stops_runaway(self):
        r = agent(["garbage"], max_steps=3).run("q")
        self.assertEqual((r.status, r.steps), ("max_steps", 3))

    def test_llm_failure_is_handled(self):
        class Boom:
            def complete(self, s, m): raise TimeoutError("slow")
        r = Agent(Boom(), build_registry(DataTools(DF))).run("q")
        self.assertEqual(r.status, "error")


class Memory(unittest.TestCase):
    def test_working_memory_bounds_context_and_pins_question(self):
        wm = WorkingMemory(keep_last=4)
        for i in range(20): wm.add("user", f"m{i}")
        v = wm.view()
        self.assertEqual(len(v), 5)
        self.assertEqual(v[0]["content"], "m0")

    def test_long_observation_truncated(self):
        wm = WorkingMemory(max_obs_chars=50)
        wm.add("user", "OBSERVATION: " + "x" * 500)
        self.assertIn("truncated", wm.messages[0]["content"])

    def test_persistent_recall_across_runs(self):
        mem = SQLiteMemory()
        a = agent(['{"final":"Month-to-month churns most"}'], memory=mem)
        a.run("which contract has most churn")
        self.assertEqual(len(mem.recall("contract churn question")), 1)
        self.assertEqual(mem.recall("completely unrelated xyz"), [])


if __name__ == "__main__":
    unittest.main()
