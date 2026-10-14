"""Memory in two layers.

WorkingMemory: the message list for ONE run. It keeps the original question
pinned, truncates long observations, and drops old middle turns so the
context stays bounded.

SQLiteMemory: persistent storage across runs. It saves each run (question,
answer, status, trace) and recalls similar earlier runs by keyword overlap.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time


class WorkingMemory:
    def __init__(self, max_obs_chars: int = 1500, keep_last: int = 8):
        self.messages: list[dict] = []
        self.max_obs_chars = max_obs_chars
        self.keep_last = keep_last

    def add(self, role: str, content: str) -> None:
        if content.startswith("OBSERVATION:") and len(content) > self.max_obs_chars:
            content = content[: self.max_obs_chars] + " ...[truncated]"
        self.messages.append({"role": role, "content": content})

    def view(self) -> list[dict]:
        """First message (question + context) is pinned; older middle turns are dropped."""
        if len(self.messages) <= self.keep_last + 1:
            return list(self.messages)
        return [self.messages[0]] + self.messages[-self.keep_last:]


_WORD = re.compile(r"[a-z0-9_]+")


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


class SQLiteMemory:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, ts REAL, question TEXT, "
            "answer TEXT, status TEXT, steps INTEGER, trace TEXT)"
        )
        self.db.commit()

    def save_run(self, question: str, answer: str, status: str, steps: int, trace: list) -> None:
        self.db.execute(
            "INSERT INTO runs (ts, question, answer, status, steps, trace) VALUES (?,?,?,?,?,?)",
            (time.time(), question, answer, status, steps, json.dumps(trace)),
        )
        self.db.commit()

    def recall(self, question: str, k: int = 2) -> list[dict]:
        """Return up to k earlier successful runs that share words with this question."""
        q = _words(question)
        scored = []
        for rid, q2, a2 in self.db.execute("SELECT id, question, answer FROM runs WHERE status='ok'"):
            overlap = len(q & _words(q2))
            if overlap:
                scored.append((overlap, rid, q2, a2))
        scored.sort(reverse=True)
        return [{"question": q2, "answer": a2} for _, _, q2, a2 in scored[:k]]
