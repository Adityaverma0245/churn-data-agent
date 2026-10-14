"""Tools the agent can call, plus a registry that validates every call.

Design rule: a tool never raises into the agent loop. Bad input or a failing
tool returns an error string, so the model can read it and recover.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TARGET = "Churn"


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # param name -> "str" | "int" | "dict"
    fn: Callable[..., Any]
    required: tuple[str, ...] = ()


class DataTools:
    """Analytics tools over a customer table with a Yes/No `Churn` column."""

    def __init__(self, df: pd.DataFrame, seed: int = 42):
        if TARGET not in df.columns:
            raise ValueError(f"DataFrame needs a '{TARGET}' column")
        self.df = df.copy()
        self.df[TARGET] = self.df[TARGET].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)
        self.seed = seed
        self._model = None
        self._columns: list[str] | None = None
        self._auc: float | None = None

    # ---- tools -------------------------------------------------------
    def describe_dataset(self) -> str:
        d = self.df
        return json.dumps({
            "rows": int(len(d)),
            "columns": [c for c in d.columns if c != TARGET],
            "churn_rate": round(float(d[TARGET].mean()), 4),
        })

    def churn_rate_by(self, column: str) -> str:
        if column not in self.df.columns or column == TARGET:
            return f"ERROR: unknown column '{column}'. Valid: {[c for c in self.df.columns if c != TARGET]}"
        s = self.df[column]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            s = pd.qcut(s, 4, duplicates="drop").astype(str)
        out = self.df.groupby(s.astype(str))[TARGET].agg(["mean", "count"]).round(4)
        return json.dumps({k: {"churn_rate": float(v["mean"]), "count": int(v["count"])} for k, v in out.iterrows()})

    def top_churn_drivers(self, n: int = 5) -> str:
        self._fit()
        imp = pd.Series(self._model.feature_importances_, index=self._columns).sort_values(ascending=False)
        top = {k: round(float(v), 4) for k, v in imp.head(int(n)).items()}
        return json.dumps({"top_features": top, "holdout_ROC-AUC": round(self._auc, 4)})

    def predict_churn(self, customer: dict) -> str:
        if not isinstance(customer, dict):
            return "ERROR: 'customer' must be an object of feature values"
        self._fit()
        base = {}
        for c in self.df.columns:
            if c == TARGET:
                continue
            s = self.df[c]
            base[c] = s.median() if pd.api.types.is_numeric_dtype(s) else s.mode().iloc[0]
        unknown = [k for k in customer if k not in base]
        if unknown:
            return f"ERROR: unknown features {unknown}. Valid: {list(base)}"
        base.update(customer)
        row = pd.get_dummies(pd.DataFrame([base])).reindex(columns=self._columns, fill_value=0)
        p = float(self._model.predict_proba(row)[0, 1])
        return json.dumps({"churn_probability": round(p, 4), "features_used": base}, default=float)

    # ---- internals ---------------------------------------------------
    def _fit(self) -> None:
        if self._model is not None:
            return
        X = pd.get_dummies(self.df.drop(columns=[TARGET]))
        y = self.df[TARGET]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=self.seed)
        m = RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(Xtr, ytr)
        self._auc = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        self._model, self._columns = m, list(X.columns)


def build_registry(dt: DataTools) -> dict[str, Tool]:
    tools = [
        Tool("describe_dataset", "Row count, column names and overall churn rate.", {}, dt.describe_dataset),
        Tool("churn_rate_by", "Churn rate and customer count for each value of one column.",
             {"column": "str"}, dt.churn_rate_by, required=("column",)),
        Tool("top_churn_drivers", "Train a Random Forest and return the top-n most important features and holdout ROC-AUC.",
             {"n": "int"}, dt.top_churn_drivers),
        Tool("predict_churn", "Churn probability for one customer. 'customer' is an object of feature values; omitted features use dataset defaults.",
             {"customer": "dict"}, dt.predict_churn, required=("customer",)),
    ]
    return {t.name: t for t in tools}


_TYPES = {"str": str, "int": int, "dict": dict}


def call_tool(registry: dict[str, Tool], name: str, args: Any) -> str:
    """Validate and run a tool call. Always returns a string observation."""
    if name not in registry:
        return f"ERROR: unknown tool '{name}'. Available: {sorted(registry)}"
    tool = registry[name]
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return "ERROR: args must be a JSON object"
    args = dict(args)
    extra = set(args) - set(tool.params)
    if extra:
        return f"ERROR: unexpected args {sorted(extra)} for '{name}'. Expected: {tool.params}"
    missing = [p for p in tool.required if p not in args]
    if missing:
        return f"ERROR: missing required args {missing} for '{name}'"
    for k, v in list(args.items()):
        want = _TYPES[tool.params[k]]
        if want is int and isinstance(v, str) and v.lstrip("-").isdigit():
            args[k] = int(v)
        elif not isinstance(args[k], want) or isinstance(args[k], bool):
            return f"ERROR: arg '{k}' must be {tool.params[k]}"
    try:
        return str(tool.fn(**args))
    except Exception as e:  # tool failure becomes an observation, not a crash
        return f"ERROR: tool '{name}' failed: {type(e).__name__}: {e}"


def describe_tools(registry: dict[str, Tool]) -> str:
    return "\n".join(f"- {t.name}({json.dumps(t.params)}): {t.description}" for t in registry.values())
