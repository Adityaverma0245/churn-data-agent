# churn-data-agent

A small, readable tool-using agent over tabular customer data, with a planner loop, validated tools, two layers of memory and an evaluation harness. Built as a learning and prototyping project for agentic AI. It is not a production system.

## What it does

Ask questions about a customer-churn table in plain English. The agent decides which tool to call, reads the result, and answers from tool output only.

```
$ python -m churn_agent.cli "Compare churn rate by Contract" --trace
```

## Architecture

```
question -> [Agent loop] -> LLM (JSON action) -> validated tool call -> observation -> ... -> final answer
                |                                        |
         WorkingMemory (bounded)                  DataTools (pandas + scikit-learn)
                |
         SQLiteMemory (past runs, recalled by keyword overlap)
```

| Part | File | What it does |
|---|---|---|
| Reasoning loop | `churn_agent/agent.py` | think -> act -> observe -> repeat, max steps, invalid-JSON recovery, repeat-call detection, full trace |
| Tools | `churn_agent/tools.py` | `describe_dataset`, `churn_rate_by`, `top_churn_drivers`, `predict_churn`; every call is schema-validated and errors come back as observations, never crashes |
| Memory | `churn_agent/memory.py` | WorkingMemory (pins the question, truncates observations, drops old turns) and SQLiteMemory (saves runs, recalls similar ones) |
| LLM backends | `churn_agent/llm.py` | `AnthropicLLM` (real model) and `RuleBasedPlanner` (offline keyword stand-in for tests and demos) |
| Evaluation | `evals/run_evals.py` | task success, tool-selection accuracy, average steps, invalid-action rate, with a per-case failure reason |

## Quick start

```bash
pip install -r requirements.txt
python data/make_sample_data.py > data/sample_churn.csv   # synthetic data
python -m churn_agent.cli "Which factors drive churn the most?"
python -m unittest discover -s tests -v                  # 15 tests
python evals/run_evals.py                                 # writes evals/results.md
```

To use a real model: `pip install anthropic`, set `ANTHROPIC_API_KEY`, then add `--llm anthropic` to the CLI or the eval runner. To use the real Telco Customer Churn dataset, pass `--data path/to/file.csv` (it needs a Yes/No `Churn` column).

## Evaluation

Each case in `evals/cases.json` has a question, the tool it should use (or none, for out-of-scope questions) and strings the answer must contain. A case passes only if the status is ok, the expected content is present and the right tool (or no tool) was used. Failures print the reason so you can fix the prompt, tools or validation and re-run.

## Honest limitations

- The included eval results use the **rule-based planner on synthetic data**. That planner was written alongside the test cases, so its 100% score shows the harness and loop work, not that an LLM agent is accurate. Run `--llm anthropic` for a meaningful measurement.
- The Anthropic backend is a thin wrapper and has not been load-tested.
- Memory recall is keyword overlap, not embeddings.
- Single-user, single-process, no authentication. Not deployed anywhere.

## Ideas for next steps

- Run the eval suite with a real model and add harder cases (multi-step questions, ambiguous column names).
- Add a tool-call budget and per-step latency and cost logging.
- Replace keyword recall with embedding search.
- Compare planners (rule-based vs. LLM vs. different prompts) on the same cases.

## License

MIT
