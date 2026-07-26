# Global rules for EDA Engine tasks 40–64

These rules apply to every task in this archive.

The EDA Engine is a **generic tabular evidence engine**.

It must support Home Credit-like competitions, but Home Credit must not define the architecture.

Important rules:

```text
- Generic logic first.
- MetricRegistry determines metric requirements.
- ValidationPolicySelector determines validation policy.
- Competition presets may provide hints, but they must not be required.
- Gini Stability is only one metric registry entry.
- A time column alone must never force temporal validation.
- Home Credit-specific behavior must emerge from metric/schema/preset evidence.
- Notebook execution remains forbidden.
```

The old mental model:

```text
WEEK_NUM exists -> temporal validation
Gini Stability -> default metric worldview
case_id -> default join key
```

must be replaced with:

```text
task_type + metric_family + schema evidence + data signals + scout hypotheses
        -> validation policy
        -> evidence-backed recommendations
```

## Recommended Codex steering prompt before Task 40

Use this prompt before continuing:

```text
Do not continue with the old task 40+ wording.

We have replaced Task 39 with a generic MetricRegistry.

From Task 40 onward, use docs/EDA_ENGINE_CODEX_TASKS_40_PLUS_GENERIC.md as the source of truth.

Important:
- A time column alone must never force temporal validation.
- Gini Stability is supported, but it is only one metric registry entry.
- Home Credit-specific behavior must come from metric/preset/schema evidence, not from global defaults.
- Implement only the next requested task.
- Do not implement future tasks.
- Do not execute Kaggle notebooks.

Use the project-local Windows virtual environment:

E:\wavebreaker\.venv-win\Scripts\python.exe

Run tests with:

E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests/eda -q
```
