# 58_full_research_to_eda_to_strategy_cli_generic

## Goal

Add an optional full workflow that runs research, writes Scout outputs, runs EDA, and synthesizes final strategy.

## Files to create/change

```text
kaggle_researcher/main.py
kaggle_researcher/eda/orchestrator.py
tests/test_full_research_eda_strategy_mocked.py
```

## Codex prompt

```text
Add optional full workflow.

CLI flags:
- --write-eda-plan
- --run-eda
- --local-dataset-path
- --eda-output-dir
- --final-synthesis

Behavior:
- Existing research-only mode remains default.
- If --write-eda-plan:
  - write Research Scout outputs.
- If --run-eda:
  - require either --local-dataset-path or dataset download config.
  - run EDA Engine using generated or provided hypotheses/task plan.
- If --final-synthesis:
  - require eda_evidence_pack from current or provided run.
  - run final_synthesizer.
  - write final_strategy.json/md/docx where practical.
- All external calls must be mockable.
- Default tests must use local fixture dataset and mocked DeepSeek.
- The workflow must work for generic tabular fixtures, not only Home Credit.
```

## Acceptance criteria

- Research-only mode still passes existing tests.
- Full mocked workflow produces:
  - research_run.json
  - research_hypotheses.json
  - eda_task_plan.json
  - eda_evidence_pack.json
  - final_strategy.json
- No real network calls in tests.
- Tests pass.

---

## Rules inherited by this task

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
