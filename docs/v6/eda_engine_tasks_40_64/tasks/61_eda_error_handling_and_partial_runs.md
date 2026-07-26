# 61_eda_error_handling_and_partial_runs

## Goal

Make partial EDA runs reproducible and useful even when some modules fail.

## Files to create/change

```text
kaggle_researcher/eda/orchestrator.py
kaggle_researcher/eda/io/artifact_writer.py
tests/eda/test_eda_partial_runs.py
```

## Codex prompt

```text
Improve EDA error handling.

Requirements:
- Define module status object:
  - module
  - status: success|failed|skipped
  - started_at
  - finished_at
  - duration_sec
  - error_message
- Write module_statuses.json.
- If a non-blocking module fails:
  - write failed placeholder JSON
  - continue
- If a blocking module fails:
  - write partial evidence_pack if possible
  - fail run unless fail_fast=false and fallback output exists
- EdaRunResult should include module_statuses.
- All exceptions should be sanitized:
  - no secrets
  - no huge stack traces in JSON by default
- Partial packs must not contain unsupported conclusions.
```

## Acceptance criteria

- Simulated P1 failure still creates evidence pack.
- Simulated blocking failure writes partial artifacts.
- module_statuses.json is written.
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
