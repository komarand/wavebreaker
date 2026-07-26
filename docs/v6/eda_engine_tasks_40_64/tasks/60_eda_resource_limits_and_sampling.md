# 60_eda_resource_limits_and_sampling

## Goal

Centralize row caps, memory-safe sampling, and module runtime limits.

## Files to create/change

```text
kaggle_researcher/eda/io/dataset_reader.py
kaggle_researcher/eda/config.py
kaggle_researcher/eda/modules/table_profiler.py
kaggle_researcher/eda/modules/drift_analyzer.py
tests/eda/test_eda_resource_limits.py
```

## Codex prompt

```text
Add resource limit handling.

Requirements:
- Add settings:
  - EDA_MAX_TABLE_BYTES
  - EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS
  - EDA_MODULE_TIMEOUT_SEC
- DatasetReader should expose lightweight file size info.
- Profiling should automatically sample if file size or row count exceeds caps.
- Drift/adversarial validation should cap train/test rows.
- Every sampled/capped result must include:
  - sampled=true
  - sample_rows
  - limitation/warning
- Sampling behavior must be generic and task-independent.
- Do not use OS-specific memory APIs unless optional.
```

## Acceptance criteria

- Low caps force sampling in tests.
- sampled=true appears in relevant outputs.
- Warnings mention caps.
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
