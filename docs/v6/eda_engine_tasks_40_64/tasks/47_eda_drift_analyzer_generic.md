# 47_eda_drift_analyzer_generic

## Goal

Analyze drift as optional evidence, not as a universal assumption.

## Files to create/change

```text
kaggle_researcher/eda/modules/drift_analyzer.py
tests/eda/test_drift_analyzer.py
```

## Codex prompt

```text
Implement generic drift_analyzer.

Function:
- analyze_drift(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
    max_rows: int = 500000,
    random_seed: int = 42,
) -> dict

Requirements:
- Compute target drift by period only when target and time exist.
- Compute row count by period when time exists.
- Compute missingness drift train vs test for shared columns.
- Compute numeric PSI for shared numeric columns.
- Compute categorical distribution shift for shared categorical columns.
- Implement adversarial validation only if sklearn is available:
  - safe features only
  - exclude target, id, prediction, query/group columns unless explicitly allowed
  - cap rows at max_rows
  - return AUC and top features if available
- If no test table exists, provide train-only drift diagnostics where possible.
- If no time columns exist, skip temporal drift with limitation.
- Drift evidence should influence validation only later or as diagnostic, not retroactively override generic policy unless explicitly wired.
```

## Acceptance criteria

- Fixture with time column returns target_drift.
- Fixture without time column skips temporal drift with limitation.
- Artificially shifted train/test fixture produces higher drift severity.
- target/id columns are excluded from adversarial features.
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
