# 42_eda_leakage_checker_generic

## Goal

Implement generic leakage checks for tabular competitions, not only Home Credit-like datasets.

## Files to create/change

```text
kaggle_researcher/eda/modules/leakage_checker.py
tests/eda/test_leakage_checker.py
```

## Codex prompt

```text
Implement generic leakage_checker.

Function:
- check_leakage(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
) -> list[LeakageCheckResult]

MVP checks:
- train/test id overlap
- train/test group/entity overlap where group columns exist
- target column present in test
- target-like column names outside target
- sample_submission structure sanity
- duplicate rows across train/test for base tables where feasible
- suspicious numeric columns with extremely high target association
- potential post-target/future-date risk when time/date columns exist
- ranking/query leakage risk when query/group identifiers appear in both train/test

Requirements:
- Confirmed leakage requires direct evidence.
- Suspicious association is warning, not proof.
- Time/date presence alone is not leakage.
- Group overlap is not always leakage; severity depends on selected validation policy.
- Missing required tables/columns should return not_testable checks.
- Must not scan huge tables without sampling/caps.
```

## Acceptance criteria

- Default fixture has passed id overlap check.
- Modified fixture with overlapping id detects overlap.
- Test target column is high/critical severity.
- Group overlap is reported with contextual severity.
- Missing id returns not_testable.
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
