# 48_eda_baseline_runner_generic

## Goal

Run an honest baseline appropriate to task_type and metric family.

## Files to create/change

```text
kaggle_researcher/eda/modules/baseline_runner.py
tests/eda/test_baseline_runner.py
requirements.txt
```

## Codex prompt

```text
Implement generic baseline_runner.

Function:
- run_baseline(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    metric_evidence: MetricEvidence,
    leakage_evidence: list[LeakageCheckResult],
    reader: DatasetReader,
    output_dir: Path,
    max_rows: int = 1000000,
    random_seed: int = 42,
) -> dict

Requirements:
- Baseline is optional and must be enabled by orchestrator flag.
- Use train_base only.
- Exclude:
  - target
  - id columns
  - raw date strings unless encoded safely
  - prediction/sample submission columns
  - columns flagged by critical leakage checks
  - query/group columns when they define validation split and should not be features
- Choose model by task_type:
  - binary_classification -> classifier
  - multiclass_classification -> classifier
  - regression -> regressor
  - ranking -> skipped/not_testable in MVP unless query-aware baseline exists
  - survival -> skipped/not_testable in MVP
  - forecasting_tabular -> simple time-aware baseline later; skipped in MVP if unsupported
- Model preference:
  - LightGBM if installed.
  - sklearn HistGradientBoostingClassifier/Regressor fallback.
  - LogisticRegression/LinearRegression fallback if needed.
- Basic preprocessing:
  - numeric fill missing
  - categorical encoding fit on train fold only
- Use validation_evidence selected policy.
- Compute metric using MetricRegistry/local metric when available.
- If local metric unavailable, train baseline can still run but metric is skipped with warning.
- Do not train on test.
- Do not use target encoding.
- Do not optimize leaderboard score.
```

## Acceptance criteria

- Binary classification fixture baseline runs.
- Regression fixture baseline runs.
- Ranking/survival fixture returns skipped/not_testable, not failure.
- target/id columns are not in feature list.
- Baseline uses selected validation policy.
- Tests pass without requiring LightGBM.

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
