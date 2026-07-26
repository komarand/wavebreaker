# 40_eda_validation_policy_and_split_helpers

## Goal

Implement generic validation policy helpers instead of assuming temporal validation by default.

This task replaces the previous temporal-only task.

The EDA Engine must support ordinary iid tabular tasks, grouped tasks, ranking tasks, temporal tasks, and forecasting-like tabular tasks.

A time column alone is **not sufficient** to make temporal validation primary.

## Files to create/change

```text
kaggle_researcher/eda/validation/
├── __init__.py
├── split_helpers.py
├── temporal_split.py
├── group_split.py
└── policy_selector.py

tests/eda/test_split_helpers.py
tests/eda/test_temporal_split.py
tests/eda/test_group_split.py
tests/eda/test_validation_policy_selector.py
```

## Codex prompt

```text
Implement generic validation policy helpers and a validation policy selector.

Use docs/EDA_ENGINE_SPEC.md as source of truth, but do not assume Home Credit or Gini Stability as the default case.

Implement split helper functions:

In validation/split_helpers.py:
- infer_class_balance(df, target_col: str) -> dict
- infer_regression_target_stats(df, target_col: str) -> dict
- infer_candidate_group_columns(schema, profiles) -> list[dict]
- infer_candidate_time_columns(schema, profiles) -> list[dict]
- summarize_column_distribution(df, col: str, target_col: str | None = None) -> list[dict]

In validation/temporal_split.py:
- infer_periods(df, time_col: str) -> list
- build_latest_period_holdout(periods: list, holdout_period_count: int = 4) -> dict
- build_expanding_window_folds(periods: list, n_folds: int = 5, min_train_periods: int = 3) -> list[dict]
- summarize_period_counts(df, time_col: str, target_col: str | None = None) -> list[dict]

In validation/group_split.py:
- summarize_group_counts(df, group_col: str, target_col: str | None = None) -> dict
- assess_group_split_feasibility(df, group_col: str, target_col: str | None = None) -> dict
- detect_group_leakage_risk(train_df, test_df, group_col: str) -> dict

In validation/policy_selector.py:
- select_validation_policy(
    task_type,
    metric_spec,
    inferred_schema,
    table_profiles,
    validation_signals: dict | None = None,
    scout_hypotheses: list | None = None,
) -> dict

Validation policies to support:
- stratified_kfold
- kfold
- group_kfold
- stratified_group_kfold
- temporal_holdout
- expanding_window
- ranking_group_cv
- custom_required

Policy selection rules:
- Binary/multiclass iid classification -> stratified_kfold.
- Regression iid -> kfold.
- Ranking/recommender with query_id/session_id/user_id -> ranking_group_cv or group_kfold.
- Group/entity leakage risk -> group_kfold or stratified_group_kfold.
- Forecasting or temporal/stability metric -> temporal_holdout or expanding_window.
- Time column alone is diagnostic, not sufficient for primary temporal validation.
- Unknown/custom metric -> conservative validation with warning and custom_required if needed.

The selector output must include:
- primary_validation
- diagnostic_validations
- rejected_validations
- confidence
- evidence_refs
- warnings
- limitations
- reasoning_summary

Do not import sklearn in split helper modules unless absolutely necessary.
Do not implement model training.
```

## Acceptance criteria

- iid binary classification with no group/time requirement selects `stratified_kfold`.
- iid regression selects `kfold`.
- ranking metric with query/group column selects `ranking_group_cv`.
- temporal/stability metric selects temporal policy when time column exists.
- time column alone does not force temporal primary validation.
- too few periods returns infeasible temporal policy with reason.
- group split feasibility is tested.
- Tests pass with:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests/eda -q
```

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
