# 50_eda_notebook_static_analysis_generic

## Goal

Statically extract patterns from notebook source text without executing notebooks.

## Files to create/change

```text
kaggle_researcher/eda/modules/notebook_static_analyzer.py
tests/eda/test_notebook_static_analyzer.py
```

## Codex prompt

```text
Implement generic notebook_static_analyzer.

Function:
- analyze_notebooks_static(
    notebook_sources: list[SourceDocument | RetrievedDocument | dict],
    output_dir: Path | None = None,
) -> dict

Requirements:
- Static text/code pattern extraction only.
- Do not execute notebook code.
- Extract patterns:
  - cv_strategy
  - feature_families
  - model_families
  - metric_code
  - postprocessing
  - suspicious_leaderboard_overfit_patterns
- Detect common strings:
  - KFold
  - StratifiedKFold
  - GroupKFold
  - StratifiedGroupKFold
  - TimeSeriesSplit
  - LightGBM, CatBoost, XGBoost
  - target encoding
  - adversarial validation
  - rank averaging
  - clipping
  - threshold tuning
  - logloss calibration
  - RMSLE target transform
  - QWK threshold optimization
- Notebook scores are observations, not truth.
- Warnings should be contextual to task/metric when available.
```

## Acceptance criteria

- Static fixture detects model and CV patterns.
- No code execution occurs.
- Metric-specific patterns are extracted.
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
