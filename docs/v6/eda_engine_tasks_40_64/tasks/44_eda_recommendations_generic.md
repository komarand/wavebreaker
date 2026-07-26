# 44_eda_recommendations_generic

## Goal

Build evidence-backed next actions without assuming temporal validation or Gini by default.

## Files to create/change

```text
kaggle_researcher/eda/modules/recommendations.py
tests/eda/test_eda_recommendations.py
```

## Codex prompt

```text
Implement generic recommended next actions builder.

Function:
- build_recommended_next_actions(
    evidence_pack_partial: dict,
    hypothesis_results: list[HypothesisResult],
) -> list[RecommendedNextAction]

Requirements:
- Generate actions from confirmed or partially_confirmed evidence only.
- Every action must include:
  - priority
  - action
  - why
  - evidence_refs
- Generic MVP rules:
  - If validation policy selected StratifiedKFold -> P0 action to use stratified CV.
  - If validation policy selected KFold -> P0 action to use KFold.
  - If validation policy selected GroupKFold/StratifiedGroupKFold -> P0 action to respect group split.
  - If validation policy selected temporal_holdout/expanding_window -> P0 action to use temporal validation.
  - If metric requires probabilities -> action to output probabilities/ranks, not hard labels.
  - If metric requires threshold -> action to tune threshold on validation only.
  - If metric requires calibration -> action to check calibration/clipping.
  - If metric is regression_error -> action to optimize regression loss and inspect target transform.
  - If leakage check warns/fails -> P0 action to fix/exclude unsafe source.
  - If secondary tables exist but relationship module not run -> P1 action to run relationship inference before aggregations.
- Do not invent actions without evidence_refs.
- Sort by priority.
```

## Acceptance criteria

- Binary iid fixture produces StratifiedKFold action.
- Home Credit-like fixture produces temporal validation action.
- F1 metric produces threshold tuning action.
- LogLoss metric produces calibration action.
- RMSE metric produces regression target/loss action.
- Every action has evidence_refs.
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
