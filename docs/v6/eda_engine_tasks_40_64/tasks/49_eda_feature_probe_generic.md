# 49_eda_feature_probe_generic

## Goal

Assess promising feature families across generic tabular tasks.

## Files to create/change

```text
kaggle_researcher/eda/modules/feature_probe.py
tests/eda/test_feature_probe.py
```

## Codex prompt

```text
Implement generic feature_probe module.

Function:
- probe_feature_families(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    relationship_evidence: dict,
    leakage_evidence: list[LeakageCheckResult],
    baseline_evidence: dict,
    metric_evidence: dict | None = None,
) -> list[dict]

Feature families:
- base_numeric_features
- base_categorical_features
- missingness_indicators
- date_features
- secondary_table_aggregations
- high_cardinality_encoding
- target_encoding_or_woe
- monotonic_or_binning_features
- ranking_group_features
- regression_target_transform

Requirements:
- Return:
  - feature_family
  - status: high_potential|medium_potential|low_potential|unsafe|not_testable
  - leakage_risk: low|medium|high
  - evidence
  - recommendation
- Target encoding / WoE is high leakage risk unless OOF/group/time-safe policy exists.
- Secondary aggregations need relationship evidence.
- Regression target transform is relevant only for regression metrics such as RMSLE/RMSE with skewed target.
- Ranking group features are relevant only for ranking tasks.
- Do not generate actual feature engineering code.
```

## Acceptance criteria

- Secondary aggregations are not_testable before relationship evidence.
- With relationship evidence, secondary aggregations become medium/high potential.
- Target encoding is high leakage risk.
- Regression fixture can recommend target transform if target is skewed.
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
