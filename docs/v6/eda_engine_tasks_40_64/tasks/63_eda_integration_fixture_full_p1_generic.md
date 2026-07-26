# 63_eda_integration_fixture_full_p1_generic

## Goal

Add offline integration tests that run MVP + P1 modules on generic fixture data.

## Files to create/change

```text
tests/eda/test_eda_integration_full_p1.py
tests/fixtures/eda/
```

## Codex prompt

```text
Add full offline EDA integration tests.

Fixtures:
- home_credit_tiny
- iid_binary_tiny
- regression_tiny
- grouped_binary_tiny

Requirements:
- Run home_credit_tiny with:
  - MVP modules
  - relationship_inferer
  - drift_analyzer
  - feature_probe
- Run iid_binary_tiny and verify:
  - primary validation is StratifiedKFold
  - temporal validation is not forced
- Run regression_tiny and verify:
  - primary validation is KFold
  - regression metric evidence is used
- Run grouped_binary_tiny and verify:
  - group-aware validation is selected
- Do not require baseline unless fallback sklearn model is available and test is stable.
- Validate:
  - eda_evidence_pack.json exists.
  - hypothesis_results cover all input hypotheses.
  - recommended_next_actions is non-empty.
  - quality gates return no critical warnings.
```

## Acceptance criteria

- Tests run offline.
- Tests do not require Kaggle credentials.
- Tests do not require DeepSeek.
- Tests prove generic tabular behavior, not only Home Credit behavior.
- Tests pass in CI.

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
