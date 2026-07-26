# 52_eda_p1_hypothesis_and_recommendation_rules_generic

## Goal

Extend hypothesis evaluation and recommendations to use generic P1 evidence.

## Files to create/change

```text
kaggle_researcher/eda/modules/hypothesis_evaluator.py
kaggle_researcher/eda/modules/recommendations.py
tests/eda/test_hypothesis_evaluator_p1.py
tests/eda/test_eda_recommendations_p1.py
```

## Codex prompt

```text
Extend hypothesis evaluator and recommended actions for P1 evidence.

Rules:
- Relationship hypotheses:
  - confirmed when join keys and coverage are found.
  - partially_confirmed when only weak candidate keys exist.
- Drift hypotheses:
  - confirmed when drift severity is medium/high for relevant drift type.
  - rejected when drift checks show stable distributions.
  - not_testable when test/shared columns unavailable.
- Baseline hypotheses:
  - confirmed when honest baseline completed for supported task_type.
  - skipped when baseline disabled.
  - not_testable for unsupported task types.
- Feature hypotheses:
  - confirmed or partially_confirmed from feature_probe statuses.
- Notebook hypotheses:
  - confirmed only as "pattern observed", not as factual performance proof.

Recommendations:
- one-to-many relationship -> aggregate before join.
- high drift -> use selected robust validation and avoid public LB overfitting.
- baseline complete -> use as sanity floor, not final solution.
- high-potential feature family -> prioritize P1/P2 experiment.
- risky notebook pattern -> audit before copying.
- threshold-sensitive metric -> validate threshold tuning policy.
- calibration-sensitive metric -> check calibration/clipping.
- regression skew -> consider target transform if evidence supports it.
```

## Acceptance criteria

- P1 hypotheses are evaluated when evidence exists.
- Baseline disabled produces skipped, not failure.
- Drift recommendation cites drift evidence.
- Relationship recommendation cites relationship evidence.
- Metric-specific recommendations cite metric evidence.
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
