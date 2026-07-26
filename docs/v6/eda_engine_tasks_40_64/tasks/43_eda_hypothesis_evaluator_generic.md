# 43_eda_hypothesis_evaluator_generic

## Goal

Evaluate Research Scout hypotheses against generic EDA evidence.

## Files to create/change

```text
kaggle_researcher/eda/modules/hypothesis_evaluator.py
tests/eda/test_hypothesis_evaluator.py
```

## Codex prompt

```text
Implement or refactor hypothesis_evaluator to be generic.

Function:
- evaluate_hypotheses(
    hypotheses: list[ResearchHypothesis],
    evidence_pack_partial: dict,
    module_statuses: dict[str, str] | None = None,
) -> list[HypothesisResult]

Requirements:
- Every input hypothesis must produce exactly one result.
- Supported categories:
  - schema
  - metric
  - validation
  - leakage
  - relationship
  - drift
  - baseline
  - feature
  - notebook
  - data_quality
- Do not assume Home Credit hypothesis IDs except in fixtures.
- For MVP, provide deterministic evaluators by category:
  - schema -> inferred_schema evidence
  - metric -> metric_evidence
  - validation -> validation_evidence.primary/diagnostic policies
  - leakage -> leakage_evidence
- Unknown category -> not_testable with limitation.
- skipped module -> skipped hypothesis.
- confirmed/rejected must include evidence_refs.
- not_testable/skipped must include limitations.
- impact_on_strategy must be concrete.

Important:
- A validation hypothesis saying "temporal CV is required" should be confirmed only if validation evidence selected temporal validation as primary or strong diagnostic.
- If a time column exists but temporal policy was rejected as primary, the hypothesis should be partially_confirmed or rejected depending on wording.
```

## Acceptance criteria

- All fixture hypotheses are evaluated.
- Binary iid validation hypothesis does not get temporal confirmation merely from date column.
- confirmed results include evidence_refs.
- skipped/not_testable include limitations.
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
