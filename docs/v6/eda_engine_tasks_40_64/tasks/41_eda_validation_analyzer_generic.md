# 41_eda_validation_analyzer_generic

## Goal

Build factual validation evidence using the generic ValidationPolicySelector.

This task replaces the previous temporal-first validation analyzer.

## Files to create/change

```text
kaggle_researcher/eda/modules/validation_analyzer.py
kaggle_researcher/eda/validation/policy_selector.py
tests/eda/test_validation_analyzer.py
```

## Codex prompt

```text
Implement generic validation_analyzer.

Function:
- analyze_validation(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    metric_evidence: MetricEvidence,
    reader: DatasetReader,
) -> ValidationEvidence

Requirements:
1. Load train_base table only as much as needed.
2. Detect and summarize:
   - target column availability
   - id column availability
   - candidate group/entity columns
   - candidate time/date columns
   - candidate query/ranking columns
3. If target exists:
   - for classification: class balance / target rate
   - for regression: target summary stats
   - for time columns: target by period as diagnostic
   - for group columns: target/group distribution as diagnostic
4. If test_base exists:
   - compare train/test time range when time columns exist
   - compare train/test group overlap when group columns exist
5. Call ValidationPolicySelector to produce:
   - primary validation
   - diagnostic validations
   - rejected validations
6. Temporal validation can be primary only when:
   - metric/task requires temporal validation, or
   - train/test relation indicates future test, or
   - scout hypothesis plus data evidence supports temporal split risk.
7. For ordinary iid classification:
   - recommend StratifiedKFold.
8. For ordinary iid regression:
   - recommend KFold.
9. For grouped tasks:
   - recommend GroupKFold or StratifiedGroupKFold.
10. For ranking tasks:
   - recommend group/query-aware validation.

Do not hard-code Home Credit as the default.
Home Credit behavior must emerge from metric_spec=gini_stability and detected WEEK_NUM.
```

## Acceptance criteria

- Fixture with gini_stability + WEEK_NUM recommends temporal validation.
- Binary iid fixture without temporal metric recommends StratifiedKFold even if a date column exists.
- Regression fixture recommends KFold.
- Grouped fixture recommends GroupKFold/StratifiedGroupKFold.
- Ranking fixture recommends query/group-aware validation.
- Validation evidence includes warnings/limitations when policy cannot be selected confidently.
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
