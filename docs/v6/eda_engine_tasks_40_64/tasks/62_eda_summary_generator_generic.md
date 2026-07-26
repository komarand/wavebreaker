# 62_eda_summary_generator_generic

## Goal

Generate a concise human-readable `eda_summary.md` from evidence pack without adding unsupported claims.

## Files to create/change

```text
kaggle_researcher/eda/summary.py
kaggle_researcher/eda/orchestrator.py
tests/eda/test_eda_summary.py
```

## Codex prompt

```text
Implement EDA summary generator.

Function:
- build_eda_summary(pack: EdaEvidencePack) -> str

Required sections:
- Dataset
- Schema
- Metric
- Validation
- Leakage
- Relationships
- Drift
- Baseline
- Feature probes
- Hypothesis results
- Recommended next actions
- Warnings
- Limitations

Requirements:
- Summary is derived only from EdaEvidencePack.
- Include evidence_refs in bullet text where useful.
- Mark skipped/not_testable clearly.
- Do not add strategy beyond recommended_next_actions.
- Do not overstate temporal validation.
- If temporal validation is diagnostic, call it diagnostic.
- Replace existing inline summary creation in orchestrator with this function.
```

## Acceptance criteria

- Summary contains all required sections.
- Summary does not mention modules that are absent except as skipped/not_testable.
- Warnings and limitations are included.
- Summary respects validation_evidence.primary_validation.
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
