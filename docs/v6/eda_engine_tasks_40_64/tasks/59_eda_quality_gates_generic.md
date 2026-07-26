# 59_eda_quality_gates_generic

## Goal

Validate EDA outputs before they are used by Final Synthesizer.

## Files to create/change

```text
kaggle_researcher/eda/quality.py
tests/eda/test_eda_quality.py
```

## Codex prompt

```text
Implement EDA quality gates.

Functions:
- validate_evidence_pack(pack: EdaEvidencePack) -> list[str]
- validate_evidence_refs(pack: EdaEvidencePack) -> list[str]
- validate_hypothesis_results(pack: EdaEvidencePack, hypotheses: ResearchHypotheses) -> list[str]
- validate_no_unsupported_summary_claims(summary_text: str, pack: EdaEvidencePack) -> list[str]

Checks:
- Every input hypothesis has exactly one result.
- confirmed/rejected hypothesis results have evidence_refs.
- evidence_refs point to existing paths in pack where practical.
- recommended_next_actions have evidence_refs.
- eda_summary.md must not claim more than evidence_pack contains.
- warnings/limitations are preserved.
- No forbidden phrases:
  - "probably confirmed" without evidence
  - "leakage found" unless leakage check status failed with evidence
  - "baseline proves final solution"
  - "temporal validation is required" when validation_evidence selected another primary policy
```

## Acceptance criteria

- Missing hypothesis result creates warning.
- Broken evidence_ref creates warning.
- Empty action evidence_refs creates warning.
- Temporal overclaim creates warning.
- Quality functions return warnings, not exceptions.
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
