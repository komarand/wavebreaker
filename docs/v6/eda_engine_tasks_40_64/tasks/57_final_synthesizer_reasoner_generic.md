# 57_final_synthesizer_reasoner_generic

## Goal

Implement the final strategy synthesizer that combines retrieved sources and generic EDA evidence.

## Files to create/change

```text
kaggle_researcher/reasoning/final_synthesizer.py
tests/test_final_synthesizer.py
```

## Codex prompt

```text
Implement final strategy synthesizer.

Function:
- async synthesize_final_strategy(
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    domain_patterns: list[dict],
    research_hypotheses: ResearchHypotheses,
    eda_evidence_pack: EdaEvidencePack,
    reasoning_outputs: dict,
    client: DeepSeekClient,
    model: str,
) -> FinalStrategyResult

Requirements:
- Use DeepSeekClient.chat_json.
- Prompt must require:
  - no raw chain-of-thought.
  - no unsupported claims.
  - no claim that notebooks were executed.
  - no claim that baseline is final solution.
  - link every important recommendation to EDA evidence_refs.
  - respect validation_evidence.primary_validation.
- Output sections:
  - executive_summary
  - metric_and_validation
  - dataset_facts_from_eda
  - leakage_and_data_quality
  - drift_and_leaderboard_risk
  - baseline_findings
  - feature_priorities
  - modeling_plan
  - experiments_queue
  - what_not_to_do
  - first_48_hours
- If EDA selected StratifiedKFold, do not override it with temporal CV.
- If temporal validation is diagnostic only, state that clearly.
- If EDA evidence is missing for a claim, mark it as hypothesis or limitation.
```

## Acceptance criteria

- Mock LLM response validates into FinalStrategyResult.
- Recommendations include evidence_refs.
- Prompt includes source -> hypothesis -> EDA -> strategy rule.
- Prompt includes "respect validation_evidence.primary_validation".
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
