# 56_final_synthesizer_schema_generic

## Goal

Define a structured contract for final strategy synthesis that can consume generic EDA evidence.

## Files to create/change

```text
kaggle_researcher/reasoning/final_synthesizer.py
tests/test_final_synthesizer_schema.py
```

## Codex prompt

```text
Create or update final_synthesizer contract.

Models or typed dicts:
- FinalStrategyResult
- FinalStrategySection
- FinalStrategyAction

Function placeholder:
- async synthesize_final_strategy(...)

Inputs:
- competition_desc
- plan_data
- retrieved_documents
- domain_patterns
- research_hypotheses
- eda_evidence_pack
- previous reasoning outputs

Requirements:
- Final strategy must link:
  source claim -> scout hypothesis -> EDA result -> strategy action.
- Actions must include:
  - priority
  - action
  - reason
  - evidence_refs
  - related_hypothesis_ids
- Do not implement LLM call yet; add schema and placeholder only.
- The schema must support generic tabular outcomes:
  - stratified CV
  - KFold
  - group CV
  - temporal CV
  - ranking group CV
  - custom validation required
```

## Acceptance criteria

- FinalStrategyResult validates.
- Missing evidence_refs on action fails validation.
- Placeholder raises NotImplementedError.
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
