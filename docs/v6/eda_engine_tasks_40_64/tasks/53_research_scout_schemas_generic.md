# 53_research_scout_schemas_generic

## Goal

Define schemas for Research Scout outputs that feed the generic EDA Engine.

## Files to create/change

```text
kaggle_researcher/research_scout/
├── __init__.py
├── schemas.py
└── prompts.py

tests/test_research_scout_schemas.py
```

## Codex prompt

```text
Create or update research_scout schemas.

Models:
- ResearchScoutOutput
- ScoutHypothesis
- ScoutEdaTask
- ScoutStructuredFinding
- ScoutLimitation
- EdaTaskPlanDraft

Requirements:
- ResearchScoutOutput can be serialized to:
  - research_hypotheses.json
  - eda_task_plan.json
  - research_scout_summary.md
- Hypotheses must include:
  - hypothesis_id
  - category
  - claim
  - rationale
  - expected_eda_checks
  - priority
  - confidence_before_eda
  - source_refs
- eda_task_plan must include:
  - competition_id
  - task_type
  - metric
  - eda_tasks
  - hypothesis_index
  - recommended_module_sequence
  - recommended_human_checklist
  - blocking_tasks
- IDs should be stable and category-prefixed:
  - schema_001
  - metric_001
  - val_001
  - leak_001
  - drift_001

Important:
- Do not assume temporal validation by default.
- Do not assume Home Credit column names.
- Hypotheses must be generic first, competition-specific second.
```

## Acceptance criteria

- Research Scout output validates.
- Generated EDA input JSON validates against EDA schemas.
- Invalid hypothesis category/status fails validation.
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
