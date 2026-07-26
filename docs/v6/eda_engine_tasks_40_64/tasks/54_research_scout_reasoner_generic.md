# 54_research_scout_reasoner_generic

## Goal

Implement the Research Scout reasoning module that generates generic EDA hypotheses from retrieved sources.

## Files to create/change

```text
kaggle_researcher/research_scout/scout.py
tests/test_research_scout.py
```

## Codex prompt

```text
Implement or update Research Scout.

Function:
- async run_research_scout(
    competition_id: str,
    competition_url: str,
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    client: DeepSeekClient,
    model: str,
) -> ResearchScoutOutput

Requirements:
- Use DeepSeek V4 Pro through DeepSeekClient.chat_json.
- Generate:
  - research_hypotheses
  - eda_task_plan
  - markdown summary text
- Prompt must:
  - convert source-backed claims into testable EDA hypotheses.
  - separate source facts from hypotheses.
  - avoid claiming EDA has already run.
  - require expected_eda_checks for every hypothesis.
  - prioritize P0 blocking checks for schema/metric/validation/leakage.
  - infer task_type and metric from plan_data.
  - generate generic tabular hypotheses first.
  - generate temporal validation hypotheses only if metric/source/description supports them.
  - generate group validation hypotheses only if group/entity/query risk is plausible.
- Must include at least:
  - schema_001
  - metric_001
  - val_001
  - leak_001
- Validate output against research_scout schemas and EDA schemas.
- On LLM failure, provide deterministic fallback from PlanData.

Do not automatically create "temporal validation is required" for every tabular task.
```

## Acceptance criteria

- Mock LLM response validates and writes expected objects.
- Fallback output includes P0 hypotheses.
- Ordinary binary classification fallback does not force temporal validation.
- Every hypothesis has expected_eda_checks.
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
