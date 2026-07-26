# 55_research_pipeline_writes_scout_outputs

## Goal

Wire Research Scout into `run_research` so the research pipeline can produce EDA input files.

## Files to create/change

```text
kaggle_researcher/main.py
tests/test_pipeline_research_scout_outputs.py
```

## Codex prompt

```text
Wire Research Scout into the existing run_research pipeline.

Requirements:
- Add optional CLI flag:
  --write-eda-plan
- When enabled, after retrieved_documents are available:
  - run Research Scout
  - write research_hypotheses.json
  - write eda_task_plan.json
  - write research_scout_summary.md
  next to research_run.json/report output.
- research_run.json should include paths to these files.
- Existing behavior without --write-eda-plan should remain unchanged.
- Do not run EDA Engine from run_research in this task.
- Scout output must use generic task_type/metric/validation wording.
```

## Acceptance criteria

- Mocked pipeline writes all three Scout outputs.
- EDA input JSON files validate.
- Without --write-eda-plan, no Scout files are written.
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
