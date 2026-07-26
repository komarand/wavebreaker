# Research Scout Mode

Research Scout mode is Stage 1 of the planned v5 architecture. It reuses the v4
research pipeline through retrieval, then converts retrieved evidence into
machine-readable hypotheses and EDA tasks for a future EDA Engine.

## What It Does

- Runs the planner.
- Collects Kaggle, arXiv, Hugging Face Papers, and GitHub sources.
- Summarizes, embeds, indexes, and retrieves evidence.
- Applies source quality scoring when available.
- Calls Research Scout reasoning with `deepseek-v4-pro`.
- Writes stable UTF-8 JSON artifacts under `runs/{competition_id}_{timestamp}/`.

Primary outputs:

- `research_hypotheses.json`
- `eda_task_plan.json`
- `research_scout_summary.md`
- `research_scout_raw.json`
- `research_scout_validation.json`
- `models_used.json`

## What It Does Not Do

- It does not download Kaggle datasets.
- It does not run EDA.
- It does not execute notebooks.
- It does not train baseline models.
- It does not generate the final DOCX report in scout mode.

## CLI

```powershell
python main.py `
  "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability" `
  "Predict credit default risk. Metric: Gini Stability. Tabular credit data." `
  --mode scout
```

The legacy `--report-mode minimal` flag is still supported. New code should use:

```text
--mode full|scout|minimal
```

## JSON Outputs

`research_hypotheses.json` contains:

- competition metadata;
- task type, metric, and domain;
- source and source quality summaries;
- testable hypotheses with priority, category, provenance, verification steps,
  success and failure conditions;
- EDA tasks linked back to hypotheses;
- recommended EDA sequence;
- model usage diagnostics.

`eda_task_plan.json` is the focused EDA Engine input. It contains:

- competition metadata;
- metric metadata;
- EDA tasks;
- a compact hypothesis index;
- recommended module sequence;
- blocking task IDs.

The future EDA Engine should consume `eda_task_plan.json` directly, produce
evidence keys named by the tasks, and pass those evidence artifacts to the
Synthesizer stage.

## Validation

Scout output is normalized and then validated. A valid scout run must include:

- at least eight hypotheses;
- at least one P0 validation hypothesis;
- at least one P0 leakage hypothesis;
- schema inference, validation analyzer, and leakage checker tasks;
- temporal validation coverage for stability metrics.

By default, invalid Scout output fails loudly. For debugging only:

```powershell
python main.py <competition_url> <description> --mode scout --allow-partial-scout-output
```

When partial output is allowed, validation errors are written next to
`research_hypotheses_partial.json`.

## Limitations

Research Scout is evidence-to-hypothesis planning. It is not proof that any
dataset property is true. Dataset-dependent claims are marked
`not_verified_on_data` and must be checked by the future EDA Engine.
