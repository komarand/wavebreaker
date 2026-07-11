# KaggleResearcher Production Runbook

This runbook uses the project-local Windows virtual environment:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe
```

Do not use global Python for these commands.

## Install Or Update

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
E:\wavebreaker\.venv-win\Scripts\python.exe -m pip install -e .
```

## 1. Run Research-Only Pipeline

Use this when you want the source-based roadmap only. This path does not inspect train/test data.

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main `
  "https://www.kaggle.com/competitions/example-competition" `
  "Short competition description, target, metric, and known constraints." `
  --competition-id example-competition `
  --output-dir reports
```

Expected outputs:

- `reports\<competition>_research_report.docx`
- `runs\<competition>_<timestamp>\roadmap.md`
- `runs\<competition>_<timestamp>\research_run.json`
- retrieval and reasoning artifacts such as `plan.json`, `retrieved_documents.json`, and `experiments.json`

## 2. Generate Research Scout EDA Plan

Use Scout mode when you need EDA inputs but are not ready to run data execution.

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main `
  "https://www.kaggle.com/competitions/example-competition" `
  "Short competition description, target, metric, and known constraints." `
  --competition-id example-competition `
  --mode scout `
  --output-dir reports
```

Expected outputs in `runs\<competition>_<timestamp>\`:

- `research_hypotheses.json`
- `eda_task_plan.json`
- `research_scout_summary.md`
- `research_scout_validation.json`
- `research_run.json`

## 3. Run EDA MVP With Local Dataset

Use local dataset mode for deterministic production checks, fixtures, or manually downloaded Kaggle data.

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id example-competition `
  --hypotheses-path runs\example-competition_YYYYMMDD_HHMMSS\research_hypotheses.json `
  --task-plan-path runs\example-competition_YYYYMMDD_HHMMSS\eda_task_plan.json `
  --local-dataset-path data\kaggle_datasets\example-competition `
  --output-dir data\eda_runs `
  --no-download-dataset
```

Expected outputs in `data\eda_runs\<competition>_<timestamp>\`:

- `eda_evidence_pack.json`
- `eda_summary.md`
- `module_statuses.json`
- `file_inventory.json`
- `inferred_schema.json`
- `table_profiles.json`
- `metric_evidence.json`
- `validation_evidence.json`
- `leakage_evidence.json`
- `hypothesis_results.json`
- `recommended_next_actions.json`

## 4. Run EDA With P1 Modules

P1 modules add relationship inference, drift diagnostics, feature probes, and optional static notebook analysis. They remain evidence modules, not leaderboard optimization.

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id example-competition `
  --hypotheses-path runs\example-competition_YYYYMMDD_HHMMSS\research_hypotheses.json `
  --task-plan-path runs\example-competition_YYYYMMDD_HHMMSS\eda_task_plan.json `
  --local-dataset-path data\kaggle_datasets\example-competition `
  --output-dir data\eda_runs `
  --no-download-dataset `
  --enable-p1-modules
```

Additional expected outputs can include:

- `relationship_evidence.json`
- `drift_evidence.json`
- `feature_probe_evidence.json`
- `notebook_static_analysis.json` when static analysis is enabled
- `artifacts\drift\`
- `artifacts\profiles\`

## 5. Run EDA With Baseline

The baseline is a sanity check for metric, split, and target handling. It is not final score optimization.

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id example-competition `
  --hypotheses-path runs\example-competition_YYYYMMDD_HHMMSS\research_hypotheses.json `
  --task-plan-path runs\example-competition_YYYYMMDD_HHMMSS\eda_task_plan.json `
  --local-dataset-path data\kaggle_datasets\example-competition `
  --output-dir data\eda_runs `
  --no-download-dataset `
  --enable-p1-modules `
  --enable-baseline
```

Additional expected outputs can include:

- `baseline_evidence.json`
- `artifacts\baseline\baseline_oof_predictions.csv`
- `artifacts\baseline\fold_metrics.csv`

## 6. Run Full Research To EDA To Final Strategy

This writes Scout inputs, runs local-dataset EDA, then synthesizes a final strategy from source evidence plus EDA evidence.

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main `
  "https://www.kaggle.com/competitions/example-competition" `
  "Short competition description, target, metric, and known constraints." `
  --competition-id example-competition `
  --write-eda-plan `
  --run-eda `
  --local-dataset-path data\kaggle_datasets\example-competition `
  --eda-output-dir data\eda_runs `
  --final-synthesis `
  --output-dir reports
```

Expected outputs:

- `research_hypotheses.json`
- `eda_task_plan.json`
- `eda_evidence_pack.json`
- `eda_summary.md`
- `final_strategy.json`
- `final_strategy.md`
- `final_strategy.docx` when document export is enabled for final strategy
- the regular research report DOCX in `reports\`

## Safety Notes

- Kaggle competition rules must be accepted before dataset download.
- Prefer `--local-dataset-path` for production repeatability.
- Notebook execution is never performed. Static notebook analysis reads text/code only.
- Baseline is a sanity check, not final score optimization.
- Large datasets may be sampled. If table profiles contain `sampled=true`, treat affected conclusions as sample-limited.
- EDA exceptions are sanitized before JSON output; secrets such as Kaggle keys and API tokens must not be logged.

## Generic Tabular Validation Behavior

- Ordinary binary or multiclass classification can use StratifiedKFold.
- Ordinary regression can use KFold.
- Grouped/entity tasks can use GroupKFold or StratifiedGroupKFold.
- Ranking tasks can use query/group-aware validation.
- Temporal validation is used as primary only when the metric, task, or data evidence supports it.
- A date column alone is diagnostic, not enough to force temporal validation.
- Gini Stability is supported, but generic tabular behavior is not Home Credit-specific.

## Regression Checks

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests\eda -q
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest -q
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main --help
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main --help
```
