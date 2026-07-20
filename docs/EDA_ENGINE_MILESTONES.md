# KaggleResearcher — EDA Engine Milestones

This file continues the implementation roadmap after `Milestone C — Trust & Memory`.

EDA Engine is introduced as a separate Data Evidence Layer. It does not replace the existing research/reasoning pipeline. It consumes Research Scout hypotheses and real competition data, then produces machine-readable evidence for the Final Synthesizer.

---

## Scoped override for EDA Engine tasks

The earlier global rule from `CODEX_TASKS.md`:

```text
Do not download Kaggle train/test datasets.
Do not perform real EDA, adversarial validation, model training, or confirmed leakage detection.
```

continues to apply to the original research/reasoning pipeline.

Starting with task `28`, data execution is allowed only inside:

```text
kaggle_researcher/eda/
kaggle_eda_engine/
```

Allowed only inside the EDA Engine scope:

```text
- downloading Kaggle competition datasets when explicitly requested by CLI/config;
- reading local train/test/sample_submission files;
- computing schema/profile/validation/leakage/drift/baseline evidence;
- writing EDA artifacts under data/eda_runs/.
```

Still forbidden everywhere:

```text
- executing Kaggle notebooks;
- executing arbitrary user/notebook code;
- optimizing public leaderboard score;
- AutoML;
- stacking/blending;
- hidden network calls in tests;
- logging Kaggle credentials or other secrets.
```

EDA Engine outputs must be machine-first and evidence-based. Any strategic claim must have `evidence_refs`, `confidence`, and a concrete finding.

---

## Milestone D — EDA Engine Foundation

### Goal

Introduce the Kaggle EDA Engine as a separate data-evidence layer without breaking the existing research/reasoning pipeline.

### Includes tasks

```text
28–35
```

### Outcome

This milestone creates the EDA package, schemas, config, artifact layout, dataset IO abstractions, and offline fixtures.

It does not yet run the full EDA pipeline on real Kaggle data.

### Tasks

```text
28_eda_docs_and_task_rules
29_eda_package_skeleton
30_eda_schemas_core
31_eda_config
32_eda_artifact_writer
33_eda_dataset_resolver
34_eda_dataset_reader
35_eda_offline_fixtures
```

### Completion criteria

By the end of this milestone:

```text
- python -m kaggle_eda_engine.main --help works.
- EDA schemas validate input/output contracts.
- EDA config does not require DeepSeek/vLLM/Postgres.
- EDA run directories can be created.
- Local dataset paths can be resolved.
- Tiny fixture datasets can be read offline.
- No real Kaggle download is required in tests.
```

---

## Milestone E — EDA MVP Evidence Modules

### Goal

Implement the P0 EDA modules and produce the first valid `eda_evidence_pack.json` from local fixtures.

### Includes tasks

```text
36–45
```

### Outcome

This milestone creates a working MVP without P1 modules such as drift, baseline, feature probe, or notebook static analysis.

### MVP modules

```text
file_inventory
schema_inferer
table_profiler
metric_analyzer
validation_analyzer
leakage_checker
hypothesis_evaluator
recommended_next_actions
```

### Tasks

```text
36_eda_file_inventory
37_eda_schema_inferer
38_eda_table_profiler
39_eda_metric_analyzer_basic
40_eda_temporal_split_helpers
41_eda_validation_analyzer
42_eda_leakage_checker_basic
43_eda_hypothesis_evaluator
44_eda_recommendations
45_eda_mvp_orchestrator_and_cli
```

### Completion criteria

By the end of this milestone, this command should work on a local fixture dataset:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id fixture_competition `
  --hypotheses-path tests\fixtures\eda\home_credit_tiny\research_hypotheses.json `
  --task-plan-path tests\fixtures\eda\home_credit_tiny\eda_task_plan.json `
  --local-dataset-path tests\fixtures\eda\home_credit_tiny `
  --no-download-dataset
```

The run should create:

```text
eda_evidence_pack.json
eda_summary.md
file_inventory.json
inferred_schema.json
table_profiles.json
metric_evidence.json
validation_evidence.json
leakage_evidence.json
hypothesis_results.json
recommended_next_actions.json
```

And `hypothesis_results.json` must contain at least:

```text
schema_001
metric_001
val_001
leak_001
```

---

## Milestone F — EDA P1 Modules

### Goal

Add relationship inference, drift analysis, honest baseline, feature probes, and static notebook pattern extraction.

### Includes tasks

```text
46–52
```

### Outcome

These modules are enabled explicitly via CLI flags or task plan. They should not block the MVP path.

### P1 modules

```text
relationship_inferer
drift_analyzer
baseline_runner
feature_probe
notebook_static_analysis
```

### Tasks

```text
46_eda_relationship_inferer
47_eda_drift_analyzer
48_eda_baseline_runner_base_table
49_eda_feature_probe
50_eda_notebook_static_analysis
51_eda_p1_orchestrator_wiring
52_eda_p1_hypothesis_and_recommendation_rules
```

### Completion criteria

By the end of this milestone:

```text
- MVP run still works without P1 flags.
- --enable-p1-modules runs relationship/drift/feature probes.
- --enable-baseline is required for baseline_runner.
- P1 failures become warnings, not fatal run errors.
- P1 evidence is included in eda_evidence_pack.json when available.
- Hypothesis evaluation and recommendations use P1 evidence.
```

---

## Milestone G — Research Scout and Final Synthesizer Integration

### Goal

Connect the existing research pipeline to the EDA Engine by producing `research_hypotheses.json` / `eda_task_plan.json`, then consuming `eda_evidence_pack.json`.

### Includes tasks

```text
53–58
```

### Outcome

The original research pipeline can generate EDA inputs, the EDA Engine can produce evidence, and the Final Synthesizer can connect:

```text
source claim -> scout hypothesis -> EDA result -> strategy action
```

### Tasks

```text
53_research_scout_schemas
54_research_scout_reasoner
55_research_pipeline_writes_scout_outputs
56_final_synthesizer_schema
57_final_synthesizer_reasoner
58_full_research_to_eda_to_strategy_cli
```

### Completion criteria

By the end of this milestone:

```text
- run_research can optionally write Research Scout outputs.
- Research Scout outputs validate against EDA input schemas.
- EDA Engine can consume generated Scout files.
- Final Synthesizer can consume eda_evidence_pack.json.
- Optional full workflow can produce final_strategy.json.
```

---

## Milestone H — Production Hardening

### Goal

Make EDA Engine reliable on larger local datasets, safer in failure modes, and enforce evidence quality.

### Includes tasks

```text
59–64
```

### Outcome

EDA Engine becomes more production-ready: resource limits, sampling, partial runs, quality gates, summary generation, integration tests, and runbook docs.

### Tasks

```text
59_eda_quality_gates
60_eda_resource_limits_and_sampling
61_eda_error_handling_and_partial_runs
62_eda_summary_generator
63_eda_integration_fixture_full_p1
64_eda_production_cli_docs
```

### Completion criteria

By the end of this milestone:

```text
- EDA evidence packs pass quality gates.
- Broken evidence_refs are detected.
- Large-table sampling is explicit.
- Partial failed runs still write useful artifacts.
- eda_summary.md is generated only from evidence_pack data.
- Full P1 offline integration test runs without Kaggle/DeepSeek.
- RUNBOOK has copy-pasteable Windows commands.
```

---

## Recommended command sequence

### EDA-only MVP on local fixture

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id fixture_competition `
  --hypotheses-path tests\fixtures\eda\home_credit_tiny\research_hypotheses.json `
  --task-plan-path tests\fixtures\eda\home_credit_tiny\eda_task_plan.json `
  --local-dataset-path tests\fixtures\eda\home_credit_tiny `
  --no-download-dataset
```

### EDA with P1 modules

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id fixture_competition `
  --hypotheses-path tests\fixtures\eda\home_credit_tiny\research_hypotheses.json `
  --task-plan-path tests\fixtures\eda\home_credit_tiny\eda_task_plan.json `
  --local-dataset-path tests\fixtures\eda\home_credit_tiny `
  --no-download-dataset `
  --enable-p1-modules
```

### EDA with baseline

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id fixture_competition `
  --hypotheses-path tests\fixtures\eda\home_credit_tiny\research_hypotheses.json `
  --task-plan-path tests\fixtures\eda\home_credit_tiny\eda_task_plan.json `
  --local-dataset-path tests\fixtures\eda\home_credit_tiny `
  --no-download-dataset `
  --enable-p1-modules `
  --enable-baseline
```

### Regression checks after each EDA task

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests/eda
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main --help
```

### Full project regression

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main --help
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main --help
```
