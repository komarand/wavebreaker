# KaggleResearcher v5 — EDA Engine Specification

**Document:** `docs/EDA_ENGINE_SPEC.md`  
**Status:** implementation specification  
**Scope:** Kaggle EDA Engine / Data Evidence Layer  
**Primary consumers:** Codex, maintainers, tests, Final DeepResearch Synthesizer

---


1. task_type должен быть enum:
   - binary_classification
   - multiclass_classification
   - regression
   - ranking
   - survival
   - forecasting_tabular
   - multilabel_classification

2. metric_analyzer должен поддерживать registry:
   - auc / roc_auc
   - gini / normalized_gini
   - logloss
   - accuracy
   - f1 / macro_f1
   - quadratic_weighted_kappa
   - rmse / rmsle / mae
   - mape / smape
   - r2
   - map@k / ndcg
   - concordance_index / c-index
   - custom / unknown

3. validation_analyzer не должен всегда тянуть temporal CV.
   Он должен выбирать policy по evidence:
   - есть time + metric/time split risk → temporal CV
   - есть group/id/entity leakage risk → GroupKFold
   - обычный iid binary/multiclass → StratifiedKFold
   - regression iid → KFold
   - ranking/query groups → GroupKFold by query/group
   - forecasting → rolling/expanding
   - survival → stratified/group/time-aware depending on columns

4. baseline_runner должен выбирать модель по task_type:
   - binary/multiclass → classifier
   - regression → regressor
   - ranking → optional skipped/not_testable in MVP
   - survival → skipped/not_testable in MVP
   - forecasting → skipped or simple lag baseline later

5. schema_inferer должен искать не только:
   target/case_id/WEEK_NUM
   но generic:
   - target/name from sample_submission
   - id column from sample_submission join
   - group/entity columns
   - time/date columns
   - query_id/session_id/user_id for ranking/recommender

## 1. Purpose

`Kaggle EDA Engine` is the data-execution layer of KaggleResearcher v5.

The existing KaggleResearcher research pipeline collects external sources, indexes them, retrieves relevant documents, and produces a reasoning-based roadmap. That pipeline is intentionally source-based: it reads Kaggle notebooks, papers, GitHub repositories, and domain memory, but it does **not** inspect the actual Kaggle train/test dataset.

EDA Engine fills this missing layer.

It consumes:

```text
research_hypotheses.json
eda_task_plan.json
Kaggle dataset or local dataset path
```

and produces:

```text
eda_evidence_pack.json
eda_summary.md
module-level JSON artifacts
artifacts/
```

The core responsibility of EDA Engine is to turn Research Scout hypotheses into factual evidence from real competition data.

Ideal flow:

```text
Research Scout:
  "Here is what should be checked."

EDA Engine:
  "Here is what the data actually shows."

Final DeepResearch Synthesizer:
  "Here is the strategy implied by sources + hypotheses + evidence."
```

EDA Engine is not another LLM report writer. It is a deterministic, testable, machine-first evidence layer.

---

## 2. Relationship to the existing KaggleResearcher pipeline

### 2.1 Existing v4 pipeline

The existing pipeline remains retrieval/reasoning-based:

```text
competition_url + competition_desc
  -> planner
  -> Kaggle/arXiv/GitHub agents
  -> PDF parser
  -> summarizer
  -> embeddings
  -> PostgreSQL + pgvector
  -> hybrid retrieval
  -> reasoning chain
  -> report composer
  -> .docx roadmap
```

The reasoning chain can discuss validation risk, leakage risk, metric implications, experiment priorities, and leaderboard risk based on retrieved text sources. It must not claim that real train/test data was analyzed.

### 2.2 v5 pipeline with EDA Engine

EDA Engine is added as a separate Stage 2:

```text
Stage 0 — Retrieval Layer
  Input:
    competition_url
    competition_desc

  Output:
    plan_data
    retrieved_documents
    domain_patterns


Stage 1 — Research Scout
  Input:
    competition_url
    competition_desc
    plan_data
    retrieved_documents
    domain_patterns

  Output:
    research_hypotheses.json
    eda_task_plan.json
    research_scout_summary.md


Stage 2 — EDA Engine
  Input:
    competition_id
    competition_url
    research_hypotheses.json
    eda_task_plan.json
    Kaggle dataset or local dataset path

  Output:
    eda_evidence_pack.json
    eda_summary.md
    module-level JSON artifacts
    artifacts/


Stage 3 — Final DeepResearch Synthesizer
  Input:
    competition_desc
    plan_data
    retrieved_documents
    domain_patterns
    research_hypotheses.json
    eda_evidence_pack.json

  Output:
    final_strategy.json
    final_strategy.md
    final_strategy.docx
```

### 2.3 Separation of responsibilities

| Layer | Sees real dataset? | Uses LLM? | Main output |
|---|---:|---:|---|
| Retrieval Layer | No | Some components | retrieved documents |
| Reasoning Layer | No | Yes | source-based roadmap |
| Research Scout | No | Yes | hypotheses and EDA task plan |
| EDA Engine | Yes | No by default | factual evidence pack |
| Final Synthesizer | Reads EDA output | Yes | final strategy |

EDA Engine should not depend on DeepSeek, vLLM, PostgreSQL, or pgvector for MVP execution. It may consume files produced by the research pipeline, but it should be runnable offline on local fixture datasets.

---

## 3. Scoped data-execution override

The original research pipeline rules still apply to the retrieval/reasoning system:

```text
Do not download Kaggle train/test datasets.
Do not execute Kaggle notebooks.
Do not perform real EDA from the research pipeline.
Do not claim confirmed leakage based only on text sources.
```

Starting with EDA Engine tasks, a scoped exception is allowed:

```text
Data execution is allowed only inside:
- kaggle_researcher/eda/
- kaggle_eda_engine/
```

Allowed inside EDA Engine:

```text
- download Kaggle competition datasets when explicitly requested;
- read local train/test/sample_submission files;
- inspect schemas and table profiles;
- compute validation evidence;
- compute basic leakage checks;
- compute relationship evidence;
- compute drift evidence;
- run an honest baseline when explicitly enabled;
- write EDA artifacts under data/eda_runs/.
```

Still forbidden everywhere, including EDA Engine:

```text
- execute Kaggle notebooks;
- execute arbitrary downloaded code;
- perform AutoML;
- optimize public leaderboard score;
- perform stacking/blending search;
- treat public notebook scores as ground truth;
- log secrets such as KAGGLE_KEY, DEEPSEEK_API_KEY, GITHUB_TOKEN;
- run hidden network calls in tests.
```

---

## 4. Non-goals

EDA Engine MVP must not:

```text
- generate the final Kaggle solution;
- replace the reasoning chain;
- write the final DeepResearch report;
- execute public notebooks;
- do strong feature engineering across all tables;
- perform leaderboard optimization;
- perform blending/stacking;
- perform expensive AutoML;
- claim more than its evidence supports.
```

EDA Engine may later support optional baseline and feature-probe modules, but these are evidence/sanity modules, not leaderboard optimization modules.

---

## 5. Design principles

### 5.1 Machine-first output

The primary output is JSON, not prose.

Good:

```json
{
  "hypothesis_id": "val_001",
  "status": "confirmed",
  "confidence_after_eda": "high",
  "finding": "WEEK_NUM detected with 92 periods. Latest-period holdout can be constructed.",
  "evidence_refs": [
    "inferred_schema.global_roles.candidate_time_columns",
    "validation_evidence.oot_holdout"
  ],
  "impact_on_strategy": "Use latest-period out-of-time holdout plus expanding-window CV as primary validation."
}
```

Bad:

```text
Looks like time validation is probably needed.
```

### 5.2 Evidence references are mandatory for strategic claims

Any output that affects strategy must include:

```text
- concrete finding;
- confidence;
- evidence_refs;
- limitations when relevant;
- impact_on_strategy when attached to a hypothesis.
```

### 5.3 Large-data safety

EDA Engine must avoid loading large datasets into memory unnecessarily.

Rules:

```text
- Prefer Polars lazy scans for CSV/Parquet.
- Use bounded sampling when full scans are unsafe.
- Preserve sampled=true in outputs.
- Store sample size and limitations when sampling occurs.
- Never hide sampling from downstream consumers.
```

### 5.4 Graceful degradation

If a module cannot test something, it should return:

```text
status: not_testable
limitations: [...]
warnings: [...]
```

rather than inventing a conclusion.

---

## 6. Project structure

Recommended package layout:

```text
kaggle_researcher/
├── eda/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── orchestrator.py
│   ├── schemas.py
│   ├── quality.py
│   ├── summary.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── artifact_writer.py
│   │   ├── dataset_resolver.py
│   │   └── dataset_reader.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── file_inventory.py
│   │   ├── schema_inferer.py
│   │   ├── table_profiler.py
│   │   ├── metric_analyzer.py
│   │   ├── validation_analyzer.py
│   │   ├── leakage_checker.py
│   │   ├── relationship_inferer.py
│   │   ├── drift_analyzer.py
│   │   ├── baseline_runner.py
│   │   ├── feature_probe.py
│   │   ├── notebook_static_analyzer.py
│   │   ├── hypothesis_evaluator.py
│   │   └── recommendations.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── gini.py
│   │   └── gini_stability.py
│   └── validation/
│       ├── __init__.py
│       └── temporal_split.py
│
└── ...

kaggle_eda_engine/
├── __init__.py
└── main.py

tests/
└── eda/
    ├── fixtures/
    ├── test_eda_schemas.py
    ├── test_file_inventory.py
    ├── test_schema_inferer.py
    ├── test_table_profiler.py
    ├── test_metric_analyzer.py
    ├── test_validation_analyzer.py
    ├── test_leakage_checker.py
    ├── test_hypothesis_evaluator.py
    ├── test_eda_recommendations.py
    └── test_eda_orchestrator_mvp.py
```

`kaggle_eda_engine` is a thin external CLI wrapper around `kaggle_researcher.eda.main`.

---

## 7. Inputs

## 7.1 `research_hypotheses.json`

This is the semantic hypothesis layer produced by Research Scout.

### Required top-level shape

```json
{
  "schema_version": "1.0",
  "competition_id": "home-credit-credit-risk-model-stability",
  "created_at": "2026-07-06T12:00:00+02:00",
  "hypotheses": [],
  "eda_tasks": [],
  "structured_findings": [],
  "scout_limitations": [],
  "models_used": {}
}
```

### Hypothesis shape

```json
{
  "hypothesis_id": "val_001",
  "category": "validation",
  "claim": "Validation should be temporal because the metric rewards stability over time.",
  "rationale": "Competition description and retrieved sources mention WEEK_NUM and stability metric.",
  "expected_eda_checks": [
    "schema_inferer.detect_time_columns",
    "validation_analyzer.period_distribution",
    "validation_analyzer.temporal_cv_feasibility"
  ],
  "priority": "P0",
  "confidence_before_eda": "medium",
  "source_refs": [
    "retrieved_documents.doc_123"
  ]
}
```

### Required hypothesis fields

| Field | Type | Required | Notes |
|---|---:|---:|---|
| `hypothesis_id` | string | yes | stable, unique |
| `category` | enum | yes | see allowed categories |
| `claim` | string | yes | testable claim |
| `rationale` | string | recommended | why Scout created it |
| `expected_eda_checks` | list[string] | yes | module/check hints |
| `priority` | enum | yes | P0/P1/P2/P3 |
| `confidence_before_eda` | enum | yes | low/medium/high |
| `source_refs` | list[string] | optional | source evidence from research stage |

### Allowed hypothesis categories

```text
schema
metric
validation
leakage
relationship
drift
baseline
feature
notebook
leaderboard
data_quality
```

### Minimum required MVP hypotheses

Research Scout should produce at least:

```text
schema_001
metric_001
val_001
leak_001
```

EDA MVP must return one `HypothesisResult` for each of these.

---

## 7.2 `eda_task_plan.json`

This is the executable task plan.

### Required top-level shape

```json
{
  "schema_version": "1.0",
  "competition_id": "home-credit-credit-risk-model-stability",
  "task_type": "binary_classification",
  "metric": {
    "name": "gini_stability",
    "greater_is_better": true,
    "requires_probabilities": true,
    "requires_groups_or_time": true
  },
  "dataset": {
    "competition_url": "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability",
    "download_required": true,
    "local_dataset_path": null
  },
  "eda_tasks": [],
  "hypothesis_index": {},
  "recommended_module_sequence": [
    "file_inventory",
    "schema_inferer",
    "table_profiler",
    "metric_analyzer",
    "validation_analyzer",
    "leakage_checker",
    "relationship_inferer",
    "drift_analyzer",
    "baseline_runner",
    "feature_probe",
    "notebook_static_analysis"
  ],
  "recommended_human_checklist": [],
  "blocking_tasks": [
    "file_inventory",
    "schema_inferer",
    "validation_analyzer",
    "leakage_checker"
  ]
}
```

### Task shape

```json
{
  "task_id": "validation_001",
  "module": "validation_analyzer",
  "priority": "P0",
  "blocking": true,
  "related_hypothesis_ids": [
    "val_001"
  ],
  "params": {}
}
```

### Invariants

```text
- competition_id must match research_hypotheses.competition_id.
- recommended_module_sequence must contain only known modules.
- blocking_tasks must be a subset of known modules.
- eda_tasks[].module must be a known module.
- Unknown modules are skipped with warning, not silently ignored.
- If dataset.local_dataset_path is set, dataset download is skipped.
```

---

## 8. Outputs

## 8.1 Main output: `eda_evidence_pack.json`

This is the primary machine-readable artifact consumed by Final Synthesizer.

### Required top-level shape

```json
{
  "schema_version": "1.0",
  "competition_id": "home-credit-credit-risk-model-stability",
  "created_at": "2026-07-06T12:00:00+02:00",
  "run_id": "home-credit-credit-risk-model-stability_20260706_120000",
  "dataset": {},
  "file_inventory": {},
  "inferred_schema": {},
  "table_profiles": [],
  "metric_evidence": {},
  "validation_evidence": {},
  "leakage_evidence": [],
  "relationship_evidence": {},
  "drift_evidence": {},
  "baseline_evidence": {},
  "feature_probe_evidence": [],
  "notebook_static_analysis": {},
  "hypothesis_results": [],
  "recommended_next_actions": [],
  "warnings": [],
  "limitations": [],
  "artifacts": {}
}
```

### Top-level field meanings

| Field | Meaning |
|---|---|
| `schema_version` | Evidence pack schema version |
| `competition_id` | Kaggle competition slug/id |
| `created_at` | ISO timestamp |
| `run_id` | deterministic-ish run identifier |
| `dataset` | dataset path/cache info |
| `file_inventory` | file list and table role hints |
| `inferred_schema` | semantic table/column roles |
| `table_profiles` | per-table profiles |
| `metric_evidence` | metric interpretation and local metric availability |
| `validation_evidence` | factual validation recommendation |
| `leakage_evidence` | list of leakage/data-risk checks |
| `relationship_evidence` | optional P1 relationship inference |
| `drift_evidence` | optional P1 drift analysis |
| `baseline_evidence` | optional P1 baseline evidence |
| `feature_probe_evidence` | optional P1 feature-family evidence |
| `notebook_static_analysis` | optional static notebook pattern extraction |
| `hypothesis_results` | one result per Scout hypothesis |
| `recommended_next_actions` | evidence-backed action list |
| `warnings` | non-fatal problems |
| `limitations` | known scope/data limitations |
| `artifacts` | paths to generated artifacts |

---

## 8.2 Human-readable output: `eda_summary.md`

`eda_summary.md` is secondary. It must be derived from `eda_evidence_pack.json` and must not add unsupported claims.

Required sections:

```text
# EDA Summary

## Dataset
## Schema
## Metric
## Validation
## Leakage
## Relationships
## Drift
## Baseline
## Feature probes
## Hypothesis results
## Recommended next actions
## Warnings
## Limitations
```

---

## 8.3 Module-level JSON artifacts

Every module must write its own artifact when run or skipped.

MVP outputs:

```text
file_inventory.json
inferred_schema.json
table_profiles.json
metric_evidence.json
validation_evidence.json
leakage_evidence.json
hypothesis_results.json
recommended_next_actions.json
warnings.json
limitations.json
```

P1 outputs:

```text
relationship_evidence.json
drift_evidence.json
baseline_evidence.json
feature_probe_evidence.json
notebook_static_analysis.json
```

If a module is skipped, its JSON artifact should still exist and contain:

```json
{
  "enabled": false,
  "status": "skipped",
  "reason": "Module disabled by config."
}
```

---

## 9. Run directory contract

Each run writes a self-contained directory:

```text
data/eda_runs/{competition_id}_{YYYYMMDD_HHMMSS}/
├── input_research_hypotheses.json
├── input_eda_task_plan.json
├── eda_evidence_pack.json
├── eda_summary.md
├── module_statuses.json
├── file_inventory.json
├── inferred_schema.json
├── table_profiles.json
├── metric_evidence.json
├── validation_evidence.json
├── leakage_evidence.json
├── relationship_evidence.json
├── drift_evidence.json
├── baseline_evidence.json
├── feature_probe_evidence.json
├── notebook_static_analysis.json
├── hypothesis_results.json
├── recommended_next_actions.json
├── warnings.json
├── limitations.json
└── artifacts/
    ├── plots/
    ├── profiles/
    ├── baseline/
    ├── drift/
    ├── validation/
    └── samples/
```

### Invariants

```text
- Input JSON files are copied as-is.
- Every module writes a JSON result or skipped/failed placeholder.
- eda_evidence_pack.json is written when possible, even for partial runs.
- module_statuses.json records success/failed/skipped status and durations.
- Warnings and limitations are preserved in the evidence pack.
```

---

## 10. Evidence reference contract

`evidence_refs` are string paths into `eda_evidence_pack.json`.

Examples:

```text
file_inventory.files
inferred_schema.global_roles.target_column
inferred_schema.global_roles.candidate_time_columns
validation_evidence.time_columns
validation_evidence.oot_holdout
validation_evidence.target_by_period
leakage_evidence[0].evidence.overlap_count
relationship_evidence.relationships
drift_evidence.adversarial_validation.auc
baseline_evidence.overall_metric.value
```

### Rules

```text
- Strategic claims must have evidence_refs.
- confirmed/rejected hypothesis results must have evidence_refs.
- recommended_next_actions must have evidence_refs.
- not_testable/skipped results may have empty evidence_refs only if limitations explain why.
- Quality gates should warn on broken evidence_refs.
```

---

## 11. EDA configuration

File:

```text
kaggle_researcher/eda/config.py
```

Settings:

```text
EDA_RUNS_DIR=./data/eda_runs
KAGGLE_DATASETS_DIR=./data/kaggle_datasets
EDA_SCHEMA_VERSION=1.0
EDA_PROFILE_SAMPLE_ROWS=200000
EDA_MAX_PROFILE_ROWS_FULL_SCAN=2000000
EDA_MAX_ADVERSARIAL_ROWS=500000
EDA_MAX_BASELINE_ROWS=1000000
EDA_RANDOM_SEED=42
EDA_MAX_TABLE_BYTES optional
EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS optional
EDA_MODULE_TIMEOUT_SEC optional

KAGGLE_USERNAME optional
KAGGLE_KEY optional
```

### Config rules

```text
- DEEPSEEK_API_KEY is not required for EDA Engine MVP.
- Kaggle credentials are required only when dataset download is requested.
- Secrets must not appear in logs, reprs, exceptions, JSON artifacts, or tests.
```

---

## 12. CLI contract

### 12.1 EDA MVP local dataset mode

```powershell
python -m kaggle_eda_engine.main `
  --competition-id home-credit-credit-risk-model-stability `
  --hypotheses-path "runs\...\research_hypotheses.json" `
  --task-plan-path "runs\...\eda_task_plan.json" `
  --local-dataset-path "data\kaggle_datasets\home-credit-credit-risk-model-stability" `
  --no-download-dataset
```

### 12.2 Dataset download mode

```powershell
python -m kaggle_eda_engine.main `
  --competition-id home-credit-credit-risk-model-stability `
  --competition-url "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability" `
  --hypotheses-path "runs\...\research_hypotheses.json" `
  --task-plan-path "runs\...\eda_task_plan.json" `
  --download-dataset
```

### 12.3 MVP-only explicit modules

```powershell
python -m kaggle_eda_engine.main `
  --competition-id home-credit-credit-risk-model-stability `
  --hypotheses-path "runs\...\research_hypotheses.json" `
  --task-plan-path "runs\...\eda_task_plan.json" `
  --local-dataset-path "data\kaggle_datasets\home-credit-credit-risk-model-stability" `
  --modules file_inventory,schema_inferer,table_profiler,metric_analyzer,validation_analyzer,leakage_checker
```

### 12.4 P1 modules

```powershell
python -m kaggle_eda_engine.main `
  --competition-id home-credit-credit-risk-model-stability `
  --hypotheses-path "runs\...\research_hypotheses.json" `
  --task-plan-path "runs\...\eda_task_plan.json" `
  --local-dataset-path "data\kaggle_datasets\home-credit-credit-risk-model-stability" `
  --enable-p1-modules
```

### 12.5 Baseline

```powershell
python -m kaggle_eda_engine.main `
  --competition-id home-credit-credit-risk-model-stability `
  --hypotheses-path "runs\...\research_hypotheses.json" `
  --task-plan-path "runs\...\eda_task_plan.json" `
  --local-dataset-path "data\kaggle_datasets\home-credit-credit-risk-model-stability" `
  --enable-p1-modules `
  --enable-baseline
```

### 12.6 Required CLI arguments

```text
--competition-id             required
--hypotheses-path            required
--task-plan-path             required
```

### 12.7 Optional CLI arguments

```text
--competition-url
--local-dataset-path
--output-dir
--download-dataset
--no-download-dataset
--force-download
--modules
--skip-modules
--enable-p1-modules
--enable-baseline
--enable-notebook-analysis
--profile-sample-rows
--max-baseline-rows
--fail-fast
--debug
```

---

## 13. Orchestrator contract

File:

```text
kaggle_researcher/eda/orchestrator.py
```

### Public function

```python
async def run_eda(config: EdaRunConfig) -> EdaRunResult:
    ...
```

### Execution order

```text
1. Load and validate research_hypotheses.json.
2. Load and validate eda_task_plan.json.
3. Verify competition_id consistency.
4. Create run directory.
5. Copy input JSON files into run directory.
6. Resolve dataset path.
7. Build DatasetReader.
8. Run file_inventory.
9. Run schema_inferer.
10. Run table_profiler.
11. Run metric_analyzer.
12. Run validation_analyzer.
13. Run leakage_checker.
14. Optionally run relationship_inferer.
15. Optionally run drift_analyzer.
16. Optionally run baseline_runner.
17. Optionally run feature_probe.
18. Optionally run notebook_static_analysis.
19. Write module-level JSON artifacts.
20. Evaluate hypotheses.
21. Build recommended_next_actions.
22. Build eda_evidence_pack.json.
23. Run EDA quality gates.
24. Build eda_summary.md.
25. Return EdaRunResult.
```

### Blocking modules

```text
file_inventory
schema_inferer
table_profiler
metric_analyzer
validation_analyzer
leakage_checker
hypothesis_evaluator
recommendations
```

### Non-blocking modules

```text
relationship_inferer
drift_analyzer
baseline_runner
feature_probe
notebook_static_analysis
```

### Failure policy

```text
If a blocking module fails:
  - record module status;
  - write partial artifacts if possible;
  - fail run if fail_fast=true;
  - otherwise produce partial evidence pack only when a safe degraded output exists.

If a non-blocking module fails:
  - write failed placeholder JSON;
  - add warning;
  - continue run.

If dataset cannot be resolved:
  - fail clearly before module execution.
```

---

## 14. Core Pydantic schemas

File:

```text
kaggle_researcher/eda/schemas.py
```

### 14.1 Common enums

```python
Confidence = Literal["low", "medium", "high"]
Priority = Literal["P0", "P1", "P2", "P3"]

HypothesisStatus = Literal[
    "confirmed",
    "partially_confirmed",
    "rejected",
    "not_testable",
    "skipped",
]

LeakageCheckStatus = Literal[
    "passed",
    "failed",
    "warning",
    "not_testable",
    "skipped",
]

Severity = Literal["low", "medium", "high", "critical"]
```

### 14.2 `EdaRunConfig`

```python
class EdaRunConfig(BaseModel):
    competition_id: str
    competition_url: str | None = None

    hypotheses_path: Path
    task_plan_path: Path

    local_dataset_path: Path | None = None
    output_dir: Path | None = None

    download_dataset: bool = True
    force_download: bool = False

    modules: list[str] | None = None
    skip_modules: list[str] = []

    profile_sample_rows: int = 200_000
    max_profile_rows_full_scan: int = 2_000_000
    max_adversarial_rows: int = 500_000
    max_baseline_rows: int = 1_000_000

    enable_p1_modules: bool = False
    enable_baseline: bool = False
    enable_notebook_static_analysis: bool = False

    random_seed: int = 42
    fail_fast: bool = False
```

### 14.3 `EdaRunResult`

```python
class EdaRunResult(BaseModel):
    competition_id: str
    run_id: str
    output_dir: Path

    evidence_pack_path: Path
    summary_path: Path

    module_statuses: dict[str, str]
    hypothesis_results_count: int

    warnings: list[str]
    limitations: list[str]

    duration_sec: float
```

### 14.4 `ResearchHypothesis`

```python
class ResearchHypothesis(BaseModel):
    hypothesis_id: str
    category: Literal[
        "schema",
        "metric",
        "validation",
        "leakage",
        "relationship",
        "drift",
        "baseline",
        "feature",
        "notebook",
        "leaderboard",
        "data_quality",
    ]
    claim: str
    rationale: str | None = None
    expected_eda_checks: list[str]
    priority: Priority
    confidence_before_eda: Confidence
    source_refs: list[str] = []
```

### 14.5 `ResearchHypotheses`

```python
class ResearchHypotheses(BaseModel):
    schema_version: str = "1.0"
    competition_id: str
    created_at: str | None = None

    hypotheses: list[ResearchHypothesis]
    eda_tasks: list[dict] = []
    structured_findings: list[dict] = []
    scout_limitations: list[str] = []
    models_used: dict = {}
```

### 14.6 `EdaTaskPlan`

```python
class EdaTask(BaseModel):
    task_id: str
    module: str
    priority: Priority
    blocking: bool = False
    related_hypothesis_ids: list[str] = []
    params: dict = {}

class EdaTaskPlan(BaseModel):
    schema_version: str = "1.0"
    competition_id: str
    task_type: str | None = None
    metric: dict = {}
    dataset: dict = {}

    eda_tasks: list[EdaTask] = []
    hypothesis_index: dict[str, list[str]] = {}
    recommended_module_sequence: list[str] = []
    recommended_human_checklist: list[str] = []
    blocking_tasks: list[str] = []
```

### 14.7 `DatasetFile`

```python
class DatasetFile(BaseModel):
    path: str
    name: str
    extension: str
    size_bytes: int
    size_mb: float

    role_hint: Literal[
        "train",
        "test",
        "sample_submission",
        "metadata",
        "unknown",
    ]

    table_hint: Literal[
        "base",
        "secondary",
        "depth_0",
        "depth_1",
        "depth_2",
        "submission",
        "unknown",
    ]

    can_read: bool
    read_error: str | None = None
```

### 14.8 `FileInventoryResult`

```python
class FileInventoryResult(BaseModel):
    dataset_path: str
    files: list[DatasetFile]

    detected_formats: dict[str, int]
    table_roles: dict[str, str]

    train_files: list[str]
    test_files: list[str]
    sample_submission_files: list[str]
    metadata_files: list[str]

    missing_train_test_pairs: list[dict]
    duplicate_format_pairs: list[dict]
    suspicious_files: list[dict]

    warnings: list[str]
```

### 14.9 `ColumnRole`

```python
class ColumnRole(BaseModel):
    name: str
    role: Literal[
        "target",
        "primary_id",
        "group",
        "time",
        "date",
        "prediction",
        "numeric_feature",
        "categorical_feature",
        "unknown",
    ]
    confidence: Confidence
    reason: str
```

### 14.10 `TableSchema`

```python
class TableSchema(BaseModel):
    table_name: str
    path: str

    role: Literal["train", "test", "submission", "metadata", "unknown"]
    table_type: Literal[
        "base",
        "secondary",
        "depth_0",
        "depth_1",
        "depth_2",
        "unknown",
    ]

    n_columns: int
    columns: list[dict]
    column_roles: list[ColumnRole]

    candidate_join_keys: list[str]
    candidate_time_columns: list[str]
    candidate_date_columns: list[str]

    confidence: Confidence
    warnings: list[str]
```

### 14.11 `InferredSchema`

```python
class InferredSchema(BaseModel):
    global_roles: dict

    tables: list[TableSchema]

    target_column: str | None
    primary_id_column: str | None
    prediction_column: str | None

    train_base_table: str | None
    test_base_table: str | None
    sample_submission_table: str | None

    candidate_time_columns: list[str]
    candidate_group_columns: list[str]
    candidate_date_columns: list[str]

    confidence: Confidence
    warnings: list[str]
```

### 14.12 `ColumnProfile`

```python
class ColumnProfile(BaseModel):
    name: str
    dtype: str

    missing_count: int | None = None
    missing_pct: float | None = None

    n_unique: int | None = None
    unique_ratio: float | None = None

    mean: float | None = None
    std: float | None = None
    min: float | int | str | None = None
    max: float | int | str | None = None
    q01: float | None = None
    q05: float | None = None
    q50: float | None = None
    q95: float | None = None
    q99: float | None = None

    top_values: list[dict] = []
    date_min: str | None = None
    date_max: str | None = None

    is_constant: bool = False
    is_mostly_missing: bool = False
    is_high_cardinality: bool = False
```

### 14.13 `TableProfile`

```python
class TableProfile(BaseModel):
    table_name: str
    path: str

    n_rows: int | None
    n_cols: int

    sampled: bool = False
    sample_rows: int | None = None

    columns: list[ColumnProfile]

    mostly_missing_columns: list[str]
    high_cardinality_columns: list[str]
    constant_columns: list[str]

    warnings: list[str]
```

### 14.14 `MetricEvidence`

```python
class MetricEvidence(BaseModel):
    metric_name: str
    base_metric: str | None = None

    requires_probabilities: bool | None = None
    rank_based: bool | None = None
    requires_time_or_groups: bool | None = None
    local_metric_available: bool = False

    threshold_search_needed: bool | None = None
    tie_sensitivity: Literal["low", "medium", "high"] | None = None

    components: dict = {}
    required_columns: dict = {}

    warnings: list[str] = []
```

### 14.15 `ValidationEvidence`

```python
class ValidationEvidence(BaseModel):
    time_columns: list[dict] = []
    target_by_period: list[dict] = []

    test_time_relation: dict = {}
    oot_holdout: dict = {}
    temporal_folds: dict = {}

    recommended_validation: dict = {}

    warnings: list[str] = []
    limitations: list[str] = []
```

### 14.16 `LeakageCheckResult`

```python
class LeakageCheckResult(BaseModel):
    check_id: str
    status: LeakageCheckStatus
    severity: Severity

    finding: str
    evidence: dict = {}

    related_hypothesis_ids: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []
```

### 14.17 `HypothesisResult`

```python
class HypothesisResult(BaseModel):
    hypothesis_id: str
    category: str

    status: HypothesisStatus
    confidence_after_eda: Confidence

    finding: str
    evidence_refs: list[str]

    impact_on_strategy: str
    limitations: list[str] = []
```

### 14.18 `RecommendedNextAction`

```python
class RecommendedNextAction(BaseModel):
    priority: Priority
    action: str
    why: str
    evidence_refs: list[str]
```

### 14.19 `EdaEvidencePack`

```python
class EdaEvidencePack(BaseModel):
    schema_version: str = "1.0"
    competition_id: str
    created_at: str
    run_id: str

    dataset: dict

    file_inventory: dict
    inferred_schema: dict
    table_profiles: list[dict]

    metric_evidence: dict
    validation_evidence: dict
    leakage_evidence: list[dict]

    relationship_evidence: dict = {}
    drift_evidence: dict = {}
    baseline_evidence: dict = {}
    feature_probe_evidence: list[dict] = []
    notebook_static_analysis: dict = {}

    hypothesis_results: list[HypothesisResult]
    recommended_next_actions: list[RecommendedNextAction]

    warnings: list[str]
    limitations: list[str]
    artifacts: dict
```

---

## 15. Dataset IO contracts

## 15.1 Dataset resolver

File:

```text
kaggle_researcher/eda/io/dataset_resolver.py
```

### Public functions

```python
def derive_competition_slug(
    competition_id: str,
    competition_url: str | None = None,
) -> str:
    ...

def resolve_dataset(
    competition_id: str,
    competition_url: str | None,
    local_dataset_path: Path | None,
    download: bool,
    force_download: bool,
    cache_dir: Path,
) -> Path:
    ...
```

### Behavior

```text
If local_dataset_path is provided:
  - validate it exists and is a directory;
  - return it;
  - do not download.

If cached dataset exists and force_download=false:
  - return cache_dir / competition_id.

If download=false and no local/cache exists:
  - raise DatasetNotFoundError.

If download=true:
  - call Kaggle CLI/API through a mockable helper;
  - download competition files;
  - unzip archives into cache_dir / competition_id.
```

### Invariants

```text
- Tests must not perform real downloads.
- Kaggle CLI/API calls must be mockable.
- KAGGLE_KEY must never be logged.
- Dataset cache is not deleted unless force_download=true.
```

---

## 15.2 Dataset reader

File:

```text
kaggle_researcher/eda/io/dataset_reader.py
```

### Public class

```python
class DatasetReader:
    def __init__(self, dataset_path: Path) -> None:
        ...

    def resolve_path(self, relative_path: str | Path) -> Path:
        ...

    def read_schema(self, relative_path: str | Path) -> list[dict]:
        ...

    def count_rows(self, relative_path: str | Path) -> int | None:
        ...

    def sample_table(
        self,
        relative_path: str | Path,
        n_rows: int,
        seed: int = 42,
    ):
        ...

    def read_columns(
        self,
        relative_path: str | Path,
        columns: list[str],
        n_rows: int | None = None,
    ):
        ...

    def file_head(
        self,
        relative_path: str | Path,
        n_rows: int = 5,
    ):
        ...
```

### Supported formats

```text
.csv
.parquet
.json
.jsonl
```

### Implementation rules

```text
- Prefer Polars.
- Use scan_parquet for parquet where practical.
- Use scan_csv for csv where practical.
- Return Polars DataFrame for sampled/head/column reads.
- Use pandas only as small-file fallback if needed.
- Raise ReaderError for unsupported/unreadable files.
```

---

## 16. Module contracts

## 16.1 Module 1 — `file_inventory`

File:

```text
kaggle_researcher/eda/modules/file_inventory.py
```

### Purpose

Determine what files exist in the dataset and infer basic table roles.

### Public function

```python
def build_file_inventory(dataset_path: Path) -> FileInventoryResult:
    ...
```

### Must determine

```text
- list of files;
- file sizes;
- extensions;
- train/test/sample_submission/metadata hints;
- base/secondary/depth hints;
- duplicate csv/parquet logical pairs;
- missing train/test pairs;
- suspicious files;
- readability status.
```

### Output example

```json
{
  "files": [
    {
      "path": "train/train_base.parquet",
      "name": "train_base.parquet",
      "extension": ".parquet",
      "size_bytes": 129394278,
      "size_mb": 123.4,
      "role_hint": "train",
      "table_hint": "base",
      "can_read": true,
      "read_error": null
    }
  ],
  "table_roles": {
    "train_base": "train_base",
    "test_base": "test_base",
    "sample_submission": "submission"
  },
  "detected_formats": {
    ".parquet": 12,
    ".csv": 3
  }
}
```

### Invariants

```text
- Must not load full data files.
- Must not fail entire module because one file is unreadable.
- Must record unreadable files with can_read=false and read_error.
```

---

## 16.2 Module 2 — `schema_inferer`

File:

```text
kaggle_researcher/eda/modules/schema_inferer.py
```

### Purpose

Infer semantic roles for tables and columns.

### Public function

```python
def infer_schema(
    file_inventory: FileInventoryResult,
    reader: DatasetReader,
) -> InferredSchema:
    ...
```

### Must find

```text
- target column;
- primary id column;
- group columns;
- time columns;
- date columns;
- prediction column;
- train/test base tables;
- sample submission table;
- secondary/depth tables;
- candidate join keys.
```

### Home Credit CRMS special handling

For `home-credit-credit-risk-model-stability`-like datasets, prefer:

```text
case_id
WEEK_NUM
date_decision
target
score
train_base
test_base
depth suffixes _0/_1/_2
```

### Invariants

```text
- Must not infer target from test table.
- Must use sample_submission to identify prediction column where possible.
- Must degrade with warnings if target/id/time cannot be found.
```

---

## 16.3 Module 3 — `table_profiler`

File:

```text
kaggle_researcher/eda/modules/table_profiler.py
```

### Purpose

Build statistical table profiles safely.

### Public function

```python
def profile_tables(
    file_inventory: FileInventoryResult,
    inferred_schema: InferredSchema,
    reader: DatasetReader,
    sample_rows: int = 200_000,
    max_full_scan_rows: int = 2_000_000,
) -> list[TableProfile]:
    ...
```

### Must compute

For each table:

```text
- row count;
- column count;
- dtypes;
- missing count;
- missing percentage;
- cardinality;
- unique ratio;
- numeric stats;
- categorical top values;
- date range;
- constant columns;
- high-missing columns;
- high-cardinality columns.
```

### Large-data policy

```text
- Full profile if row count is known and <= max_full_scan_rows.
- Otherwise sample sample_rows and set sampled=true.
- Preserve warnings/limitations for sampled profiles.
```

---

## 16.4 Module 4 — `metric_analyzer`

File:

```text
kaggle_researcher/eda/modules/metric_analyzer.py
```

### Purpose

Determine local metric properties and whether metric components can be computed.

### Public function

```python
def analyze_metric(
    task_plan: EdaTaskPlan,
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
) -> MetricEvidence:
    ...
```

### For Gini/AUC

```text
rank_based=true
requires_probabilities=true
threshold_search_needed=false
```

### For `gini_stability`

```text
base_metric=gini
rank_based=true
requires_probabilities=true
requires_time_or_groups=true
local_metric_available=true
components:
  - overall_gini
  - weekly_gini
  - trend_penalty
  - residual_std_penalty
threshold_search_needed=false
```

### Invariants

```text
- Unknown metrics must not crash.
- Unknown metrics return local_metric_available=false with warning.
- AUC/Gini-like metrics must not suggest threshold search.
```

---

## 16.5 Module 5 — `validation_analyzer`

File:

```text
kaggle_researcher/eda/modules/validation_analyzer.py
```

### Purpose

Build factual validation policy from data.

### Public function

```python
def analyze_validation(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    metric_evidence: MetricEvidence,
    reader: DatasetReader,
) -> ValidationEvidence:
    ...
```

### Must check

```text
- target availability;
- time column availability;
- group column availability;
- min/max/n_periods for time columns;
- row count by period;
- target rate by period;
- train/test time relation;
- latest-period holdout feasibility;
- rolling/expanding fold feasibility;
- whether random CV is likely unsafe;
- whether StratifiedGroupKFold may be diagnostic only.
```

### Primary rule

If a reliable temporal dimension exists and the metric is temporal/stability-sensitive:

```text
primary validation = out-of-time holdout + rolling/expanding temporal CV
```

`StratifiedGroupKFold` may be diagnostic only unless EDA finds no meaningful temporal dimension.

### Output example

```json
{
  "time_columns": [
    {
      "name": "WEEK_NUM",
      "source_table": "train_base",
      "min": 0,
      "max": 91,
      "n_periods": 92,
      "n_missing": 0,
      "confidence": "high"
    }
  ],
  "target_by_period": [],
  "test_time_relation": {
    "available": true,
    "interpretation": "test_after_train"
  },
  "oot_holdout": {
    "feasible": true,
    "recommended_holdout_periods": [88, 89, 90, 91]
  },
  "temporal_folds": {
    "feasible": true,
    "strategy": "expanding_window"
  },
  "recommended_validation": {
    "primary": "out_of_time_holdout_plus_rolling_cv",
    "holdout_strategy": "latest_periods",
    "cv_strategy": "expanding_window",
    "stratified_group_kfold_allowed_as_primary": false,
    "stratified_group_kfold_allowed_as_diagnostic": true,
    "random_kfold_risk": "high"
  }
}
```

---

## 16.6 Module 6 — `leakage_checker`

File:

```text
kaggle_researcher/eda/modules/leakage_checker.py
```

### Purpose

Perform basic factual leakage and data-risk checks.

### Public function

```python
def check_leakage(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
) -> list[LeakageCheckResult]:
    ...
```

### MVP checks

```text
- train/test id overlap;
- target column present in test;
- target-like column names outside target;
- sample_submission structure sanity;
- duplicate rows across train/test base tables where feasible;
- suspicious numeric columns with extremely high target association.
```

### Important distinction

```text
Confirmed issue:
  target column appears in test; train/test id overlap when it should not; exact duplicate rows where unsafe.

Risk warning:
  target-like name; very high target association; possible global target encoding risk.
```

EDA Engine must distinguish confirmed leakage from possible leakage risk.

---

## 16.7 Module 7 — `relationship_inferer` P1

File:

```text
kaggle_researcher/eda/modules/relationship_inferer.py
```

### Purpose

Infer relationships between base and secondary tables.

### Public function

```python
def infer_relationships(
    inferred_schema: InferredSchema,
    file_inventory: FileInventoryResult,
    reader: DatasetReader,
) -> dict:
    ...
```

### Must determine

```text
- base table;
- base id column;
- secondary tables;
- candidate join keys;
- relationship type;
- coverage;
- orphan rate;
- avg/max rows per base id;
- row multiplication risk;
- date cutoff feasibility.
```

### Relationship types

```text
one_to_one
one_to_many
many_to_one
many_to_many
unknown
```

### Invariants

```text
- Must not recommend direct one-to-many joins without aggregation.
- Must warn about row multiplication risk.
```

---

## 16.8 Module 8 — `drift_analyzer` P1

File:

```text
kaggle_researcher/eda/modules/drift_analyzer.py
```

### Purpose

Estimate train/test and period-level drift.

### Public function

```python
def analyze_drift(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
    max_rows: int = 500_000,
    random_seed: int = 42,
) -> dict:
    ...
```

### Must compute

```text
- target drift by period;
- row count by period;
- missingness drift;
- numeric PSI;
- categorical distribution shift;
- adversarial validation train vs test when sklearn is available;
- top adversarial features when available;
- overall drift risk.
```

### Invariants

```text
- Exclude target/id/prediction columns from adversarial features.
- Mark sampled=true when caps are used.
- If sklearn is unavailable, return enabled=false with warning.
```

---

## 16.9 Module 9 — `baseline_runner` P1

File:

```text
kaggle_researcher/eda/modules/baseline_runner.py
```

### Purpose

Run an honest minimal baseline to verify metric, validation, and sanity floor.

### Public function

```python
def run_baseline(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    metric_evidence: MetricEvidence,
    leakage_evidence: list[LeakageCheckResult],
    reader: DatasetReader,
    output_dir: Path,
    max_rows: int = 1_000_000,
    random_seed: int = 42,
) -> dict:
    ...
```

### MVP baseline rules

```text
- base-table-only;
- no target encoding;
- no secondary aggregations initially;
- no test target;
- no sample submission features;
- temporal validation if available;
- per-period metric if available;
- feature importance if model supports it.
```

### Model fallback order

```text
1. LightGBMClassifier if installed.
2. sklearn HistGradientBoostingClassifier.
3. sklearn LogisticRegression.
```

### Baseline artifacts

```text
artifacts/baseline/fold_metrics.csv
artifacts/baseline/feature_importance.csv
artifacts/baseline/oof_predictions.parquet or .csv
```

### Invariants

```text
- Baseline is disabled by default.
- Baseline requires --enable-baseline.
- Baseline is a sanity floor, not final leaderboard optimization.
```

---

## 16.10 Module 10 — `feature_probe` P1

File:

```text
kaggle_researcher/eda/modules/feature_probe.py
```

### Purpose

Assess which feature families should be built first.

### Public function

```python
def probe_feature_families(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    relationship_evidence: dict,
    leakage_evidence: list[LeakageCheckResult],
    baseline_evidence: dict,
) -> list[dict]:
    ...
```

### Feature families

```text
base_numeric_features
base_categorical_features
missingness_indicators
date_features
secondary_table_aggregations
high_cardinality_encoding
target_encoding_or_woe
```

### Output shape

```json
{
  "feature_family": "secondary_table_aggregations",
  "status": "high_potential",
  "leakage_risk": "medium",
  "evidence": {
    "num_secondary_tables": 7,
    "high_coverage_relationships": 5
  },
  "recommendation": "Start with count/mean/max/min/n_unique aggregations for high-coverage one-to-many secondary tables."
}
```

### Invariants

```text
- Must not generate final feature engineering code.
- Must mark target encoding/WoE as high leakage risk unless OOF/temporal encoding policy exists.
```

---

## 16.11 Module 11 — `notebook_static_analysis` P2

File:

```text
kaggle_researcher/eda/modules/notebook_static_analyzer.py
```

### Purpose

Statically extract patterns from notebook text/code collected by the research pipeline.

### Public function

```python
def analyze_notebooks_static(
    notebook_sources: list[SourceDocument | RetrievedDocument | dict],
    output_dir: Path | None = None,
) -> dict:
    ...
```

### Must extract

```text
- CV strategy;
- feature families;
- model families;
- metric code;
- postprocessing;
- metric tricks;
- suspicious leaderboard-overfit patterns.
```

### Must not do

```text
- do not execute notebooks;
- do not execute arbitrary code;
- do not trust notebook score as truth.
```

---

## 17. Metric helper contracts

## 17.1 Gini

File:

```text
kaggle_researcher/eda/metrics/gini.py
```

Functions:

```python
def gini_from_predictions(y_true, y_pred) -> float:
    ...

def normalized_gini(y_true, y_pred) -> float:
    ...
```

Rules:

```text
- Perfect ranking should return close to 1.0 for normalized gini.
- Reversed ranking should return close to -1.0 where applicable.
- Ties should be handled deterministically.
```

## 17.2 Gini stability

File:

```text
kaggle_researcher/eda/metrics/gini_stability.py
```

Functions:

```python
def weekly_gini(y_true, y_pred, week) -> list[dict]:
    ...

def simple_gini_stability_score(y_true, y_pred, week) -> dict:
    ...
```

Rules:

```text
- Return per-period Gini.
- Return trend/residual diagnostics when possible.
- If periods are too sparse, return warning/limitation.
```

---

## 18. Validation helper contracts

File:

```text
kaggle_researcher/eda/validation/temporal_split.py
```

Functions:

```python
def infer_periods(df, time_col: str) -> list:
    ...

def build_latest_period_holdout(
    periods: list,
    holdout_period_count: int = 4,
) -> dict:
    ...

def build_expanding_window_folds(
    periods: list,
    n_folds: int = 5,
    min_train_periods: int = 3,
) -> list[dict]:
    ...

def summarize_period_counts(
    df,
    time_col: str,
    target_col: str | None = None,
) -> list[dict]:
    ...
```

Invariants:

```text
- Deterministic output.
- Natural chronological sorting.
- Too few periods returns infeasible result with reason.
- No sklearn dependency here.
```

---

## 19. Hypothesis evaluation contract

File:

```text
kaggle_researcher/eda/modules/hypothesis_evaluator.py
```

### Public function

```python
def evaluate_hypotheses(
    hypotheses: list[ResearchHypothesis],
    evidence_pack_partial: dict,
    module_statuses: dict[str, str] | None = None,
) -> list[HypothesisResult]:
    ...
```

### Required behavior

Every input hypothesis must produce exactly one output result.

### Status rules

```text
confirmed:
  Evidence directly supports the hypothesis.

partially_confirmed:
  Some evidence supports it, but a required piece is missing or ambiguous.

rejected:
  Evidence contradicts the hypothesis.

not_testable:
  Required data/evidence is absent or module cannot safely test it.

skipped:
  Related module was disabled or failed non-fatally.
```

### MVP deterministic rules

Implement deterministic rules for at least:

```text
schema_001
metric_001
val_001
leak_001
```

### Invariants

```text
- No hypothesis may disappear.
- No duplicate hypothesis results.
- confirmed/rejected must include evidence_refs.
- not_testable/skipped must include limitations.
- impact_on_strategy must be concrete.
```

---

## 20. Recommended next actions contract

File:

```text
kaggle_researcher/eda/modules/recommendations.py
```

### Public function

```python
def build_recommended_next_actions(
    evidence_pack_partial: dict,
    hypothesis_results: list[HypothesisResult],
) -> list[RecommendedNextAction]:
    ...
```

### MVP rules

Generate actions such as:

```text
If temporal validation is feasible:
  P0: Use out-of-time holdout + expanding CV.

If metric is rank-based:
  P0/P1: Use probabilities/ranks, not hard labels.

If leakage check fails/warns:
  P0: Exclude/fix unsafe column/source before modeling.

If secondary tables exist but relationships not analyzed:
  P1: Run relationship inference before aggregation features.
```

### Invariants

```text
- Every action must include evidence_refs.
- Actions must be sorted P0 -> P3.
- No action without evidence.
```

---

## 21. Quality gates

File:

```text
kaggle_researcher/eda/quality.py
```

Functions:

```python
def validate_evidence_pack(pack: EdaEvidencePack) -> list[str]:
    ...

def validate_evidence_refs(pack: EdaEvidencePack) -> list[str]:
    ...

def validate_hypothesis_results(
    pack: EdaEvidencePack,
    hypotheses: ResearchHypotheses,
) -> list[str]:
    ...

def validate_no_unsupported_summary_claims(
    summary_text: str,
    pack: EdaEvidencePack,
) -> list[str]:
    ...
```

Checks:

```text
- Every input hypothesis has exactly one result.
- confirmed/rejected hypothesis results have evidence_refs.
- evidence_refs point to existing paths where practical.
- recommended_next_actions have evidence_refs.
- summary does not claim more than evidence pack contains.
- warnings and limitations are preserved.
```

Forbidden summary/report claims unless directly supported:

```text
"leakage found"
"baseline proves final solution"
"EDA confirmed" without evidence_refs
"public LB is reliable" without evidence
"notebook score proves"
```

Quality functions return warnings, not exceptions.

---

## 22. Module status contract

File:

```text
module_statuses.json
```

Shape:

```json
{
  "file_inventory": {
    "module": "file_inventory",
    "status": "success",
    "started_at": "2026-07-06T12:00:00+02:00",
    "finished_at": "2026-07-06T12:00:01+02:00",
    "duration_sec": 1.23,
    "error_message": null
  }
}
```

Allowed status values:

```text
success
failed
skipped
```

Rules:

```text
- Every planned module must have status.
- Error messages must be sanitized.
- No huge stack traces in JSON by default.
- No secrets in error messages.
```

---

## 23. MVP scope

MVP includes:

```text
1. dataset resolution local/cache/download shell
2. file_inventory
3. schema_inferer
4. table_profiler
5. metric_analyzer basic
6. validation_analyzer
7. leakage_checker basic
8. hypothesis_evaluator
9. recommended_next_actions
10. eda_evidence_pack.json
11. eda_summary.md
12. quality gates basic
```

MVP excludes:

```text
- relationship_inferer
- drift_analyzer
- baseline_runner
- feature_probe
- notebook_static_analysis
```

MVP should still write skipped placeholders for excluded P1/P2 modules.

---

## 24. P1 scope

P1 includes:

```text
1. relationship_inferer
2. drift_analyzer
3. baseline_runner
4. feature_probe
```

Rules:

```text
- P1 modules are non-blocking.
- P1 modules run only when enabled by CLI/config/task plan.
- Baseline requires explicit --enable-baseline.
```

---

## 25. P2 scope

P2 includes:

```text
1. notebook_static_analysis
2. simple secondary aggregation baseline
3. metric trick risk analysis
4. stronger feature-family probes
5. OOF storage and correlation diagnostics
```

P2 remains optional and non-blocking.

---

## 26. MVP acceptance criteria

This command must work with local fixtures:

```powershell
python -m kaggle_eda_engine.main `
  --competition-id fixture_competition `
  --hypotheses-path "tests\fixtures\eda\home_credit_tiny\research_hypotheses.json" `
  --task-plan-path "tests\fixtures\eda\home_credit_tiny\eda_task_plan.json" `
  --local-dataset-path "tests\fixtures\eda\home_credit_tiny" `
  --no-download-dataset
```

It must create:

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
warnings.json
limitations.json
module_statuses.json
```

`hypothesis_results.json` must include at least:

```text
schema_001
metric_001
val_001
leak_001
```

No default test may require:

```text
- Kaggle credentials;
- DeepSeek API;
- vLLM;
- PostgreSQL;
- internet access;
- LightGBM;
- notebook execution.
```

---

## 27. Offline fixture contract

Recommended fixture directory:

```text
tests/fixtures/eda/home_credit_tiny/
├── train_base.csv
├── test_base.csv
├── sample_submission.csv
├── train_static_0.csv
├── test_static_0.csv
├── research_hypotheses.json
└── eda_task_plan.json
```

Fixture requirements:

```text
- train_base has case_id, WEEK_NUM, date_decision, target.
- test_base has case_id, WEEK_NUM, date_decision and no target.
- sample_submission has case_id and score.
- static tables share case_id.
- target rate varies slightly by WEEK_NUM.
- default fixture has no train/test id overlap.
- at least one modified test fixture can introduce id overlap.
```

---

## 28. Integration with Research Scout

Research Scout should write:

```text
research_hypotheses.json
eda_task_plan.json
research_scout_summary.md
```

The existing research pipeline should support an optional flag:

```text
--write-eda-plan
```

When enabled:

```text
- run Research Scout after retrieved_documents are available;
- write EDA input files next to research_run.json;
- do not run EDA automatically unless --run-eda is also specified.
```

---

## 29. Integration with Final Synthesizer

Final Synthesizer consumes:

```text
competition_desc
plan_data
retrieved_documents
domain_patterns
research_hypotheses
eda_evidence_pack
previous reasoning outputs
```

Its strategy recommendations must connect:

```text
source claim
  -> scout hypothesis
  -> EDA result
  -> strategy action
```

Final strategy actions must include:

```text
priority
action
reason
evidence_refs
related_hypothesis_ids
```

If EDA evidence is missing, Final Synthesizer must mark a claim as hypothesis/limitation rather than fact.

---

## 30. Full workflow CLI

Optional full workflow flags in the main research CLI:

```text
--write-eda-plan
--run-eda
--local-dataset-path
--eda-output-dir
--final-synthesis
```

Behavior:

```text
Default:
  research-only mode.

With --write-eda-plan:
  write Research Scout outputs.

With --run-eda:
  run EDA Engine from generated/provided hypotheses and task plan.

With --final-synthesis:
  run Final Synthesizer using eda_evidence_pack.
```

Full workflow must remain mockable in tests and must not require real network calls by default.

---

## 31. Testing strategy

### Unit tests

```text
tests/eda/test_eda_schemas.py
tests/eda/test_eda_config.py
tests/eda/test_artifact_writer.py
tests/eda/test_dataset_resolver.py
tests/eda/test_dataset_reader.py
tests/eda/test_file_inventory.py
tests/eda/test_schema_inferer.py
tests/eda/test_table_profiler.py
tests/eda/test_metric_analyzer.py
tests/eda/test_gini_metrics.py
tests/eda/test_temporal_split.py
tests/eda/test_validation_analyzer.py
tests/eda/test_leakage_checker.py
tests/eda/test_hypothesis_evaluator.py
tests/eda/test_eda_recommendations.py
tests/eda/test_eda_quality.py
```

### Integration tests

```text
tests/eda/test_eda_orchestrator_mvp.py
tests/eda/test_eda_orchestrator_p1.py
tests/eda/test_eda_integration_full_p1.py
```

### Test rules

```text
- No default test may call real Kaggle API.
- No default test may call DeepSeek.
- No default test may require internet.
- No default test may require PostgreSQL.
- Real API tests must be explicitly marked integration.
- External calls must be mockable.
```

---

## 32. Production runbook examples

The copy-pasteable production runbook lives in `docs/RUNBOOK.md`. All Windows commands use the project-local virtual environment:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe
```

### 32.1 Install dependencies

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
E:\wavebreaker\.venv-win\Scripts\python.exe -m pip install -e .
```

### 32.2 Run research-only pipeline

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main `
  "https://www.kaggle.com/competitions/example-competition" `
  "Short competition description, target, metric, and known constraints." `
  --competition-id example-competition `
  --output-dir reports
```

Expected outputs include the research DOCX report, `roadmap.md`, `research_run.json`, and retrieval/reasoning artifacts. This mode does not inspect train/test data.

### 32.3 Generate Research Scout EDA plan

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main `
  "https://www.kaggle.com/competitions/example-competition" `
  "Short competition description, target, metric, and known constraints." `
  --competition-id example-competition `
  --mode scout `
  --output-dir reports
```

Expected outputs include `research_hypotheses.json`, `eda_task_plan.json`, `research_scout_summary.md`, and `research_scout_validation.json`.

### 32.4 Run EDA on local dataset

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id example-competition `
  --hypotheses-path runs\example-competition_YYYYMMDD_HHMMSS\research_hypotheses.json `
  --task-plan-path runs\example-competition_YYYYMMDD_HHMMSS\eda_task_plan.json `
  --local-dataset-path data\kaggle_datasets\example-competition `
  --output-dir data\eda_runs `
  --no-download-dataset
```

Expected outputs include `eda_evidence_pack.json`, `eda_summary.md`, `module_statuses.json`, module JSON artifacts, and `artifacts/`.

### 32.5 Run EDA with P1 modules

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

Additional outputs can include `relationship_evidence.json`, `drift_evidence.json`, `feature_probe_evidence.json`, and optional static notebook analysis artifacts. Notebook execution is still forbidden.

### 32.6 Run EDA with baseline

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

Baseline output is a sanity check for metric, split, and target handling. It is not a final model or public leaderboard optimization.

### 32.7 Full research to EDA to final strategy workflow

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

Expected outputs include `research_hypotheses.json`, `eda_task_plan.json`, `eda_evidence_pack.json`, `eda_summary.md`, `final_strategy.json`, `final_strategy.md`, and `final_strategy.docx` when document export is enabled for final strategy.

### 32.8 Production safety notes

```text
- Kaggle rules must be accepted before dataset download.
- Prefer local dataset mode for repeatability.
- Notebook execution is never performed.
- Baseline is a sanity check, not final score optimization.
- Large datasets may be sampled; sampled=true must be preserved in conclusions.
- Ordinary classification can use StratifiedKFold.
- Ordinary regression can use KFold.
- Grouped tasks can use group-aware validation.
- Temporal validation is primary only when evidence supports it.
- Gini Stability is supported but not the default worldview.
```

---

## 33. Requirements additions

EDA Engine may require:

```text
polars
pyarrow
scikit-learn
numpy
pandas
```

Optional:

```text
lightgbm
```

Rules:

```text
- LightGBM must not be required for tests.
- scikit-learn-dependent modules must degrade gracefully if unavailable.
- Dataset download code may depend on kaggle package or Kaggle CLI, but tests must mock it.
```

---

## 34. Codex implementation rules for EDA tasks

For tasks implementing this spec:

```text
Use docs/SPEC.md, docs/EDA_ENGINE_SPEC.md, and docs/CODEX_TASKS.md as the source of truth.
Implement only the current task.
Do not implement later tasks.
Data execution is allowed only inside kaggle_researcher/eda and kaggle_eda_engine.
Do not execute Kaggle notebooks.
Do not perform AutoML or leaderboard optimization.
Keep functions small, typed, and testable.
Add or update tests for every task.
External APIs must be mockable.
Never log secrets.
```

Recommended prompt:

```text
Use docs/SPEC.md, docs/EDA_ENGINE_SPEC.md, and docs/CODEX_TASKS.md as the source of truth.
Implement only task <TASK_ID>.
For tasks 28+, data execution is allowed only inside kaggle_researcher/eda and kaggle_eda_engine.
Do not execute Kaggle notebooks.
Keep the change small and add tests.
```

---

## 35. Final quality principle

EDA Engine must be stricter than the LLM reasoning layer.

If Scout says:

```text
"Temporal drift may matter."
```

EDA Engine must answer one of:

```text
confirmed:
  "WEEK_NUM exists, target rate changes by period, and OOT holdout is feasible."

rejected:
  "No meaningful target/feature drift was found across periods under the implemented checks."

partially_confirmed:
  "WEEK_NUM exists and folds can be built, but test time relation is unavailable."

not_testable:
  "No time column was found."

skipped:
  "Drift analyzer was disabled."
```

Every meaningful EDA conclusion must answer:

```text
What was checked?
What data supports it?
How confident are we?
What should Stage 3 do with it?
What are the limitations?
```
