# KaggleResearcher — EDA Engine Codex Tasks 28+

This file continues `docs/CODEX_TASKS.md` from task `28`.

Use this file together with:

```text
docs/SPEC.md
docs/EDA_ENGINE_SPEC.md
docs/CODEX_TASKS.md
```

---

## Scoped override for tasks `28+`

The previous global rule “Do not download Kaggle train/test datasets / Do not perform real EDA” remains valid for the original Research pipeline.

For tasks `28+`, Codex may implement data-execution features only inside the EDA Engine scope:

```text
Allowed only under kaggle_researcher/eda and kaggle_eda_engine:
- downloading Kaggle competition datasets when explicitly requested by CLI/config;
- reading local train/test/sample_submission files;
- computing schema/profile/validation/leakage/drift/baseline evidence;
- writing EDA artifacts under data/eda_runs/.

Still forbidden:
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

## 28_eda_docs_and_task_rules

### Goal

Add the EDA Engine specification to project docs and update task rules so Codex can safely implement data-execution only inside the EDA package.

### Files to create/change

```text
docs/EDA_ENGINE_SPEC.md
docs/CODEX_TASKS.md
README.md
```

### Codex prompt

```text
Add docs/EDA_ENGINE_SPEC.md based on the EDA Engine requirements.

Requirements:
- Describe EDA Engine as Stage 2 / Data Evidence Layer.
- Make clear that existing research/reasoning pipeline remains text-source based.
- Define inputs:
  - research_hypotheses.json
  - eda_task_plan.json
  - Kaggle dataset or local dataset path
- Define outputs:
  - eda_evidence_pack.json
  - eda_summary.md
  - module-level JSON artifacts
  - artifacts/
- Document the scoped override:
  - data execution is allowed only inside kaggle_researcher/eda and kaggle_eda_engine.
  - notebook execution remains forbidden.
- Document MVP modules:
  - file_inventory
  - schema_inferer
  - table_profiler
  - metric_analyzer basic
  - validation_analyzer
  - leakage_checker basic
  - hypothesis_evaluator
  - recommended_next_actions
- Update README with a short EDA Engine section and a placeholder CLI command.

Do not implement code in this task.
```

### Acceptance criteria

- `docs/EDA_ENGINE_SPEC.md` exists.
- README explains that EDA Engine is separate from the research pipeline.
- `docs/CODEX_TASKS.md` contains the scoped override for tasks `28+`.
- No Python code is changed.

---

## 29_eda_package_skeleton

### Goal

Create the EDA Engine package structure and a runnable placeholder CLI.

### Files to create/change

```text
kaggle_researcher/eda/
├── __init__.py
├── main.py
├── config.py
├── orchestrator.py
├── schemas.py
├── quality.py
├── io/
│   ├── __init__.py
│   ├── artifact_writer.py
│   ├── dataset_resolver.py
│   └── dataset_reader.py
├── modules/
│   ├── __init__.py
│   ├── file_inventory.py
│   ├── schema_inferer.py
│   ├── table_profiler.py
│   ├── metric_analyzer.py
│   ├── validation_analyzer.py
│   ├── leakage_checker.py
│   ├── hypothesis_evaluator.py
│   └── recommendations.py
├── metrics/
│   ├── __init__.py
│   ├── gini.py
│   └── gini_stability.py
└── validation/
    ├── __init__.py
    └── temporal_split.py

kaggle_eda_engine/
├── __init__.py
└── main.py

tests/eda/
```

### Codex prompt

```text
Create the EDA Engine package skeleton.

Requirements:
- Add kaggle_researcher/eda as an internal package.
- Add kaggle_eda_engine as a thin CLI wrapper around kaggle_researcher.eda.main.
- main.py should expose:
  python -m kaggle_eda_engine.main --help
- The CLI should accept but not yet execute:
  --competition-id
  --competition-url
  --hypotheses-path
  --task-plan-path
  --local-dataset-path
  --output-dir
  --download-dataset / --no-download-dataset
  --modules
  --skip-modules
  --enable-p1-modules
  --enable-baseline
  --fail-fast
- For now, running without --help may print NotImplemented.

Do not implement real dataset reading yet.
```

### Acceptance criteria

- `python -m kaggle_eda_engine.main --help` works.
- `import kaggle_researcher.eda` works.
- `pytest` passes.
- Existing `python -m kaggle_researcher.main --help` still works.

---

## 30_eda_schemas_core

### Goal

Define stable typed contracts for all EDA inputs, outputs, module results, and evidence references.

### Files to create/change

```text
kaggle_researcher/eda/schemas.py
tests/eda/test_eda_schemas.py
requirements.txt
```

### Codex prompt

```text
Implement kaggle_researcher/eda/schemas.py using Pydantic v2.

Models:
- EdaRunConfig
- EdaRunResult
- EdaTaskPlan
- EdaTask
- ResearchHypotheses
- ResearchHypothesis
- DatasetInfo
- DatasetFile
- FileInventoryResult
- ColumnRole
- TableSchema
- InferredSchema
- ColumnProfile
- TableProfile
- MetricEvidence
- ValidationEvidence
- LeakageCheckResult
- HypothesisResult
- RecommendedNextAction
- EdaEvidencePack

Enums or Literal types:
- confidence: low|medium|high
- priority: P0|P1|P2|P3
- hypothesis status: confirmed|partially_confirmed|rejected|not_testable|skipped
- leakage check status: passed|failed|warning|not_testable|skipped
- severity: low|medium|high|critical

Requirements:
- Use safe defaults for optional list/dict fields.
- Validate that competition_id in ResearchHypotheses and EdaTaskPlan can be compared by orchestrator later.
- EdaEvidencePack must include:
  schema_version, competition_id, created_at, run_id, dataset,
  file_inventory, inferred_schema, table_profiles, metric_evidence,
  validation_evidence, leakage_evidence, relationship_evidence,
  drift_evidence, baseline_evidence, feature_probe_evidence,
  notebook_static_analysis, hypothesis_results, recommended_next_actions,
  warnings, limitations, artifacts.
- Evidence refs are strings for now, but create a helper type alias or validator to reject empty refs.

Do not implement module logic yet.
```

### Acceptance criteria

- Invalid confidence/status/priority raises validation error.
- Mutable defaults are independent per instance.
- Minimal valid EdaEvidencePack validates.
- Empty evidence ref is rejected.
- Tests pass.

---

## 31_eda_config

### Goal

Add EDA-specific configuration without coupling it to DeepSeek, vLLM, or PostgreSQL.

### Files to create/change

```text
kaggle_researcher/eda/config.py
.env.example
tests/eda/test_eda_config.py
```

### Codex prompt

```text
Implement kaggle_researcher/eda/config.py.

Create:
- EdaConfigError
- EdaSettings
- load_eda_config() -> EdaSettings

Settings:
- EDA_RUNS_DIR default ./data/eda_runs
- KAGGLE_DATASETS_DIR default ./data/kaggle_datasets
- EDA_SCHEMA_VERSION default 1.0
- EDA_PROFILE_SAMPLE_ROWS default 200000
- EDA_MAX_PROFILE_ROWS_FULL_SCAN default 2000000
- EDA_MAX_ADVERSARIAL_ROWS default 500000
- EDA_MAX_BASELINE_ROWS default 1000000
- EDA_RANDOM_SEED default 42
- KAGGLE_USERNAME optional
- KAGGLE_KEY optional

Requirements:
- Do not require DEEPSEEK_API_KEY.
- Do not log secrets.
- Update .env.example with EDA settings.
- Tests must isolate environment variables with monkeypatch.
```

### Acceptance criteria

- Defaults are applied correctly.
- Missing Kaggle credentials do not fail config loading.
- Secret values are not printed in repr/logs.
- Tests pass.

---

## 32_eda_artifact_writer

### Goal

Implement the artifact writer that creates reproducible run directories and writes JSON/Markdown artifacts.

### Files to create/change

```text
kaggle_researcher/eda/io/artifact_writer.py
tests/eda/test_artifact_writer.py
```

### Codex prompt

```text
Implement ArtifactWriter.

Class:
- ArtifactWriter(output_dir: Path)

Methods:
- create_run_dir(competition_id: str, timestamp: datetime | None = None) -> Path
- write_json(name: str, data: Any) -> Path
- write_markdown(name: str, text: str) -> Path
- copy_input(path: Path, name: str) -> Path
- artifact_path(*parts: str) -> Path

Requirements:
- Run directory format:
  data/eda_runs/{competition_id}_{YYYYMMDD_HHMMSS}/
- Create subdirectories:
  artifacts/
  artifacts/plots/
  artifacts/profiles/
  artifacts/baseline/
  artifacts/drift/
  artifacts/validation/
  artifacts/samples/
- JSON should be pretty-printed with UTF-8 and stable key ordering where practical.
- Existing files can be overwritten only inside the current run dir.
- copy_input should preserve the original file content.

Do not implement EDA modules yet.
```

### Acceptance criteria

- Run directory structure is created.
- JSON and Markdown files are written.
- Input files are copied exactly.
- Tests use tmp_path and pass.

---

## 33_eda_dataset_resolver

### Goal

Implement local dataset resolution and safe Kaggle download/cache shell, fully mockable in tests.

### Files to create/change

```text
kaggle_researcher/eda/io/dataset_resolver.py
tests/eda/test_dataset_resolver.py
```

### Codex prompt

```text
Implement dataset resolver.

Functions:
- derive_competition_slug(competition_id: str, competition_url: str | None = None) -> str
- resolve_dataset(
    competition_id: str,
    competition_url: str | None,
    local_dataset_path: Path | None,
    download: bool,
    force_download: bool,
    cache_dir: Path,
) -> Path

Requirements:
- If local_dataset_path is provided:
  - validate it exists and is a directory
  - return it
  - do not download
- If cached dataset exists and force_download=false:
  - return cache_dir / competition_id
- If download=false and no local/cache exists:
  - raise clear DatasetNotFoundError
- If download=true:
  - call Kaggle CLI or Kaggle API through a small helper function that can be mocked.
  - download competition files into cache_dir / competition_id.
  - unzip downloaded archives.
- Never log KAGGLE_KEY.
- Do not download anything in tests.

Implementation note:
- Put actual Kaggle CLI invocation in a small private function `_download_with_kaggle_cli(...)` so tests can monkeypatch it.
```

### Acceptance criteria

- Local dataset path returns without download.
- Cached dataset path returns without download.
- Missing dataset with download=false raises DatasetNotFoundError.
- Download path calls mocked helper.
- Tests perform no network calls.

---

## 34_eda_dataset_reader

### Goal

Implement a safe tabular reader abstraction for CSV/Parquet/JSON files using lazy or bounded reads.

### Files to create/change

```text
kaggle_researcher/eda/io/dataset_reader.py
requirements.txt
tests/eda/test_dataset_reader.py
```

### Codex prompt

```text
Implement DatasetReader.

Class:
- DatasetReader(dataset_path: Path)

Methods:
- resolve_path(relative_path: str | Path) -> Path
- read_schema(relative_path: str | Path) -> list[dict]
- count_rows(relative_path: str | Path) -> int | None
- sample_table(relative_path: str | Path, n_rows: int, seed: int = 42)
- read_columns(relative_path: str | Path, columns: list[str], n_rows: int | None = None)
- file_head(relative_path: str | Path, n_rows: int = 5)

Requirements:
- Support .csv, .parquet, .json, .jsonl.
- Prefer polars for CSV/Parquet.
- Use lazy scan where practical.
- Never load full large files into pandas by default.
- Raise ReaderError with useful message for unsupported or unreadable files.
- Return data as Polars DataFrame for sampled/head/columns operations.
- Tests use tiny fixture files only.

Add polars and pyarrow to requirements if not present.
```

### Acceptance criteria

- CSV schema/head/count works.
- Parquet schema/head/count works.
- Unsupported extension raises ReaderError.
- read_columns returns only requested columns.
- Tests pass without large files.

---

## 35_eda_offline_fixtures

### Goal

Create deterministic tiny datasets and input JSON fixtures for EDA module development.

### Files to create/change

```text
tests/fixtures/eda/home_credit_tiny/
├── train_base.csv
├── test_base.csv
├── sample_submission.csv
├── train_static_0.csv
├── test_static_0.csv
├── research_hypotheses.json
└── eda_task_plan.json

tests/eda/test_eda_fixtures.py
```

### Codex prompt

```text
Add offline EDA fixtures.

Create a tiny Home Credit-like dataset:
- train_base.csv:
  case_id, WEEK_NUM, date_decision, target, income, ext_score, product_type
- test_base.csv:
  case_id, WEEK_NUM, date_decision, income, ext_score, product_type
- sample_submission.csv:
  case_id, score
- train_static_0.csv:
  case_id, num_group1, credit_amount, status
- test_static_0.csv:
  case_id, num_group1, credit_amount, status

Fixture requirements:
- target exists only in train_base.
- case_id exists in all tables.
- WEEK_NUM exists in train/test base.
- target rate varies slightly by WEEK_NUM.
- no train/test id overlap in default fixture.
- include one mostly-missing column and one high-cardinality-like small fixture column if practical.

Create research_hypotheses.json with at least:
- schema_001
- metric_001
- val_001
- leak_001

Create eda_task_plan.json with P0 modules:
- file_inventory
- schema_inferer
- table_profiler
- metric_analyzer
- validation_analyzer
- leakage_checker

Add tests that validate:
- fixture files exist.
- input JSON validates against EDA schemas.
- DatasetReader can read every fixture table.
```

### Acceptance criteria

- All fixture CSV files are readable.
- research_hypotheses.json validates.
- eda_task_plan.json validates.
- Tests pass offline.

---

## 36_eda_file_inventory

### Goal

Implement dataset file inventory and table role hints.

### Files to create/change

```text
kaggle_researcher/eda/modules/file_inventory.py
tests/eda/test_file_inventory.py
```

### Codex prompt

```text
Implement file_inventory module.

Function:
- build_file_inventory(dataset_path: Path) -> FileInventoryResult

Requirements:
- Recursively list files under dataset_path.
- Include:
  - relative path
  - name
  - extension
  - size_bytes
  - size_mb
  - role_hint
  - table_hint
  - can_read
  - read_error
- Detect formats:
  .csv, .parquet, .json, .jsonl, .zip
- Detect role_hint:
  train, test, sample_submission, metadata, unknown
- Detect table_hint:
  base, secondary, depth_0, depth_1, depth_2, submission, unknown
- Detect duplicate logical tables available in both csv/parquet.
- Detect missing train/test pairs by logical table name.
- Do not load full data files.
- Do not fail the whole module on one unreadable file; record can_read=false.

Use DatasetReader only for lightweight schema/header readability checks.
```

### Acceptance criteria

- Fixture train/test/sample_submission are classified correctly.
- detected_formats counts files correctly.
- unreadable file is recorded with warning instead of crashing.
- duplicate csv/parquet logical pair detection is tested.
- Tests pass.

---

## 37_eda_schema_inferer

### Goal

Infer semantic roles for tables and columns.

### Files to create/change

```text
kaggle_researcher/eda/modules/schema_inferer.py
tests/eda/test_schema_inferer.py
```

### Codex prompt

```text
Implement schema_inferer module.

Function:
- infer_schema(file_inventory: FileInventoryResult, reader: DatasetReader) -> InferredSchema

Requirements:
- Identify:
  - train_base_table
  - test_base_table
  - sample_submission_table
  - target_column
  - primary_id_column
  - prediction_column
  - candidate_time_columns
  - candidate_date_columns
  - candidate_group_columns
  - candidate_join_keys
- Use filename hints and schema/header inspection.
- Prefer:
  - target for target column
  - case_id or id for primary id
  - WEEK_NUM, date_decision, date/timestamp-like names for time/date
  - score/prediction from sample_submission as prediction column
- For Home Credit-like datasets, explicitly handle:
  - case_id
  - WEEK_NUM
  - date_decision
  - target
  - score
  - depth tables with suffix _0/_1/_2
- Assign confidence low/medium/high with reasons.
- Add warnings if target, id, or train/test base cannot be found.
- Must not infer target from test table.
```

### Acceptance criteria

- Fixture schema identifies target, case_id, WEEK_NUM, date_decision, score.
- target is not inferred from test.
- train_base and test_base are identified.
- Missing target fixture produces warning and degraded confidence.
- Tests pass.

---

## 38_eda_table_profiler

### Goal

Compute safe table profiles with sampling awareness.

### Files to create/change

```text
kaggle_researcher/eda/modules/table_profiler.py
tests/eda/test_table_profiler.py
```

### Codex prompt

```text
Implement table_profiler module.

Function:
- profile_tables(
    file_inventory: FileInventoryResult,
    inferred_schema: InferredSchema,
    reader: DatasetReader,
    sample_rows: int = 200000,
    max_full_scan_rows: int = 2000000,
) -> list[TableProfile]

Requirements:
For each readable tabular file compute:
- n_rows
- n_cols
- sampled
- sample_rows
- per-column:
  - name
  - dtype
  - missing_count
  - missing_pct
  - n_unique
  - unique_ratio
  - numeric mean/std/min/max/quantiles where applicable
  - categorical top values where applicable
  - date_min/date_max where parseable
  - is_constant
  - is_mostly_missing
  - is_high_cardinality
- table-level:
  - mostly_missing_columns
  - high_cardinality_columns
  - constant_columns
  - warnings

Large data policy:
- If row count is known and <= max_full_scan_rows, full profile is allowed.
- Otherwise sample sample_rows rows and set sampled=true.
- Never use pandas for full large-table profiling.
```

### Acceptance criteria

- Fixture profiles include all tables.
- target column profile has n_unique=2.
- mostly-missing and constant columns are detected in tests.
- sampled flag is true when max_full_scan_rows is set below fixture row count.
- Tests pass.

---

## 39_eda_metric_analyzer_basic

### Goal

Implement basic metric evidence and local Gini/Gini stability helpers.

### Files to create/change

```text
kaggle_researcher/eda/modules/metric_analyzer.py
kaggle_researcher/eda/metrics/gini.py
kaggle_researcher/eda/metrics/gini_stability.py
tests/eda/test_metric_analyzer.py
tests/eda/test_gini_metrics.py
```

### Codex prompt

```text
Implement basic metric analyzer.

Functions:
In metrics/gini.py:
- gini_from_predictions(y_true, y_pred) -> float
- normalized_gini(y_true, y_pred) -> float

In metrics/gini_stability.py:
- weekly_gini(y_true, y_pred, week) -> list[dict]
- simple_gini_stability_score(y_true, y_pred, week) -> dict

In modules/metric_analyzer.py:
- analyze_metric(task_plan: EdaTaskPlan, inferred_schema: InferredSchema, table_profiles: list[TableProfile]) -> MetricEvidence

Requirements:
- For metric name gini or auc:
  - rank_based=true
  - requires_probabilities=true
  - threshold_search_needed=false
- For gini_stability:
  - base_metric=gini
  - rank_based=true
  - requires_probabilities=true
  - requires_time_or_groups=true
  - components include weekly_gini, trend_penalty, residual_std_penalty
- If required time column is missing, add warning.
- Unknown metric should not fail; return local_metric_available=false with warning.

Do not train models in this task.
```

### Acceptance criteria

- Gini is 1.0 for perfect ranking and near -1.0 for reversed ranking where applicable.
- gini_stability returns per-week components.
- gini_stability evidence requires WEEK_NUM/time.
- Unknown metric is handled gracefully.
- Tests pass.

---

## 40_eda_temporal_split_helpers

### Goal

Implement deterministic temporal holdout and rolling/expanding fold construction helpers.

### Files to create/change

```text
kaggle_researcher/eda/validation/temporal_split.py
tests/eda/test_temporal_split.py
```

### Codex prompt

```text
Implement temporal validation helpers.

Functions:
- infer_periods(df, time_col: str) -> list
- build_latest_period_holdout(periods: list, holdout_period_count: int = 4) -> dict
- build_expanding_window_folds(periods: list, n_folds: int = 5, min_train_periods: int = 3) -> list[dict]
- summarize_period_counts(df, time_col: str, target_col: str | None = None) -> list[dict]

Requirements:
- Deterministic output.
- Sort periods naturally.
- For target_col:
  - include n_rows and target_mean by period.
- If there are too few periods, return infeasible result with reason.
- Do not import sklearn here.
```

### Acceptance criteria

- Latest-period holdout picks latest periods.
- Expanding folds are chronological and non-overlapping in validation periods.
- Too few periods returns infeasible result.
- Period summary includes target mean when target exists.
- Tests pass.

---

## 41_eda_validation_analyzer

### Goal

Build factual validation evidence from schema, metric, and real train/test base tables.

### Files to create/change

```text
kaggle_researcher/eda/modules/validation_analyzer.py
tests/eda/test_validation_analyzer.py
```

### Codex prompt

```text
Implement validation_analyzer module.

Function:
- analyze_validation(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    metric_evidence: MetricEvidence,
    reader: DatasetReader,
) -> ValidationEvidence

Requirements:
- Use train_base_table from inferred_schema.
- Determine available time columns.
- For each time column:
  - min
  - max
  - n_periods
  - n_missing
  - confidence
- If target and time exist:
  - compute row count by period
  - compute target rate by period
- If test_base has same time column:
  - compute train/test time relation.
- Build latest-period holdout recommendation when feasible.
- Build expanding-window CV recommendation when feasible.
- If metric requires time/groups and time exists:
  - recommend out_of_time_holdout_plus_rolling_cv as primary.
- StratifiedGroupKFold can be diagnostic but not primary when temporal validation is feasible.
- If no time column exists:
  - explain fallback and limitations.
```

### Acceptance criteria

- Fixture recommends out_of_time_holdout_plus_rolling_cv.
- target_by_period is populated.
- test_time_relation is populated when test has WEEK_NUM.
- No time column fixture returns limitation, not crash.
- Tests pass.

---

## 42_eda_leakage_checker_basic

### Goal

Implement basic factual leakage checks.

### Files to create/change

```text
kaggle_researcher/eda/modules/leakage_checker.py
tests/eda/test_leakage_checker.py
```

### Codex prompt

```text
Implement leakage_checker basic module.

Function:
- check_leakage(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
) -> list[LeakageCheckResult]

MVP checks:
- train/test id overlap
- target column present in test
- target-like column names outside target
- sample_submission structure sanity
- duplicate rows across train/test for base tables where feasible
- suspicious columns with extremely high target association for numeric columns in train_base

Requirements:
- Distinguish confirmed issue from risk warning.
- id overlap:
  - passed when overlap_count=0
  - failed or warning when overlap_count>0 depending on competition context
- target present in test is high/critical severity.
- suspicious high target association is warning, not confirmed leakage.
- If required columns/tables are missing, return not_testable check.
- Do not scan huge tables without sampling; for MVP base fixture full scan is OK.
```

### Acceptance criteria

- Default fixture has passed id overlap check.
- Modified fixture with overlapping case_id detects overlap.
- Test target column is detected as high severity.
- Missing id returns not_testable.
- Tests pass.

---

## 43_eda_hypothesis_evaluator

### Goal

Evaluate every Research Scout hypothesis against collected EDA evidence.

### Files to create/change

```text
kaggle_researcher/eda/modules/hypothesis_evaluator.py
tests/eda/test_hypothesis_evaluator.py
```

### Codex prompt

```text
Implement hypothesis_evaluator.

Function:
- evaluate_hypotheses(
    hypotheses: list[ResearchHypothesis],
    evidence_pack_partial: dict,
    module_statuses: dict[str, str] | None = None,
) -> list[HypothesisResult]

Requirements:
- Every input hypothesis must produce exactly one HypothesisResult.
- Support categories:
  - schema
  - metric
  - validation
  - leakage
  - relationship
  - drift
  - baseline
  - feature
  - notebook
  - data_quality
- Status rules:
  - confirmed if evidence directly supports claim.
  - partially_confirmed if only part is supported.
  - rejected if evidence contradicts claim.
  - not_testable if required data/evidence is absent.
  - skipped if related module was disabled/skipped.
- For MVP, implement deterministic rules for:
  - schema_001
  - metric_001
  - val_001
  - leak_001
- Generic fallback:
  - not_testable with limitation if category cannot be evaluated.
- confirmed/rejected must include evidence_refs.
- not_testable/skipped must include limitations.
- impact_on_strategy must be concrete.
```

### Acceptance criteria

- All fixture hypotheses are evaluated.
- No hypothesis ID is missing or duplicated.
- confirmed results include evidence_refs.
- skipped/not_testable include limitations.
- Tests pass.

---

## 44_eda_recommendations

### Goal

Build evidence-backed recommended next actions for Stage 3.

### Files to create/change

```text
kaggle_researcher/eda/modules/recommendations.py
tests/eda/test_eda_recommendations.py
```

### Codex prompt

```text
Implement recommended next actions builder.

Function:
- build_recommended_next_actions(
    evidence_pack_partial: dict,
    hypothesis_results: list[HypothesisResult],
) -> list[RecommendedNextAction]

Requirements:
- Generate P0/P1/P2/P3 actions from confirmed or partially_confirmed evidence.
- MVP rules:
  - If temporal validation feasible, P0 action recommends OOT holdout + expanding CV.
  - If metric is rank-based, P0/P1 action says use probabilities/ranks, not class labels.
  - If leakage check warns/fails, P0 action recommends fixing or excluding unsafe feature/source.
  - If schema has secondary tables but relationship module not run, P1 action recommends running relationship inference before aggregation features.
- Every action must include:
  - priority
  - action
  - why
  - evidence_refs
- Sort by priority.
- Do not invent actions without evidence_refs.
```

### Acceptance criteria

- Fixture produces validation and metric actions.
- Actions are sorted P0 -> P3.
- Every action has non-empty evidence_refs.
- Tests pass.

---

## 45_eda_mvp_orchestrator_and_cli

### Goal

Wire P0 EDA modules into a working local-dataset MVP that writes `eda_evidence_pack.json`.

### Files to create/change

```text
kaggle_researcher/eda/orchestrator.py
kaggle_researcher/eda/main.py
kaggle_eda_engine/main.py
tests/eda/test_eda_orchestrator_mvp.py
tests/eda/test_eda_cli.py
```

### Codex prompt

```text
Implement the EDA MVP orchestrator and CLI.

Function:
- async run_eda(config: EdaRunConfig) -> EdaRunResult

Execution order:
1. load and validate research_hypotheses.json
2. load and validate eda_task_plan.json
3. verify competition_id consistency
4. create run directory
5. copy input JSON files into run directory
6. resolve dataset path
7. build DatasetReader
8. file_inventory
9. schema_inferer
10. table_profiler
11. metric_analyzer
12. validation_analyzer
13. leakage_checker
14. write module JSON artifacts
15. evaluate_hypotheses
16. build recommended_next_actions
17. build eda_evidence_pack.json
18. build eda_summary.md
19. return EdaRunResult

Requirements:
- Support local dataset path mode.
- Dataset download path may exist but tests must use local dataset.
- Non-blocking module failures become warnings.
- MVP modules are blocking unless fail_fast=false and partial evidence is available.
- Write skipped placeholders for P1 modules:
  - relationship_evidence
  - drift_evidence
  - baseline_evidence
  - feature_probe_evidence
  - notebook_static_analysis
- CLI command:
  python -m kaggle_eda_engine.main --competition-id fixture_competition --hypotheses-path ... --task-plan-path ... --local-dataset-path ...
```

### Acceptance criteria

- Offline fixture run creates:
  - eda_evidence_pack.json
  - eda_summary.md
  - file_inventory.json
  - inferred_schema.json
  - table_profiles.json
  - metric_evidence.json
  - validation_evidence.json
  - leakage_evidence.json
  - hypothesis_results.json
  - recommended_next_actions.json
- `hypothesis_results.json` includes schema_001, metric_001, val_001, leak_001.
- CLI test runs without network calls.
- Tests pass.

---

## 46_eda_relationship_inferer

### Goal

Infer relationships between base and secondary tables.

### Files to create/change

```text
kaggle_researcher/eda/modules/relationship_inferer.py
tests/eda/test_relationship_inferer.py
```

### Codex prompt

```text
Implement relationship_inferer module.

Function:
- infer_relationships(
    inferred_schema: InferredSchema,
    file_inventory: FileInventoryResult,
    reader: DatasetReader,
) -> dict

Requirements:
- Identify base_table and base_id_column.
- For each secondary train table:
  - find candidate join keys shared with base table.
  - compute relationship_type:
    one_to_one, one_to_many, many_to_one, many_to_many, unknown
  - compute coverage_left_to_right.
  - compute orphan_rate_right.
  - compute avg_rows_per_left and max_rows_per_left.
  - compute row_multiplication_risk low/medium/high.
  - detect candidate date cutoff columns.
  - assign confidence.
- Must not recommend direct one-to-many joins without aggregation.
- Must support sampled checks for large tables.
```

### Acceptance criteria

- Fixture train_static_0 is detected as related to train_base by case_id.
- one-to-many relationship is detected when fixture has multiple rows per case_id.
- row_multiplication_risk is populated.
- Missing join key returns unknown relationship with warning.
- Tests pass.

---

## 47_eda_drift_analyzer

### Goal

Analyze period drift and train/test distribution shift.

### Files to create/change

```text
kaggle_researcher/eda/modules/drift_analyzer.py
tests/eda/test_drift_analyzer.py
```

### Codex prompt

```text
Implement drift_analyzer module.

Function:
- analyze_drift(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
    max_rows: int = 500000,
    random_seed: int = 42,
) -> dict

Requirements:
- Compute target drift by period when target and time exist:
  - target_by_period
  - slope or simple trend indicator
  - severity
- Compute row count by period.
- Compute missingness drift train vs test for shared columns.
- Compute numeric PSI for shared numeric columns.
- Compute categorical shift for shared categorical columns.
- Implement adversarial validation only if sklearn is available:
  - safe features only
  - exclude target, id, prediction columns
  - cap rows at max_rows
  - return AUC and top features if model supports it
- If sklearn unavailable, return enabled=false with warning.
- Mark sampled=true when max_rows cap is used.
```

### Acceptance criteria

- Fixture drift output includes target_drift and missingness_drift.
- Artificially shifted fixture produces higher drift severity.
- Adversarial validation can be disabled gracefully.
- target/id columns are excluded from adversarial features.
- Tests pass.

---

## 48_eda_baseline_runner_base_table

### Goal

Run an honest base-table-only baseline using the recommended validation policy.

### Files to create/change

```text
kaggle_researcher/eda/modules/baseline_runner.py
tests/eda/test_baseline_runner.py
requirements.txt
```

### Codex prompt

```text
Implement baseline_runner module for base-table-only baseline.

Function:
- run_baseline(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    metric_evidence: MetricEvidence,
    leakage_evidence: list[LeakageCheckResult],
    reader: DatasetReader,
    output_dir: Path,
    max_rows: int = 1000000,
    random_seed: int = 42,
) -> dict

Requirements:
- Baseline is optional and must be enabled by orchestrator flag.
- Use train_base only.
- Exclude:
  - target
  - id columns
  - time columns only if they are unsafe; otherwise allow as explicit option? For MVP, exclude raw date strings and keep numeric WEEK_NUM only if needed for fold diagnostics, not as feature.
  - prediction/sample submission columns
  - columns flagged by critical leakage checks
- Use validation_evidence recommended holdout/folds.
- Model preference:
  - LightGBMClassifier if lightgbm installed.
  - fallback to sklearn HistGradientBoostingClassifier.
  - fallback to LogisticRegression if needed.
- Basic preprocessing:
  - numeric fill missing
  - categorical one-hot or ordinal encoding with safe train-only fit
- Compute:
  - overall metric
  - base Gini/AUC where applicable
  - per-period metric where applicable
  - feature importance if available
- Write artifacts:
  - artifacts/baseline/fold_metrics.csv
  - artifacts/baseline/feature_importance.csv
  - artifacts/baseline/oof_predictions.parquet or csv
- Do not train on test.
- Do not use target encoding.
- Do not optimize leaderboard score.
```

### Acceptance criteria

- Fixture baseline runs with fallback model.
- Metrics are written to artifacts/baseline.
- target/id columns are not in feature list.
- Baseline uses temporal holdout when available.
- Tests pass without requiring LightGBM.

---

## 49_eda_feature_probe

### Goal

Assess which feature families are promising and safe to build first.

### Files to create/change

```text
kaggle_researcher/eda/modules/feature_probe.py
tests/eda/test_feature_probe.py
```

### Codex prompt

```text
Implement feature_probe module.

Function:
- probe_feature_families(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    relationship_evidence: dict,
    leakage_evidence: list[LeakageCheckResult],
    baseline_evidence: dict,
) -> list[dict]

Feature families to evaluate:
- base_numeric_features
- base_categorical_features
- missingness_indicators
- date_features
- secondary_table_aggregations
- high_cardinality_encoding
- target_encoding_or_woe

Requirements:
- Return for each family:
  - feature_family
  - status: high_potential|medium_potential|low_potential|unsafe|not_testable
  - leakage_risk: low|medium|high
  - evidence
  - recommendation
- Mark target_encoding_or_woe as high leakage risk unless OOF/temporal encoding policy exists.
- Mark secondary_table_aggregations high/medium potential only when relationships are known.
- Do not generate actual feature engineering code.
```

### Acceptance criteria

- Fixture returns secondary_table_aggregations as not_testable before relationship evidence.
- With relationship evidence, secondary_table_aggregations becomes medium/high potential.
- target_encoding_or_woe is high leakage risk.
- Tests pass.

---

## 50_eda_notebook_static_analysis

### Goal

Statically extract CV/model/feature patterns from collected notebook source text without executing notebooks.

### Files to create/change

```text
kaggle_researcher/eda/modules/notebook_static_analyzer.py
tests/eda/test_notebook_static_analyzer.py
```

### Codex prompt

```text
Implement notebook_static_analyzer.

Function:
- analyze_notebooks_static(
    notebook_sources: list[SourceDocument | RetrievedDocument | dict],
    output_dir: Path | None = None,
) -> dict

Requirements:
- Static text/code pattern extraction only.
- Do not execute notebook code.
- Extract patterns:
  - cv_strategy
  - feature_families
  - model_families
  - metric_code
  - postprocessing
  - suspicious_leaderboard_overfit_patterns
- Detect common strings:
  - StratifiedKFold
  - GroupKFold
  - StratifiedGroupKFold
  - TimeSeriesSplit
  - WEEK_NUM
  - LightGBM, CatBoost, XGBoost
  - target encoding
  - adversarial validation
  - rank averaging
  - clipping
- Return warnings when notebooks use non-temporal CV in temporal/stability tasks.
- Notebook scores are observations, not truth.
```

### Acceptance criteria

- Static fixture notebook text detects model and CV patterns.
- No code execution occurs.
- Warnings are produced for risky CV pattern.
- Tests pass.

---

## 51_eda_p1_orchestrator_wiring

### Goal

Wire optional P1 modules into the EDA orchestrator and CLI.

### Files to create/change

```text
kaggle_researcher/eda/orchestrator.py
kaggle_researcher/eda/main.py
tests/eda/test_eda_orchestrator_p1.py
```

### Codex prompt

```text
Wire P1 modules into run_eda.

Modules:
- relationship_inferer
- drift_analyzer
- baseline_runner
- feature_probe
- notebook_static_analysis

Requirements:
- P1 modules run only when:
  - --enable-p1-modules is set, or
  - module is explicitly listed in --modules, or
  - eda_task_plan recommended_module_sequence includes it and config allows P1.
- baseline_runner additionally requires --enable-baseline.
- Non-blocking P1 failures should:
  - write module JSON with status=failed/skipped
  - add warning
  - continue run
- Update EdaEvidencePack fields:
  - relationship_evidence
  - drift_evidence
  - baseline_evidence
  - feature_probe_evidence
  - notebook_static_analysis
- Update eda_summary.md with P1 sections when available.
```

### Acceptance criteria

- MVP run still works without P1 flags.
- P1 fixture run writes relationship_evidence.json and drift_evidence.json.
- Baseline does not run unless --enable-baseline.
- P1 failure is warning, not fatal.
- Tests pass.

---

## 52_eda_p1_hypothesis_and_recommendation_rules

### Goal

Extend hypothesis evaluation and recommendations to use P1 evidence.

### Files to create/change

```text
kaggle_researcher/eda/modules/hypothesis_evaluator.py
kaggle_researcher/eda/modules/recommendations.py
tests/eda/test_hypothesis_evaluator_p1.py
tests/eda/test_eda_recommendations_p1.py
```

### Codex prompt

```text
Extend hypothesis evaluator and recommended actions for P1 evidence.

Add deterministic evaluation rules for:
- relationship hypotheses:
  - confirmed when join keys and coverage are found.
  - partially_confirmed when only weak candidate keys exist.
- drift hypotheses:
  - confirmed when drift severity is medium/high.
  - rejected when drift checks show stable distributions.
  - not_testable when test/shared columns unavailable.
- baseline hypotheses:
  - confirmed when honest baseline completed.
  - skipped when baseline disabled.
- feature hypotheses:
  - confirmed or partially_confirmed from feature_probe statuses.
- notebook hypotheses:
  - confirmed only as "pattern observed", not as factual performance proof.

Extend recommendations:
- relationship one-to-many -> aggregate before join.
- high drift -> trust temporal CV over random CV/public LB.
- baseline complete -> use as sanity floor, not final solution.
- high-potential feature family -> prioritize P1/P2 experiment.
- risky notebook pattern -> audit before copying.
```

### Acceptance criteria

- P1 hypotheses are evaluated when evidence exists.
- Baseline disabled produces skipped, not failure.
- Drift recommendation cites drift evidence.
- Relationship recommendation cites relationship evidence.
- Tests pass.

---

## 53_research_scout_schemas

### Goal

Define schemas for Research Scout outputs that feed EDA Engine.

### Files to create/change

```text
kaggle_researcher/research_scout/
├── __init__.py
├── schemas.py
└── prompts.py

tests/test_research_scout_schemas.py
```

### Codex prompt

```text
Create research_scout schemas.

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
```

### Acceptance criteria

- Research Scout output validates.
- Generated EDA input JSON validates against EDA schemas.
- Invalid hypothesis status/category fails validation.
- Tests pass.

---

## 54_research_scout_reasoner

### Goal

Implement the Research Scout reasoning module that generates EDA hypotheses from retrieved sources.

### Files to create/change

```text
kaggle_researcher/research_scout/scout.py
tests/test_research_scout.py
```

### Codex prompt

```text
Implement Research Scout.

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
- Must include at least:
  - schema_001
  - metric_001
  - val_001
  - leak_001
- Validate output against research_scout schemas and EDA schemas.
- On LLM failure, provide deterministic fallback from PlanData.
```

### Acceptance criteria

- Mock LLM response validates and writes expected objects.
- Fallback output includes P0 hypotheses.
- Every hypothesis has expected_eda_checks.
- Tests pass.

---

## 55_research_pipeline_writes_scout_outputs

### Goal

Wire Research Scout into `run_research` so the research pipeline can produce EDA input files.

### Files to create/change

```text
kaggle_researcher/main.py
tests/test_pipeline_research_scout_outputs.py
```

### Codex prompt

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
```

### Acceptance criteria

- Mocked pipeline writes all three Scout outputs.
- EDA input JSON files validate.
- Without --write-eda-plan, no Scout files are written.
- Tests pass.

---

## 56_final_synthesizer_schema

### Goal

Define a structured contract for final strategy synthesis that can consume EDA evidence.

### Files to create/change

```text
kaggle_researcher/reasoning/final_synthesizer.py
tests/test_final_synthesizer_schema.py
```

### Codex prompt

```text
Create final_synthesizer contract.

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
```

### Acceptance criteria

- FinalStrategyResult validates.
- Missing evidence_refs on action fails validation.
- Placeholder raises NotImplementedError.
- Tests pass.

---

## 57_final_synthesizer_reasoner

### Goal

Implement the final strategy synthesizer that combines retrieved sources and EDA evidence.

### Files to create/change

```text
kaggle_researcher/reasoning/final_synthesizer.py
tests/test_final_synthesizer.py
```

### Codex prompt

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
- If EDA evidence is missing for a claim, mark it as hypothesis or limitation.
```

### Acceptance criteria

- Mock LLM response validates into FinalStrategyResult.
- Recommendations include evidence_refs.
- Prompt includes source -> hypothesis -> EDA -> strategy rule.
- Tests pass.

---

## 58_full_research_to_eda_to_strategy_cli

### Goal

Add an optional full workflow that runs research, writes Scout outputs, runs EDA, and synthesizes final strategy.

### Files to create/change

```text
kaggle_researcher/main.py
kaggle_researcher/eda/orchestrator.py
tests/test_full_research_eda_strategy_mocked.py
```

### Codex prompt

```text
Add optional full workflow.

CLI flags:
- --write-eda-plan
- --run-eda
- --local-dataset-path
- --eda-output-dir
- --final-synthesis

Behavior:
- Existing research-only mode remains default.
- If --write-eda-plan:
  - write Research Scout outputs.
- If --run-eda:
  - require either --local-dataset-path or dataset download config.
  - run EDA Engine using generated or provided hypotheses/task plan.
- If --final-synthesis:
  - require eda_evidence_pack from current or provided run.
  - run final_synthesizer.
  - write final_strategy.json/md/docx where practical.
- All external calls must be mockable.
- Default tests must use local fixture dataset and mocked DeepSeek.
```

### Acceptance criteria

- Research-only mode still passes existing tests.
- Full mocked workflow produces:
  - research_run.json
  - research_hypotheses.json
  - eda_task_plan.json
  - eda_evidence_pack.json
  - final_strategy.json
- No real network calls in tests.
- Tests pass.

---

## 59_eda_quality_gates

### Goal

Validate EDA outputs before they are used by Final Synthesizer.

### Files to create/change

```text
kaggle_researcher/eda/quality.py
tests/eda/test_eda_quality.py
```

### Codex prompt

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
```

### Acceptance criteria

- Missing hypothesis result creates warning.
- Broken evidence_ref creates warning.
- Empty action evidence_refs creates warning.
- Quality functions return warnings, not exceptions.
- Tests pass.

---

## 60_eda_resource_limits_and_sampling

### Goal

Centralize row caps, memory-safe sampling, and module runtime limits.

### Files to create/change

```text
kaggle_researcher/eda/io/dataset_reader.py
kaggle_researcher/eda/config.py
kaggle_researcher/eda/modules/table_profiler.py
kaggle_researcher/eda/modules/drift_analyzer.py
tests/eda/test_eda_resource_limits.py
```

### Codex prompt

```text
Add resource limit handling.

Requirements:
- Add settings:
  - EDA_MAX_TABLE_BYTES
  - EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS
  - EDA_MODULE_TIMEOUT_SEC
- DatasetReader should expose lightweight file size info.
- Profiling should automatically sample if file size or row count exceeds caps.
- Drift/adversarial validation should cap train/test rows.
- Every sampled/capped result must include:
  - sampled=true
  - sample_rows
  - limitation/warning
- Do not use OS-specific memory APIs unless optional.
```

### Acceptance criteria

- Low caps force sampling in tests.
- sampled=true appears in relevant outputs.
- Warnings mention caps.
- Tests pass.

---

## 61_eda_error_handling_and_partial_runs

### Goal

Make partial EDA runs reproducible and useful even when some modules fail.

### Files to create/change

```text
kaggle_researcher/eda/orchestrator.py
kaggle_researcher/eda/io/artifact_writer.py
tests/eda/test_eda_partial_runs.py
```

### Codex prompt

```text
Improve EDA error handling.

Requirements:
- Define module status object:
  - module
  - status: success|failed|skipped
  - started_at
  - finished_at
  - duration_sec
  - error_message
- Write module_statuses.json.
- If a non-blocking module fails:
  - write failed placeholder JSON
  - continue
- If a blocking module fails:
  - write partial evidence_pack if possible
  - fail run unless fail_fast=false and fallback output exists
- EdaRunResult should include module_statuses.
- All exceptions should be sanitized: no secrets, no huge stack traces in JSON by default.
```

### Acceptance criteria

- Simulated P1 failure still creates evidence pack.
- Simulated blocking failure writes partial artifacts.
- module_statuses.json is written.
- Tests pass.

---

## 62_eda_summary_generator

### Goal

Generate a concise human-readable `eda_summary.md` from evidence pack without adding unsupported claims.

### Files to create/change

```text
kaggle_researcher/eda/summary.py
kaggle_researcher/eda/orchestrator.py
tests/eda/test_eda_summary.py
```

### Codex prompt

```text
Implement EDA summary generator.

Function:
- build_eda_summary(pack: EdaEvidencePack) -> str

Required sections:
- Dataset
- Schema
- Metric
- Validation
- Leakage
- Relationships
- Drift
- Baseline
- Feature probes
- Hypothesis results
- Recommended next actions
- Warnings
- Limitations

Requirements:
- Summary is derived only from EdaEvidencePack.
- Include evidence_refs in bullet text where useful.
- Mark skipped/not_testable clearly.
- Do not add strategy beyond recommended_next_actions.
- Replace existing inline summary creation in orchestrator with this function.
```

### Acceptance criteria

- Summary contains all required sections.
- Summary does not mention modules that are absent except as skipped/not_testable.
- Warnings and limitations are included.
- Tests pass.

---

## 63_eda_integration_fixture_full_p1

### Goal

Add an offline integration test that runs MVP + P1 modules on fixture data.

### Files to create/change

```text
tests/eda/test_eda_integration_full_p1.py
tests/fixtures/eda/home_credit_tiny/
```

### Codex prompt

```text
Add full offline EDA integration test.

Requirements:
- Use home_credit_tiny fixture.
- Run:
  - MVP modules
  - relationship_inferer
  - drift_analyzer
  - feature_probe
- Do not require baseline unless fallback sklearn model is available and test is stable.
- Validate:
  - eda_evidence_pack.json exists.
  - relationship_evidence has relationships.
  - drift_evidence has target_drift.
  - feature_probe_evidence is non-empty.
  - hypothesis_results cover all input hypotheses.
  - recommended_next_actions is non-empty.
  - quality gates return no critical warnings.
```

### Acceptance criteria

- Test runs offline.
- Test does not require Kaggle credentials.
- Test does not require DeepSeek.
- Test passes in CI.

---

## 64_eda_production_cli_docs

### Goal

Document practical commands and expected outputs for EDA Engine production use.

### Files to create/change

```text
README.md
docs/EDA_ENGINE_SPEC.md
docs/RUNBOOK.md
```

### Codex prompt

```text
Add production usage documentation for EDA Engine.

Include commands:
1. Run research-only pipeline.
2. Generate Research Scout EDA plan.
3. Run EDA MVP with local dataset.
4. Run EDA with P1 modules.
5. Run EDA with baseline.
6. Run full research -> EDA -> final strategy workflow.

Use Windows PowerShell examples with:
E:\wavebreaker\.venv-win\Scripts\python.exe

Document expected outputs:
- research_hypotheses.json
- eda_task_plan.json
- eda_evidence_pack.json
- eda_summary.md
- final_strategy.json/md/docx

Document safety notes:
- Kaggle rules must be accepted before dataset download.
- Notebook execution is never performed.
- Baseline is sanity check, not final score optimization.
- Large datasets may be sampled and sampled=true must be respected.
```

### Acceptance criteria

- README has EDA section.
- RUNBOOK has copy-pasteable commands.
- Docs mention local dataset mode.
- Docs mention no notebook execution.
- No tests required unless docs lint exists.

---

# Recommended Codex usage pattern for EDA tasks

For each EDA task, use a prompt like this:

```text
Use docs/SPEC.md, docs/EDA_ENGINE_SPEC.md, and docs/CODEX_TASKS.md as the source of truth.
Implement only task <TASK_ID>.
Do not implement later tasks.
For tasks 28+, data execution is allowed only inside kaggle_researcher/eda and kaggle_eda_engine.
Do not execute Kaggle notebooks.
Keep the change small and add tests.
```

After each task:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests/eda
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main --help
```

For full regression:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main --help
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main --help
```
