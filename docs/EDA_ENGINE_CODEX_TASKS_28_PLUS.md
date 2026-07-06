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

## 39a_eda_metric_registry_and_validation_policy

### Goal

Make EDA Engine generic for tabular Kaggle tasks, not Home Credit-specific.

### Codex prompt

Implement:
- MetricSpec
- MetricRegistry
- ValidationPolicySelector

Support metric families:
- ranking_metric: auc, gini, map@k, ndcg
- probabilistic_metric: logloss
- threshold_metric: f1, accuracy, precision, recall
- regression_metric: rmse, rmsle, mae, mape, smape, r2
- ordinal_metric: quadratic_weighted_kappa
- survival_metric: concordance_index
- unknown_metric

Validation policies:
- stratified_kfold
- kfold
- group_kfold
- stratified_group_kfold
- temporal_holdout
- expanding_window
- ranking_group_cv
- custom_required

Rules:
- Temporal CV is primary only if time evidence exists or metric/task requires temporal validation.
- Group CV is primary if group/entity leakage risk exists.
- StratifiedKFold is valid for ordinary iid classification.
- KFold is valid for ordinary iid regression.
- Unknown/custom metric should produce local_metric_available=false and recommended manual metric implementation.

Do not remove Home Credit heuristics; move them into presets.

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

## 40_eda_validation_policy_and_split_helpers

### Goal

Implement generic validation policy helpers instead of assuming temporal validation by default.

This task replaces the previous temporal-only task.

The EDA Engine must support ordinary iid tabular tasks, grouped tasks, ranking tasks, temporal tasks, and forecasting-like tabular tasks.

A time column alone is **not sufficient** to make temporal validation primary.

### Files to create/change

```text
kaggle_researcher/eda/validation/
├── __init__.py
├── split_helpers.py
├── temporal_split.py
├── group_split.py
└── policy_selector.py

tests/eda/test_split_helpers.py
tests/eda/test_temporal_split.py
tests/eda/test_group_split.py
tests/eda/test_validation_policy_selector.py
```

### Codex prompt

```text
Implement generic validation policy helpers and a validation policy selector.

Use docs/EDA_ENGINE_SPEC.md as source of truth, but do not assume Home Credit or Gini Stability as the default case.

Implement split helper functions:

In validation/split_helpers.py:
- infer_class_balance(df, target_col: str) -> dict
- infer_regression_target_stats(df, target_col: str) -> dict
- infer_candidate_group_columns(schema, profiles) -> list[dict]
- infer_candidate_time_columns(schema, profiles) -> list[dict]
- summarize_column_distribution(df, col: str, target_col: str | None = None) -> list[dict]

In validation/temporal_split.py:
- infer_periods(df, time_col: str) -> list
- build_latest_period_holdout(periods: list, holdout_period_count: int = 4) -> dict
- build_expanding_window_folds(periods: list, n_folds: int = 5, min_train_periods: int = 3) -> list[dict]
- summarize_period_counts(df, time_col: str, target_col: str | None = None) -> list[dict]

In validation/group_split.py:
- summarize_group_counts(df, group_col: str, target_col: str | None = None) -> dict
- assess_group_split_feasibility(df, group_col: str, target_col: str | None = None) -> dict
- detect_group_leakage_risk(train_df, test_df, group_col: str) -> dict

In validation/policy_selector.py:
- select_validation_policy(
    task_type,
    metric_spec,
    inferred_schema,
    table_profiles,
    validation_signals: dict | None = None,
    scout_hypotheses: list | None = None,
) -> dict

Validation policies to support:
- stratified_kfold
- kfold
- group_kfold
- stratified_group_kfold
- temporal_holdout
- expanding_window
- ranking_group_cv
- custom_required

Policy selection rules:
- Binary/multiclass iid classification -> stratified_kfold.
- Regression iid -> kfold.
- Ranking/recommender with query_id/session_id/user_id -> ranking_group_cv or group_kfold.
- Group/entity leakage risk -> group_kfold or stratified_group_kfold.
- Forecasting or temporal/stability metric -> temporal_holdout or expanding_window.
- Time column alone is diagnostic, not sufficient for primary temporal validation.
- Unknown/custom metric -> conservative validation with warning and custom_required if needed.

The selector output must include:
- primary_validation
- diagnostic_validations
- rejected_validations
- confidence
- evidence_refs
- warnings
- limitations
- reasoning_summary

Do not import sklearn in split helper modules unless absolutely necessary.
Do not implement model training.
```

### Acceptance criteria

- iid binary classification with no group/time requirement selects `stratified_kfold`.
- iid regression selects `kfold`.
- ranking metric with query/group column selects `ranking_group_cv`.
- temporal/stability metric selects temporal policy when time column exists.
- time column alone does not force temporal primary validation.
- too few periods returns infeasible temporal policy with reason.
- group split feasibility is tested.
- Tests pass with:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests/eda -q
```

---

## 41_eda_validation_analyzer_generic

### Goal

Build factual validation evidence using the generic ValidationPolicySelector.

This task replaces the previous temporal-first validation analyzer.

### Files to create/change

```text
kaggle_researcher/eda/modules/validation_analyzer.py
kaggle_researcher/eda/validation/policy_selector.py
tests/eda/test_validation_analyzer.py
```

### Codex prompt

```text
Implement generic validation_analyzer.

Function:
- analyze_validation(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    metric_evidence: MetricEvidence,
    reader: DatasetReader,
) -> ValidationEvidence

Requirements:
1. Load train_base table only as much as needed.
2. Detect and summarize:
   - target column availability
   - id column availability
   - candidate group/entity columns
   - candidate time/date columns
   - candidate query/ranking columns
3. If target exists:
   - for classification: class balance / target rate
   - for regression: target summary stats
   - for time columns: target by period as diagnostic
   - for group columns: target/group distribution as diagnostic
4. If test_base exists:
   - compare train/test time range when time columns exist
   - compare train/test group overlap when group columns exist
5. Call ValidationPolicySelector to produce:
   - primary validation
   - diagnostic validations
   - rejected validations
6. Temporal validation can be primary only when:
   - metric/task requires temporal validation, or
   - train/test relation indicates future test, or
   - scout hypothesis plus data evidence supports temporal split risk.
7. For ordinary iid classification:
   - recommend StratifiedKFold.
8. For ordinary iid regression:
   - recommend KFold.
9. For grouped tasks:
   - recommend GroupKFold or StratifiedGroupKFold.
10. For ranking tasks:
   - recommend group/query-aware validation.

Do not hard-code Home Credit as the default.
Home Credit behavior must emerge from metric_spec=gini_stability and detected WEEK_NUM.
```

### Acceptance criteria

- Fixture with gini_stability + WEEK_NUM recommends temporal validation.
- Binary iid fixture without temporal metric recommends StratifiedKFold even if a date column exists.
- Regression fixture recommends KFold.
- Grouped fixture recommends GroupKFold/StratifiedGroupKFold.
- Ranking fixture recommends query/group-aware validation.
- Validation evidence includes warnings/limitations when policy cannot be selected confidently.
- Tests pass.

---

## 42_eda_leakage_checker_generic

### Goal

Implement generic leakage checks for tabular competitions, not only Home Credit-like datasets.

### Files to create/change

```text
kaggle_researcher/eda/modules/leakage_checker.py
tests/eda/test_leakage_checker.py
```

### Codex prompt

```text
Implement generic leakage_checker.

Function:
- check_leakage(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
) -> list[LeakageCheckResult]

MVP checks:
- train/test id overlap
- train/test group/entity overlap where group columns exist
- target column present in test
- target-like column names outside target
- sample_submission structure sanity
- duplicate rows across train/test for base tables where feasible
- suspicious numeric columns with extremely high target association
- potential post-target/future-date risk when time/date columns exist
- ranking/query leakage risk when query/group identifiers appear in both train/test

Requirements:
- Confirmed leakage requires direct evidence.
- Suspicious association is warning, not proof.
- Time/date presence alone is not leakage.
- Group overlap is not always leakage; severity depends on selected validation policy.
- Missing required tables/columns should return not_testable checks.
- Must not scan huge tables without sampling/caps.
```

### Acceptance criteria

- Default fixture has passed id overlap check.
- Modified fixture with overlapping id detects overlap.
- Test target column is high/critical severity.
- Group overlap is reported with contextual severity.
- Missing id returns not_testable.
- Tests pass.

---

## 43_eda_hypothesis_evaluator_generic

### Goal

Evaluate Research Scout hypotheses against generic EDA evidence.

### Files to create/change

```text
kaggle_researcher/eda/modules/hypothesis_evaluator.py
tests/eda/test_hypothesis_evaluator.py
```

### Codex prompt

```text
Implement or refactor hypothesis_evaluator to be generic.

Function:
- evaluate_hypotheses(
    hypotheses: list[ResearchHypothesis],
    evidence_pack_partial: dict,
    module_statuses: dict[str, str] | None = None,
) -> list[HypothesisResult]

Requirements:
- Every input hypothesis must produce exactly one result.
- Supported categories:
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
- Do not assume Home Credit hypothesis IDs except in fixtures.
- For MVP, provide deterministic evaluators by category:
  - schema -> inferred_schema evidence
  - metric -> metric_evidence
  - validation -> validation_evidence.primary/diagnostic policies
  - leakage -> leakage_evidence
- Unknown category -> not_testable with limitation.
- skipped module -> skipped hypothesis.
- confirmed/rejected must include evidence_refs.
- not_testable/skipped must include limitations.
- impact_on_strategy must be concrete.

Important:
- A validation hypothesis saying "temporal CV is required" should be confirmed only if validation evidence selected temporal validation as primary or strong diagnostic.
- If a time column exists but temporal policy was rejected as primary, the hypothesis should be partially_confirmed or rejected depending on wording.
```

### Acceptance criteria

- All fixture hypotheses are evaluated.
- Binary iid validation hypothesis does not get temporal confirmation merely from date column.
- confirmed results include evidence_refs.
- skipped/not_testable include limitations.
- Tests pass.

---

## 44_eda_recommendations_generic

### Goal

Build evidence-backed next actions without assuming temporal validation or Gini by default.

### Files to create/change

```text
kaggle_researcher/eda/modules/recommendations.py
tests/eda/test_eda_recommendations.py
```

### Codex prompt

```text
Implement generic recommended next actions builder.

Function:
- build_recommended_next_actions(
    evidence_pack_partial: dict,
    hypothesis_results: list[HypothesisResult],
) -> list[RecommendedNextAction]

Requirements:
- Generate actions from confirmed or partially_confirmed evidence only.
- Every action must include:
  - priority
  - action
  - why
  - evidence_refs
- Generic MVP rules:
  - If validation policy selected StratifiedKFold -> P0 action to use stratified CV.
  - If validation policy selected KFold -> P0 action to use KFold.
  - If validation policy selected GroupKFold/StratifiedGroupKFold -> P0 action to respect group split.
  - If validation policy selected temporal_holdout/expanding_window -> P0 action to use temporal validation.
  - If metric requires probabilities -> action to output probabilities/ranks, not hard labels.
  - If metric requires threshold -> action to tune threshold on validation only.
  - If metric requires calibration -> action to check calibration/clipping.
  - If metric is regression_error -> action to optimize regression loss and inspect target transform.
  - If leakage check warns/fails -> P0 action to fix/exclude unsafe source.
  - If secondary tables exist but relationship module not run -> P1 action to run relationship inference before aggregations.
- Do not invent actions without evidence_refs.
- Sort by priority.
```

### Acceptance criteria

- Binary iid fixture produces StratifiedKFold action.
- Home Credit-like fixture produces temporal validation action.
- F1 metric produces threshold tuning action.
- LogLoss metric produces calibration action.
- RMSE metric produces regression target/loss action.
- Every action has evidence_refs.
- Tests pass.

---

## 45_eda_mvp_orchestrator_and_cli_generic

### Goal

Wire P0 generic EDA modules into a working local-dataset MVP.

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
Implement the generic EDA MVP orchestrator and CLI.

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
11. metric_analyzer using MetricRegistry
12. validation_analyzer using ValidationPolicySelector
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
- Write skipped placeholders for P1 modules:
  - relationship_evidence
  - drift_evidence
  - baseline_evidence
  - feature_probe_evidence
  - notebook_static_analysis
- Do not assume Home Credit as default.
- Home Credit fixture should still work through metric/preset/schema evidence.
```

### Acceptance criteria

- Offline Home Credit-like fixture creates eda_evidence_pack.json.
- Additional iid classification fixture selects StratifiedKFold.
- Additional regression fixture selects KFold.
- CLI test runs without network calls.
- Tests pass.

---

## 46_eda_relationship_inferer_generic

### Goal

Infer relationships between base and secondary tables for generic multi-table tabular competitions.

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
- For each secondary train/test table:
  - find candidate join keys shared with base table.
  - compute relationship_type:
    one_to_one, one_to_many, many_to_one, many_to_many, unknown
  - compute coverage_left_to_right.
  - compute orphan_rate_right.
  - compute avg_rows_per_left and max_rows_per_left.
  - compute row_multiplication_risk low/medium/high.
  - detect candidate group/query/entity keys.
  - detect candidate date cutoff columns.
  - assign confidence.
- Do not recommend direct one-to-many joins without aggregation.
- Do not assume case_id as the only possible join key.
- Use schema hints, sample_submission keys, and shared column patterns.
- Must support sampled checks for large tables.
```

### Acceptance criteria

- Home Credit fixture detects relationship by case_id.
- Generic fixture detects relationship by customer_id/order_id.
- one-to-many relationship is detected when multiple rows per base id exist.
- Missing join key returns unknown relationship with warning.
- Tests pass.

---

## 47_eda_drift_analyzer_generic

### Goal

Analyze drift as optional evidence, not as a universal assumption.

### Files to create/change

```text
kaggle_researcher/eda/modules/drift_analyzer.py
tests/eda/test_drift_analyzer.py
```

### Codex prompt

```text
Implement generic drift_analyzer.

Function:
- analyze_drift(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
    max_rows: int = 500000,
    random_seed: int = 42,
) -> dict

Requirements:
- Compute target drift by period only when target and time exist.
- Compute row count by period when time exists.
- Compute missingness drift train vs test for shared columns.
- Compute numeric PSI for shared numeric columns.
- Compute categorical distribution shift for shared categorical columns.
- Implement adversarial validation only if sklearn is available:
  - safe features only
  - exclude target, id, prediction, query/group columns unless explicitly allowed
  - cap rows at max_rows
  - return AUC and top features if available
- If no test table exists, provide train-only drift diagnostics where possible.
- If no time columns exist, skip temporal drift with limitation.
- Drift evidence should influence validation only later or as diagnostic, not retroactively override generic policy unless explicitly wired.
```

### Acceptance criteria

- Fixture with time column returns target_drift.
- Fixture without time column skips temporal drift with limitation.
- Artificially shifted train/test fixture produces higher drift severity.
- target/id columns are excluded from adversarial features.
- Tests pass.

---

## 48_eda_baseline_runner_generic

### Goal

Run an honest baseline appropriate to task_type and metric family.

### Files to create/change

```text
kaggle_researcher/eda/modules/baseline_runner.py
tests/eda/test_baseline_runner.py
requirements.txt
```

### Codex prompt

```text
Implement generic baseline_runner.

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
  - raw date strings unless encoded safely
  - prediction/sample submission columns
  - columns flagged by critical leakage checks
  - query/group columns when they define validation split and should not be features
- Choose model by task_type:
  - binary_classification -> classifier
  - multiclass_classification -> classifier
  - regression -> regressor
  - ranking -> skipped/not_testable in MVP unless query-aware baseline exists
  - survival -> skipped/not_testable in MVP
  - forecasting_tabular -> simple time-aware baseline later; skipped in MVP if unsupported
- Model preference:
  - LightGBM if installed.
  - sklearn HistGradientBoostingClassifier/Regressor fallback.
  - LogisticRegression/LinearRegression fallback if needed.
- Basic preprocessing:
  - numeric fill missing
  - categorical encoding fit on train fold only
- Use validation_evidence selected policy.
- Compute metric using MetricRegistry/local metric when available.
- If local metric unavailable, train baseline can still run but metric is skipped with warning.
- Do not train on test.
- Do not use target encoding.
- Do not optimize leaderboard score.
```

### Acceptance criteria

- Binary classification fixture baseline runs.
- Regression fixture baseline runs.
- Ranking/survival fixture returns skipped/not_testable, not failure.
- target/id columns are not in feature list.
- Baseline uses selected validation policy.
- Tests pass without requiring LightGBM.

---

## 49_eda_feature_probe_generic

### Goal

Assess promising feature families across generic tabular tasks.

### Files to create/change

```text
kaggle_researcher/eda/modules/feature_probe.py
tests/eda/test_feature_probe.py
```

### Codex prompt

```text
Implement generic feature_probe module.

Function:
- probe_feature_families(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    relationship_evidence: dict,
    leakage_evidence: list[LeakageCheckResult],
    baseline_evidence: dict,
    metric_evidence: dict | None = None,
) -> list[dict]

Feature families:
- base_numeric_features
- base_categorical_features
- missingness_indicators
- date_features
- secondary_table_aggregations
- high_cardinality_encoding
- target_encoding_or_woe
- monotonic_or_binning_features
- ranking_group_features
- regression_target_transform

Requirements:
- Return:
  - feature_family
  - status: high_potential|medium_potential|low_potential|unsafe|not_testable
  - leakage_risk: low|medium|high
  - evidence
  - recommendation
- Target encoding / WoE is high leakage risk unless OOF/group/time-safe policy exists.
- Secondary aggregations need relationship evidence.
- Regression target transform is relevant only for regression metrics such as RMSLE/RMSE with skewed target.
- Ranking group features are relevant only for ranking tasks.
- Do not generate actual feature engineering code.
```

### Acceptance criteria

- Secondary aggregations are not_testable before relationship evidence.
- With relationship evidence, secondary aggregations become medium/high potential.
- Target encoding is high leakage risk.
- Regression fixture can recommend target transform if target is skewed.
- Tests pass.

---

## 50_eda_notebook_static_analysis_generic

### Goal

Statically extract patterns from notebook source text without executing notebooks.

### Files to create/change

```text
kaggle_researcher/eda/modules/notebook_static_analyzer.py
tests/eda/test_notebook_static_analyzer.py
```

### Codex prompt

```text
Implement generic notebook_static_analyzer.

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
  - KFold
  - StratifiedKFold
  - GroupKFold
  - StratifiedGroupKFold
  - TimeSeriesSplit
  - LightGBM, CatBoost, XGBoost
  - target encoding
  - adversarial validation
  - rank averaging
  - clipping
  - threshold tuning
  - logloss calibration
  - RMSLE target transform
  - QWK threshold optimization
- Notebook scores are observations, not truth.
- Warnings should be contextual to task/metric when available.
```

### Acceptance criteria

- Static fixture detects model and CV patterns.
- No code execution occurs.
- Metric-specific patterns are extracted.
- Tests pass.

---

## 51_eda_p1_orchestrator_wiring_generic

### Goal

Wire optional P1 modules into the EDA orchestrator without making them mandatory.

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
- Update EdaEvidencePack fields.
- Update eda_summary.md with P1 sections when available.
- P1 modules must respect generic task_type and metric_evidence.
```

### Acceptance criteria

- MVP run still works without P1 flags.
- P1 fixture run writes relationship_evidence.json and drift_evidence.json.
- Baseline does not run unless --enable-baseline.
- P1 failure is warning, not fatal.
- Tests pass.

---

## 52_eda_p1_hypothesis_and_recommendation_rules_generic

### Goal

Extend hypothesis evaluation and recommendations to use generic P1 evidence.

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

Rules:
- Relationship hypotheses:
  - confirmed when join keys and coverage are found.
  - partially_confirmed when only weak candidate keys exist.
- Drift hypotheses:
  - confirmed when drift severity is medium/high for relevant drift type.
  - rejected when drift checks show stable distributions.
  - not_testable when test/shared columns unavailable.
- Baseline hypotheses:
  - confirmed when honest baseline completed for supported task_type.
  - skipped when baseline disabled.
  - not_testable for unsupported task types.
- Feature hypotheses:
  - confirmed or partially_confirmed from feature_probe statuses.
- Notebook hypotheses:
  - confirmed only as "pattern observed", not as factual performance proof.

Recommendations:
- one-to-many relationship -> aggregate before join.
- high drift -> use selected robust validation and avoid public LB overfitting.
- baseline complete -> use as sanity floor, not final solution.
- high-potential feature family -> prioritize P1/P2 experiment.
- risky notebook pattern -> audit before copying.
- threshold-sensitive metric -> validate threshold tuning policy.
- calibration-sensitive metric -> check calibration/clipping.
- regression skew -> consider target transform if evidence supports it.
```

### Acceptance criteria

- P1 hypotheses are evaluated when evidence exists.
- Baseline disabled produces skipped, not failure.
- Drift recommendation cites drift evidence.
- Relationship recommendation cites relationship evidence.
- Metric-specific recommendations cite metric evidence.
- Tests pass.

---

## 53_research_scout_schemas_generic

### Goal

Define schemas for Research Scout outputs that feed the generic EDA Engine.

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

### Acceptance criteria

- Research Scout output validates.
- Generated EDA input JSON validates against EDA schemas.
- Invalid hypothesis category/status fails validation.
- Tests pass.

---

## 54_research_scout_reasoner_generic

### Goal

Implement the Research Scout reasoning module that generates generic EDA hypotheses from retrieved sources.

### Files to create/change

```text
kaggle_researcher/research_scout/scout.py
tests/test_research_scout.py
```

### Codex prompt

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

### Acceptance criteria

- Mock LLM response validates and writes expected objects.
- Fallback output includes P0 hypotheses.
- Ordinary binary classification fallback does not force temporal validation.
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
- Scout output must use generic task_type/metric/validation wording.
```

### Acceptance criteria

- Mocked pipeline writes all three Scout outputs.
- EDA input JSON files validate.
- Without --write-eda-plan, no Scout files are written.
- Tests pass.

---

## 56_final_synthesizer_schema_generic

### Goal

Define a structured contract for final strategy synthesis that can consume generic EDA evidence.

### Files to create/change

```text
kaggle_researcher/reasoning/final_synthesizer.py
tests/test_final_synthesizer_schema.py
```

### Codex prompt

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

### Acceptance criteria

- FinalStrategyResult validates.
- Missing evidence_refs on action fails validation.
- Placeholder raises NotImplementedError.
- Tests pass.

---

## 57_final_synthesizer_reasoner_generic

### Goal

Implement the final strategy synthesizer that combines retrieved sources and generic EDA evidence.

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

### Acceptance criteria

- Mock LLM response validates into FinalStrategyResult.
- Recommendations include evidence_refs.
- Prompt includes source -> hypothesis -> EDA -> strategy rule.
- Prompt includes "respect validation_evidence.primary_validation".
- Tests pass.

---

## 58_full_research_to_eda_to_strategy_cli_generic

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
- The workflow must work for generic tabular fixtures, not only Home Credit.
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

## 59_eda_quality_gates_generic

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
  - "temporal validation is required" when validation_evidence selected another primary policy
```

### Acceptance criteria

- Missing hypothesis result creates warning.
- Broken evidence_ref creates warning.
- Empty action evidence_refs creates warning.
- Temporal overclaim creates warning.
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
- Sampling behavior must be generic and task-independent.
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
- All exceptions should be sanitized:
  - no secrets
  - no huge stack traces in JSON by default
- Partial packs must not contain unsupported conclusions.
```

### Acceptance criteria

- Simulated P1 failure still creates evidence pack.
- Simulated blocking failure writes partial artifacts.
- module_statuses.json is written.
- Tests pass.

---

## 62_eda_summary_generator_generic

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
- Do not overstate temporal validation.
- If temporal validation is diagnostic, call it diagnostic.
- Replace existing inline summary creation in orchestrator with this function.
```

### Acceptance criteria

- Summary contains all required sections.
- Summary does not mention modules that are absent except as skipped/not_testable.
- Warnings and limitations are included.
- Summary respects validation_evidence.primary_validation.
- Tests pass.

---

## 63_eda_integration_fixture_full_p1_generic

### Goal

Add offline integration tests that run MVP + P1 modules on generic fixture data.

### Files to create/change

```text
tests/eda/test_eda_integration_full_p1.py
tests/fixtures/eda/
```

### Codex prompt

```text
Add full offline EDA integration tests.

Fixtures:
- home_credit_tiny
- iid_binary_tiny
- regression_tiny
- grouped_binary_tiny

Requirements:
- Run home_credit_tiny with:
  - MVP modules
  - relationship_inferer
  - drift_analyzer
  - feature_probe
- Run iid_binary_tiny and verify:
  - primary validation is StratifiedKFold
  - temporal validation is not forced
- Run regression_tiny and verify:
  - primary validation is KFold
  - regression metric evidence is used
- Run grouped_binary_tiny and verify:
  - group-aware validation is selected
- Do not require baseline unless fallback sklearn model is available and test is stable.
- Validate:
  - eda_evidence_pack.json exists.
  - hypothesis_results cover all input hypotheses.
  - recommended_next_actions is non-empty.
  - quality gates return no critical warnings.
```

### Acceptance criteria

- Tests run offline.
- Tests do not require Kaggle credentials.
- Tests do not require DeepSeek.
- Tests prove generic tabular behavior, not only Home Credit behavior.
- Tests pass in CI.

---

## 64_eda_production_cli_docs_generic

### Goal

Document practical commands and expected outputs for generic EDA Engine production use.

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

Document generic tabular behavior:
- ordinary classification can use StratifiedKFold.
- ordinary regression can use KFold.
- grouped tasks can use group-aware validation.
- temporal validation is used only when evidence supports it.
- Gini Stability is supported but not the default worldview.
```

### Acceptance criteria

- README has EDA section.
- RUNBOOK has copy-pasteable commands.
- Docs mention local dataset mode.
- Docs mention no notebook execution.
- Docs mention generic tabular validation behavior.
- No tests required unless docs lint exists.

---

# Recommended Codex steering prompt before Task 40

Use this prompt before continuing:

```text
Do not continue with the old task 40+ wording.

We have replaced Task 39 with a generic MetricRegistry.

From Task 40 onward, use docs/EDA_ENGINE_CODEX_TASKS_40_PLUS_GENERIC.md as the source of truth.

Important:
- A time column alone must never force temporal validation.
- Gini Stability is supported, but it is only one metric registry entry.
- Home Credit-specific behavior must come from metric/preset/schema evidence, not from global defaults.
- Implement only the next requested task.
- Do not implement future tasks.
- Do not execute Kaggle notebooks.

Use the project-local Windows virtual environment:

E:\wavebreaker\.venv-win\Scripts\python.exe

Run tests with:

E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests/eda -q
```
