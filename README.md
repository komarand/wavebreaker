# KaggleResearcher

Minimal bootstrap for the research/reasoning layer described in `docs/SPEC.md`.

## Requirements

- Python 3.11+
- PostgreSQL with pgvector for retrieval storage
- CUDA is optional but recommended for faster local embeddings

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

```bash
python -m kaggle_researcher.main --help
```

## Full Run

The canonical end-to-end command is `full-run`. It creates one manifest-backed
parent run, passes Scout artifacts into EDA and reasoning automatically, and
writes stable outputs under `research/`, `eda/`, `reasoning/`, and `final/`.

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main full-run `
  --competition-id titanic `
  --competition-url "https://www.kaggle.com/competitions/titanic" `
  --competition-description "Binary classification of passenger survival. Metric: accuracy." `
  --local-dataset-path "E:\wavebreaker\data\kaggle_datasets\titanic" `
  --no-download-dataset
```

Use `--profile minimal|standard|full` to choose the EDA profile; `standard` is
the default. Resume a verified run with `--resume-run-dir <run-directory>` or
rerun a stage and its downstream work with repeated `--force-rerun-stage`.

Full-run orchestration uses typed stage results rather than a shared context map.
`run_manifest.json` is versioned and records integrity-checked relative artifact
pointers; supported legacy manifests are backed up and migrated atomically on
resume. Final Strategy keeps risk, validation-requirement, and safety-constraint
IDs in dedicated namespaces. See `docs/contracts.md` and
`docs/contract_cleanup_audit.md` for the contract and migration rules.

### Research Hypotheses Contract

`research_hypotheses.json` uses schema version `1.0`. Each hypothesis has a
canonical `hypothesis_id`, one of the documented EDA categories (`schema`,
`relationship`, `feature`, `notebook`, and related categories), and
`confidence_before_eda`. Scout validates and atomically writes this contract
before EDA starts. Older unversioned artifacts are migrated deterministically
for known field and category aliases, with a preserved `.legacy.json` backup
inside a resumed full run. Unknown categories and future schema versions fail
clearly instead of being guessed.

`eda_task_plan.json` uses the same versioned boundary. Tasks use `task_id`
(not `id`), and each `hypothesis_index` value is an array of task IDs so a
hypothesis can safely map to multiple checks. The full-run gate validates both
artifacts together before EDA starts, migrating supported legacy task plans and
preserving an `eda_task_plan.legacy.json` backup when it does.

### Reasoning Result Nulls

Reasoning artifacts preserve meaningful optional absence. In particular,
`ValidationResult.secondary_validation: null` means no distinct secondary
validation policy is justified. `primary_validation` remains required and must
contain a policy method; empty objects are not treated as a substitute for
either policy. Plural fields such as failure modes and policy notes use `[]`.

Embeddings are computed inside Python with SentenceTransformers. The default embedding model is `Qwen/Qwen3-Embedding-0.6B`, with `EMBED_DIM=1024` and `MAX_EMBED_BATCH_SIZE=8`.

Current status: bootstrap only; external APIs, retrieval, embeddings, and report generation are not implemented yet.

## Kaggle EDA Engine

KaggleResearcher v5 includes the Kaggle EDA Engine as a separate Stage 2 Data Evidence Layer. The research/reasoning pipeline remains source-based; it reads public sources and does not inspect train/test data unless EDA is explicitly run.

EDA consumes:

- `research_hypotheses.json`
- `eda_task_plan.json`
- a Kaggle competition dataset or `--local-dataset-path`

EDA produces:

- `eda_evidence_pack.json`
- `eda_summary.md`
- `module_statuses.json`
- module-level JSON artifacts and `artifacts/`

Local fixture run:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id home_credit_tiny `
  --hypotheses-path tests\fixtures\eda\home_credit_tiny\research_hypotheses.json `
  --task-plan-path tests\fixtures\eda\home_credit_tiny\eda_task_plan.json `
  --local-dataset-path tests\fixtures\eda\home_credit_tiny `
  --no-download-dataset
```

P1 evidence modules:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_eda_engine.main `
  --competition-id home_credit_tiny `
  --hypotheses-path tests\fixtures\eda\home_credit_tiny\research_hypotheses.json `
  --task-plan-path tests\fixtures\eda\home_credit_tiny\eda_task_plan.json `
  --local-dataset-path tests\fixtures\eda\home_credit_tiny `
  --no-download-dataset `
  --enable-p1-modules
```

Dataset execution is allowed only inside `kaggle_researcher/eda/` and `kaggle_eda_engine/`. Notebook execution is never performed. Baseline output is an evidence sanity check, not final score optimization. Large datasets may be sampled; when `sampled=true` appears in table profiles, downstream conclusions must respect that limit.

Generic tabular behavior is evidence-based:

- ordinary classification can use StratifiedKFold;
- ordinary regression can use KFold;
- grouped tasks can use group-aware validation;
- temporal validation is primary only when evidence supports it;
- Gini Stability is supported, but it is not the default worldview.

See `docs/RUNBOOK.md` for production command sequences and expected outputs.
