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
