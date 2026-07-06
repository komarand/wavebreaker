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

KaggleResearcher v5 adds the Kaggle EDA Engine as a separate Stage 2 Data Evidence Layer. The existing research/reasoning pipeline remains text-source based and does not inspect train/test data by default.

The EDA Engine consumes Research Scout outputs plus a Kaggle dataset or local dataset path:

- `research_hypotheses.json`
- `eda_task_plan.json`
- Kaggle competition data or a local fixture/dataset directory

It produces machine-readable evidence for the Final Synthesizer:

- `eda_evidence_pack.json`
- `eda_summary.md`
- module-level JSON artifacts
- `artifacts/`

Dataset execution is allowed only inside `kaggle_researcher/eda/` and `kaggle_eda_engine/`. Notebook execution remains forbidden.

Placeholder CLI:

```powershell
python -m kaggle_eda_engine.main `
  --competition-id fixture_competition `
  --hypotheses-path tests\fixtures\eda\home_credit_tiny\research_hypotheses.json `
  --task-plan-path tests\fixtures\eda\home_credit_tiny\eda_task_plan.json `
  --local-dataset-path tests\fixtures\eda\home_credit_tiny `
  --no-download-dataset
```
