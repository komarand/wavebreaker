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
