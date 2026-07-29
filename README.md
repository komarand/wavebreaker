# KaggleResearcher

KaggleResearcher is a competition-research system. The v4 research and
reasoning pipeline is implemented and remains available through the legacy
`kaggle_researcher.main` entry point. Wavebreaker B5 adds a deterministic
competition-facts pipeline and a single-call competition brief through
`kaggle_researcher.wave`.

## Documentation

- [v4 specification](docs/SPEC.md)
- [v4 implementation tasks](docs/CODEX_TASKS.md)
- [Wavebreaker B5 specification](docs/SPEC_B5.md)
- [Wavebreaker B5 implementation tasks](docs/CODEX_TASKS_B5.md)

## Requirements

- Python 3.11+
- The legacy v4 pipeline uses PostgreSQL with pgvector and local
  SentenceTransformers embeddings. CUDA is optional.
- B5 requires no PostgreSQL, embeddings, or GPU.

## Installation

```bash
pip install -r requirements.txt
```

## Entry points

Legacy v4 pipeline:

```bash
python -m kaggle_researcher.main --help
```

Wavebreaker B5:

```bash
python -m kaggle_researcher.wave --help
```

The B5 `facts` command is designed to run without a model API key. The B5
implementation is delivered incrementally by the tasks in
[`docs/CODEX_TASKS_B5.md`](docs/CODEX_TASKS_B5.md), beginning with the
documentation contract in task 65.
