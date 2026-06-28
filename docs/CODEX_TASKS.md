# KaggleResearcher — Codex implementation tasks

This file decomposes `docs/SPEC.md` into small, ordered implementation tasks. Codex should implement one task at a time and avoid changing files outside the task scope unless required by tests.

## Global rules for Codex

Always follow these rules unless a later human instruction explicitly overrides them:

- Use `docs/SPEC.md` as the source of truth.
- Implement only the current task.
- Do not implement out-of-scope data-execution features.
- Do not download Kaggle train/test datasets.
- Do not execute Kaggle notebooks.
- Do not perform real EDA, adversarial validation, model training, or confirmed leakage detection.
- Do not introduce chunking. Use `retrieved_documents`, not `chunks`.
- Prefer explicit schemas over arbitrary dictionaries.
- Keep functions small, typed, and testable.
- Add or update tests for every task.
- External APIs must be mockable in tests.
- Never log secrets such as `DEEPSEEK_API_KEY`, `KAGGLE_KEY`, or `GITHUB_TOKEN`.

## Milestones

### Milestone A — Research MVP

Goal: one CLI run creates a basic `.docx` report from collected sources.

Includes tasks: `00`–`13`.

Does not include full reasoning-chain, DomainMemory, SkepticalReviewer, or quality gates.

### Milestone B — Kaggle Analyst Layer

Goal: report includes metric interpretation, validation recommendation, leakage-risk hypotheses, experiment priorities, and 15-section roadmap.

Includes tasks: `14`–`21`.

### Milestone C — Trust & Memory

Goal: report becomes traceable and reviewable with evidence IDs, DomainMemory, quality gates, and `research_run.json`.

Includes tasks: `22`–`27`.

---

## 00_bootstrap_project

### Goal

Create the repository structure and minimal runnable package without business logic.

### Files to create/change

```text
kaggle_researcher/
├── __init__.py
├── main.py
├── config.py
├── schemas.py
├── logging_utils.py
├── clients/
│   ├── __init__.py
│   └── deepseek_client.py
├── agents/
│   └── __init__.py
├── parsers/
│   └── __init__.py
├── store/
│   └── __init__.py
├── reasoning/
│   └── __init__.py
└── report/
    └── __init__.py
tests/
.env.example
requirements.txt
README.md
```

### Codex prompt

```text
Create the initial Python project skeleton for KaggleResearcher according to docs/SPEC.md.

Requirements:
- Python 3.11+.
- Async-first architecture.
- Do not implement external API logic yet.
- Add empty packages: clients, agents, parsers, store, reasoning, report.
- Add .env.example with all environment variables from the spec.
- Add README.md with a short project description and run placeholder.
- Add logging_utils.py with a basic get_logger(name) helper.
- main.py should expose a CLI with --help and print NotImplemented for now.

Do not implement Kaggle, DeepSeek, pgvector, or vLLM yet.
```

### Acceptance criteria

- `python -m kaggle_researcher.main --help` works.
- `pytest` runs successfully.
- `import kaggle_researcher` works.

---

## 01_schemas

### Goal

Define stable typed contracts so later tasks do not invent incompatible dict shapes.

### Files to create/change

```text
kaggle_researcher/schemas.py
tests/test_schemas.py
requirements.txt
```

### Codex prompt

```text
Implement kaggle_researcher/schemas.py using Pydantic v2.

Models:
- SourceDocument
- RetrievedDocument
- PlanData
- ResearchRunResult
- ReasoningBaseResult
- ValidationResult
- LeakageRiskResult
- MetricResult
- LeaderboardAuditResult
- ExperimentItem
- ReviewResult

Requirements:
- SourceDocument fields: id, competition_id, source, title, url, content, summary, metadata.
- RetrievedDocument fields: id, competition_id, source, title, url, content, score, rrf_score, metadata.
- source must be one of: kaggle, arxiv, papers_with_code, github.
- All reasoning results must include confidence: low|medium|high and evidence_ids: list[str].
- Use defaults for optional list/dict fields to avoid mutable default bugs.
- Add unit tests for validation and defaults.
```

### Acceptance criteria

- Invalid `source` raises a validation error.
- Invalid `confidence` raises a validation error.
- Optional list/dict defaults are independent per instance.
- Tests pass.

---

## 02_config

### Goal

Implement central configuration and environment loading.

### Files to create/change

```text
kaggle_researcher/config.py
tests/test_config.py
```

### Codex prompt

```text
Implement kaggle_researcher/config.py.

Create:
- ConfigError
- Settings Pydantic model or dataclass
- load_config() -> Settings

Requirements:
- DEEPSEEK_API_KEY is required from env; no default; missing value raises ConfigError.
- KAGGLE_USERNAME and KAGGLE_KEY are optional.
- GITHUB_TOKEN is optional.
- PG_DSN default: postgresql://researcher:researcher@localhost:5432/kaggle_research
- Constants/defaults:
  - DEEPSEEK_V4_PRO = deepseek-v4-pro
  - DEEPSEEK_V4_FLASH = deepseek-v4-flash
  - EMBED_MODEL = Qwen/Qwen3-Embedding-0.6B
  - EMBED_DIM = 1024
  - MAX_EMBED_BATCH_SIZE = 8
  - TOP_K = 10
  - MAX_NOTEBOOKS = 20
  - MAX_PAPERS = 15
  - MAX_REPOS = 10
  - PDF_CACHE_DIR = ./data/pdfs

Do not log secrets.
```

### Acceptance criteria

- Missing `DEEPSEEK_API_KEY` raises `ConfigError`.
- Defaults are applied correctly.
- Tests use monkeypatch/env isolation.

---

## 03_deepseek_client

### Goal

Create one mockable client wrapper for all DeepSeek calls.

### Files to create/change

```text
kaggle_researcher/clients/deepseek_client.py
tests/test_deepseek_client.py
```

### Codex prompt

```text
Implement kaggle_researcher/clients/deepseek_client.py.

Create class DeepSeekClient:
- __init__(api_key: str, base_url: str = "https://api.deepseek.com")
- async chat_json(model, system_prompt, user_prompt, timeout=90) -> dict
- async chat_text(model, system_prompt, user_prompt, timeout=90, max_tokens=None) -> str

Requirements:
- Use httpx.AsyncClient.
- Use OpenAI-compatible /chat/completions endpoint.
- chat_json should request JSON object output.
- Retry with exponential backoff for 429, 5xx, and network errors; max 3 attempts.
- If chat_json receives invalid JSON, attempt one JSON-repair call.
- Never log the API key.
- Tests must use httpx.MockTransport; no real network calls.
```

### Acceptance criteria

- `chat_json` returns a dict from a mocked JSON response.
- 429 triggers retry.
- Invalid JSON triggers one repair attempt.
- API key is not present in logs or exceptions.

---

## 04_db_ddl

### Goal

Create Docker Compose and SQL DDL used by PgStore and DomainMemory.

### Files to create/change

```text
docker-compose.yml
kaggle_researcher/store/sql.py
```

### Codex prompt

```text
Create docker-compose.yml and kaggle_researcher/store/sql.py.

Docker:
- PostgreSQL 16 with pgvector image: pgvector/pgvector:pg16
- user/password/db: researcher/researcher/kaggle_research
- port 5432:5432
- volume ./data/postgres:/var/lib/postgresql/data

SQL constants:
- CREATE EXTENSION IF NOT EXISTS vector;
- CREATE TABLE documents with fields from docs/SPEC.md.
- CREATE TABLE competition_patterns with fields from docs/SPEC.md.
- HNSW index on documents.embedding.
- GIN index on documents.ts_content.
- B-tree index on documents.competition_id.
- HNSW index on competition_patterns.embedding.

Do not implement PgStore methods in this task.
```

### Acceptance criteria

- SQL strings are importable constants.
- Docker Compose matches the spec.

---

## 05_pg_store

### Goal

Implement document storage and retrieval over PostgreSQL + pgvector.

### Files to create/change

```text
kaggle_researcher/store/pg_store.py
tests/test_pg_store_unit.py
```

### Codex prompt

```text
Implement kaggle_researcher/store/pg_store.py.

Create class PgStore:
- __init__(competition_id: str, dsn: str)
- async init() -> None
- async upsert(docs: list[SourceDocument], embeddings: list[list[float]]) -> None
- async vector_search(embedding: list[float], top_k: int = 10) -> list[RetrievedDocument]
- async fts_search(query: str, top_k: int = 10) -> list[RetrievedDocument]
- async close() -> None

Requirements:
- Use asyncpg pool.
- Run DDL from store/sql.py in init().
- upsert must run in one transaction.
- Validate len(docs) == len(embeddings).
- Filter all searches by competition_id.
- RetrievedDocument.content = summary if present else content.
- Do not implement DomainMemory here.
```

### Acceptance criteria

- Unit tests verify len mismatch raises an error.
- Unit tests verify expected SQL calls using mocks.
- Integration tests, if added, must be marked with `@pytest.mark.integration`.

---

## 06_embedder

### Goal

Implement vLLM OpenAI-compatible embedding calls with batching and order guarantees.

### Files to create/change

```text
kaggle_researcher/embedder.py
tests/test_embedder.py
```

### Codex prompt

```text
Implement kaggle_researcher/embedder.py.

Functions:
- embed_texts(texts: list[str], batch_size: int = 8) -> list[list[float]]
- embed_one(text: str) -> list[float]
- get_embedding_dim() -> int

Requirements:
- If len(texts) > batch_size, split into microbatches.
- Preserve output order exactly.
- Load SentenceTransformer lazily.
- Normalize embeddings.
- Tests mock SentenceTransformer; no real model download required.
```

### Acceptance criteria

- Mock model returns embeddings in input order.
- Empty input returns [].
- get_embedding_dim returns the detected vector length.

---

## 07_planner

### Goal

Implement competition query planning.

### Files to create/change

```text
kaggle_researcher/planner.py
tests/test_planner.py
```

### Codex prompt

```text
Implement kaggle_researcher/planner.py.

Functions:
- async plan(description: str, client: DeepSeekClient, model: str) -> PlanData
- fallback_plan(description: str) -> PlanData

Requirements:
- Use DeepSeek V4 Pro through client.chat_json.
- System prompt decomposes a Kaggle competition description into search queries.
- Expected keys: task_type, metric, domain, kaggle_queries, arxiv_queries, github_queries, key_techniques, similar_competitions.
- If optional list fields are missing, fill them with empty lists.
- On LLM failure, expose fallback_plan for caller to use; do not silently swallow unless caller requested fallback.
```

### Acceptance criteria

- Mock LLM response becomes PlanData.
- Missing optional lists become empty lists.
- fallback_plan returns useful non-empty query lists.

---

## 08_kaggle_agent

### Goal

Collect Kaggle notebooks as text sources without executing or reverse-engineering them.

### Files to create/change

```text
kaggle_researcher/agents/kaggle_agent.py
tests/test_kaggle_agent.py
```

### Codex prompt

```text
Implement kaggle_researcher/agents/kaggle_agent.py.

Functions:
- search_notebooks(queries: list[str], max_notebooks: int) -> list[dict]
- get_notebook_content(kernel_ref: str, max_chars: int = 8000) -> str
- build_kaggle_documents(raw_results: list[dict], competition_id: str) -> list[SourceDocument]

Requirements:
- Use Kaggle API/CLI where appropriate, but tests must not call the real API.
- Deduplicate by kernel_ref/ref.
- Sort by total_votes desc.
- get_notebook_content downloads the kernel into a temporary directory.
- Extract markdown cells fully and first 500 characters of each code cell.
- Return empty content with warning if notebook cannot be downloaded or parsed.
- Do not execute notebooks.
- Do not perform notebook reverse engineering.
```

### Acceptance criteria

- Deduplication works.
- Sorting by votes works.
- build_kaggle_documents creates valid SourceDocument instances.
- Tests mock subprocess/API calls.

---

## 09_arxiv_pdf_agent

### Goal

Collect arXiv papers and parse PDF text/tables with fallback to abstracts.

### Files to create/change

```text
kaggle_researcher/agents/arxiv_agent.py
kaggle_researcher/parsers/pdf_parser.py
tests/test_arxiv_agent.py
tests/test_pdf_parser.py
```

### Codex prompt

```text
Implement PDF parsing and arXiv collection.

In parsers/pdf_parser.py:
- async download_pdf(url: str, paper_id: str, cache_dir: str) -> Path | None
- extract_tables_as_text(page) -> str
- parse_pdf(pdf_path: Path, max_chars: int = 8000) -> str

In agents/arxiv_agent.py:
- search_arxiv(queries: list[str], max_papers: int) -> list[dict]
- enrich_with_pdf(papers: list[dict], cache_dir: str) -> list[dict]
- search_papers_with_code(query: str) -> list[dict]
- build_arxiv_documents(papers: list[dict], competition_id: str) -> list[SourceDocument]

Requirements:
- Use pdfplumber for parsing.
- Include first page, first 3 pages, last 2 pages, and any page with tables.
- Convert tables to rows joined by ` | `.
- Deduplicate arXiv papers by entry_id.
- If PDF download or parse fails, set content = abstract.
- Tests must not call real arXiv/Papers with Code unless marked integration.
```

### Acceptance criteria

- parse_pdf returns text for a small fixture PDF.
- PDF failure falls back to abstract.
- SourceDocument objects are valid.

---

## 10_github_agent

### Goal

Collect GitHub repositories as text sources from README and metadata.

### Files to create/change

```text
kaggle_researcher/agents/github_agent.py
tests/test_github_agent.py
```

### Codex prompt

```text
Implement kaggle_researcher/agents/github_agent.py.

Functions:
- async search_repos(queries: list[str], token: str | None = None, max_repos: int = 10) -> list[dict]
- build_github_documents(raw_repos: list[dict], competition_id: str) -> list[SourceDocument]

Requirements:
- Use GitHub Search API for repositories.
- Deduplicate by full_name.
- Sort by stars desc.
- content = description + README text if README is available.
- Metadata includes stars, language, full_name, updated_at.
- Work without token, but expose warning/rate-limit context.
- Tests must use mocked HTTP responses.
```

### Acceptance criteria

- Mock GitHub search response returns valid documents.
- Stars are stored in metadata.
- Deduplication works.

---

## 11_summarizer

### Goal

Summarize sources through DeepSeek V4 Flash with safe fallbacks.

### Files to create/change

```text
kaggle_researcher/summarizer.py
tests/test_summarizer.py
```

### Codex prompt

```text
Implement kaggle_researcher/summarizer.py.

Functions:
- async summarize_one(client: DeepSeekClient, doc: SourceDocument, model: str) -> SourceDocument
- async summarize_all(client: DeepSeekClient, docs: list[SourceDocument], model: str, concurrency: int = 8) -> list[SourceDocument]

Requirements:
- Use DeepSeek V4 Flash via client.chat_text.
- Target summary length: 250–300 words.
- If doc.content has fewer than 150 characters, set summary = content without API call.
- On API error, fallback to first 800 characters of content.
- Preserve input order.
- Limit concurrency with asyncio.Semaphore.
```

### Acceptance criteria

- Short content does not call API.
- API failure uses fallback.
- summarize_all preserves order.

---

## 12_retriever

### Goal

Implement vector + FTS hybrid retrieval with RRF.

### Files to create/change

```text
kaggle_researcher/retriever.py
tests/test_retriever.py
```

### Codex prompt

```text
Implement kaggle_researcher/retriever.py.

Functions:
- reciprocal_rank_fusion(vector_results: list[RetrievedDocument], fts_results: list[RetrievedDocument], k: int = 60) -> list[RetrievedDocument]
- async hybrid_search(store: PgStore, query: str, top_k: int = 10) -> list[RetrievedDocument]

Requirements:
- RRF score = sum(1 / (k + rank + 1)).
- Deduplicate by id.
- Preserve best available document metadata.
- hybrid_search calls embed_one(query), then store.vector_search and store.fts_search concurrently.
- Each underlying search gets top_k * 2.
- Return top_k after fusion.
```

### Acceptance criteria

- RRF correctly sums duplicate ids.
- hybrid_search calls both vector and FTS search.
- Results are sorted by rrf_score desc.

---

## 13_minimal_e2e

### Goal

Produce the first working vertical slice without the full reasoning-chain.

### Files to create/change

```text
kaggle_researcher/main.py
kaggle_researcher/report/docx_generator.py
tests/test_pipeline_minimal.py
```

### Codex prompt

```text
Implement a minimal end-to-end pipeline in run_research without full reasoning-chain.

run_research should:
1. load_config
2. derive competition_id from URL if not provided
3. initialize PgStore
4. call planner.plan or fallback_plan
5. collect Kaggle and arXiv sources
6. normalize to SourceDocument
7. summarize_all
8. embed indexed content (summary if present else content)
9. PgStore.upsert
10. run hybrid_search for several queries from plan_data
11. generate a simple docx report
12. return ResearchRunResult

Do not wire GitHub, DomainMemory, or reasoning modules in this task.
Warnings should collect non-critical source failures.
```

### Acceptance criteria

- A mocked end-to-end run creates a `.docx` file.
- ResearchRunResult contains competition_id, report_path, num_documents, num_sources, warnings, duration_sec.
- No real external API calls in tests.

---

## 14_reasoning_common

### Goal

Add shared utilities and prompt rules for all reasoning modules.

### Files to create/change

```text
kaggle_researcher/reasoning/common.py
kaggle_researcher/reasoning/prompts.py
tests/test_reasoning_common.py
```

### Codex prompt

```text
Create the common reasoning layer.

Functions/helpers:
- format_retrieved_documents(docs: list[RetrievedDocument]) -> str
- validate_evidence_ids(result, docs) -> list[str]
- call_reasoning_json(client, model, system_prompt, user_payload, result_model)

Add SYSTEM_RULES:
- return JSON only
- separate facts, hypotheses, recommendations
- include confidence
- include evidence_ids
- do not claim real train/test analysis
- do not confirm leakage based only on text sources
- do not implement or imply data-execution features

Use retrieved_documents terminology everywhere.
```

### Acceptance criteria

- Formatting includes document ids, source, title, url, content snippet, rrf_score.
- Evidence validation detects unknown ids.
- Tests pass.

---

## 15_metric_specialist

### Goal

Analyze the competition metric and modeling implications.

### Files to create/change

```text
kaggle_researcher/reasoning/metric_specialist.py
tests/test_metric_specialist.py
```

### Codex prompt

```text
Implement reasoning/metric_specialist.py.

Function:
- async analyze_metric(plan_data: PlanData, retrieved_documents: list[RetrievedDocument], client: DeepSeekClient, model: str) -> MetricResult

Requirements:
- Use DeepSeekClient.chat_json through reasoning/common helper.
- Prompt includes metric guidance:
  - AUC/Gini -> ranking, rank averaging useful
  - LogLoss -> calibration, clipping
  - F1/Dice -> threshold search
  - RMSE/RMSLE -> target transform/clipping
  - MAP@K/NDCG -> ranking/candidate generation
- Result includes confidence and evidence_ids.
- Do not claim dataset was analyzed.
```

### Acceptance criteria

- Mock client response validates into MetricResult.
- Missing evidence ids produce warnings or validation errors according to common helper.

---

## 16_validation_architect

### Goal

Recommend validation strategy and split risks from text sources.

### Files to create/change

```text
kaggle_researcher/reasoning/validation_architect.py
tests/test_validation_architect.py
```

### Codex prompt

```text
Implement reasoning/validation_architect.py.

Function:
- async design_validation(competition_desc: str, plan_data: PlanData, retrieved_documents: list[RetrievedDocument], client: DeepSeekClient, model: str) -> ValidationResult

Requirements:
- Focus only on CV/split risk.
- Do not propose models or feature engineering.
- Separate facts/hypotheses/recommendations.
- confidence should not be high if there are no sources about split/time/group structure.
- evidence_ids are required for source-backed claims.
```

### Acceptance criteria

- Mock response validates into ValidationResult.
- Prompt contains explicit prohibition on model/feature recommendations.

---

## 17_leakage_risk_analyst

### Goal

Generate leakage-risk hypotheses, not confirmed leakage claims.

### Files to create/change

```text
kaggle_researcher/reasoning/leakage_risk_analyst.py
tests/test_leakage_risk_analyst.py
```

### Codex prompt

```text
Implement reasoning/leakage_risk_analyst.py.

Function:
- async analyze_leakage_risk(competition_desc: str, plan_data: PlanData, retrieved_documents: list[RetrievedDocument], client: DeepSeekClient, model: str) -> LeakageRiskResult

Requirements:
- This module produces hypotheses, not detections.
- Prompt must forbid phrases like "leakage found" or "leakage confirmed".
- Use language like "possible risk", "hypothesis", "recommended check".
- confidence should usually be low or medium because real data is not visible.
- Return possible_issues and recommended_checks.
- Include evidence_ids where relevant.
```

### Acceptance criteria

- Prompt includes strict non-confirmation rule.
- Mock response validates into LeakageRiskResult.

---

## 18_leaderboard_auditor

### Goal

Assess public/private leaderboard risk and submission selection rule.

### Files to create/change

```text
kaggle_researcher/reasoning/leaderboard_auditor.py
tests/test_leaderboard_auditor.py
```

### Codex prompt

```text
Implement reasoning/leaderboard_auditor.py.

Function:
- async audit_leaderboard_risk(competition_desc: str, plan_data: PlanData, validation_result: ValidationResult, retrieved_documents: list[RetrievedDocument], client: DeepSeekClient, model: str) -> LeaderboardAuditResult

Requirements:
- Use task_type and metric from plan_data.
- Use validation_result as context.
- Return shake_up_risk, public_lb_trust, submission_selection_rule, warnings, confidence, evidence_ids.
- Do not claim actual LB/CV correlation was measured.
```

### Acceptance criteria

- Mock response validates into LeaderboardAuditResult.
- Prompt warns against public LB overfitting.

---

## 19_experiment_planner

### Goal

Build a prioritized experiment queue with ROI logic.

### Files to create/change

```text
kaggle_researcher/reasoning/experiment_planner.py
tests/test_experiment_planner.py
```

### Codex prompt

```text
Implement reasoning/experiment_planner.py.

Function:
- async plan_experiments(validation_result: ValidationResult, leakage_result: LeakageRiskResult, metric_result: MetricResult, retrieved_documents: list[RetrievedDocument], client: DeepSeekClient, model: str) -> list[ExperimentItem]

Requirements:
- priority must be P0/P1/P2/P3.
- Each experiment includes experiment, why, cost, expected_gain, risk, evidence_ids.
- P0 should include honest validation and baseline if not already covered.
- Do not present EDA/adversarial validation/leakage detection as already executed.
- Sort output by priority.
```

### Acceptance criteria

- Invalid priority fails validation.
- Output is sorted P0 -> P3.
- Tests use mocked LLM response.

---

## 20_skeptical_reviewer

### Goal

Add a review pass that removes unsupported and generic claims.

### Files to create/change

```text
kaggle_researcher/reasoning/skeptical_reviewer.py
tests/test_skeptical_reviewer.py
```

### Codex prompt

```text
Implement reasoning/skeptical_reviewer.py.

Function:
- async review(draft_sections: dict, retrieved_documents: list[RetrievedDocument], client: DeepSeekClient, model: str) -> ReviewResult

Requirements:
- Do not add new unsupported claims.
- Identify unsupported_claims, too_generic, unnecessary_experiments.
- Return revised_sections with the same high-level keys as draft_sections.
- If a key claim has no evidence_ids, mark it unsupported.
- Prompt should use a critical Kaggle Grandmaster reviewer persona without adding new facts.
```

### Acceptance criteria

- Mock response validates into ReviewResult.
- Prompt includes no-new-facts rule.

---

## 21_report_composer

### Goal

Compose the 15-section Kaggle analyst report.

### Files to create/change

```text
kaggle_researcher/reasoning/report_composer.py
tests/test_report_composer.py
```

### Codex prompt

```text
Implement reasoning/report_composer.py.

Function:
- async compose_report(competition_desc, plan_data, domain_patterns, validation_result, leakage_result, metric_result, experiments, lb_audit, review, client, model) -> str

Requirements:
- Output markdown-like text.
- Use exactly 15 sections from docs/SPEC.md.
- Include confidence in key sections.
- Include a "Чего не делать" section.
- Do not claim real EDA or train/test analysis was performed.
- Do not include raw chain-of-thought.
```

### Acceptance criteria

- Report contains all 15 required section headings.
- Report does not contain forbidden data-execution claims.

---

## 22_domain_memory

### Goal

Implement global pattern memory for similar competition families.

### Files to create/change

```text
kaggle_researcher/store/domain_memory.py
patterns/seed_patterns.json
tests/test_domain_memory_unit.py
```

### Codex prompt

```text
Implement store/domain_memory.py.

Class DomainMemory:
- __init__(dsn: str, embed_dim: int)
- async init() -> None
- async find_similar(task_type: str, domain: str, top_k: int = 5) -> list[dict]
- async save_pattern(pattern: dict) -> None
- async seed_from_file(path: str | Path) -> int
- async close() -> None

Requirements:
- Use competition_patterns table from store/sql.py.
- pattern_text = deterministic concat of competition_family, task_type, domain, typical_models.
- embedding via embed_one.
- find_similar does not filter by competition_id.
- save_pattern upserts by id = stable hash(competition_family + source_competition_id).
- Add seed patterns for credit_risk_tabular, medical_imaging, recommender, time_series, nlp_classification.
```

### Acceptance criteria

- pattern_text generation is deterministic.
- seed_from_file returns inserted count.
- Unit tests mock DB and embedder.

---

## 23_full_orchestration

### Goal

Wire all modules into the full `run_research` pipeline.

### Files to create/change

```text
kaggle_researcher/main.py
tests/test_pipeline_full_mocked.py
```

### Codex prompt

```text
Wire the full pipeline in run_research.

Order:
1. load_config
2. build competition_id
3. init PgStore
4. planner.plan with fallback option
5. collect Kaggle/arXiv/GitHub sources
6. summarize documents
7. embed indexed content
8. PgStore.upsert
9. build retrieval queries from plan_data
10. hybrid_search for each query
11. deduplicate retrieved_documents
12. DomainMemory.find_similar
13. metric_specialist
14. validation_architect
15. leakage_risk_analyst
16. leaderboard_auditor
17. experiment_planner
18. skeptical_reviewer
19. report_composer
20. docx_generator
21. save research_run.json
22. return ResearchRunResult

Requirements:
- Non-critical source failures become warnings.
- Critical errors: DB unavailable, no documents, no embeddings, report generation failure.
- Close PgStore/DomainMemory in finally.
- Do not implement data-execution features.
```

### Acceptance criteria

- Fully mocked pipeline completes.
- GitHub failure does not abort pipeline; warning is recorded.
- No train/test dataset functions exist or are called.

---

## 24_docx_generator

### Goal

Make report generation robust and readable.

### Files to create/change

```text
kaggle_researcher/report/docx_generator.py
tests/test_docx_generator.py
```

### Codex prompt

```text
Implement report/docx_generator.py.

Function:
- generate_report(competition_name: str, roadmap_text: str, sources: list[RetrievedDocument], output_path: str | Path) -> Path

Requirements:
- Use python-docx.
- Markdown-like parsing:
  - "# " -> Heading 1
  - "## " -> Heading 2
  - numbered section headings -> Heading 2
  - bullets -> List Bullet
  - fenced code blocks -> monospace paragraphs
- Add final Sources section with title, source, url, rrf_score.
- Create output directory if missing.
```

### Acceptance criteria

- Function creates a valid .docx.
- Sources section is present.
- Tests can open generated document with python-docx.

---

## 25_cli_and_run_json

### Goal

Add a practical CLI and machine-readable run artifact.

### Files to create/change

```text
kaggle_researcher/main.py
tests/test_cli.py
```

### Codex prompt

```text
Add CLI polish and research_run.json output.

Command:
python -m kaggle_researcher.main <competition_url> <competition_desc> \
  --output-dir ./reports \
  --debug \
  --no-github \
  --fast

Requirements:
- --output-dir controls report and JSON output directory.
- --debug enables verbose logging.
- --no-github skips GitHub agent.
- --fast reduces MAX_NOTEBOOKS/MAX_PAPERS/MAX_REPOS and may skip skeptical_reviewer.
- Save research_run.json next to .docx.
- JSON contains competition_id, plan_data, num_sources, warnings, retrieved_document ids, reasoning outputs summary, report_path.
```

### Acceptance criteria

- CLI parses all flags.
- --no-github prevents GitHub agent call.
- research_run.json is written in mocked pipeline test.

---

## 26_quality_gates

### Goal

Validate reasoning outputs and report text before delivery.

### Files to create/change

```text
kaggle_researcher/quality.py
tests/test_quality.py
```

### Codex prompt

```text
Create quality.py.

Functions:
- validate_reasoning_outputs(outputs: dict) -> list[str]
- validate_report_text(report_text: str) -> list[str]
- validate_retrieved_documents(docs: list[RetrievedDocument]) -> list[str]

Checks:
- reasoning outputs include confidence.
- key reasoning outputs include evidence_ids.
- report contains all 15 required sections.
- report must not contain claims such as "we analyzed train/test", "EDA showed", or "adversarial validation found".
- retrieved_documents is not empty.
- Leakage Risk Analyst output must not say leakage is confirmed.
```

### Acceptance criteria

- Quality functions return warnings, not exceptions.
- Tests cover missing sections, forbidden phrases, missing evidence_ids.

---

## 27_offline_eval_fixtures

### Goal

Add offline fixtures to test the full pipeline without external APIs.

### Files to create/change

```text
tests/fixtures/
tests/test_pipeline_offline.py
```

### Codex prompt

```text
Add offline fixtures for a full mocked run.

Create fixtures:
- competition_desc
- mock PlanData
- 3 Kaggle SourceDocument objects
- 2 arXiv SourceDocument objects
- 1 GitHub SourceDocument object
- mock embeddings
- mock RetrievedDocument objects
- mock reasoning outputs

Add test_pipeline_offline.py that runs the pipeline with mocked external clients and verifies:
- ResearchRunResult is returned.
- report text or .docx is generated.
- no real network calls occur.
- quality gates pass or produce only expected warnings.
```

### Acceptance criteria

- Offline test suite runs without API keys.
- No network access is required.
- The pipeline can be validated in CI.

---

# Recommended Codex usage pattern

For each task, use a prompt like this:

```text
Use docs/SPEC.md and docs/CODEX_TASKS.md as the source of truth.
Implement only task <TASK_ID>.
Do not implement later tasks.
Do not implement out-of-scope data-execution features.
Keep the change small and add tests.
```

After each task:

```bash
pytest
python -m kaggle_researcher.main --help
```

For integration-only tasks, keep real API tests behind explicit markers and never require credentials in default CI.
