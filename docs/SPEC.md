# KaggleResearcher

### Спецификация v4
DeepSeek V4 · Qwen3-Embedding-0.6B · PostgreSQL + pgvector · pdfplumber · Reasoning Chain  
2025

> **Codex source of truth.** This Markdown file is the implementation-oriented version of `KaggleResearcher_spec_v4.docx`. Codex should prefer this file over the `.docx` when implementing.
>
> **Scope boundary:** implement only the research/reasoning layer. Do not download train/test datasets, do not run real EDA, do not run adversarial validation, do not execute notebooks, and do not claim confirmed leakage detection.

## Обзор

KaggleResearcher автоматически исследует Kaggle-соревнование: собирает
notebooks, скачивает PDF статей и парсит их через pdfplumber (текст +
таблицы), индексирует в PostgreSQL + pgvector, ищет через гибридный
поиск (векторный + полнотекстовый + RRF), и строит roadmap через цепочку
специализированных reasoning-агентов на DeepSeek V4 Pro.

Ключевое отличие от простого RAG-конвейера: система отвечает не только
на вопрос «что писали другие», но и на вопросы опытного Kaggle-аналитика
— какая валидация честная, где риск утечки, что проверить в первую
очередь, какие эксперименты дадут максимальный прирост.

|                                                                                                                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Вся обработка локальна — кроме DeepSeek API. Эмбеддинги считаются локально через SentenceTransformers; CUDA опциональна, но рекомендуется. PostgreSQL хранит и векторы, и метаданные, и память о паттернах прошлых соревнований в одном месте. |

## Изменения: v1 → v2 (инфраструктура)

|               |                                       |                                                 |
|---------------|---------------------------------------|-------------------------------------------------|
| **Компонент** | **v1**                                | **v2**                                          |
| Embeddings    | nomic-embed-text (Ollama, CPU, 270MB) | Qwen3-Embedding-0.6B (SentenceTransformers)     |
| Векторная БД  | ChromaDB (файловая)                   | PostgreSQL + pgvector (SQL, надёжно)            |
| Тип поиска    | только косинусный                     | гибридный: vector + tsvector + RRF              |
| PDF статьи    | только abstract из API                | полный текст + таблицы через pdfplumber         |
| Батчинг embed | по одному запросу                     | локальные батчи через SentenceTransformers      |
| PDF кэш       | нет                                   | ./data/pdfs/ — повторный запуск не перекачивает |

## Изменения: v2 → v3 (reasoning-слой)

|                                |                                |                                                                          |
|--------------------------------|--------------------------------|--------------------------------------------------------------------------|
| **Компонент**                  | **v2**                         | **v3**                                                                   |
| Синтез отчёта                  | один reasoner.py, один вызов   | цепочка из 7 reasoning-модулей, каждый отвечает за свой вопрос           |
| Валидация                      | не выделена отдельно           | validation_architect.py — CV-схема и риск сплита до выбора модели        |
| Анализ утечек                  | отсутствовал                   | leakage_risk_analyst.py — гипотезы по текстовым источникам, не по данным |
| Метрика                        | упоминалась вскользь в roadmap | metric_specialist.py — calibration, rank averaging, threshold search     |
| План экспериментов             | плоский список идей            | experiment_planner.py — приоритеты P0-P3 с оценкой cost/expected_gain    |
| Leaderboard risk               | не учитывался                  | leaderboard_auditor.py — shake-up risk, правило выбора сабмитов          |
| Самопроверка                   | отсутствовала                  | skeptical_reviewer.py — критика черновика перед финальной сборкой        |
| Память о прошлых соревнованиях | отсутствовала                  | domain_memory.py — паттерны across runs, отдельная таблица               |
| Структура отчёта               | 7 секций (анализ → источники)  | 15 секций (executive summary → 48-hour action plan)                      |
| Уверенность вывода             | не указывалась                 | поле confidence (low/medium/high) в каждом reasoning-результате          |

## Технический стек

|                           |                                |                                                                     |
|---------------------------|--------------------------------|---------------------------------------------------------------------|
| **Компонент**             | **Инструмент**                 | **Роль**                                                            |
| LLM: планировщик + синтез | DeepSeek V4 Pro API            | Chain-of-thought, не грузит GPU                                     |
| LLM: суммаризация         | DeepSeek V4 Flash API          | Дешевле V4 Pro для рутинной суммаризации                            |
| Embeddings                | Qwen3-Embedding-0.6B (SentenceTransformers) | Локально, CUDA опциональна                                            |
| Векторная БД              | PostgreSQL 16 + pgvector       | Векторный + полнотекстовый поиск в SQL                              |
| PDF парсинг               | pdfplumber                     | Текст + таблицы из arXiv PDF                                        |
| Источник: ноутбуки        | Kaggle Python API              | Notebooks, writeups, сортировка по голосам                          |
| Источник: статьи          | arxiv + Papers with Code API   | Академические источники, без ключей                                 |
| Источник: код             | GitHub REST API                | Репо по звёздам                                                     |
| HTTP-клиент               | httpx + asyncio                | Параллельные запросы                                                |
| Reasoning-цепочка         | DeepSeek V4 Pro, 7 промптов    | Validation, Leakage, Metric, Experiments, LB Audit, Review, Compose |
| Память паттернов          | PostgreSQL (отдельная таблица) | Паттерны прошлых соревнований across runs                           |
| Отчёт                     | python-docx                    | Финальный .docx с 15-секционным roadmap                             |
| БД контейнер              | docker compose                 | pgvector/pgvector:pg16                                              |

## Архитектура

Изменение относительно v2: вместо одного reasoner — цепочка
специализированных reasoning-агентов. Каждый отвечает на свой вопрос
вместо одного общего 'дай roadmap'. Все работают над одним и тем же
набором retrieved_documents, но с разными системными промптами.
Дополнительная инфраструктура не нужна — это всё DeepSeek V4 Pro вызовы
поверх существующего retrieval.

- Шаг 1 — Planner: DeepSeek V4 Pro декомпозирует описание → JSON с
  запросами для каждого агента

- Шаг 2 — Агенты параллельно: Kaggle notebooks, arXiv PDF, GitHub репо

- Шаг 3 — PDF Parser: pdfplumber скачивает и парсит PDF (текст + таблицы
  → строки)

- Шаг 4 — Summarizer: DeepSeek V4 Flash сжимает каждый документ до 300
  токенов параллельно

- Шаг 5 — Embedder: Qwen3-Embedding-0.6B через SentenceTransformers — все документы батчем,
  CUDA опциональна

- Шаг 6 — pgvector upsert: записывает документы + эмбеддинги + tsvector
  в Postgres

- Шаг 7 — Hybrid Retriever: векторный + FTS параллельно, RRF слияние

- Шаг 8 — Domain Memory: подтягивает паттерны похожих прошлых
  соревнований из pgvector

- Шаг 9 — Reasoning-цепочка: Validation Architect → Leakage Risk Analyst
  → Metric Specialist → Experiment Planner → Leaderboard Auditor,
  параллельно где возможно

- Шаг 10 — Skeptical Reviewer: критикует и дорабатывает вывод шага 9

- Шаг 11 — Report Composer: собирает все разделы в единый roadmap

- Шаг 12 — docx: генерирует report\_{competition_id}.docx

|                                                                                                                                                                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data-execution функциональность (реальный EDA на train/test, adversarial validation, парсинг/выполнение notebooks) сознательно не включена в текущий scope. Этот проект реализует только research/reasoning layer. |

## Структура проекта

```text
kaggle_researcher/
├── main.py                   # точка входа
├── config.py                 # ключи и параметры
├── schemas.py                # единые Pydantic-схемы
├── planner.py                # DeepSeek V4 Pro: декомпозиция
├── summarizer.py             # DeepSeek V4 Flash: сжатие документов
├── embedder.py               # Qwen3-Embedding-0.6B via SentenceTransformers: батч-эмбеддинги
├── retriever.py              # гибридный поиск + RRF
├── agents/
│   ├── kaggle_agent.py       # поиск и скачивание notebooks
│   ├── arxiv_agent.py        # поиск + скачивание PDF
│   └── github_agent.py       # поиск репо
├── parsers/
│   └── pdf_parser.py         # pdfplumber: текст + таблицы
├── reasoning/
│   ├── validation_architect.py   # CV-схема, риски сплита
│   ├── leakage_risk_analyst.py   # гипотезы о leakage-рисках по источникам
│   ├── metric_specialist.py      # интерпретация метрики, calibration/threshold
│   ├── experiment_planner.py     # приоритизированная очередь экспериментов
│   ├── leaderboard_auditor.py    # public/private LB риски, выбор сабмитов
│   ├── skeptical_reviewer.py     # критика и доработка отчёта
│   └── report_composer.py        # сборка всех секций в единый roadmap
├── store/
│   ├── pg_store.py           # PostgreSQL + pgvector: документы
│   └── domain_memory.py      # паттерны прошлых соревнований
├── report/
│   └── docx_generator.py     # генерация .docx
├── docker-compose.yml        # PostgreSQL с pgvector
├── requirements.txt
└── data/
    ├── pdfs/                 # кэш PDF файлов
    └── postgres/             # данные PostgreSQL
```

## Инфраструктура

### docker-compose.yml

```yaml
version: "3.8"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: researcher
      POSTGRES_PASSWORD: researcher
      POSTGRES_DB: kaggle_research
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    restart: unless-stopped
```

### Local embeddings — Qwen3-Embedding-0.6B

Embeddings are computed directly inside Python with SentenceTransformers. CUDA is optional but recommended.

```bash
pip install -r requirements.txt
```

The default model is `Qwen/Qwen3-Embedding-0.6B`.

## Контракты модулей

Каждый модуль описан через публичный интерфейс: сигнатуры функций и
классов, параметры, возвращаемые значения, побочные эффекты и
инварианты.

## Единая схема данных

Все агенты возвращают разные наборы полей, но к моменту передачи в
PgStore.upsert и далее в reasoning-слой все объекты приводятся к двум
единым схемам. Chunking не планируется — один документ хранится как одна
запись. Термин 'retrieved_documents' используется вместо 'чанков' везде
кроме RRF (где это технический термин ранжирования).

### SourceDocument — схема на входе в PgStore

|                |             |                  |                                                                          |
|----------------|-------------|------------------|--------------------------------------------------------------------------|
| **Поле**       | **Тип**     | **Обязательное** | **Описание**                                                             |
| id             | str         | да               | Уникальный идентификатор: kaggle ref / arxiv entry_id / github full_name |
| competition_id | str         | да               | Идентификатор соревнования — ключ изоляции в pgvector                    |
| source         | str         | да               | Одно из: 'kaggle' \| 'arxiv' \| 'papers_with_code' \| 'github'           |
| title          | str         | да               | Название документа                                                       |
| url            | str \| None | нет              | Ссылка на источник                                                       |
| content        | str         | да               | Сырой контент (текст notebook / PDF / README), лимит 8000 символов       |
| summary        | str \| None | нет              | Заполняется summarizer.py, до этого None                                 |
| metadata       | dict        | нет              | Источник-специфичные поля: total_votes, stars, abstract, pdf_url         |

### RetrievedDocument — схема на выходе из retriever и входе в reasoning-слой

|                |             |                                                                |
|----------------|-------------|----------------------------------------------------------------|
| **Поле**       | **Тип**     | **Описание**                                                   |
| id             | str         | Идентификатор из SourceDocument                                |
| competition_id | str         | Идентификатор соревнования                                     |
| source         | str         | Источник: kaggle / arxiv / papers_with_code / github           |
| title          | str         | Название документа                                             |
| url            | str \| None | Ссылка                                                         |
| content        | str         | Резюме от summarizer (или сырой content если summary пуст)     |
| score          | float       | Косинусная близость из vector_search или ts_rank из fts_search |
| rrf_score      | float       | Итоговый RRF score после слияния двух списков                  |

### embedder.py

Единственная точка взаимодействия с SentenceTransformers. Все остальные модули получают
эмбеддинги только через этот модуль. MAX_EMBED_BATCH_SIZE (из config)
ограничивает размер одного запроса.

|                    |                                                    |                                                                                      |                                                                                                                                                                                                                                                         |
|--------------------|----------------------------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Функция**        | **Параметры**                                      | **Возврат**                                                                          | **Эффекты / инварианты**                                                                                                                                                                                                                                |
| embed_texts(texts) | texts: list\[str\] — список строк для векторизации | list\[list\[float\]\] — список векторов размерности EMBED_DIM, порядок совпадает с входом | Если len(texts) \> MAX_EMBED_BATCH_SIZE (default 8) — автоматически бьёт на батчи и конкатенирует. Эмбеддинги нормализуются. |
| embed_one(text)    | text: str                                          | list\[float\] — один вектор EMBED_DIM                                                 | Обёртка над embed_texts(\[text\]). Используется в retriever для query-вектора.                                                                                                                                                                          |

### config.py

Все настройки системы. Ключи читаются из переменных окружения — нет
значений по умолчанию для секретов, при отсутствии — KeyError на старте.

|                      |         |               |                                                                        |
|----------------------|---------|---------------|------------------------------------------------------------------------|
| **Константа**        | **Тип** | **Источник**  | **Назначение**                                                         |
| DEEPSEEK_API_KEY     | str     | env           | Ключ DeepSeek API                                                      |
| DEEPSEEK_V4_PRO      | str     | hardcoded     | Model ID: deepseek-v4-pro (планировщик, синтез, reasoning-цепочка)     |
| DEEPSEEK_V4_FLASH    | str     | hardcoded     | Model ID: deepseek-v4-flash (суммаризация — дешевле V4 Pro для рутины) |
| EMBED_MODEL          | str     | env / default | Qwen/Qwen3-Embedding-0.6B                                              |
| EMBED_DIM            | int     | env / default | 1024 — размерность вектора модели                                      |
| MAX_EMBED_BATCH_SIZE | int     | env / default | 8 — максимум строк в одном локальном батче                             |
| PG_DSN               | str     | env / default | postgresql://researcher:researcher@localhost:5432/kaggle_research      |
| TOP_K                | int     | hardcoded     | 10 — документов из pgvector на один retrieval-запрос                   |
| MAX_NOTEBOOKS        | int     | hardcoded     | 20 — лимит notebooks с Kaggle                                          |
| MAX_PAPERS           | int     | hardcoded     | 15 — лимит статей с arXiv                                              |
| MAX_REPOS            | int     | hardcoded     | 10 — лимит репо с GitHub                                               |
| PDF_CACHE_DIR        | str     | hardcoded     | ./data/pdfs — локальный кэш PDF                                        |

### planner.py

Декомпозиция соревнования на поисковые запросы. Единственный вызов
DeepSeek V4 Pro с JSON-режимом на входе в пайплайн.

|                   |                                                                             |                                                                                                                                                                   |                                                                                                         |
|-------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **Функция**       | **Параметры**                                                               | **Возврат**                                                                                                                                                       | **Эффекты / инварианты**                                                                                |
| plan(description) | description: str — текстовое описание соревнования (URL + задача + метрика) | dict с ключами: task_type, metric, domain, kaggle_queries (list), arxiv_queries (list), github_queries (list), key_techniques (list), similar_competitions (list) | Один HTTP-запрос к DeepSeek V4 Pro. response_format=json_object гарантирует валидный JSON. Timeout 90с. |

### agents/kaggle_agent.py

|                                  |                                                     |                                                                                                                          |                                                                                                             |
|----------------------------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Функция**                      | **Параметры**                                       | **Возврат**                                                                                                              | **Эффекты / инварианты**                                                                                    |
| search_notebooks(queries)        | queries: list\[str\] — поисковые запросы из planner | list\[dict\]: {id, title, url, total_votes, source='kaggle'} — отсортировано по голосам по убыванию, лимит MAX_NOTEBOOKS | Kaggle API (синхронный). Дедупликация по ref. Требует ~/.kaggle/kaggle.json или env vars.                   |
| get_notebook_content(kernel_ref) | kernel_ref: str — формат 'username/kernel-name'     | str — текст ячеек notebook, лимит 8000 символов. Пустая строка если .ipynb не найден.                                    | subprocess: kaggle kernels pull в /tmp. Извлекает markdown-ячейки и первые 500 символов каждой code-ячейки. |

### agents/arxiv_agent.py

|                                |                                               |                                                                                                                |                                                                                                                             |
|--------------------------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **Функция**                    | **Параметры**                                 | **Возврат**                                                                                                    | **Эффекты / инварианты**                                                                                                    |
| search_arxiv(queries)          | queries: list\[str\]                          | list\[dict\]: {id, title, abstract, pdf_url, url, source='arxiv'} — лимит MAX_PAPERS, дедупликация по entry_id | Использует библиотеку arxiv (без ключа). Сортировка по релевантности.                                                       |
| enrich_with_pdf(papers)        | papers: list\[dict\] — результат search_arxiv | list\[dict\] — те же объекты с добавленным полем content: str (полный текст PDF или abstract как fallback)     | Параллельно скачивает PDF в PDF_CACHE_DIR. Если файл уже есть — не перекачивает. При ошибке скачивания: content = abstract. |
| search_papers_with_code(query) | query: str                                    | list\[dict\]: {id, title, content, url, source='papers_with_code'}                                             | Papers with Code REST API, без ключа. page_size=10.                                                                         |

### agents/github_agent.py

|                       |                      |                                                                                                           |                                                                                                             |
|-----------------------|----------------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Функция**           | **Параметры**        | **Возврат**                                                                                               | **Эффекты / инварианты**                                                                                    |
| search_repos(queries) | queries: list\[str\] | list\[dict\]: {id, title, content, url, stars, source='github'} — отсортировано по stars, лимит MAX_REPOS | GitHub Search API. Без токена: 60 req/час. С GITHUB_TOKEN: 5000 req/час. Запросы параллельны через asyncio. |

### parsers/pdf_parser.py

Приоритизирует страницы: abstract (стр. 1), conclusions (последние 2),
любые страницы с таблицами. Таблицы конвертируются в текст 'cell1 \|
cell2 \| cell3' — попадают в эмбеддинг и FTS.

|                                     |                                                                             |                                                                                                    |                                                                                                      |
|-------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Функция**                         | **Параметры**                                                               | **Возврат**                                                                                        | **Эффекты / инварианты**                                                                             |
| download_pdf(url, paper_id)         | url: str — ссылка на PDF; paper_id: str — используется как имя файла в кэше | Path \| None — путь к файлу. None если скачать не удалось.                                         | async. Пишет в PDF_CACHE_DIR/{paper_id}.pdf. Если файл уже существует — возвращает путь без запроса. |
| extract_tables_as_text(page)        | page — объект pdfplumber.Page                                               | str — все таблицы страницы, строки через '\n', ячейки через ' \| '. Пустая строка если таблиц нет. | Чистый CPU. None-ячейки заменяются пустой строкой.                                                   |
| parse_pdf(pdf_path, max_chars=8000) | pdf_path: Path; max_chars: int — лимит символов на выходе                   | str — извлечённый текст + таблицы, усечённый до max_chars                                          | Открывает PDF через pdfplumber. Обходит все страницы, включает приоритетные. Без сетевых вызовов.    |

### summarizer.py

Сжимает каждый документ до 250-300 слов через DeepSeek V4 Flash. Запросы
параллельны. При ошибке API — fallback на первые 800 символов оригинала.

|                            |                                                                  |                                                                      |                                                                                                          |
|----------------------------|------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Функция**                | **Параметры**                                                    | **Возврат**                                                          | **Эффекты / инварианты**                                                                                 |
| summarize_one(client, doc) | client: httpx.AsyncClient; doc: dict с полем content или summary | dict — тот же объект с добавленным/перезаписанным полем summary: str | Один HTTP-запрос к DeepSeek V4 Flash. Если текст короче 150 символов — возвращает как есть. Timeout 30с. |
| summarize_all(docs)        | docs: list\[dict\]                                               | list\[dict\] — все документы с полем summary                         | asyncio.gather — все запросы параллельно. Порядок совпадает с входом.                                    |

### store/pg_store.py — класс PgStore

Одна таблица documents хранит текст, эмбеддинг (vector(EMBED_DIM)), tsvector
(вычисляемый из content), метаданные. Изоляция соревнований по полю
competition_id.

Индексы: HNSW (m=16, ef_construction=64) для косинусного поиска; GIN для
полнотекстового; B-tree по competition_id.

|                                 |                                                                          |                                                                                      |                                                                                                                 |
|---------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Метод**                       | **Параметры**                                                            | **Возврат**                                                                          | **Эффекты / инварианты**                                                                                        |
| \_\_init\_\_(competition_id)    | competition_id: str — идентификатор соревнования                         | —                                                                                    | Сохраняет competition_id. Не открывает соединение.                                                              |
| init()                          | —                                                                        | —                                                                                    | async. Создаёт пул asyncpg. Выполняет CREATE EXTENSION, CREATE TABLE, CREATE INDEX IF NOT EXISTS. Идемпотентно. |
| upsert(docs, embeddings)        | docs: list\[dict\]; embeddings: list\[list\[float\]\] — одинаковой длины | —                                                                                    | async. INSERT ... ON CONFLICT (id) DO UPDATE — безопасен при повторном запуске. executemany одной транзакцией.  |
| vector_search(embedding, top_k) | embedding: list\[float\]; top_k: int = 10                                | list\[dict\]: {id, title, url, source, content, score} — score = 1 - cosine_distance | async. Фильтрует по competition_id. Использует HNSW индекс.                                                     |
| fts_search(query, top_k)        | query: str; top_k: int = 10                                              | list\[dict\]: {id, title, url, source, content, score} — score = ts_rank             | async. plainto_tsquery('english', query). Фильтрует по competition_id. Использует GIN индекс.                   |
| close()                         | —                                                                        | —                                                                                    | async. Закрывает пул соединений.                                                                                |

### retriever.py

RRF (Reciprocal Rank Fusion): score = Σ 1/(60 + rank). k=60 — значение
из оригинальной статьи. Документы, хорошо ранжированные в обоих списках,
получают наибольший итоговый score.

|                                                           |                                                                 |                                                                                          |                                                                                                                                                   |
|-----------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **Функция**                                               | **Параметры**                                                   | **Возврат**                                                                              | **Эффекты / инварианты**                                                                                                                          |
| reciprocal_rank_fusion(vector_results, fts_results, k=60) | vector_results: list\[dict\]; fts_results: list\[dict\]; k: int | list\[dict\] — объединённый список с полем rrf_score: float, отсортированный по убыванию | Чистая функция, без IO. Дедупликация по id. При совпадении id — суммирует очки.                                                                   |
| hybrid_search(store, query, top_k=10)                     | store: PgStore; query: str; top_k: int                          | list\[dict\] — top_k документов с rrf_score                                              | async. Вызывает embed_one(query), затем asyncio.gather(vector_search, fts_search) — параллельно. Передаёт top_k\*2 в каждый поиск перед слиянием. |

## Reasoning-слой

### Общий контракт reasoning-модулей

Каждый reasoning-модуль обязан возвращать JSON и соблюдать единые правила:

- явно отделять `facts`, `hypotheses` и `recommendations`;
- добавлять `confidence: low | medium | high`;
- добавлять `evidence_ids: list[str]` для ключевых выводов;
- не утверждать, что реальные `train/test` данные были проанализированы;
- не выдавать leakage-гипотезы за подтверждённые факты;
- не предлагать data-execution действия как уже выполненные проверки;
- опираться только на `competition_desc`, `plan_data`, `retrieved_documents` и `domain_patterns`.

Рекомендуемая базовая JSON-оболочка для reasoning-результатов:

```json
{
  "facts": [],
  "hypotheses": [],
  "recommendations": [],
  "evidence_ids": [],
  "confidence": "low|medium|high"
}
```

Вместо одного reasoner — цепочка специализированных модулей. Каждый
получает retrieved_documents (и где нужно — план из planner.py) и отвечает
на свой узкий вопрос. Все модули — это DeepSeek V4 Pro с разными
системными промптами, новой инфраструктуры не требуют.

### store/domain_memory.py

Память о паттернах прошлых соревнований. Отдельная таблица в том же
Postgres — не per-competition, а across all competitions. Заполняется
вручную (seed) или автоматически по итогам каждого успешного прогона.

```sql
CREATE TABLE IF NOT EXISTS competition_patterns (
    id                   TEXT PRIMARY KEY,
    competition_family   TEXT NOT NULL,   -- напр. 'credit_risk_tabular'
    task_type            TEXT,            -- из plan_data.task_type
    domain               TEXT,            -- из plan_data.domain
    pattern_text         TEXT NOT NULL,    -- текст, из которого считан embedding
    embedding            vector(EMBED_DIM),
    typical_models       JSONB,           -- ["LightGBM", "CatBoost", ...]
    typical_features     JSONB,           -- ["bureau aggregations", ...]
    typical_validation   TEXT,            -- напр. 'time/group split'
    common_traps         JSONB,           -- ["random KFold overestimates", ...]
    source_competition_id TEXT,           -- какое соревнование породило паттерн
    created_at           TIMESTAMPTZ DEFAULT now(),
    updated_at           TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON competition_patterns USING hnsw (embedding vector_cosine_ops);
```

pattern_text — конкатенация competition_family + task_type + domain +
typical_models, формируется детерминированно перед вызовом embed_one().
Это то, что реально векторизуется и участвует в косинусном поиске.

|                                                       |                                                                                        |                                                                                                                                       |                                                                                                                                                                                                   |
|-------------------------------------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Метод / функция**                                   | **Параметры**                                                                          | **Возврат**                                                                                                                           | **Эффекты / инварианты**                                                                                                                                                                          |
| DomainMemory.find_similar(task_type, domain, top_k=5) | task_type: str; domain: str; top_k: int                                                | list\[dict\]: {competition_family, typical_models, typical_features, typical_validation, common_traps} — top_k по косинусной близости | async. embed_one(f'{task_type} {domain}') → vector_search по competition_patterns. Не фильтрует по competition_id — это глобальная таблица.                                                       |
| DomainMemory.save_pattern(pattern)                    | pattern: dict — поля как в DDL, кроме id/embedding/created_at/updated_at (вычисляются) | —                                                                                                                                     | async. id = hash(competition_family + source_competition_id). pattern_text собирается перед вызовом embed_one(). upsert по id — повторный прогон того же competition_family обновляет updated_at. |

### reasoning/validation_architect.py

Главный модуль слоя. Предлагает CV-схему и оценивает риск incorrect
split до того, как предложена любая модель.

|                                                        |                                                                                                       |                                                                                                                                                 |                                                                                                                  |
|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **Функция**                                            | **Параметры**                                                                                         | **Возврат**                                                                                                                                     | **Эффекты / инварианты**                                                                                         |
| design_validation(competition_desc, plan_data, retrieved_documents) | competition_desc: str; plan_data: dict — вывод planner.py; retrieved_documents: list[dict] — retrieved источники | dict: {recommended_cv, validation_risk: low/medium/high, likely_split, failure_modes: list\[str\], reasoning: str, confidence: low/medium/high} | Один вызов DeepSeek V4 Pro. Системный промпт фокусирует модель только на вопросах валидации, не на модели/фичах. |

### reasoning/leakage_risk_analyst.py

Формирует гипотезы о возможных leakage-рисках по текстовым источникам —
НЕ выполнение кода на реальных данных. Ищет сигналы риска в описаниях
фичей и подходов, упомянутых в notebooks/статьях (например упоминания
id-полей, target encoding без OOF, random KFold там где вероятен
group/time split). Намеренно назван «analyst», а не «hunter» — модуль
выдвигает проверяемые гипотезы, а не находит подтверждённые утечки.

|                                                           |                                                              |                                                                                                                                 |                                                                                                                                                                |
|-----------------------------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Функция**                                               | **Параметры**                                                | **Возврат**                                                                                                                     | **Эффекты / инварианты**                                                                                                                                       |
| analyze_leakage_risk(competition_desc, plan_data, retrieved_documents) | competition_desc: str; plan_data: dict; retrieved_documents: list[dict] | dict: {risk_level: low/medium/high, possible_issues: list\[str\], recommended_checks: list\[str\], confidence: low/medium/high} | Один вызов DeepSeek V4 Pro. Анализирует только текст источников — не датасет. confidence обязателен и обычно не выше medium, так как реальные данные не видны. |

### reasoning/metric_specialist.py

|                                   |                                                                       |                                                                                                                                                                         |                                                                                                                             |
|-----------------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **Функция**                       | **Параметры**                                                         | **Возврат**                                                                                                                                                             | **Эффекты / инварианты**                                                                                                    |
| analyze_metric(plan_data, retrieved_documents) | plan_data: dict — содержит metric из planner.py; retrieved_documents: list[dict] | dict: {metric_explanation, needs_calibration: bool, rank_averaging_useful: bool, threshold_search_needed: bool, surrogate_loss_suggestion, confidence: low/medium/high} | Один вызов DeepSeek V4 Pro. Промпт содержит таблицу метрика→рекомендация (AUC/Gini, LogLoss, MAP@K, F1, RMSE) как few-shot. |

### reasoning/experiment_planner.py

Roadmap как приоритизированная очередь экспериментов с оценкой ROI, а не
плоский список идей.

|                                                                            |                                                                                          |                                                                                                         |                                                                                                                               |
|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Функция**                                                                | **Параметры**                                                                            | **Возврат**                                                                                             | **Эффекты / инварианты**                                                                                                      |
| plan_experiments(validation_result, leakage_result, metric_result, retrieved_documents) | validation_result: dict; leakage_result: dict; metric_result: dict; retrieved_documents: list[dict] | list\[dict\]: {priority: P0..P3, experiment, why, cost, expected_gain, risk}, отсортировано по priority | Один вызов DeepSeek V4 Pro. Получает выводы трёх предыдущих модулей как контекст — это финальная агрегация перед reviewer'ом. |

### reasoning/leaderboard_auditor.py

Оценивает риск переобучения на public leaderboard и формулирует критерий
выбора финальных сабмитов.

|                                                                                |                                                                                       |                                                                                                                                                  |                                                                                                                                |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Функция**                                                                    | **Параметры**                                                                         | **Возврат**                                                                                                                                      | **Эффекты / инварианты**                                                                                                       |
| audit_leaderboard_risk(competition_desc, plan_data, validation_result, retrieved_documents) | competition_desc: str; plan_data: dict; validation_result: dict; retrieved_documents: list[dict] | dict: {shake_up_risk: low/medium/high, submission_selection_rule: str, public_lb_trust: str, warnings: list\[str\], confidence: low/medium/high} | Один вызов DeepSeek V4 Pro. Опирается на public/private split упоминания в источниках и тип соревнования/метрику из plan_data. |

### reasoning/skeptical_reviewer.py

Второй проход поверх вывода experiment_planner. Не добавляет новые
неподтверждённые утверждения; может переписывать, удалять или помечать
слабые места в существующих секциях, опираясь только на то, что
подтверждено retrieved-источниками.

|                                |                                                                                                                                 |                                                                                                                                 |                                                                                                                                                      |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Функция**                    | **Параметры**                                                                                                                   | **Возврат**                                                                                                                     | **Эффекты / инварианты**                                                                                                                             |
| review(draft_sections, retrieved_documents) | draft_sections: dict — все выводы предыдущих модулей объединённые; retrieved_documents: list[dict] — для проверки утверждений на источники | dict: {unsupported_claims: list\[str\], too_generic: list\[str\], unnecessary_experiments: list\[str\], revised_sections: dict} | Один вызов DeepSeek V4 Pro с системным промптом 'критикуй как Kaggle Grandmaster'. revised_sections — те же ключи что draft_sections, но с правками. |

### reasoning/report_composer.py

Финальная сборка: берёт выводы всех reasoning-модулей после ревью и
компонует единый текст отчёта по 15-секционной структуре (executive
summary → ... → first 48 hours plan).

|                                                                                                                                               |                                                    |                                                                     |                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| **Функция**                                                                                                                                   | **Параметры**                                      | **Возврат**                                                         | **Эффекты / инварианты**                                                                                                              |
| compose_report(competition_desc, plan_data, domain_patterns, validation_result, leakage_result, metric_result, experiments, lb_audit, review) | все dict-объекты — выводы предыдущих шагов цепочки | str — итоговый текст roadmap в markdown-подобном формате, 15 секций | Один вызов DeepSeek V4 Pro. Prompt задаёт фиксированную структуру секций. max_tokens=6000 (отчёт длиннее чем в v2 из-за доп. секций). |

### report/docx_generator.py

|                                                                       |                                                                                                                                                      |             |                                                                                                                                                                                                                            |
|-----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Функция**                                                           | **Параметры**                                                                                                                                        | **Возврат** | **Эффекты / инварианты**                                                                                                                                                                                                   |
| generate_report(competition_name, roadmap_text, sources, output_path) | competition_name: str; roadmap_text: str — вывод report_composer; sources: list[RetrievedDocument] — документы с rrf_score и url; output_path: str = 'roadmap.docx' | —           | Пишет .docx на диск. Парсит roadmap_text построчно: строки вида '1. ЗАГОЛОВОК' и '## заголовок' → Heading 2; строки с '–'/'•' → List Bullet; остальное → Normal. Последняя страница — список источников с RRF score и URL. |

### main.py — оркестрация

Единственная публичная точка входа в систему. Связывает все модули в
единый пайплайн. Возвращает ResearchRunResult — структурированный
результат прогона, который можно логировать или проверять в тестах.

|                                                                      |                                                                                                                                                |                                                                                                                                             |                                                                                                                                                                                                                                                                                                                    |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Функция**                                                          | **Параметры**                                                                                                                                  | **Возврат**                                                                                                                                 | **Эффекты / инварианты**                                                                                                                                                                                                                                                                                           |
| run_research(competition_url, competition_desc, competition_id=None) | competition_url: str; competition_desc: str — текстовое описание задачи + метрики; competition_id: str \| None — если None, извлекается из URL | dict: {competition_id: str, report_path: str, num_documents: int, num_sources: dict\[str,int\], warnings: list\[str\], duration_sec: float} | async. Единственная функция с side-effects на весь пайплайн: создаёт pgvector коллекцию, пишет PDF в кэш, генерирует .docx. warnings собирает некритичные проблемы (PDF недоступен, GitHub rate limit). При критической ошибке — exception, partial state в БД остаётся (повторный запуск безопасен через upsert). |

### num_sources — пример значения

|                  |              |                                         |
|------------------|--------------|-----------------------------------------|
| **Ключ**         | **Значение** | **Описание**                            |
| kaggle           | int          | Количество проиндексированных notebooks |
| arxiv            | int          | Количество статей с arXiv               |
| papers_with_code | int          | Количество статей с Papers with Code    |
| github           | int          | Количество репозиториев                 |

## Структура финального отчёта

report_composer.py собирает не плоский roadmap, а структурированный
документ из 15 секций — это и есть отличие 'research bot' от
'Kaggle-аналитика'.

- 1\. Executive summary

- 2\. Тип соревнования и интерпретация метрики

- 3\. Анатомия датасета (по доступным описаниям, без EDA на реальных
  данных)

- 4\. Стратегия валидации

- 5\. Риски утечки и shake-up

- 6\. Разведка по публичным notebooks

- 7\. Паттерны похожих прошлых соревнований

- 8\. План baseline

- 9\. План feature engineering

- 10\. План моделей

- 11\. План ансамблирования

- 12\. Очередь экспериментов с приоритетами

- 13\. Стратегия выбора финальных сабмитов

- 14\. Чего не делать

- 15\. План первых 48 часов

|                                                                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Секция «Чего не делать» формируется из выводов skeptical_reviewer (unnecessary_experiments) — это явный фильтр от траты времени на типовые ловушки соревнования. |

## Запуск

```bash
# 1. PostgreSQL с pgvector
docker compose up -d

# 2. Зависимости
pip install -r requirements.txt

# 3. Переменные окружения
export DEEPSEEK_API_KEY='sk-...'
export KAGGLE_USERNAME='username'
export KAGGLE_KEY='key'
export GITHUB_TOKEN='ghp_...'   # опционально

# 4. Запуск
python main.py \
  'https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability' \
  'Предсказание дефолта по кредиту. Метрика: Gini. Табличные данные.'
```

## requirements.txt

```text
httpx==0.27.0
kaggle==1.6.14
arxiv==2.1.0
asyncpg==0.29.0
pgvector==0.3.2
pdfplumber==0.11.0
python-docx==1.1.2
sentence-transformers
torch
transformers
accelerate
```

## Ограничения

- Qwen3-Embedding-0.6B runs locally through SentenceTransformers. CUDA is optional but recommended.

- Kaggle API требует принятия правил соревнования для скачивания
  notebooks

- GitHub без токена: 60 req/час; с токеном: 5000 req/час

- PDF некоторых статей недоступны — автоматический fallback на abstract
  из API

- DeepSeek V4 Pro думает 30-90 секунд на вызов — нормально, он делает
  chain-of-thought. Reasoning-цепочка из 7 модулей кратно увеличивает
  суммарное время прогона относительно v2

- Повторный запуск безопасен: upsert не дублирует данные, PDF кэшируются
  в ./data/pdfs/

- Leakage Risk Analyst анализирует только текст источников — не реальные
  данные соревнования. Это эвристика по описаниям подходов в
  notebooks/статьях, не детектор утечек на факте

- Validation Architect и Leaderboard Auditor дают рекомендации на основе
  текстовых источников и описания задачи — не на основе фактического
  анализа train/test


## Kaggle EDA Engine / Data Evidence Layer

Starting from v5, KaggleResearcher includes an optional EDA Engine described in `docs/EDA_ENGINE_SPEC.md`.

The existing research pipeline remains source/retrieval/reasoning-based and does not execute Kaggle datasets. The EDA Engine is a separate data-execution layer under:

- `kaggle_researcher/eda/`
- `kaggle_eda_engine/`

EDA Engine consumes:

- `research_hypotheses.json`
- `eda_task_plan.json`
- Kaggle dataset or local dataset path

EDA Engine produces:

- `eda_evidence_pack.json`
- `eda_summary.md`
- module-level JSON artifacts
- `artifacts/`

Notebook execution remains forbidden. Dataset execution is allowed only inside the EDA Engine scope.