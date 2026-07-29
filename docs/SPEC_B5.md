# Wavebreaker B5 — Competition Brief

<!-- Canonical B5 specification: docs/SPEC_B5.md -->

**Спецификация под существующую кодовую базу** `komarand/wavebreaker@main`
**Цель системы:** выигрывать соревнования, а не быть полной архитектурой
**Дата:** 27 июля 2026

---

## 0. Что уже есть в репозитории

Обзор `main` (37 коммитов, ~8000 LOC) существенно расходится с README, который заявляет «bootstrap only». Реализовано больше, и часть моих прежних замечаний уже закрыта кодом.

**Закрыто в коде, снимаю из реестра:**

| Пункт | Как закрыто |
|---|---|
| R-34 retry | `deepseek_client._post_chat_completions` — обработка 429 и 5xx с backoff |
| R-36 subprocess | `kaggle_agent` использует `KaggleApi` напрямую, не subprocess |
| R-31/R-32 GPU | vLLM убран, `embedder` на SentenceTransformers, модель 0.6B/1024 |
| R-46 контракт | `ResearchRunResult` — полноценная pydantic-модель |
| R-04 тесты | ~40 тестовых модулей, unit/integration/smoke/network |
| R-03 частично | `_create_run_dir`, `_write_json_artifact` — артефакты прогона пишутся |
| R-71 частично | `EdaTask`, `ResearchHypothesis`, `VerificationStep` с success/failure criteria уже есть |

**Что переиспользуется в B5 без изменений:** `clients/deepseek_client.py`, `config.py`, `logging_utils.py`, `report/docx_generator.py`, механика `runs/{competition_id}_{timestamp}/`, тестовая инфраструктура.

### 0.1. Главная находка: `research_scout.py`

2541 строка, из которых подавляющая часть — постобработка вывода модели:

```
enforce_stratified_groupkfold_caveat   cleanup_generic_eda_tasks
correct_hypothesis_categories          is_generic_verification_text
expand_verification_steps              relink_eda_tasks
_temporal_validation_hypothesis        _rewrite_temporal_validation_task
enforce_scout_validation_policy        _unsafe_or_missing_temporal_policy
```

`_temporal_validation_hypothesis` возвращает захардкоженную гипотезу. `enforce_stratified_groupkfold_caveat` вшивает доменное знание как патч поверх ответа. `correct_hypothesis_categories` переклассифицирует то, что модель отнесла не туда.

Это спираль: промпт не сработал → допишем правило поверх вывода → правило конфликтует с другим случаем → допишем ещё. Без измерения нельзя узнать, помогает патч или вредит, поэтому патчи только накапливаются. Две трети файла — доменное знание автора, выдаваемое системой за найденное в источниках.

**Решение:** доменное знание переносится в промпт и в детерминированный слой фактов. Постобработка сводится к валидации схемы. Если правило вроде «StratifiedGroupKFold требует оговорки» действительно верно, оно живёт в промпте одной строкой, а не в 40 строках патча.

---

## 1. Что такое B5

Один вызов reasoning-модели над полным набором собранных фактов. Без retrieval, без эмбеддингов, без Postgres, без цепочки агентов.

```
Phase 0 (без LLM)                     Phase 1 (один вызов)      Phase 2
──────────────────────                ──────────────────        ────────
competition metadata      ┐
file manifest             │
historical leaderboards   ├──→ CompetitionFacts ──→ LLM ──→ CompetitionBrief ──→ render
notebooks + AST + scores  │         + тексты
discussions + writeups    ┘
```

Обоснование: потолок корпуса — 300–600 документов на соревнование. После Phase 0 факты занимают 3–6К токенов, тексты обсуждений и writeup'ов — ещё 40–80К. Это влезает в контекст целиком. Retrieval существует для случая, когда не влезает.

**B5 является одновременно продуктом и baseline.** Любая будущая сложность обязана его побить, иначе она не нужна.

---

## 2. Структура модулей

```
kaggle_researcher/
  facts/                    # НОВОЕ — детерминированный слой, ноль LLM
    __init__.py
    models.py               # pydantic-контракты фактов
    competition.py          # метаданные
    files.py                # манифест + sample_submission
    leaderboard.py          # public/private, shake-up
    notebooks.py            # список, скоры, pull
    notebook_ast.py         # AST-извлечение + fingerprint + дедуп
    discussions.py          # форумы, writeup'ы
    collect.py              # оркестрация Phase 0
  brief.py                  # НОВОЕ — один вызов, сборка контекста
  brief_schemas.py          # НОВОЕ — CompetitionBrief
  render.py                 # НОВОЕ — детерминированный markdown

  clients/deepseek_client.py    # без изменений
  config.py                     # правки, см. §7
  report/docx_generator.py      # без изменений
```

**Паркуется** (не удаляется, выводится из пути исполнения): `retriever.py`, `embedder.py`, `store/`, `summarizer.py`, `reasoning/*` (семь модулей), `research_scout.py`, `parsers/pdf_parser.py`, `agents/arxiv_agent.py`, `agents/paper_search_agent.py`.

**Переносится частично:** `agents/kaggle_agent.py` → `facts/notebooks.py` (логика `kernels_list`, `kernels_pull`, нормализация ядра сохраняется).

**Сохраняется как целевой выход:** `research_scout_schemas.EdaTask` и `ResearchHypothesis` — они уже правильные и становятся частью `CompetitionBrief`.

---

## 3. Phase 0 — контракты фактов

Ноль вызовов LLM. Каждое поле либо прочитано из источника, либо `None`. Отсутствующее поле никогда не заполняется догадкой.

### 3.1. `CompetitionFacts`

```python
class CompetitionFacts(BaseModel):
    schema_version: str = "1.0"
    competition_id: str
    collected_at: datetime

    metadata: CompetitionMetadata
    files: FileManifest
    notebooks: list[NotebookFacts]
    discussions: list[DiscussionFacts]
    similar_competitions: list[LeaderboardStability]
    cv_lb_pairs: list[CvLbPair]

    user_constraints: UserConstraints
    collection_errors: list[str]
```

### 3.2. `CompetitionMetadata`

```python
class CompetitionMetadata(BaseModel):
    competition_id: str
    title: str | None
    metric_name: str | None
    is_code_competition: bool | None
    submissions_per_day: int | None
    max_team_size: int | None
    deadline: datetime | None
    reward: str | None
    category: str | None
    num_teams: int | None
    unavailable_fields: list[str]
```

Источник: `KaggleApi.competitions_list(search=slug)` и поля объекта соревнования. Всё, что API не отдаёт, попадает в `unavailable_fields`, а не заполняется.

### 3.3. `FileManifest`

```python
class FileManifest(BaseModel):
    files: list[FileInfo]                # name, size_bytes, role_hint
    train_test_size_ratio: float | None
    sample_submission_columns: list[str]
    sample_submission_source: Literal["api", "header_download", "unavailable"]
    limitations: list[str]
```

Источник: `KaggleApi.competition_list_files(competition)`. Для колонок `sample_submission` — порядок попыток: метаданные API → загрузка файла при размере ниже `MAX_SAMPLE_SUB_BYTES` (по умолчанию 5 МБ) → `unavailable`. Если правила соревнования не приняты, API вернёт 403: это записывается в `limitations`, прогон не падает.

`role_hint` определяется правилами по имени: `train*` → train, `test*` → test, `sample_submission*` → submission, иначе `auxiliary`.

### 3.4. `NotebookFacts`

```python
class NotebookFacts(BaseModel):
    ref: str
    title: str
    author: str | None
    votes: int
    public_score: float | None
    last_run: datetime | None

    ast_fingerprint: str
    lineage_cluster_id: str

    splitters: list[CodeObservation]
    models: list[CodeObservation]
    metrics: list[CodeObservation]
    feature_ops: list[CodeObservation]
    declared_cv: list[str]

    parse_status: Literal["ok", "partial", "failed"]
```

```python
class CodeObservation(BaseModel):
    name: str                    # "StratifiedGroupKFold"
    kwargs: dict[str, str]       # {"n_splits": "5", "groups": "customer_id"}
    locator: str                 # "cell_18"
```

### 3.5. Извлечение AST — что именно ищем

`nbformat` читает `.ipynb`, `ast.parse` разбирает каждую code-ячейку. Ячейка с синтаксической ошибкой пропускается, `parse_status = "partial"`.

**Сплиттеры** (самое ценное, прямой ответ на главный вопрос соревнования):
`KFold`, `StratifiedKFold`, `GroupKFold`, `StratifiedGroupKFold`, `TimeSeriesSplit`, `train_test_split`, `PurgedGroupTimeSeriesSplit`.
Извлекаются keyword-аргументы: `n_splits`, `shuffle`, `random_state`, `groups`. Аргумент `groups=` — важнее самого сплиттера: он называет колонку, по которой утечка вероятна.

**Модели:** `LGBM*`, `XGB*`, `CatBoost*`, `RandomForest*`, `ExtraTrees*`, `Ridge`, `Lasso`, `LogisticRegression`, `AutoModel*`, `nn.Module`-наследники. Плюс `n_estimators`, `learning_rate`, `max_depth`, если заданы литералом.

**Метрики:** любой вызов из `sklearn.metrics` плюс имена, совпадающие с `metadata.metric_name`.

**Feature ops:** `.groupby(...).agg(...)`, `.merge(...)`, `.rolling(...)`, `.shift(...)`, `TargetEncoder`, `LabelEncoder`, `OneHotEncoder`, `.fillna(...)`.

**Заявленный CV:** регулярное выражение по markdown-ячейкам и строковым литералам — `(cv|oof|local)\D{0,12}(0\.\d{3,})`. Даёт `declared_cv`, из которого вместе с `public_score` собирается `CvLbPair`.

**Fingerprint:** `ast.dump` каждой code-ячейки с вырезанными числовыми и строковыми литералами, конкатенация, sha256. Одинаковый fingerprint → один `lineage_cluster_id`. Двадцать форков одного baseline считаются за один источник.

### 3.6. `DiscussionFacts`

```python
class DiscussionFacts(BaseModel):
    topic_id: str
    title: str
    author: str | None
    author_is_host: bool
    votes: int
    created_at: datetime | None
    source_type: Literal["discussion", "winner_writeup"]
    competition_id: str          # может отличаться от текущего
    text: str
```

Источник: `kaggle forums topics list` / `topics show`, категория `competition_write_ups` для writeup'ов победителей. Работать через `kagglesdk`, не через subprocess.

Собирается: все темы текущего соревнования (пагинация до потолка), плюс writeup'ы победителей 3–8 похожих завершённых соревнований, заданных в `--similar`.

### 3.7. `LeaderboardStability` и `CvLbPair`

```python
class LeaderboardStability(BaseModel):
    competition_id: str
    status: Literal["computed", "not_computable"]
    public_private_spearman: float | None
    top10_retention: float | None
    median_rank_change: float | None
    matched_teams: int
    source: Literal["meta_kaggle", "api_final_only", "unavailable"]
    not_computable_reason: str | None
```

```python
class CvLbPair(BaseModel):
    notebook_ref: str
    declared_cv: float
    public_score: float
    lineage_cluster_id: str
```

Парные public/private скоры доступны в Meta Kaggle; API после завершения отдаёт финальный лидерборд. Если Meta Kaggle не подключён — `status = not_computable`, и модель обязана это увидеть, а не оценить shake-up на глаз.

`CvLbPair` — самый недооценённый артефакт. Систематический разрыв между заявленным CV и публичным скором по 40 ноутбукам определяет стратегию сильнее любого текста.

### 3.8. `UserConstraints`

```python
class UserConstraints(BaseModel):
    vram_gb: float | None
    hours_per_week: float | None
    cloud_budget_usd: float | None
    objective: Literal["medal", "top_percent", "learn", "fast_baseline"] = "medal"
```

Задаётся флагами CLI или `.env`. Незаданное остаётся `None`; модель обязана пометить feasibility как неизвестную, а не предположить.

---

## 4. Phase 1 — один вызов

### 4.1. Сборка контекста

`brief.py` формирует одно сообщение с явными границами блоков. Порядок и приоритет при усечении:

| Приоритет | Блок | Оценка |
|---|---|---|
| 1 | `CompetitionFacts` без текстов обсуждений (JSON) | 3–6К |
| 2 | Writeup'ы победителей похожих соревнований | 10–25К |
| 3 | Обсуждения текущего: хост → голоса → свежесть | 20–50К |
| 4 | Сводка AST по кластерам происхождения | 2–5К |

Потолок `MAX_CONTEXT_TOKENS` в конфиге. При превышении отбрасывается хвост блока 3, и в `CompetitionBrief.limitations` записывается сколько документов отброшено. Молчаливое усечение запрещено.

Недоверенный контент (§8) оборачивается разметкой с id и явной инструкцией не исполнять содержащиеся в нём указания.

### 4.2. `CompetitionBrief`

```python
class Claim(BaseModel):
    text: str
    source_ids: list[str]          # пусто = отбрасывается рендером
    kind: Literal["fact", "claim", "inference"]

class CompetitionBrief(BaseModel):
    schema_version: str = "1.0"
    competition_id: str

    thesis: str
    thesis_support: list[str]      # id аспектов или находок

    validation: list[Claim]
    metric_notes: list[Claim]
    leakage_risks: list[Claim]
    what_works: list[Claim]
    time_wasters: list[Claim]

    hypotheses: list[ResearchHypothesis]   # из research_scout_schemas
    eda_tasks: list[EdaTask]               # из research_scout_schemas

    first_moves: list[str]
    unknowns: list[str]
    limitations: list[str]
```

Промпт требует: каждый `Claim` несёт `source_ids`; `kind="fact"` допустим только со ссылкой на `CompetitionFacts`; при отсутствии оснований возвращается запись в `unknowns`, а не правдоподобный текст.

### 4.3. Постобработка

Только валидация, без содержательных правок:

1. Схема проходит pydantic — иначе один повторный вызов с текстом ошибки, затем отказ.
2. `Claim` с пустыми `source_ids` перемещается в `unknowns` с пометкой.
3. `source_ids`, не встречающиеся в `CompetitionFacts`, — удаляются, факт фиксируется в `limitations`.
4. `hypotheses` и `eda_tasks` проверяются существующим валидатором из `research_scout_schemas`.

Ни `enforce_*`, ни `correct_*`, ни `_rewrite_*`. Если модель систематически ошибается — правится промпт.

---

## 5. Phase 2 — рендер

`render.py` детерминированно собирает markdown из `CompetitionBrief`. Ноль LLM, ноль регулярок по прозе.

```
1. Соревнование в цифрах        # из CompetitionFacts, таблица
2. Тезис
3. Валидация
4. Метрика
5. Риски утечки
6. Что работает у других
7. Чего избегать
8. Первые шаги
9. Что проверить на данных    # eda_tasks
10. Неизвестное
Приложение: источники
```

Раздел 1 печатается всегда, даже при провале Phase 1: метрика, тип соревнования, лимиты, распределение сплиттеров по кластерам, разрыв CV↔LB, shake-up похожих. Это работает без модели и уже полезно.

DOCX — через существующий `docx_generator`, вызываемый из markdown, а не из вывода LLM.

---

## 6. CLI

```bash
# только факты, без LLM и без ключей к модели
python -m kaggle_researcher.main facts <slug>

# полный бриф
python -m kaggle_researcher.main brief <slug> \
    --similar comp-a,comp-b,comp-c \
    --vram 12 --hours 10 --objective medal \
    --max-notebooks 60 --max-discussions 200

# запись факта участия — вход для будущей памяти
python -m kaggle_researcher.main journal <slug> \
    --used-validation "GroupKFold(customer_id)" \
    --final-rank 47 --num-teams 1200 --brief-was-useful yes
```

`facts` не требует `DEEPSEEK_API_KEY`. Это отдельный полезный инструмент и одновременно способ отладить Phase 0 без затрат.

---

## 7. Правки существующего кода

**`config.py`.** `load_config` игнорирует env для `top_k`, `max_notebooks`, `max_papers`, `max_repos`, `pdf_cache_dir` — значения захардкожены в конструкторе `Settings`, притом что у поля есть default. Читать из env. Добавить: `MAX_DISCUSSIONS`, `MAX_CONTEXT_TOKENS`, `MAX_SAMPLE_SUB_BYTES`, `META_KAGGLE_DIR`, `RUN_BUDGET_TOKENS`.

**Авторизация Kaggle.** `KAGGLE_USERNAME`/`KAGGLE_KEY` — legacy. Перейти на `KAGGLE_API_TOKEN`, старый путь оставить как fallback.

**Бюджет.** `RUN_BUDGET_TOKENS` с деградацией: при приближении к потолку усекается блок 3 контекста, факт записывается в `limitations`. Не исключение.

**Checkpoints.** `CompetitionFacts` сериализуется в `runs/{id}_{ts}/facts.json` до вызова модели. Флаг `--facts-from <path>` пропускает Phase 0 целиком. Итерации над промптом становятся бесплатными по сбору.

**README.** Привести в соответствие с состоянием кода.

---

## 8. Безопасность

Тексты обсуждений и README приходят из интернета и идут в модель. Каждый блок недоверенного контента обрамляется маркером с id источника, системная инструкция объявляет содержимое данными. Валидация вывода (§4.3) отбрасывает `source_ids`, отсутствующие в фактах, — это ловит попытку внедрить фальшивый источник.

`docker-compose` вместе с Postgres выводится из основного пути; если остаётся для будущего retrieval — пароль в env, порт на `127.0.0.1`.

---

## 9. Порядок реализации

| Шаг | Содержание | Проверка |
|---|---|---|
| 1 | `facts/competition.py`, `facts/files.py`, CLI `facts` | Печатает метрику, code/не code, лимиты, отношение train/test на трёх реальных соревнованиях |
| 2 | `facts/notebooks.py` + `notebook_ast.py` | На 40 ноутбуках: распределение сплиттеров, число кластеров происхождения меньше числа ноутбуков |
| 3 | `facts/discussions.py` | Собираются темы текущего и writeup'ы категории `competition_write_ups` |
| 4 | `facts/leaderboard.py` | Три завершённых соревнования дают `status=computed` либо честное `not_computable` |
| 5 | `brief.py` + `brief_schemas.py` + `render.py` | Бриф на реальном соревновании; каждый claim имеет источник |
| 6 | `journal` | Запись результата участия |

Шаги 1–4 не требуют ключа к модели и полезны сами по себе.

---

## 10. Проверка полезности

Формальный eval-harness не строится: он больше самой системы и цель — выигрывать соревнования.

Вместо него — участие. На каждом соревновании фиксируется:

- изменил ли бриф план по сравнению с намерением до его прочтения;
- какая схема валидации была выбрана и совпала ли с рекомендованной;
- финальное место и разница public/private.

Три-четыре записи `journal` дают то, чего не даст никакая синтетическая разметка: реальный `outcome`, привязанный к рекомендации. Это же — единственный корректный источник для `domain_memory`, если она вернётся.

Отрицательный результат тоже полезен: если после трёх соревнований бриф ни разу не поменял план, значит слой не нужен, а `facts` — нужен.
