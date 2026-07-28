# KaggleResearcher v6 — Modular Monolith Specification

**Document:** `docs/specs/KAGGLE_RESEARCHER_V6_SPEC.md`  
**Status:** implementation-ready canonical specification  
**Version:** 0.2.6  
**Date:** 2026-07-26  
**Audience:** maintainers, Codex implementation agents, test authors  
**Architecturally supersedes:** v4 pipeline integration and v5 EDA integration model  
**Revision basis:** 0.2 plus the architecture/migration review grounded in the `kaggle_eda` branch, four successive contract-kernel implementability reviews, and the v6 baseline audit. They are reconciled in Appendices B through F.

This file is the sole normative v6 specification. Documents under `docs/archive/`,
legacy EDA task files numbered 28–64, and older versioned copies are historical
context only and cannot override this document.

---

## 1. Executive decision

KaggleResearcher v6 is a controlled rebuild of the integration architecture, not a full rewrite of every algorithm.

Existing retrieval, parsing, Research Scout, profiling, metric, validation, leakage, drift, baseline, feature-probe, and rendering code may be reused after it is placed behind v6 contracts and passes v6 acceptance tests.

The following parts are rebuilt from first principles:

1. public contracts between modules and stages;
2. evidence identity and the canonical `EvidenceCatalog`;
3. Research → Dataset Intelligence and Dataset Intelligence → Synthesis adapters;
4. module composition and the in-process runner;
5. checkpoint, resume, and retention rules;
6. Final Synthesis input, output, and publication gates;
7. failure handling and partial-run semantics.

v6 is deliberately a **local modular monolith**:

- one Python process by default;
- independently testable modules;
- typed Python objects passed in memory during a run;
- serialized checkpoints for expensive or externally consumed results;
- one canonical evidence catalog;
- a small in-process DAG runner;
- no distributed scheduler, custom asset platform, or content-addressable store in the MVP.

The central invariant is:

> Replacing a module with another implementation of the same public contract must not require changes to Final Synthesis, presentation, the central runner, or unrelated modules.

---

## 2. Motivation

v5 introduced useful separation between source research, dataset execution, and final synthesis. Its integration model nevertheless allowed the same fact space to be represented independently in:

- Pydantic models;
- module JSON files;
- the aggregated `EdaEvidencePack`;
- evidence-path generators;
- synthesis registries;
- prompt-level allowed-reference lists;
- post-LLM reference validators.

This created a distributed monolith inside one repository. A field rename or list reordering could propagate through the pack builder, prompt builder, context builder, validators, reports, and tests. Contract violations were often detected only after expensive retrieval and EDA.

v6 removes that failure class without imposing distributed-system ceremony on a local tool. It keeps strict contracts at meaningful module boundaries, but does not require every in-process handoff to be serialized and reread from disk.

---

## 3. Goals

v6 must:

- support independent implementation and testing of each module;
- make cross-module inputs and outputs explicit and typed;
- validate incompatible stage inputs before dataset IO or model calls;
- preserve provenance for every dataset fact used by Final Synthesis;
- use one implementation to register, select, resolve, and validate evidence;
- support safe partial execution and resume of expensive work;
- distinguish blocking failures from valid degraded results;
- support generic tabular Kaggle competitions first;
- keep competition-specific behavior outside the generic core;
- preserve working v4/v5 algorithms where they meet the new contract;
- support Windows and Linux local execution;
- keep core tests offline and independent of PostgreSQL, pgvector, vLLM, Kaggle, and external LLMs;
- bound dataset, compute, storage, and LLM costs;
- allow Synthesis to request a small, validated set of additional deterministic investigations when evidence is insufficient.

---

## 4. Non-goals

The MVP does not aim to:

- provide distributed or remote execution;
- implement a general workflow platform;
- provide a scheduler daemon or orchestration UI;
- implement arbitrary dynamic workflows;
- build a content-addressable or recursive Merkle artifact store;
- implement AutoML or unrestricted hyperparameter search;
- execute downloaded notebooks, scripts, or repository code;
- guarantee byte-identical floating-point output across operating systems;
- make every legacy schema permanently compatible;
- use an LLM to repair deterministic contract violations;
- make Markdown, DOCX, or a legacy pack a source of truth;
- optimize public leaderboard score automatically.

Dagster, Prefect, or another external orchestrator may be evaluated later only if the project acquires scheduled runs, remote executors, multiple concurrent users, dynamic graphs, or a clear operational need for an orchestration UI.

---

## 5. Architectural principles

### 5.1 Public contracts, private implementations

Every module has a public input contract, output contract, typed configuration, declared capabilities, and failure policy. Internal helper functions and intermediate objects are private and may use dataclasses, protocols, typed dictionaries, arrays, or data frames.

Public cross-module contracts use Pydantic and reject unknown fields by default. Private implementation details do not become public contracts merely because two functions share them.

### 5.2 In-memory execution, selective persistence

During a normal run, a downstream module receives the validated Python result object produced by its dependency. Publication may checkpoint the same public result, but the next module does not reread it merely to continue in the same process.

Checkpoints exist for:

- resume after interruption;
- expensive deterministic results;
- stage boundaries;
- LLM boundaries;
- final machine-readable outputs;
- tables, models, plots, and other bulk artifacts.

### 5.3 No expanding global state

There is no mutable `PipelineContext`, `EdaEvidencePack`, or nested dictionary into which unrelated modules append fields. A runner-owned execution context may carry infrastructure services such as logging, cancellation, artifact publication, and resource limits, but never analytical results.

### 5.4 One canonical evidence catalog

`EvidenceCatalog` is the only public service that registers, resolves, filters, and validates evidence references. Prompt construction and Final Strategy validation receive the same catalog or the same immutable catalog subset.

There is no standalone `generate_allowed_evidence_refs(pack)` or consumer-specific evidence registry.

### 5.5 Evidence identity is independent of storage layout

Public claims reference an opaque stable `evidence_id`. Evidence semantics are represented separately by a structured key and dimensions. Physical JSON paths, list indexes, filenames, table rows, and blob locations may appear only in catalog locators, never as public references.

### 5.6 Fail early at trust boundaries

Research outputs, run requests, module configs, resumed checkpoints, catalog fragments, and LLM responses are validated before expensive dependent work. Invalid required inputs produce typed blocking errors, not warnings.

### 5.7 Orchestrator coordinates; domain modules decide

The runner may resolve dependencies, execute nodes, checkpoint results, and propagate failures. It must not contain competition-specific branches, metric interpretation, validation selection, leakage rules, feature logic, or claims.

### 5.8 Deterministic core, bounded LLM edges

Dataset facts and gates are produced by deterministic code. Research Scout and Strategy Synthesis may use LLMs, but their outputs are typed, bounded, validated, and excluded from byte-deterministic promises.

### 5.9 Explicit degradation

A degraded result is still contract-valid. It states what was computed, what was not computed, and why. Missing required fields, broken refs, duplicate IDs, and invalid payloads are failures, never degraded success.

### 5.10 Derived views are not inputs

Markdown, DOCX, legacy v5 packs, human summaries, and dashboards are derived projections. Core modules must not consume them.

---

## 6. System boundaries

v6 contains five stages:

```text
Source Intelligence
    collect -> parse -> normalize -> index -> retrieve

Research Scout
    retrieved source evidence -> hypotheses + investigation plan

Dataset Intelligence
    dataset + validated plan -> dataset evidence + diagnostics

Strategy Synthesis
    source evidence + dataset evidence -> validated strategy

Presentation
    validated strategy -> Markdown/DOCX/legacy exports
```

Each stage can run against stored fixtures from the previous stage. Full web retrieval, Kaggle download, EDA, and LLM synthesis are not required to be rerun together for routine contract tests.

---

## 7. Bounded responsibilities

### 7.1 Source Intelligence

Owns source discovery, safe fetching, article/PDF/repository/notebook static parsing, normalization, chunking, indexing, retrieval, and source provenance.

It must not inspect the competition dataset, execute downloaded code, or promote source authors' observations into facts about the local dataset.

### 7.2 Research Scout

Owns interpretation of the competition brief and retrieved source evidence. It produces source-backed considerations, testable hypotheses, and requested dataset capabilities.

It must not invent dataset observations, name private EDA functions, or address physical artifact paths.

### 7.3 Dataset Intelligence

Owns dataset resolution, safe IO, inventory, schema and role inference, metric interpretation, validation selection, leakage checks, relationships, drift, baselines, feature probes, submission analysis, and hypothesis evaluation.

It publishes observations and diagnostics, not final narrative strategy.

### 7.4 Strategy Synthesis

Owns evidence-backed strategy, claim classification, uncertainty, experiment proposals, and bounded requests for missing investigations.

It must not modify evidence, execute arbitrary analysis, or cite evidence outside the supplied catalog.

### 7.5 Presentation

Owns rendering and exports. It cannot add or reinterpret claims. If a report needs new content, the validated strategy contract must contain that content first.

---

## 8. Package boundaries

Recommended structure:

```text
kaggle_researcher/
├── contracts/
│   ├── common.py
│   ├── artifacts.py
│   ├── evidence.py
│   ├── research.py
│   ├── dataset.py
│   └── synthesis.py
├── source_intelligence/
├── research_scout/
├── dataset_intelligence/
│   ├── gateway/
│   ├── modules/
│   ├── capabilities.py
│   └── publication.py
├── synthesis/
├── presentation/
├── runtime/
│   ├── module.py
│   ├── plan.py
│   ├── runner.py
│   ├── checkpoint.py
│   └── retention.py
├── adapters/
│   ├── research_to_dataset.py
│   ├── dataset_to_synthesis.py
│   └── legacy_v5.py
└── cli/
```

Rules:

- `contracts/` contains public types and no analytical implementations;
- a module may import public contracts but not another module's private models;
- adapters contain deterministic mapping and validation, not business analysis;
- `runtime/` knows module metadata and infrastructure interfaces, not domain rules;
- legacy code is reached through adapters, never imported into new public contracts.

---

## 9. Common contracts

Every **persisted top-level contract** — anything written to disk, checkpointed, or exchanged across a package boundary as a whole document — carries an integer `schema_version`. All public models, top-level or nested, use `extra="forbid"`.

Nested value types such as `RunIdentity`, `Diagnostic`, `Limitation`, `Claim`, `StrategyConstraint`, and the port specs deliberately carry no version of their own: they are versioned by the document that contains them. Giving every nested model an independent version would create version pairs that can disagree with no rule to resolve them, and would make a field addition require bumping several numbers at once.

```python
class RunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    competition_id: str
    created_at: datetime


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: DiagnosticSeverity
    message: str
    details: dict[str, JsonValue] = {}


class Limitation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    scope: Literal["module", "evidence", "run"]
    severity: FindingSeverity
    message: str
    affected_capabilities: tuple[str, ...] = ()


class ModuleResult[T](BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    module_id: str
    module_contract_version: int
    module_implementation_version: str
    quality: Literal["complete", "degraded"]
    value: T
    diagnostics: tuple[Diagnostic, ...] = ()
    limitations: tuple[Limitation, ...] = ()
```

Two severity scales exist and are never merged:

```python
DiagnosticSeverity = Literal["info", "warning", "error", "critical"]
"""Execution state of a module run. Not a statement about the dataset."""

FindingSeverity = Literal["info", "notable", "serious", "critical"]
"""Importance of a dataset observation. Not a statement about execution."""
```

A dataset with critical leakage is a *successful* run reporting `FindingSeverity="critical"`. A module that crashed while looking for leakage is `DiagnosticSeverity="critical"` with no finding at all. One combined scale leaves the runner and the claim gate unable to distinguish an analytical alarm from an execution failure, so `Diagnostic` carries only execution severity and analytical importance lives on the finding/evidence side.

`skipped` and `failed` are execution outcomes and do not masquerade as `ModuleResult`. Degradation is a checkable invariant, not a convention:

```text
complete → no error or critical diagnostic, and no module-scope limitation
degraded → at least one module-scope limitation or degradation diagnostic,
           and the declared required output is still contract-valid
failed   → no ModuleResult is produced at all
```

A module that cannot satisfy the `degraded` row fails instead of returning a result. Contract tests assert this invariant directly; it is never left to each `value` type to express limitations in its own way.

### 9.1 Schema versioning

- versions are positive integers: `1`, `2`, `3`;
- any public shape change requires a new integer version;
- consumers declare accepted versions explicitly;
- compatibility is implemented by named deterministic adapters such as `ValidationEvidenceV1ToV2`;
- no forward-compatibility promise is inferred from optional fields;
- adapters never silently invent missing semantics.

### 9.2 Compatibility and release policy

`extra="forbid"` is retained deliberately: it turns accidental contract drift into a loud, immediate failure instead of a silent late one. The cost of that guarantee is stated here explicitly so it is a policy, not a surprise.

- v6 provides **no forward compatibility** for public contracts.
- Any change to a public shape, including adding an optional field, creates a new integer schema version.
- A producer and its direct consumers are updated **atomically in one monorepo change**.
- An older consumer must **reject** data of a newer version; ignoring unknown fields is forbidden.
- Reading an artifact of a different version is permitted **only** through a named deterministic adapter.
- An adapter is required only where a cross-version read is actually supported. Otherwise a stale checkpoint is invalidated with an explicit diagnostic.
- Adapters are **directional**: the existence of `V1ToV2` does not imply `V2ToV1`.
- Automatic removal of unknown fields and automatic version downgrade are forbidden.
- Supporting multiple concurrent versions is **not** an MVP goal.

Behavior matrix:

| Producer | Consumer | Result |
|---|---|---|
| v1 | v1 | Allowed |
| v2 | v2 | Allowed |
| v2 | v1 | Hard rejection |
| v1 | v2 | Only via `V1ToV2` |
| checkpoint v1 | same module, code now emits v2 | Adapter, or explicit invalidation per §18.5 |

The last row is not two components disagreeing at runtime. It arises only when a module's code is upgraded **between runs**, so a checkpoint written by the old code is read back by the new code on resume. The resume rule in §18.5 already governs it: the checkpoint is reused only if its contract version is accepted or a named adapter exists; otherwise the node is re-executed and the stale checkpoint is invalidated with a diagnostic. Within a single run there is never a cross-version checkpoint read, because the producing and consuming code are identical.

---

## 10. Artifact model

Evidence and bulk data are different concerns. v6 does not use a universal `payload: dict` envelope.

### 10.1 JSON artifacts

Small public contracts and checkpoints use JSON.

```python
class JsonArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["json"] = "json"
    artifact_id: str
    artifact_type: str
    schema_version: int
    relative_path: str
    file_hash: str
```

Examples: inventory result, metric evidence, validation policy, leakage summary, hypothesis results, Final Strategy.

### 10.2 Table artifacts

Large rectangular data uses Parquet or Arrow.

```python
class TableArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["table"] = "table"
    artifact_id: str
    artifact_type: str
    format: Literal["parquet", "arrow"]
    relative_path: str
    schema_fingerprint: str
    file_hash: str
    row_count: int
    column_count: int
```

Examples: sampled profiles, OOF predictions, transformed feature tables, relation edges when too large for JSON.

Table schema and integrity are validated at publication. A same-run consumer may receive an in-memory table plus its ref. A resumed consumer loads the table by ref and validates file integrity and expected schema fingerprint; it does not rerun full semantic validation on every read.

The table backend is not a new dependency: the v5 branch already reads datasets through polars with `scan_parquet` and ships `pyarrow`. §10.2 formalizes typed refs over an existing capability, and migration estimates should price it accordingly.

### 10.3 Blob artifacts

Models, plots, PDFs, binary reports, and other non-tabular outputs use blobs.

```python
class BlobArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["blob"] = "blob"
    artifact_id: str
    artifact_type: str
    media_type: str
    relative_path: str
    file_hash: str
    size_bytes: int
```

### 10.4 Identity and integrity

`artifact_id` is based on run, node, and **attempt** identity. Attempt is part of the identity because two immutable attempts of the same `(run_id, node_id)` are different artifacts; omitting it would give them one identity and make the immutability rule unenforceable. It is not a content address. Rewriting a published artifact is forbidden; rerunning the node publishes a new attempt.

`file_hash` covers the bytes actually written and is used for:

- integrity checks;
- resume validation;
- detection of manual corruption;
- dataset and input change detection.

It is not used to build a recursive Merkle identity, and output float serialization is not promised to be byte-identical across platforms.

#### 10.4.1 Canonical bytes

Every field named `*_digest`, `*_fingerprint`, or `*_hash` has one normative
preimage and one normative algorithm. Implementations may cache canonical bytes,
but may not substitute a library's default serializer.

`canonical_json_bytes(value)` is defined as follows:

1. validate `value` against its declared public contract;
2. dump the complete JSON representation with aliases enabled and with explicit
   `null` values retained;
3. normalize every string value and object key to Unicode NFC; reject an object
   if two keys become equal after normalization;
4. emit contract fields in their declared preimage order; emit arbitrary mapping
   keys in ascending normalized UTF-8 byte order; preserve tuple/list order;
5. unordered collections are forbidden in a digest preimage until the owning
   contract has converted them to a tuple sorted by normalized UTF-8 bytes;
6. emit booleans and null as the JSON literals `true`, `false`, and `null`;
7. emit integers in base ten with no leading zero; reject non-finite floats and
   serialize finite floats using the RFC 8785 / ECMAScript shortest
   round-trippable JSON number form, with negative zero serialized as `0`;
8. serialize datetimes in UTC as RFC 3339 with exactly six fractional digits
   and the `Z` suffix; naive datetimes are invalid at a persistence boundary;
9. use `separators=(",", ":")`, `ensure_ascii=False`, UTF-8, and no BOM or
   trailing newline.

The special `EvidenceKey` preimage in §11.3 declares its own member order
(`name`, then `dimensions`) and uses the same scalar and UTF-8 rules.

The core digest registry is normative:

| Field | Preimage | Algorithm and encoding |
|---|---|---|
| `file_hash` | exact file bytes | BLAKE2b-256, 64 lowercase hex characters |
| `schema_fingerprint` | canonical logical table schema: ordered fields of `{name, logical_type, nullable, metadata}`, with metadata keys byte-sorted | BLAKE2b-256 over canonical JSON bytes |
| `evidence_fragment_digest` | complete validated `EvidenceFragment` | BLAKE2b-256 over canonical JSON bytes |
| `plan_digest` | complete validated `PipelinePlan` with `plan_digest` omitted | BLAKE2b-256 over canonical JSON bytes |
| `manifest_digest` | complete validated `RunManifest` with `manifest_digest` omitted | BLAKE2b-256 over canonical JSON bytes |
| `snapshot_digest` | `{plan_digest, generation, node_states}` in that order | BLAKE2b-256 over canonical JSON bytes |
| `input_snapshot_digest` | the `InputSnapshotPreimage` defined in §12.1 | BLAKE2b-256 over canonical JSON bytes |
| `config_fingerprint` | `{module_id, cache_fingerprint_version, config}` in that order | BLAKE2b-256 over canonical JSON bytes |
| `cache_fingerprint` | the complete cache preimage in §12.1 | BLAKE2b-256 over canonical JSON bytes |
| `artifact_id` | `{run_id, node_id, attempt_id, kind, artifact_type, ordinal}` in that order | `"art_" + first 24 hex characters of BLAKE2b-128(canonical bytes)` |

`ordinal` is the zero-based position of the artifact in the node's validated
publication list. Reordering that list is therefore an artifact-identity change
and must be deliberate.

`dataset_fingerprint` is a typed discriminated contract rather than an
underspecified string:

```python
class LocalDatasetFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["local_files"] = "local_files"
    files: tuple[DatasetFileFingerprint, ...]  # relative path byte-order
    digest: str


class DatasetFileFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    size_bytes: int
    file_hash: str
```

Its `digest` is BLAKE2b-256 over canonical JSON bytes of `files`. Paths are
relative to the validated dataset root, use `/` separators, contain no `.` or
`..` segments, and are sorted by normalized UTF-8 byte order. Modification time
is not an identity input. A future remote-dataset fingerprint requires a new
discriminator value and an explicit contract version.

No digest may be defined as "hash the object" or "hash the config" without the
named preimage above. Adding a digest field requires adding one row to this
registry in the same change.

### 10.5 Publication boundary

A module publishes one node bundle containing:

```text
node_manifest.json
result.json
evidence_fragment.json
tables/*
blobs/*
```

A directory rename and a catalog mutation are two different resources and do not form one transaction. Claiming both are atomic together is not implementable, and the two failure windows are real:

```text
bundle renamed  → catalog registration failed   (orphan bundle)
catalog updated → bundle rename failed          (dangling reference)
```

No database is required to close this. The protocol is ordered so that the durable filesystem state is always the truth:

1. write the complete bundle into a staging directory and validate it there;
2. write `node_manifest.json` inside the staging bundle with `status="prepared"`, including the evidence fragment digest;
3. atomically rename the staging directory into its final location;
4. create an immutable `PUBLISHED` marker inside the bundle by writing a temporary file and renaming it into place;
5. the catalog derives canonical state **only** from bundles that carry a valid manifest, matching hashes, and a `PUBLISHED` marker;
6. recovery on startup discards or completes any bundle lacking the marker.

The `prepared` manifest is never rewritten. An in-place rewrite is not atomic — a crash mid-write can leave a truncated or torn manifest, which would contradict the recovery guarantee this protocol exists to provide. Publication is therefore expressed by *adding* an immutable artifact, never by mutating one. Both mutating steps in the protocol are renames, which are atomic on a single filesystem.

`evidence_fragment.json` files inside published bundles are the durable source of truth. `evidence_catalog.json` is a **materialized view** that can be deleted and rebuilt by scanning published bundles; it is never the only copy of a registration. A crash between steps 3 and 4 leaves a bundle that is present but invisible — safe, and resolvable by recovery — and a crash before step 3 leaves staging garbage that is never consumable.

Evidence registration is all-or-nothing per bundle: a fragment is either wholly visible to the catalog or wholly absent. An incomplete staging directory is never consumable and may be removed by recovery or garbage collection.

The two files the protocol turns on are contracts, not conventions. Recovery, resume, and integrity checking all read them, so leaving them undefined would let two implementations build incompatible recovery mechanisms while both passing their own tests:

```python
class NodeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run_id: str
    plan_id: str
    plan_digest: str
    generation: int
    node_id: str
    attempt_id: str
    module_id: str
    module_contract_version: int
    module_implementation_version: str
    cache_fingerprint: str
    input_snapshot_digest: str
    output_contract: str
    output_contract_version: int
    artifact_refs: tuple[JsonArtifactRef | TableArtifactRef | BlobArtifactRef, ...]
    evidence_fragment_digest: str
    status: Literal["prepared"]


class EvidenceFragment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run_id: str
    competition_id: str
    plan_id: str
    plan_digest: str
    generation: int
    node_id: str
    attempt_id: str
    records: tuple[EvidenceRecord, ...]
```

`status` is `prepared` and stays `prepared`: publication is expressed by adding the `PUBLISHED` marker, never by rewriting this file. `plan_digest` prevents a bundle produced for a different plan body from being admitted merely because a reused `plan_id` matches. `input_snapshot_digest` records the exact typed input snapshot the attempt was computed against, which is what lets resume decide whether a checkpoint is still applicable and lets recovery detect an attempt computed against inputs that no longer exist. `evidence_fragment_digest` binds the manifest to its fragment so a torn or truncated fragment is detected rather than partially registered.

### 10.6 Attempts and canonical selection

Bundles are immutable and a rerun publishes a new attempt, so a node may legitimately own several published bundles inside one run. Without a selection rule this collides with evidence identity: `evidence_id` is derived from the contract namespace, key, and dimensions (§11.3) and deliberately does **not** include the attempt, so a second successful attempt would publish a fragment carrying the same IDs, and rebuilding the catalog from all published bundles would fail duplicate detection. Immutability, catalog rebuild, and rerun are only mutually consistent once one attempt per node is declared canonical.

The generation state owns that declaration. There is no second
`active_attempts` mapping:

```python
class NodeGenerationState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "pending",
        "running",
        "success",
        "failed",
        "skipped",
        "inactive",
    ]
    attempt_id: str | None = None
    attempt_generation: int | None = None
    quality: Literal["complete", "degraded"] | None = None
    reason_code: str | None = None
    blocked_by_node_ids: tuple[str, ...] = ()
```

Every plan node has exactly one state in every candidate or committed
generation. The invariants are:

- `success` requires `attempt_id`, `attempt_generation`, and `quality`; the
  selected attempt must identify a valid published bundle for the same node and
  plan digest, produced in `attempt_generation <=` the selecting generation;
- a non-closure node may retain an attempt from an earlier generation; a newly
  executed or invalidated node must select an attempt produced in the current
  candidate generation;
- `failed` means execution was attempted but no valid `ModuleResult` was
  published; it has no `quality`, `attempt_id`, or `attempt_generation`;
- `skipped` means the node was activated but intentionally not executed because
  of dependency, policy, or budget; it requires `reason_code`;
- `inactive` is valid only for `NodeSpec.activation="on_request"` when no granted
  request activated the node in this generation;
- `pending` and `running` occur only in a candidate and are forbidden in a
  committed generation;
- `blocked_by_node_ids` is non-empty only for a dependency-caused `skipped`
  state and is sorted by normalized UTF-8 node ID;
- only a `success` state contributes a bundle or evidence to a snapshot.

`failed`, `skipped`, and `inactive` are not aliases for missing dictionary
entries. Their explicit presence is what lets resume, reporting, and
investigation distinguish a failed recompute from a node that was never
requested.

The run manifest owns the committed declaration:

```python
class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run: RunIdentity
    plan_id: str
    plan_digest: str
    generation: int
    node_states: Mapping[str, NodeGenerationState]
    manifest_digest: str
```

`manifest_digest` is self-referential unless its input is stated, and "hash the manifest" is not an algorithm when the digest lives inside the manifest. The normative rule:

```text
manifest_digest = blake2b(canonical_json_bytes(
    manifest without manifest_digest
), digest_size=32).hexdigest()
```

using §10.4.1. The digest field is excluded from its own input; every other field
is included.

Rules:

- at most one attempt is active per node because each node has one state and only
  `success` may name an attempt;
- the catalog is built **only** from bundles selected by `success` states in the
  committed generation;
- superseded attempts remain physically stored and inspectable, but contribute no evidence and satisfy no reference;
- retention treats superseded attempts as `ephemeral` unless the run is pinned (§19.1).

Promotion is per generation, not per node. Promoting one node alone is unsafe: because `evidence_id` is stable across attempts, recomputing an upstream node and immediately activating it would leave downstream results that were derived from *different values behind the same IDs*. A crash before the downstream recompute would leave a run that passes every structural check while mixing new upstream evidence with stale downstream conclusions — valid in form, wrong in content.

The commit protocol therefore is:

1. compute the **dependency closure** of the invalidated nodes — the nodes themselves and everything transitively downstream;
2. open a **candidate generation** numbered `generation + 1`;
3. initialize every invalidated closure node as `pending`, carrying forward no
   old attempt for those nodes; non-closure nodes retain their prior terminal
   state and successful attempt;
4. publish new attempts for closure nodes; a valid published result changes its
   state to `success`, while a terminal execution failure changes it to
   `failed` or `skipped`;
5. execute each closure node against a typed input snapshot derived from the
   candidate, never from the committed generation directly;
6. commit by writing the new manifest and renaming it into place — one
   replacement, never a per-node edit;
7. on a blocking failure, abandon the candidate; the previous generation
   remains committed and fully consistent.

An interrupted recompute leaves orphaned unpublished or unselected attempts,
which are garbage, not partial truth. The previous successful attempt is not
silently reused inside the failed candidate: it survives only as part of the
previous committed generation. A resume of that failed candidate must retry the
invalidated node or explicitly abandon the candidate; it may not treat the
previous generation's attempt as a cache hit. User-facing status must report
that the latest recompute failed even when an older committed generation remains
available as last-known-good output.

### 10.7 Committed and candidate generations

A single "committed" view is not sufficient, because it makes the protocol circular. Synthesis is itself a closure node: it must read the newly recomputed upstream evidence, that evidence becomes canonical only at commit, and the commit cannot happen until Synthesis has published. On a first run the contradiction is starker still — no committed generation containing any evidence exists yet, so a strictly committed-only view would leave Synthesis nothing to read.

Two snapshot kinds resolve this, and they differ only in visibility:

```python
class GenerationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run: RunIdentity
    plan_id: str
    plan_digest: str
    generation: int
    status: Literal["candidate", "committed"]
    node_states: Mapping[str, NodeGenerationState]
    snapshot_digest: str
```

- a **candidate snapshot** starts from the committed snapshot, removes the
  closure's old attempt selections, and tracks all plan nodes through
  `NodeGenerationState`; it grows as closure nodes publish;
- closure nodes, Synthesis included, read the candidate snapshot; nothing outside the run observes it;
- the **committed snapshot** is what presentation, exports, resume, and any external reader see;
- commit is the single atomic act that turns the candidate into the committed snapshot.

The full candidate `snapshot_digest` changes as node states change, so it is not
the digest supplied to a running node. Each node receives a separate immutable
`InputSnapshotPreimage` (§12.1), and its `NodeManifest.input_snapshot_digest`
binds the result to those exact inputs. This removes the Synthesis self-reference:
publishing the Synthesis attempt changes the full candidate snapshot but does
not change the input digest against which the strategy was generated.

Gates split along the same line. The pre-commit gate validates a strategy
against its `target_generation` and the Synthesis node's
`input_snapshot_digest`; it does not compare the strategy with the final
candidate `snapshot_digest`. Presentation validates the committed generation,
the selected Synthesis attempt, and the input digest recorded in that attempt's
manifest. A candidate that never commits is discarded whole; there is no state
in which half of it is observable.

`snapshot_digest` uses the exact preimage and algorithm in §10.4.1. Candidate
snapshots may be checkpointed for recovery; committed snapshots contain no
`pending` or `running` state.

`exports/` belongs to a generation. `evidence_catalog.json` records the `generation` and `manifest_digest` it was built from and is rebuilt when either differs from the committed manifest; a materialized view that disagrees with the manifest is stale by definition, never authoritative. Final exports — `final_strategy.json`, rendered reports, v5 exports — carry the generation they were produced from and are not valid for a later one. Presentation refuses to serve an export whose generation is behind the committed manifest rather than silently showing an older strategy.

This makes targeted invalidation well-defined: invalidating a node means recomputing its closure and committing a new generation, not deleting history and not editing one entry.

---

## 11. Evidence model

### 11.1 Structured semantic key

```python
class EvidenceKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dimensions: dict[str, str] = {}
```

Examples:

```json
{
  "name": "eda.validation.primary_policy",
  "dimensions": {}
}
```

```json
{
  "name": "eda.validation.period.target_rate",
  "dimensions": {"period": "2025_q1", "split": "train"}
}
```

Dimensions are data selectors, not embedded pseudo-paths in the semantic name.

### 11.2 Evidence record

Locators are typed, internal to catalog resolution, and never exposed as public evidence refs:

```python
class JsonEvidenceLocator(BaseModel):
    kind: Literal["json_pointer"] = "json_pointer"
    pointer: str


class TableEvidenceLocator(BaseModel):
    kind: Literal["table_selector"] = "table_selector"
    columns: list[str]
    dimensions: dict[str, str] = {}


class BlobEvidenceLocator(BaseModel):
    kind: Literal["whole_blob"] = "whole_blob"


EvidenceLocator = Annotated[
    JsonEvidenceLocator | TableEvidenceLocator | BlobEvidenceLocator,
    Field(discriminator="kind"),
]
```

```python
class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    evidence_id: str
    key: EvidenceKey
    domain: Literal["source", "dataset", "system"]
    producer_node_id: str
    producer_module: str
    producer_implementation_version: str
    run_id: str
    competition_id: str
    artifact_ref: Annotated[
        JsonArtifactRef | TableArtifactRef | BlobArtifactRef,
        Field(discriminator="kind"),
    ]
    locator: EvidenceLocator
    value_kind: Literal["scalar", "distribution", "table", "text", "diagnostic"]
    finding_severity: FindingSeverity = "info"
    summary: str
    quality: Literal["measured", "derived", "estimated", "degraded"]
    derived_from_refs: tuple[str, ...] = ()
    derivation_method: str | None = None
    limitations: tuple[Limitation, ...] = ()
```

`domain` makes the claim rule in §23 machine-checkable. Without it, "source evidence cannot prove an unmeasured dataset fact" is a sentence no validator can enforce; with it, the gate is a comparison. `domain="source"` is external literature or competition discussion, `domain="dataset"` is measured from the actual data, `domain="system"` is about the run itself.

`quality="derived"` requires a provenance edge: `derived_from_refs` must be non-empty and `derivation_method` must name a registered deterministic transform. Registration rejects a derived record with no parents, so the catalog forms an inspectable provenance graph rather than an unsourced assertion. `measured` records must have empty `derived_from_refs`.

`producer_node_id` is mandatory provenance and is not part of evidence identity.
It identifies the plan node instance that published the record. During fragment
registration it must equal `EvidenceFragment.node_id`; `producer_module` and
`producer_implementation_version` must equal the corresponding
`NodeManifest` fields for the same attempt. This is what makes two differently
configured instances of one module distinguishable without coupling
`evidence_id` to an implementation.

The locator describes where the catalog implementation can resolve the value. It may contain a JSON pointer or table selector internally. Claims never expose or cite that locator.

### 11.3 Evidence ID

`evidence_id` is a catalog reference optimized for safe LLM use:

```text
ev_validation_primary_policy_a31f9c2d
ev_period_target_rate_7b2e1881
ev_leakage_train_test_overlap_912cf6aa
```

Rules:

- the ID is deterministically derived from `EvidenceKey.name` and its dimensions;
- the producing implementation is **not** an input to identity;
- it does not contain a physical path, array index, run directory, or raw value;
- collisions are blocking registration errors;
- consumers treat IDs as opaque and do not parse meaning from them.

`EvidenceKey.name` **is** the evidence contract namespace. No separate namespace field exists, because two names for one concept is exactly the duplication this specification forbids elsewhere. A name is a dotted path in a declared registry, for example `dataset.leakage.train_test_overlap`.

Ownership is declared on the producer so the uniqueness rule is checkable rather than aspirational:

```python
class ModuleSpec(BaseModel):
    ...
    provides_evidence_namespaces: frozenset[str]
```

Two modules may not declare the same evidence namespace in one plan; preflight rejects the plan, and registration rejects a record whose `key.name` falls outside its producer's declared namespaces.

Identity is contract-scoped, not producer-scoped:

```text
identity   = dataset.leakage.train_test_overlap + dimensions
provenance = producer_module="leakage_checker_v2", producer_implementation_version="2.1.0"
```

This is a direct consequence of §38. If `evidence_id` embedded the producing module, then replacing `leakage_checker` with a different implementation of the same contract would change every ID it emits, which would in turn break synthesis fixtures, stored claims, and downstream references — exactly the coupling the architecture claims to remove. The implementation is recorded as provenance and is free to change; the evidence contract namespace is the stable identity.

The derivation is specified exactly, because "deterministic" without an algorithm produces one implementation per agent:

```text
1. normalize name and every dimension key and value to Unicode NFC
2. validate grammar, which differs for names and dimension keys:
     name          ^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$
     dimension key ^[a-z][a-z0-9_]*$
3. sort dimensions by key using byte order of the normalized UTF-8 key
4. build the canonical payload as JSON with exactly these settings:
     member order  fixed: "name" then "dimensions"
     dimensions    array of [key, value] pairs, sorted as above
     separators    (",", ":")   — no whitespace
     ensure_ascii  False        — real UTF-8 bytes, never \uXXXX escapes
     encoding      UTF-8, no BOM
5. digest = blake2b(payload, digest_size=16)
6. evidence_id = "ev_" + slug(name) + "_" + digest.hexdigest()[:12]
   where slug(name) replaces "." with "_" and truncates to 48 characters
```

Both grammars allow underscores in every segment, including the first. An earlier draft required the first segment to be alphanumeric only, which rejected ordinary dimension keys such as `ablation_id` — including one used by this document's own example in §11.7.

`ensure_ascii` and separator behavior are stated explicitly rather than left to "compact JSON," because the default in several JSON libraries is to escape non-ASCII, and two implementations disagreeing on that produce different digests for the same logical key while both believing they are canonical.

The digest is 48 bits of hex, not 32. Widening the suffix costs nothing and materially reduces the chance of a blocking collision in a catalog that may hold thousands of records.

The readable prefix exists for human debugging only and carries no semantics; the suffix carries identity. A genuine collision — two distinct canonical payloads yielding the same suffix — is a blocking registration error that names both keys, never a silent overwrite and never an automatic suffix extension, since silently lengthening the suffix would make IDs depend on registration order.

### 11.4 Catalog scope

One `EvidenceCatalog` instance belongs to exactly one `(run_id, competition_id)` pair. This is the MVP model and it is stated as a single rule to remove ambiguity:

- `evidence_id` is unique and resolvable **within its owning run**;
- the same semantic observation in a different run legitimately carries the same local ID, because the two IDs live in different catalogs;
- claims cite the local `evidence_id` and nothing else;
- cross-run evidence resolution is **not supported**; a reference that cannot resolve in the owning catalog is a blocking error, never a lookup in another run;
- `run_id` and `competition_id` remain on `EvidenceRecord` for provenance and scope validation, not as part of the reference syntax.

If a future revision introduces a global catalog, the public reference must become composite and that is an explicit breaking change:

```python
class GlobalEvidenceRef(BaseModel):
    run_id: str
    evidence_id: str
```

Mixing the two models — local IDs in claims while the catalog spans runs — is forbidden, because it makes duplicate detection and reference validation depend on which run happened to be loaded.

### 11.5 Catalog invariants

The canonical `EvidenceCatalog` owns:

- ID construction;
- atomic fragment registration;
- duplicate and collision detection;
- scope validation;
- record resolution;
- catalog filtering and context bounding;
- allowed-reference enumeration;
- reference validation for claims.

Registration fails if:

- an ID already maps to a different key;
- two records have the same contract namespace/key/dimensions in one scope without an explicit revision policy;
- two different modules register the same evidence contract namespace in one run;
- the source artifact is absent or unpublished;
- the locator cannot resolve;
- run or competition scope does not match;
- required provenance is missing or `producer_node_id`, module, implementation,
  plan, generation, or attempt disagrees with the enclosing published bundle;
- `quality="derived"` with empty `derived_from_refs` or an unregistered `derivation_method`;
- `derived_from_refs` cites an ID absent from the same catalog.

### 11.6 Single allowed-reference source

The synthesis adapter builds an immutable `EvidenceCatalogView`. The prompt builder renders allowed refs from this view. The response validator validates against the same view instance or its serialized checkpoint.

No second generator is permitted.

### 11.7 Legacy path promotion

Existing v5 dotted evidence addresses migrate through a declared deterministic map, never by string reuse. A v5 address such as `baseline_ablation_evidence.ablations.<ablation_id>` becomes:

```python
EvidenceKey(
    name="dataset.baseline.ablation",
    dimensions={"ablation_id": "<id>"},
)
```

with a derived opaque `evidence_id`. Rules:

- the mapping is a tested table owned by the legacy adapter;
- an unmapped legacy path is a blocking migration error, never silently carried forward as an ID;
- a legacy path that cannot be expressed as `(name, dimensions)` without embedding an array index is rejected, which forces the producer to emit a stable dimension instead;
- promotion is one-directional: v6 IDs are never converted back into v5 paths except by the export view in §32.

This closes the gap between "physical paths are forbidden as refs" and the fact that the v5 codebase currently treats exact dictionary paths as the official evidence address.

---

## 12. Module contract

Each module implements a small generic protocol:

```python
class Module[InputT, OutputT, ConfigT](Protocol):
    spec: ModuleSpec

    def run(
        self,
        input: InputT,
        config: ConfigT,
        services: ModuleServices,
    ) -> ModuleResult[OutputT]: ...
```

```python
class InputPortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    capability: str
    contract: str
    accepted_schema_versions: frozenset[int]
    required: bool
    cardinality: Literal["one", "many"]


class OutputPortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    capability: str
    contract: str
    contract_schema_version: int


class ModuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    module_id: str
    module_contract_version: int
    module_implementation_version: str
    cache_fingerprint_version: str
    input_ports: tuple[InputPortSpec, ...]
    output_ports: tuple[OutputPortSpec, ...]
    provides_evidence_namespaces: frozenset[str]
    default_failure_policy: Literal["blocking", "optional"]
    cacheable: bool
```

Ports are symmetric on purpose. A bare `output_schema_version` plus a capability set cannot answer the question preflight must answer — *which named contract, at which version, satisfies this consumer port* — because it never states the output contract's name. `OutputPortSpec` closes the triple:

```text
capability → contract → contract_schema_version
```

Preflight then proves compatibility structurally: for each edge, the producer's named output port must expose the contract the consumer port names, at a version inside the consumer's `accepted_schema_versions`. `provides_capabilities` is subsumed by `output_ports` and removed rather than kept as a second source of the same fact.

Edges name both ends, and they connect **nodes**, not modules:

```python
class DependencyEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    producer_node_id: str
    producer_port: str
    consumer_node_id: str
    consumer_port: str
    failure_policy: Literal["blocking", "optional"] | None = None


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    module_id: str
    config: JsonValue
    activation: Literal["always", "on_request"] = "always"


class PipelinePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    plan_id: str
    plan_digest: str
    nodes: tuple[NodeSpec, ...]
    edges: tuple[DependencyEdge, ...]
    input_bindings: tuple[PlanInputBinding, ...]
```

A node is an instance of a module with its own configuration. The distinction is load-bearing and was implied throughout the document before being defined here: attempts are indexed by `node_id`, bundle paths use `node_id`, and configuration belongs to a node rather than to a module — otherwise one module could not appear twice in a plan with different settings, which profiling and leakage checks over different table groups plainly require. The run manifest commits to both the logical `plan_id` and the exact `plan_digest`.

Ports must also be deliverable at runtime, not only comparable at preflight. `Module.run()` returns one `ModuleResult[OutputT]`, so a module with several output ports has no defined way to say which part of `OutputT` belongs to which port. Preflight would match ports that the runner then cannot actually connect. The MVP resolves this by construction:

```text
A node has exactly one output port in the MVP.
OutputT is that port's payload.
```

Multi-output modules are deferred rather than half-specified. When they are needed, the extension is a typed bundle, not a path selector:

```python
class PortBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ports: Mapping[str, JsonValue]
```

A selector-based alternative — naming a field path inside `OutputT` — is rejected because it reintroduces exactly the physical addressing that §5.5 removes from evidence refs.

Input delivery needs the same treatment, and closing only the output side would leave preflight matching ports the runner still cannot bind. The mapping is normative:

```text
Each InputPortSpec.name maps to an identically named field of InputT.

one  + required   → exactly 1 source → payload
one  + optional   → 0 or 1 source   → payload | None
many + required   → 1 or more       → tuple[payload, ...]
many + optional   → 0 or more       → tuple[payload, ...]
```

`source` means a dependency edge or an external binding. A consumer port may use
edges or bindings, but never a mixture of both. Duplicate edges and duplicate
binding IDs are rejected. For edge-backed `many`, payloads are ordered by
`(producer_node_id, producer_port)` normalized UTF-8 byte order. For
binding-backed `many`, payloads are ordered by `binding_id` normalized UTF-8
byte order. Plan order and completion order are never inputs to runtime
delivery.

Not every input has a producer node. Dataset paths, the run request, and competition metadata enter the plan from outside:

```python
class RunRequestInputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["run_request"] = "run_request"
    binding_id: str
    consumer_node_id: str
    consumer_port: str
    source_contract: Literal["RunRequest"] = "RunRequest"
    source_schema_version: int


class DatasetRootInputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["dataset_root"] = "dataset_root"
    binding_id: str
    consumer_node_id: str
    consumer_port: str
    source_contract: Literal["DatasetRootRef"] = "DatasetRootRef"
    source_schema_version: int
    dataset_role: Literal["primary"] = "primary"


class CompetitionBriefInputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["competition_brief"] = "competition_brief"
    binding_id: str
    consumer_node_id: str
    consumer_port: str
    source_contract: Literal["CompetitionBrief"] = "CompetitionBrief"
    source_schema_version: int


PlanInputBinding = Annotated[
    RunRequestInputBinding
    | DatasetRootInputBinding
    | CompetitionBriefInputBinding,
    Field(discriminator="kind"),
]
```

The binding declares the exact public contract and version supplied at runtime.
Preflight compares those fields with the consumer port exactly as it compares an
output port. The runner resolves the corresponding validated object from the
run's immutable input snapshot; the plan never carries a raw untyped path or
free-form config binding. Module configuration is delivered only through
`NodeSpec.config`.

Preflight applies the cardinality matrix above after validating every edge and
binding. An optional `many` port with no source receives `()`, while an optional
`one` port with no source receives `None`. Modeling external inputs as bindings
rather than synthetic nodes keeps the node set equal to executable things that
produce attempts.

#### 12.0.1 Plan identity and lifecycle

`plan_id` is a run-local logical identifier; `plan_digest` is the integrity and
content identity of the exact plan body. The digest uses §10.4.1 and excludes
its own field.

Plan lifecycle is deliberately immutable in the MVP:

1. construct and validate the whole plan, including all `on_request` nodes;
2. compute and insert `plan_digest`;
3. run preflight against the exact plan;
4. persist that exact body as `plan.json` before Dataset Intelligence IO or any
   node attempt; the run root may already exist from Source Intelligence or
   Research Scout, but it has no generation manifest until this step completes;
5. bind both `plan_id` and `plan_digest` into every generation, manifest, node
   bundle, and evidence fragment;
6. reject resume, publication, or continuation if either value differs.

Activating an `on_request` node changes generation state, not the plan body, so
neither plan field changes. Editing a node, edge, binding, module config, port,
or activation policy creates a different plan digest. In the MVP a different
plan digest requires a new `run_id`; in-run plan migration is not supported.
Replacing `plan.json` in place is forbidden.

Three version numbers appear near a module output and mean different things. They are separate fields and are never derived from one another:

| Version | Lives on | Governs |
|---|---|---|
| `ModuleResult.schema_version` | the envelope | the shape of the result wrapper itself |
| `OutputPortSpec.contract_schema_version` | the port | the shape of `value` carried in that envelope |
| `ModuleSpec.module_contract_version` | the module | the module's overall public surface, ports included |

Adding a field to a payload advances the output contract version and the module contract version, while the envelope version stays put. Changing `ModuleResult` itself advances the envelope version for every module at once. Preflight compares the port-level version, never the envelope or module number, when deciding whether two nodes may be connected.

`requires_capabilities: set[str]` is insufficient for any module with more than one input: it cannot express which input is which, whether an input is optional, whether several providers may satisfy one capability, or which contract versions are accepted. `input_ports` states all of it explicitly. A port with `cardinality="many"` accepts multiple providers of the same capability; a port with `required=False` may resolve to nothing and obliges the module to declare a limitation when it does.

Three version fields exist because three different questions are being asked, and one hand-maintained number cannot answer all of them:

| Field | Question | Changes when |
|---|---|---|
| `module_contract_version` | Is my input/output shape compatible? | The public contract shape changes (§9.2) |
| `module_implementation_version` | Which code produced this? | Any behavioral change worth recording as provenance |
| `cache_fingerprint_version` | May I reuse a cached result? | Any change that invalidates prior outputs |

A contract change always implies a cache change; the reverse is not true — an internal algorithm fix changes the cache fingerprint and the implementation version while leaving the contract version untouched.

Obligation is a property of the edge, not only of the module. Drift analysis may be optional in a routine run and blocking when a critical research hypothesis depends on it. `ModuleSpec.default_failure_policy` is the module's own default; `PipelinePlan` may override it per dependency edge via `DependencyEdge.failure_policy`.

Conflict resolution is one rule with no exceptions: the effective policy is the edge override when present, otherwise the module default; **an edge may escalate `optional` to `blocking`, but may never downgrade a module's `blocking` default to `optional`.** A module declaring itself blocking is asserting that its own output cannot be safely absent, and no plan may overrule that. Escalation is recorded in the plan so a blocked run can explain which edge demanded it.

`ModuleServices` may expose dataset reads, artifact publication, the evidence catalog, resource limits, safe logging, cancellation, and clocks. It must not expose a mutable analytical state.

### 12.1 Configuration fingerprint

Every module declares and receives its own typed config model.
`config_fingerprint` uses the registry in §10.4.1.

Every node execution receives one input snapshot:

```python
class InputSnapshotItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    consumer_port: str
    source_kind: Literal["edge", "binding"]
    source_id: str
    producer_node_id: str | None = None
    producer_port: str | None = None
    attempt_id: str | None = None
    attempt_generation: int | None = None
    contract: str
    contract_schema_version: int
    value_digest: str


class InputSnapshotPreimage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_digest: str
    consumer_node_id: str
    inputs: tuple[InputSnapshotItem, ...]
```

For an edge, `source_id` is
`"{producer_node_id}:{producer_port}"`, `attempt_id` and
`attempt_generation` are required, and `value_digest` is the published result
artifact's `file_hash`. For a binding,
`source_id` is `binding_id`, producer fields are `None`, and `value_digest` is
BLAKE2b-256 over canonical JSON bytes of the validated external contract.
Dataset-reading bindings include the typed dataset fingerprint from §10.4.1 in
that validated contract.

Items are sorted first by consumer port and then by `source_id`, both in
normalized UTF-8 byte order. `input_snapshot_digest` is computed from the whole
preimage by §10.4.1. A node may read no upstream or external value absent from
this list. Generation is deliberately excluded: the digest proves input
identity, while `NodeManifest.generation` proves execution placement. Identical
inputs under the same immutable plan may therefore produce a legitimate cache
hit in a later generation.

The node `cache_fingerprint` preimage is:

```text
{
  "module_id": ...,
  "cache_fingerprint_version": ...,
  "config_fingerprint": ...,
  "input_snapshot_digest": ...,
  "relevant_library_versions": [[name, version], ...]
}
```

Library pairs are present only when the module declares that a library affects
semantics and are sorted by normalized UTF-8 package name. The fingerprint uses
§10.4.1. Actual input contract versions and content hashes are already bound
inside `input_snapshot_digest`; they are not reintroduced as a second source of
truth.

Changing Final Synthesis temperature must not invalidate profiling. Changing a leakage threshold must not invalidate file inventory. A module cannot read configuration outside its declared typed subset.

### 12.2 Module result rules

- input is validated before module code executes;
- output is validated before publication;
- evidence registration occurs only after output validation;
- a module cannot publish evidence after failure;
- `quality="degraded"` requires at least one module-scope `Limitation` or degradation diagnostic, per the invariant in §9;
- modules do not mutate input contracts;
- modules do not import another module's private types;
- optional failure cannot corrupt or erase prior successful outputs.

---

## 13. Adapters

Adapters are explicit transition components, not compatibility dumping grounds.

### 13.1 Research to Dataset Intelligence

`ResearchToDatasetAdapter`:

- accepts only the public Research Scout output;
- validates schema and competition identity;
- maps hypotheses to registered Dataset Intelligence capabilities;
- rejects unknown capabilities before dataset resolution;
- preserves source evidence refs and uncertainty;
- does not infer dataset facts or choose module internals.

### 13.2 Dataset Intelligence to Synthesis

`DatasetToSynthesisAdapter`:

- accepts published module results and the canonical catalog;
- requires mandatory diagnostics and hypothesis results;
- selects and bounds evidence deterministically;
- creates one immutable `SynthesisContext`;
- exposes one `EvidenceCatalogView` to both prompt and validator;
- does not regenerate evidence IDs or read a legacy pack.

### 13.3 Version and legacy adapters

Version adapters convert one explicit integer contract version to another. Legacy adapters may import selected v5 data for migration fixtures, but imported facts are marked with legacy provenance and ambiguous JSON-path refs are rejected.

---

## 14. Dataset Intelligence capabilities

Research and Synthesis request capabilities, not filenames or implementation functions.

Core capability names:

```text
dataset.inventory
dataset.schema
dataset.submission_contract
dataset.profile
metric.interpretation
validation.policy
leakage.basic
leakage.temporal
hypothesis.evaluate
```

Optional capability names:

```text
relationships.infer
drift.univariate
drift.adversarial
baseline.evaluate
feature_probe.evaluate
panel.validate
ranking.validate
```

A capability registry maps names to module providers and accepted request contracts. The registry is static at process startup in the MVP.

---

## 15. Dataset Intelligence module set

### 15.1 Blocking core

1. **dataset_gateway** — resolves the dataset root, fingerprints allowed files, and provides safe reads.
2. **file_inventory** — inventories supported files and basic dimensions without assuming table roles.
3. **schema_inferer** — infers tables, column types, roles, candidate targets, IDs, group keys, and time candidates.
4. **submission_analyzer** — interprets sample submission shape, including multi-output and multilabel targets.
5. **table_profiler** — computes bounded profiles and missingness/cardinality summaries.
6. **metric_analyzer** — resolves task type, metric family, prediction contract, direction, and local evaluator availability.
7. **validation_analyzer** — selects a validation policy from evidence.
8. **leakage_checker** — runs blocking leakage checks required for a usable strategy.
9. **hypothesis_evaluator** — maps every requested hypothesis to one explicit result.

### 15.2 Optional analytical modules

- relationship inferer;
- drift analyzer;
- baseline runner;
- feature probe;
- extended temporal leakage checker;
- panel-data validator;
- ranking/query validator;
- high-cardinality and duplicate-group diagnostics.

Optional modules can fail without invalidating unrelated core evidence. Their absence must be represented in limitations.

### 15.3 Time and panel policy

A time-like column alone never forces temporal validation. Temporal or panel validation requires evidence such as:

- competition evaluation semantics;
- train/test chronology;
- entity × time structure;
- future-unavailable features;
- period-sensitive target or drift behavior;
- an explicit source-backed hypothesis confirmed by dataset structure.

This is the single canonical rule. Other modules reference the resulting `ValidationPolicy`; they do not restate or independently reimplement it.

### 15.4 Fit-scope safety

Any learned transform, encoding, aggregation, feature selection, or imputation used by baselines or probes declares one fit scope:

```text
none
fold_train_only
group_train_only
time_past_only
full_train_allowed
```

Unsafe full-data fitting before validation is a blocking error.

---

## 16. Research contracts

```python
class ResearchHypothesis(BaseModel):
    schema_version: int
    hypothesis_id: str
    statement: str
    rationale: str
    source_evidence_refs: list[str]
    requested_capabilities: list[str]
    required_observations: list[str]
    priority: Literal["critical", "high", "medium", "low"]
```

```python
class InvestigationPlan(BaseModel):
    schema_version: int
    competition_id: str
    hypotheses: list[ResearchHypothesis]
    resource_budget: InvestigationBudget
```

Rules:

- IDs are unique;
- every requested capability is registered;
- required observations are testable and do not prescribe private code;
- source refs resolve in the one canonical run catalog as records with `domain="source"`;
- invalid plans fail before dataset IO;
- generic mandatory Dataset Intelligence capabilities may be added only by the deterministic adapter and must be recorded as system requirements.

There is no separate source catalog. Earlier drafts referred to a "Source Evidence Catalog" that was never defined, which reintroduced the second registry §5.4 exists to prevent. Source-derived findings are registered in the same run-scoped canonical catalog with `domain="source"`, carrying the same IDs, the same registration invariants, and the same allowed-reference view. The domain field — not a separate store — is what keeps a literature claim from being used as proof of a dataset fact (§23).

---

## 17. Hypothesis evaluation

```python
class HypothesisResult(BaseModel):
    schema_version: int
    hypothesis_id: str
    status: Literal[
        "confirmed",
        "rejected",
        "partially_supported",
        "not_testable",
        "skipped",
    ]
    conclusion: str
    evidence_refs: list[str]
    diagnostics: list[Diagnostic]
```

Rules:

- exactly one result exists for every input hypothesis;
- `confirmed`, `rejected`, and `partially_supported` require dataset evidence;
- `not_testable` identifies the missing capability, observation, or data;
- `skipped` identifies an execution or budget reason;
- source evidence alone cannot confirm a local dataset hypothesis;
- evaluation consumes catalog records, not raw module dictionaries.

---

## 18. In-process runner

### 18.1 Scope

The MVP runner is a small coordinator for a static graph. It provides:

- capability resolution;
- dependency validation and topological ordering;
- sequential execution by default;
- typed input assembly;
- per-node configuration;
- success/failure/skipped status;
- checkpoint reuse;
- bundle publication;
- failure propagation;
- structured events.

It does not provide distributed execution, daemon scheduling, a UI, arbitrary plugins, complex retry workflows, or a general asset abstraction.

### 18.2 Execution status

Runner statuses are:

```text
pending
running
success
failed
skipped
inactive
```

These are the statuses in `NodeGenerationState` (§10.6). Degradation belongs to
`ModuleResult.quality`, not the execution state machine. `inactive` is reserved
for an `on_request` node that was not activated in this generation and is never
used for a dependency failure. A node skipped because a required dependency
failed records that dependency in `blocked_by_node_ids`.

### 18.3 Preflight

Before run-directory creation and dataset IO where feasible:

1. validate the run request;
2. normalize competition identity;
3. validate Research → Dataset input;
4. resolve requested capabilities and module providers;
5. verify required contract versions and adapters;
6. apply the exact `one`/`many`, required/optional cardinality matrix in §12,
   reject mixed edge/binding sources, and build and validate the static DAG;
7. validate module-specific configuration;
8. verify `plan_digest`, evidence namespace ownership, and the complete
   `on_request` capability registry;
9. check required credential presence without logging values;
10. estimate resource policy and reject impossible requests.

### 18.4 Failure propagation

- a blocking module failure fails its requested stage and skips dependants;
- an optional module failure records a diagnostic and independent nodes continue;
- deterministic input/contract errors are not retried;
- transient IO or provider errors may use a small module-declared retry count;
- retry never changes input, configuration, or evidence semantics;
- failed attempts cannot publish a visible bundle.

An invalidated node's successful attempt from the prior committed generation is
not a fallback inside the candidate. If its recompute fails, the candidate state
is `failed`; downstream active nodes become `skipped` as dictated by edge
policy. A candidate containing a blocking failure cannot commit. A candidate
with only optional failures may commit only when all remaining required ports,
quality gates, and Synthesis rules are satisfied.

### 18.5 Resume

A checkpoint is reusable only when:

- module ID matches and `cache_fingerprint_version` matches;
- output contract version is accepted or an explicit adapter exists;
- module-specific config fingerprint matches;
- required input identities/hashes match;
- dataset fingerprint matches for dataset-reading modules;
- the published bundle and hashes pass integrity checks;
- the node is cacheable.

Resume loads and validates the public checkpoint once. Subsequent same-run consumers receive the in-memory object.

---

## 19. Run layout, retention, and garbage collection

```text
data/runs/{run_id}/
├── run_manifest.json
├── plan.json
├── status.json
├── nodes/
│   └── {node_id}/{attempt_id}/
│       ├── node_manifest.json
│       ├── result.json
│       ├── evidence_fragment.json
│       ├── tables/
│       └── blobs/
├── checkpoints/
├── logs/
└── exports/
    ├── evidence_catalog.json
    ├── final_strategy.json
    ├── final_strategy.md
    ├── final_strategy.docx
    └── eda_evidence_pack_v5.json
```

`exports/evidence_catalog.json` is a materialized view created by the canonical catalog service. It is not an independently maintained registry.

### 19.1 Retention classes

Every file belongs to one class:

| Class | Examples | Default policy |
|---|---|---|
| ephemeral | staging dirs, failed partial attempts | remove on recovery or GC |
| checkpoint | reusable node bundles | keep for retained runs |
| final | strategy and presentation outputs | keep with run |
| shared cache | downloaded dataset archives, normalized sources | manage separately by cache policy |
| pinned | explicitly protected run | never delete automatically |

### 19.2 Required GC behavior

The CLI must support dry-run and explicit deletion, for example:

```text
runs gc --older-than 30d --keep-latest 5 --exclude-pinned --dry-run
```

Rules:

- no automatic deletion of pinned runs;
- shared dataset/source cache is not deleted as a side effect of run GC;
- incomplete staging directories may be cleaned after a safety age;
- GC reports reclaimed and retained paths;
- deletion targets are resolved under the configured run root only.

---

## 20. Determinism budget

The system does not make one universal determinism promise.

### 20.1 Byte deterministic

The following must match exactly for identical accepted inputs and versions:

- evidence ID construction;
- semantic keys and dimensions;
- module-specific config fingerprints;
- DAG structure and capability resolution;
- integer/enumerated policy decisions;
- normalized contract serialization where explicitly designated canonical;
- catalog reference sets and claim-to-evidence mappings.

### 20.2 Semantically deterministic

The following are compared after normalization or within declared tolerances:

- floating-point metrics;
- distribution summaries;
- sampled statistics;
- model probe scores;
- ordering of equally ranked numeric candidates.

Every affected module declares:

- random seed policy;
- sampling algorithm and seed;
- tolerance or normalization rule;
- relevant numerical library versions when materially important.

### 20.3 Intentionally non-deterministic

The following are not compared byte-for-byte and do not form artifact identity:

- LLM narrative wording;
- ordering of semantically equivalent recommendations;
- provider-generated request IDs;
- retry timing and safe log timestamps.

Golden Synthesis tests compare contract shape, claim types, required sections, evidence refs, diagnostics, and invariants—not exact prose.

---

## 21. Resource and sampling policy

All dataset-reading modules receive a typed resource budget:

```python
class ResourceBudget(BaseModel):
    max_memory_mb: int
    max_wall_time_seconds: int
    max_rows_full_scan: int
    max_sample_rows: int
    random_seed: int
    max_output_bytes: int
```

Rules:

- inventory and schema inference prefer metadata and bounded reads;
- sampling is reproducible for the same seed and algorithm version;
- evidence identifies whether it came from full data or a sample;
- sampled evidence records sampling fraction and limitations;
- a budget breach produces a typed diagnostic or failure according to module policy;
- silent truncation is forbidden;
- LLM context and additional-investigation budgets are separate from dataset compute budgets.

---

## 22. Synthesis context

```python
class SynthesisContext(BaseModel):
    schema_version: int
    run: RunIdentity
    plan_id: str
    plan_digest: str
    target_generation: int
    input_snapshot_digest: str
    competition: CompetitionBrief
    research_summary: ResearchSummary
    hypotheses: tuple[ResearchHypothesis, ...]
    hypothesis_results: tuple[HypothesisResult, ...]
    evidence: EvidenceCatalogView
    constraints: tuple[StrategyConstraint, ...]
    diagnostics: tuple[Diagnostic, ...]
    upstream_issues: tuple[UpstreamIssue, ...]
    investigation_decisions: tuple[InvestigationRequestDecision, ...]
    investigation_round: int
```

`constraints` and `upstream_issues` are carried explicitly because the publication gate (§23.1) requires the strategy to reconcile them. Without a field, they were generated by Dataset Intelligence and demanded by the gate while no contract moved them in between — the model was being judged against requirements it never received.

The same single-source rule that governs evidence governs both: the prompt builder and the publication validator read the identical `constraints` and `upstream_issues` from the one validated context object. A constraint reaching the validator but not the prompt would make failure unavoidable and inexplicable to the model.

`plan_id`, `plan_digest`, `target_generation`, and
`input_snapshot_digest` are present because the strategy must declare exactly
which immutable plan, generation, and Synthesis input snapshot it belongs to.
The digest is the Synthesis node's `InputSnapshotPreimage` digest (§12.1), not
the full candidate snapshot digest, which changes when the Synthesis attempt is
published.

The adapter builds the context once for a synthesis attempt. Prompt construction and response validation use that exact validated context.

`EvidenceCatalogView` is a **typed projection**, not the storage catalog and not raw payloads. The catalog defines storage and refs; the view defines what a model is allowed to see:

```python
class EvidenceViewEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    key: EvidenceKey
    domain: Literal["source", "dataset", "system"]
    value_kind: Literal["scalar", "distribution", "table", "text", "diagnostic"]
    finding_severity: FindingSeverity
    summary: str
    rendered_value: str | None
    quality: Literal["measured", "derived", "estimated", "degraded"]
    limitations: tuple[Limitation, ...] = ()


class EvidenceCatalogView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run: RunIdentity
    plan_id: str
    plan_digest: str
    generation: int
    snapshot_status: Literal["candidate", "committed"]
    input_snapshot_digest: str
    entries: tuple[EvidenceViewEntry, ...]
    excluded_ids: tuple[str, ...] = ()
```

Projection rules:

- scalars render their exact value; distributions and tables render a bounded deterministic summary, never a full dump;
- blobs are never inlined — they appear as an entry with `rendered_value=None`;
- rendering is deterministic and byte-stable for identical input, so the same context produces the same prompt;
- the view never contains locators, physical paths, run directories, or artifact filenames;
- `allowed_refs` is exactly `{entry.evidence_id for entry in entries}` — the same view object that renders the prompt answers reference validation, per §11.6.
- the view's run, plan, generation, and `input_snapshot_digest` exactly match
  its owning `SynthesisContext`; a view cannot be reused across attempts.

### 22.1 Context bounding

If evidence exceeds the model context budget, deterministic selection:

- preserves evidence referenced by hypothesis results;
- preserves critical diagnostics and limitations;
- preserves core metric, validation, leakage, and submission evidence;
- scores optional evidence by declared relevance rules;
- records included and excluded IDs;
- never rewrites IDs;
- exposes truncation as a limitation.

---

## 23. Claim and Final Strategy contract

```python
class ConstraintResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_id: str
    disposition: Literal["satisfied", "acknowledged", "unresolved"]
    reason: str | None = None


class Claim(BaseModel):
    claim_id: str
    kind: Literal["fact", "inference", "recommendation", "uncertainty"]
    subject_domain: Literal["source", "dataset", "system", "strategy"]
    text: str
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
    validation_prerequisites: tuple[str, ...] = ()
    missing_observation: str | None = None
    limitations: tuple[str, ...] = ()
```

Rules:

- facts require direct supporting evidence;
- inferences require evidence and make the inferential step explicit;
- recommendations require evidence, or both non-empty `assumptions` and
  `validation_prerequisites`;
- uncertainties may have no supporting evidence only when
  `missing_observation` is non-empty;
- evidence references must resolve in the supplied catalog view;
- a claim of kind `fact` or `inference` with `subject_domain="dataset"` requires **at least one** evidence record with `domain="dataset"`.

The last rule is stated positively on purpose. An earlier negative phrasing — a dataset claim may not rest *solely* on source evidence — was satisfied by citing one source record and one system record, leaving a claim about the data supported by no measurement of the data. Requiring the presence of dataset-domain evidence closes that, and closes it without any judgement about what the text says.

`subject_domain` is what makes the last rule checkable. Adding `domain` to evidence alone was not enough: the validator still had to decide whether a sentence asserted a local dataset fact, which is natural-language interpretation. The claim now declares its own subject, so the gate compares two enumerations. A claim about published literature is `source`; a claim about the actual data is `dataset`; a claim about the run is `system`; a recommendation about how to proceed is `strategy`.

`ConstraintResolution` replaces the earlier pair of parallel lists. Two lists could not express which reason belonged to which code, and constraint codes were being stored in a field named for diagnostics although a constraint is not a diagnostic. One record binds code, disposition, and reason together, and `reason` is required whenever `disposition="acknowledged"`.

Resolutions live on `FinalStrategy`, not on individual claims. A constraint is reconciled once per strategy; attaching resolutions to claims would allow the same code to be resolved differently in two sections and would give the gate no single place to check. Claims may of course discuss a constraint in prose — that is narrative, not reconciliation.

Critical deterministic conclusions are carried as machine-checkable constraints rather than as prose to be interpreted:

```python
class StrategyConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_id: str
    code: str
    rule: str
    expected_value: JsonValue
    origin_evidence_refs: tuple[str, ...]
    severity: FindingSeverity
```

```python
StrategyConstraint(
    code="temporal_split_required",
    rule="validation_policy",
    expected_value="purged_group_time_series",
    origin_evidence_refs=("ev_validation_primary_policy_a31f9c2d",),
    severity="critical",
)
```

`rule` is not free text. It is a key in a declared constraint-rule registry that binds each rule to the decision it constrains and the comparison used:

```text
rule                → decision field                        → comparator
validation_policy   → FinalStrategy.validation_decision.policy      equals
leakage_controls    → FinalStrategy.leakage_controls_decision.controls  superset
metric_alignment    → FinalStrategy.validation_decision.metric      equals
```

The registry is the single source for both the emitter and the gate. A constraint whose `rule` is absent from the registry is rejected at emission, not discovered at publication. Without this table, `expected_value` has no defined meaning and two implementations will compare different things — one checking string equality, another checking membership — while both appear to satisfy the specification.

Deterministic modules emit constraints; Synthesis must satisfy or explicitly acknowledge them. This replaces natural-language contradiction detection with a comparison.

```python
class FinalStrategy(BaseModel):
    schema_version: int
    run: RunIdentity
    plan_id: str
    plan_digest: str
    generation: int
    input_snapshot_digest: str
    executive_summary: tuple[Claim, ...]
    problem_definition: tuple[Claim, ...]
    validation_strategy: tuple[Claim, ...]
    leakage_controls: tuple[Claim, ...]
    modeling_strategy: tuple[Claim, ...]
    feature_strategy: tuple[Claim, ...]
    validation_decision: ValidationDecision
    leakage_controls_decision: LeakageControlsDecision
    experiment_plan: tuple[ExperimentRecommendation, ...]
    risks: tuple[RiskAssessment, ...]
    unresolved_questions: tuple[OpenQuestion, ...]
    constraint_resolutions: tuple[ConstraintResolution, ...]
    upstream_issue_resolutions: tuple[UpstreamIssueResolution, ...]
    global_limitations: tuple[str, ...]
```

Upstream diagnostics and limitations are reconciled structurally, exactly as constraints are:
```python
class UpstreamIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_id: str
    issue_code: str
    issue_kind: Literal["diagnostic", "limitation"]
    severity: DiagnosticSeverity | FindingSeverity
    origin_node_id: str
    message: str


class UpstreamIssueResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_id: str
    disposition: Literal["preserved", "acknowledged"]
    reason: str | None = None
```

Codes classify; identifiers identify. One `temporal_split_required` may legitimately arise for two different tables, and one limitation code may be emitted by two nodes, so keying reconciliation on `code` alone makes "exactly one resolution per constraint" ambiguous precisely when it matters. `constraint_id` and `issue_id` are the reconciliation keys; `code` remains the classification used for registry lookup and reporting. Both are assigned by the emitting deterministic module and are unique within a generation.

`UpstreamIssue.severity` is derived normatively, not by convention: for `issue_kind="diagnostic"` it is the `Diagnostic.severity` verbatim, and for `issue_kind="limitation"` it is the `Limitation.severity` verbatim. `Limitation` therefore carries a `FindingSeverity` — without it the adapter had no defined source for the severity the gate thresholds on, and each implementation would have invented a mapping.

`global_limitations` remains prose for the reader. It is not the mechanism by which preservation is checked, because verifying that a typed `Limitation` survived into free text is the same natural-language judgement this specification removed from the constraint path. Constraint reconciliation and issue preservation now use one structural pattern rather than one structural and one interpretive.

### 23.1 Publication gate

The pre-publication gate is a deterministic function of exactly four validated
inputs:

```text
StrategyAttempt
SynthesisContext
the Synthesis node's NodeManifest
the constraint-rule registry version named by the context schema
```

It performs the following checks in order and publishes only if all pass:

1. validate the `StrategyAttempt` discriminator, wrapper schema version, and
   complete `FinalStrategy` with `extra="forbid"`;
2. require exact equality of `run_id`, `competition_id`, `plan_id`,
   `plan_digest`, `generation`, and `input_snapshot_digest` across the strategy,
   context, and catalog view; require the Synthesis `NodeManifest` to match all
   of those fields it carries (`run_id`, plan fields, generation, and input
   digest), and require the context view to have
   `snapshot_status="candidate"`;
3. require the node manifest to name the same generation and to be a valid
   published Synthesis bundle in that candidate;
4. require non-empty `executive_summary`, `problem_definition`,
   `validation_strategy`, `leakage_controls`, `modeling_strategy`,
   `feature_strategy`, and `experiment_plan`;
5. collect claims from all claim-bearing sections and require globally unique,
   non-empty `claim_id` values;
6. require every evidence reference to match
   `^ev_[a-z][a-z0-9_]{0,47}_[0-9a-f]{12}$`, occur in
   `context.evidence.entries`, and not occur in `excluded_ids`; any path
   separator, JSON pointer, dotted physical path, or unknown ID therefore fails
   by construction;
7. require every `fact` and `inference` to have at least one evidence ref;
   require every dataset-domain `fact` or `inference` to cite at least one
   entry with `domain="dataset"`;
8. require every `recommendation` either to cite evidence or to have both
   non-empty `assumptions` and non-empty `validation_prerequisites`;
9. require an `uncertainty` with no evidence to have a non-empty
   `missing_observation`;
10. require `constraint_id` values in the context to be unique and every
    resolution ID in the strategy to identify exactly one context constraint;
    duplicate or foreign resolutions fail;
11. for every constraint of `serious` or `critical` severity, require exactly
    one resolution; `satisfied` passes only when the registered typed comparator
    succeeds, while `acknowledged` requires a non-whitespace reason;
    `unresolved` fails;
12. require `issue_id` values in the context to be unique and every issue
    resolution ID in the strategy to identify exactly one context issue;
    duplicate or foreign resolutions fail;
13. for every diagnostic issue with severity `error` or `critical`, and every
    limitation issue with severity `serious` or `critical`, require exactly one
    resolution; `acknowledged` requires a non-whitespace reason;
14. reject non-finite numeric output, unresolved catalog refs, schema-version
    mismatches, and any gate diagnostic of `error` or `critical`.

These rules replace the earlier requirements that "deterministic critical findings are not contradicted without explicit acknowledgment" and that "critical upstream diagnostics and limitations are preserved." Both required deciding whether free text contradicted or retained a typed record — semantic judging, which no deterministic validator can perform and which §20 forbids putting on the LLM's side of the boundary.

A claim citing a constraint is a narrative reference; the typed decision is what the gate compares. `disposition="satisfied"` is accepted only when the decision matches `expected_value` under the registry comparator; otherwise the resolution must be `acknowledged` with a reason, or the strategy does not publish. An unresolved or missing resolution for a critical constraint or issue blocks publication.

After generation commit, presentation performs a separate visibility check:
the committed manifest must select the exact Synthesis `attempt_id`; the export's
run, plan, generation, and `input_snapshot_digest` must equal that bundle; and
all evidence refs must resolve through the catalog rebuilt from the committed
`node_states`. This check never reruns the LLM and never validates against a
candidate that is no longer selected.

Formatting or reference errors may receive bounded validation-feedback retries. Missing evidence is not repaired by repeated prose generation; it enters the additional-investigation protocol.

---

## 24. Bounded additional-investigation loop

Synthesis may request deterministic evidence that is absent from the current catalog.

```python
class AdditionalInvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    request_id: str
    run: RunIdentity
    capability: str
    parameters: CapabilityParameters
    expected_key: EvidenceKey
    question: str
    related_claim_id: str | None
    existing_evidence_refs: tuple[str, ...] = ()
    priority: Literal["critical", "high", "medium", "low"]
```

`expected_key` is mandatory because deduplication is defined in terms of semantic key and dimensions. A free-text `required_observation` cannot be deterministically reduced to a key, so without this field the dedup rule below is unenforceable. The requester must name the evidence it expects to exist afterwards.

`parameters` is a typed discriminated union per whitelisted capability, never a free-form dict:

```python
class PeriodTargetRateParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["period_target_rate"] = "period_target_rate"
    period_column: ColumnRef
    target_column: ColumnRef
    split: Literal["train", "test"]


CapabilityParameters = Annotated[
    PeriodTargetRateParams | ColumnOverlapParams | GroupLeakageParams,
    Field(discriminator="kind"),
]
```

A capability with no registered parameter model cannot be requested. This is what makes "no arbitrary query" a type-level guarantee rather than a policy sentence.

The outcome of a synthesis attempt is itself a contract, not an implicit control-flow branch:

```python
class InvestigationRequestBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run: RunIdentity
    plan_id: str
    plan_digest: str
    target_generation: int
    originating_input_snapshot_digest: str
    round: int
    requests: tuple[AdditionalInvestigationRequest, ...]


class InvestigationRequestDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    disposition: Literal[
        "granted",
        "already_satisfied",
        "duplicate",
        "unsupported",
        "over_budget",
        "rejected",
    ]
    activated_node_ids: tuple[str, ...] = ()
    reason_code: str
    message: str


class InvestigationDecisionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run: RunIdentity
    plan_id: str
    plan_digest: str
    target_generation: int
    originating_input_snapshot_digest: str
    round: int
    decisions: tuple[InvestigationRequestDecision, ...]


class StrategyAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    kind: Literal["strategy"] = "strategy"
    strategy: FinalStrategy


class InvestigationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    kind: Literal["investigation"] = "investigation"
    batch: InvestigationRequestBatch


SynthesisAttemptResult = Annotated[
    StrategyAttempt | InvestigationAttempt,
    Field(discriminator="kind"),
]
```

The discriminator lives on wrapper types rather than on `FinalStrategy` and `InvestigationRequestBatch` directly, so neither payload contract has to carry a tag that exists only for this union, and both remain usable on their own.

An attempt either yields a publishable strategy or a bounded request batch. There is no third state in which synthesis partially succeeds and silently continues.

Flow:

```text
Synthesis attempt
    -> missing-evidence requests
    -> deterministic request validator
    -> whitelisted Dataset Intelligence capabilities
    -> atomic catalog extension
    -> rebuilt SynthesisContext
    -> one new synthesis attempt
```

MVP limits:

- at most one additional round by default, configurable to zero;
- at most three requests in the round;
- only registered whitelisted capabilities;
- no arbitrary Python, SQL, notebook, shell, or free-form data query;
- no request that duplicates an executed semantic key and dimensions;
- a separate wall-time and row budget;
- format errors do not trigger Dataset Intelligence;
- unsupported or over-budget requests become explicit uncertainties;
- evidence registration uses the same immutable bundle and catalog invariants;
- a second unresolved request ends the loop rather than recursing.

An investigation round cannot bypass the generation protocol. Appending evidence outside a generation commit would recreate precisely the defect §10.6 exists to prevent: new upstream evidence coexisting with a Synthesis result derived from the earlier state. Nor can the round add nodes to a plan that §18 declares static.

The MVP resolves this by pre-declaring, not by dynamic graph construction.
Investigation capabilities are ordinary `activation="on_request"` nodes present
in the immutable plan from the start. Preflight owns a versioned registry:

```text
(capability, CapabilityParameters.kind) -> one or more node_id values
```

Every mapped node must be `on_request`, expose the requested capability, and
have bindings capable of receiving the typed parameters. Missing, ambiguous, or
non-`on_request` mappings fail preflight before dataset IO.

The lifecycle is normative:

1. Synthesis runs inside an already open candidate generation `G` and receives
   `investigation_round=0`.
2. If it returns `InvestigationAttempt`, the batch must match the context's run,
   plan, generation, and `input_snapshot_digest`; request IDs and expected keys
   must be unique.
3. The deterministic validator evaluates every request in request-ID byte
   order, checks the capability registry, typed parameters, semantic-key
   deduplication, round limit, and the separate investigation budget, then emits
   one `InvestigationRequestDecision` per request.
4. `granted` requires at least one mapped `on_request` node; its
   `activated_node_ids` are sorted and non-empty. Every other disposition
   requires an empty node list and a non-empty reason.
5. The decision batch is persisted under the same candidate. The Synthesis node
   remains `pending` with `reason_code="investigation_requested"`; an
   `InvestigationAttempt` is not a publishable Final Strategy and cannot become
   the committed Synthesis attempt.
6. Granted nodes change from `inactive` to `pending` in candidate `G`. They and
   their transitive downstream closure execute in that same generation. Old
   attempts for activated or invalidated closure nodes are not carried into the
   candidate.
7. Rejected, unsupported, duplicate, already-satisfied, and over-budget
   decisions are inserted into the rebuilt `SynthesisContext` as
   `investigation_decisions`; they do not create fake evidence.
8. After granted nodes reach terminal state, rebuild the catalog view and
   Synthesis input snapshot, set `investigation_round=1`, and execute Synthesis
   once more in candidate `G`.
9. The round-1 attempt must return `StrategyAttempt`. A second
   `InvestigationAttempt` is a typed round-limit error; bounded format feedback
   may ask for a strategy, but no additional Dataset Intelligence work runs.
10. The candidate may commit only after the round-1 strategy passes §23.1. If a
    granted investigation fails optionally, Synthesis receives the corresponding
    issue and must express the unresolved observation. A blocking failure
    abandons the candidate.

If no request is granted, steps 7–10 still run in the same candidate and
generation; the final strategy carries explicit uncertainties. No empty
generation is created merely to record a rejection.

`plan_id` and `plan_digest` never change during this flow. Only
`NodeGenerationState` changes. The catalog is not extended "in place":
successful investigation evidence becomes externally visible only when the
whole generation commits.

The alternative — a request producing a new `PipelinePlan` version and a candidate generation against it — is deferred. It is more general and will eventually be needed for capabilities nobody anticipated, but it makes plan identity mutable within a run, which every other part of this specification currently assumes it is not.

The Research Scout `InvestigationPlan` in §16 carries run-local source refs, so
`dataset run --plan ...` must identify and continue the originating run. On the
first Dataset Intelligence execution it persists the one immutable
`PipelinePlan` selected by the deterministic adapter. On resume or an additional
investigation, the command must load that persisted plan and require the stored
`plan_digest`; supplying a different plan body is rejected and requires a new
run. Continuation therefore means the same `run_id`, catalog, plan digest, and a
candidate generation. It never imports fragments into a fresh run.

---

## 25. Quality gates

### 25.1 Contract gate

Validates public schema, types, required fields, unknown fields, and accepted integer versions. Failure is blocking.

### 25.2 Artifact gate

Validates publication completeness, file integrity, expected schema fingerprints, path containment, and manifest consistency. Failure is blocking for consumers of that artifact.

### 25.3 Evidence gate

Validates catalog identity, uniqueness, scope, artifact existence, locator resolvability, provenance, and limitations. Failure blocks registration of the complete node bundle.

### 25.4 Claim gate

Validates claim type rules, subject-domain consistency, allowed refs, preserved limitations, unique IDs, and constraint reconciliation per §23.1. Failure blocks strategy publication.

### 25.5 Presentation gate

Validates that rendering preserves the validated strategy and adds no claims. Failure blocks only the affected presentation artifact.

Warnings may represent valid incompleteness or uncertainty. They must not replace failures for missing required fields, invalid refs, duplicate IDs, incompatible versions, scope contamination, or invalid output.

---

## 26. Error model

Public typed errors include:

```text
ContractValidationError
ContractVersionError
CapabilityResolutionError
DatasetResolutionError
DatasetSafetyError
ResourceBudgetExceeded
ArtifactIntegrityError
ArtifactPublicationError
EvidenceRegistrationError
EvidenceResolutionError
ModuleExecutionError
SynthesisValidationError
InvestigationRequestError
```

Every error records a stable code, safe message, stage/module identity, retryability, and optional structured details. Secrets, raw provider responses, and unsafe filesystem data do not enter public diagnostics.

---

## 27. Configuration

```python
class PipelineConfig(BaseModel):
    runtime: RuntimeConfig
    source: SourceIntelligenceConfig
    research: ResearchScoutConfig
    dataset: DatasetIntelligenceConfig
    synthesis: SynthesisConfig
    presentation: PresentationConfig
    retention: RetentionConfig
```

Rules:

- each module receives only its typed config;
- unknown fields are forbidden;
- defaults are explicit and tested;
- environment variables are resolved at the outer boundary;
- secrets are secret handles and are never serialized;
- CLI overrides are deterministic and recorded without secret values;
- module fingerprints use only declared config fields;
- configuration changes that do not affect a module do not invalidate it.

---

## 28. CLI contract

Required top-level commands:

```text
kaggle-researcher run --competition <slug> [--dataset-path <path>]
kaggle-researcher dataset run --plan <path> --dataset-path <path>
kaggle-researcher synthesize --run-id <id>
kaggle-researcher resume --run-id <id>
kaggle-researcher runs inspect --run-id <id>
kaggle-researcher runs gc ...
```

CLI rules:

- local dataset mode works without Kaggle credentials;
- full-pipeline mode validates Research → Dataset input before dataset resolution;
- stage-only commands accept stored public checkpoints;
- resume reports reused and invalidated nodes with safe reasons;
- offline fixtures require no external services;
- destructive cleanup has dry-run support and explicit validated scope;
- legacy v5 execution is explicit and never silently selected.

---

## 29. Testing strategy

### 29.1 Contract tests

For every public contract:

- minimal and full valid fixtures;
- missing required field;
- unknown field;
- invalid enum or dimension;
- unsupported integer schema version;
- serialization round-trip;
- canonical byte golden vectors, including NFC, non-ASCII, mapping order,
  finite float edge cases, and negative zero;
- explicit version-adapter tests where supported.

### 29.2 Module tests

- valid fixture input and expected typed output;
- invalid input fails before implementation work;
- evidence fragment registration;
- empty, malformed, and ambiguous data;
- sampling and resource-budget behavior;
- degraded result requirements;
- module-specific config fingerprint;
- no undeclared config or dataset reads;
- numerical tolerance where applicable.

### 29.3 Adapter tests

- source and target validation;
- preservation of scope, diagnostics, uncertainty, and refs;
- unknown capability rejection before dataset IO;
- no invented semantics;
- supported legacy conversion;
- ambiguous legacy input rejection.

### 29.4 Runner and checkpoint tests

- dependency ordering and cycle detection;
- all four cardinality cases and deterministic `many` ordering;
- rejection of mixed edge/binding sources and untyped external bindings;
- missing capability provider;
- plan digest golden, immutable plan enforcement, and resume mismatch rejection;
- every plan node represented in candidate and committed `node_states`;
- `inactive` accepted only for an unrequested `on_request` node;
- blocking and optional failure propagation;
- failed recompute cannot reuse the prior generation's attempt inside the
  candidate;
- invisible failed publication;
- all-or-nothing evidence registration;
- resume hit and targeted invalidation;
- synthesis config change does not invalidate EDA;
- corrupted hash or schema rejects reuse;
- recovery ignores incomplete staging bundles.
- `NodeManifest.input_snapshot_digest` matches the exact typed input preimage.

### 29.5 Synthesis tests

- prompt and validator use one catalog view;
- unknown ID and physical-path ref rejection;
- fact without evidence rejection;
- valid inference, recommendation, and uncertainty rules;
- deterministic context bounding;
- mandatory diagnostic preservation;
- exact run/plan/generation/input-digest equality across context, view, strategy,
  and node manifest;
- structural golden comparisons without prose equality;
- additional request validation, budget, deduplication, and round limit;
- malformed LLM output cannot mutate canonical evidence.

### 29.6 Integration fixtures

Minimum fixtures:

1. binary IID classification;
2. regression;
3. temporal/stability classification;
4. grouped classification;
5. panel entity × time data;
6. ranking/query-grouped task;
7. multilabel or multi-output submission;
8. multi-table relational data;
9. malformed data with safe partial inventory;
10. unknown or custom metric.

Golden Dataset Intelligence and Synthesis tests use stored Research outputs. They do not repeat web retrieval, LLM calls, or Kaggle downloads.

The dataset fixtures above vary the *data*. They do not exercise the run lifecycle, and the defects found in revisions 0.2.2 and 0.2.3 lived precisely there — in the interaction between publication, attempts, and the catalog, where every individual section was correct. The following lifecycle scenarios are mandatory and are owned by no single section:

1. **Node executed twice.** A node reruns and publishes a second attempt; the catalog remains buildable and resolves to the active attempt only.
2. **Crash between publication and promotion.** A new attempt is published but the generation is never committed; recovery leaves the previous generation intact and fully consistent, and the orphaned attempt contributes no evidence.
3. **Upstream rerun with stale downstream.** An upstream node is recomputed; no committed generation may exist in which new upstream evidence coexists with a downstream strategy derived from the old values.
4. **Stale materialized catalog.** `evidence_catalog.json` is left behind from an earlier generation; it is detected by digest mismatch and rebuilt rather than trusted.
5. **Stale export.** A rendered strategy from an earlier generation is present; presentation refuses it rather than serving it silently.
6. **Cross-implementation ID equality.** Two independent implementations of §11.3 produce identical IDs for a shared key corpus, including non-ASCII dimension values.

7. **First run has no committed generation.** Synthesis executes against a candidate snapshot on a run with no prior committed generation, and publishes successfully.
8. **Candidate never commits.** A candidate generation is abandoned mid-closure; no part of it is observable to presentation, exports, or a later resume.
9. **Superseded candidate.** A strategy is produced against a snapshot that is then superseded; the pre-commit gate rejects it on digest mismatch rather than committing it.

10. **Failed recompute after prior success.** The invalidated node's old attempt
is absent from the candidate; failure cannot turn it into a candidate cache hit.
The previous committed generation remains inspectable and user-facing status
reports the failed update.
11. **Complete node-state map.** Every plan node is present in a committed
generation as `success`, `failed`, `skipped`, or `inactive`; no state is inferred
from a missing key.
12. **Synthesis self-publication.** Synthesis publishes into its candidate
without invalidating its own `input_snapshot_digest`; the full
`snapshot_digest` may change and is not used by the pre-publication gate.
13. **Cardinality and external binding matrix.** `one` and `many`,
required and optional ports receive exactly the shapes in §12, with typed
external contracts and stable ordering.
14. **Investigation in one candidate.** A round-0 request activates a
predeclared node, produces evidence, rebuilds context, and commits a round-1
strategy in the same generation and with the same plan digest.
15. **Rejected and repeated investigation.** Unsupported or over-budget
requests become typed decisions and uncertainties; a round-1 request cannot
activate more work or recurse.
16. **Cross-node provenance.** Two nodes using the same module publish distinct
records whose `producer_node_id` matches their fragment and manifest while the
semantic ID remains producer-independent.

Scenarios 1 through 3, 7 through 8, and 10 through 15 must exist before Phase 4
resume or Phase 6 additional-investigation work begins. Scenario 6 is a gate for
`V6-P0-001`.

---

## 30. Security and safety

- downloaded notebook and repository code is parse-only and never executed;
- archive extraction rejects path traversal and enforces size/file limits;
- dataset reads are confined to a resolved dataset root;
- every network-capable component is declared;
- tests are offline by default;
- tokens, passwords, headers, and provider secrets are never logged or serialized;
- prompts and raw responses are stored only under explicit privacy configuration;
- errors are sanitized before publication;
- destructive cleanup validates exact targets under the configured run root;
- Synthesis cannot issue shell, Python, SQL, or notebook code as an investigation request.

---

## 31. Observability

The runner emits compact structured events:

```text
node_started
input_validated
checkpoint_reused
bundle_published
evidence_registered
node_degraded
node_failed
node_skipped
node_completed
```

Events contain run ID, node/module ID, code version, duration, safe row/sample counts, diagnostic codes, checkpoint status, and published artifact IDs. Logs are not evidence and cannot be cited by Synthesis.

---

## 32. Legacy v5 compatibility

Compatibility is one-way for production:

```text
v6 canonical results -> v5 export views
```

`LegacyV5EdaPackExporter` may build `eda_evidence_pack_v5.json` for comparison or downstream transition. The v6 core must never consume that pack.

Temporary v5 import adapters are allowed only for migration fixtures. Imported records carry `origin="legacy_v5_import"`. Ambiguous physical evidence paths are not automatically promoted to v6 IDs.

Legacy import support may be removed when:

- active modules publish v6 results;
- Final Synthesis consumes only `SynthesisContext`;
- representative golden fixtures exist in v6 format;
- no supported command requires a v5 pack as input.

---

## 33. Migration plan

Migration follows vertical slices. v5 receives only critical fixes while a slice is being replaced.

### Phase 0 — Freeze and characterize v5

- capture representative full and stage-level fixtures;
- inventory current public and accidental contracts;
- record known allowed-ref and pack drift failures;
- identify reusable pure algorithms;
- freeze new cross-stage fields.

Exit: current behavior is reproducible offline from fixtures.

### Phase 1 — Minimal contract and evidence kernel

- common contracts and integer version rules;
- artifact refs for JSON, table, and blob outputs;
- `EvidenceKey`, `EvidenceRecord`, catalog, and catalog view;
- deterministic ID construction and atomic fragment registration;
- typed errors.

Exit: duplicate, broken, cross-scope, and physical-path refs are rejected without dataset or LLM dependencies.

### Phase 2 — First vertical slice

Implement or adapt:

```text
dataset gateway -> inventory -> schema -> submission -> profile
```

Add the small runner, node bundles, and resume only to the degree required by this slice.

Exit: the slice runs independently, passes fixtures, publishes valid evidence, and does not use `EdaEvidencePack` internally.

### Phase 3 — Decision slice

Implement or adapt:

```text
metric -> validation -> leakage -> hypothesis evaluation
```

Exit: IID, grouped, temporal, panel, ranking, and multi-output fixtures produce contract-valid decisions; the time-column rule has one implementation.

### Phase 4 — Research boundary

- v6 Research Scout output;
- capability registry;
- `ResearchToDatasetAdapter`;
- preflight rejection before dataset IO.

Exit: every hypothesis produces exactly one result and unknown capabilities fail early.

### Phase 5 — Synthesis boundary

This is the heaviest migration step in the project and is deliberately split. In the v5 branch the synthesis layer addresses evidence by pack key (`final_synthesizer.py` and `deterministic_strategy.py` together are roughly a quarter-million characters of code reading `evidence_pack.get("<name>_evidence")`), and a standalone `generate_allowed_evidence_refs(pack)` already exists. Treating this as one step invites a half-finished state in which two allowed-reference sources coexist — which is precisely what §11.6 forbids.

**Phase 5a — Evidence ID indirection.**

- introduce `EvidenceCatalog` and `EvidenceCatalogView` *alongside* the existing pack, without deleting it;
- register the same values under contract-namespaced IDs (§11.3);
- migrate consumers from `pack.get("x_evidence")` to `catalog.resolve(evidence_id)` **one consumer at a time**, each under a golden-equivalence test proving identical output before and after;
- replace `generate_allowed_evidence_refs(pack)` with `EvidenceCatalogView.allowed_refs()` only after **all** consumers are migrated, so a second generator never exists even temporarily.

Exit: `grep -rn 'generate_allowed_evidence_refs' src/` returns nothing, and reference validation and prompt rendering demonstrably share one view instance.

**Phase 5b — Retire the pack as an input.**

- demote the legacy pack to an export view (§32), written but never read by core modules;
- typed claims, `StrategyConstraint` reconciliation, and the Final Strategy publication gate;
- structural offline golden tests comparing claim structure and refs, never prose.

Exit: `grep -rnE 'evidence_pack\.get|\.get\("[a-z_]+_evidence"\)' src/synthesis/ src/reasoning/` returns nothing, no independent allowed-ref generator exists, and invalid refs cannot reach rendering.

### Phase 6 — Additional-investigation loop

- whitelisted request contract;
- deterministic validator and budgets;
- one bounded additional round;
- deduplication and termination rules.

Exit: missing evidence can trigger targeted deterministic analysis without arbitrary execution or recursion.

### Phase 7 — Optional modules and presentation

Migrate relationships, drift, baseline, feature probes, extended leakage, and other optional modules one at a time. Render Markdown and DOCX only from validated Final Strategy. Add v5 exports.

Exit: each optional module has independent fixtures and its failure does not corrupt core output.

### Phase 8 — Default switch and cleanup

- run v5/v6 comparisons on representative fixtures;
- make v6 the default CLI;
- retain v5 behind an explicit temporary legacy command;
- apply retention and GC policy;
- remove v5 internal dependencies after exit conditions hold.

---

## 34. MVP definition

The MVP is defined as ordered slices rather than one flat list, so that a usable end-to-end system exists before operational and investigative features are added. Each slice is independently testable and leaves the system in a working state.

**Slice 1 — Kernel.** Public contract kernel, JSON/Table/Blob refs, publication protocol (§10.5), canonical evidence catalog, local dataset gateway, inventory and schema modules.

**Slice 2 — Decision.** Submission, metric, validation, and basic leakage modules; `StrategyConstraint` emission; hypothesis evaluation.

**Slice 3 — End to end.** Research adapter, Dataset Intelligence composition, Synthesis adapter and bounded context, strict Final Strategy publication, Markdown rendering. The first genuinely usable release ends here.

**Slice 4 — Operational.** Small in-process sequential runner with checkpoint publication and targeted resume, retention classes, GC dry-run, broader integration fixtures.

**Slice 5 — Investigation.** The bounded additional-investigation round and optional capabilities.

Deferred out of the first usable release, and explicitly not cut permanently:

- the additional-investigation loop, which is valuable but is not required for a first strategy to be produced;
- automatic garbage collection; retention metadata and `--dry-run` inspection ship earlier, deletion later;
- part of the ten integration fixture classes — they remain **mandatory before the default switch** in Phase 8, but do not gate slice 3.

Not required for the MVP at all:

- distributed execution;
- external orchestration framework;
- runtime UI or daemon;
- relationship, adversarial drift, trained baseline, or feature probes;
- DOCX rendering;
- complete source-retrieval rewrite;
- remote artifact storage;
- more than one additional-investigation round;
- notebook or repository code execution.

---

## 35. System acceptance criteria

v6 is accepted only when:

1. No analytical module receives or mutates a global result state.
2. Every cross-module boundary has a versioned public input and output contract.
3. Same-run consumers receive typed in-memory results; disk reread is not required between nodes.
4. JSON, table, and blob artifacts have distinct refs and validation rules.
5. Artifact identity is not a recursive content address; file hashes protect integrity.
6. Exactly one catalog implementation owns ID construction, registration, selection, and validation.
7. Public evidence refs contain no physical paths or array indexes.
8. Prompt construction and response validation use the same catalog view.
9. Broken evidence refs and invalid required contracts are blocking errors, not warnings.
10. The runner contains no metric, validation, leakage, or competition rules.
11. Invalid Research → Dataset input fails before dataset IO.
12. A Synthesis-only config change does not invalidate Dataset Intelligence checkpoints.
13. Optional module failure cannot corrupt core artifacts or evidence.
14. Final reports derive only from validated Final Strategy.
15. Golden Synthesis tests compare structure and refs, not exact prose.
16. Numerical modules declare seed and tolerance policies.
17. Additional investigation is whitelisted, budgeted, deduplicated, and bounded.
18. v5 compatibility is implemented only through adapters and exporters.
19. Core tests run without external databases, Kaggle, vLLM, or live LLM calls.
20. No downloaded code is executed.
21. All ten integration fixture classes are covered before default switch.
22. Unpinned run artifacts have an explicit inspectable retention/GC policy.
23. Public contracts reject unknown fields, use integer versions, make every public shape change a version change, and permit cross-version reads only through tested explicit adapters.
24. Replacing one module implementation does not require edits to non-adjacent consumers.
25. Evidence identity derives from the evidence contract namespace, never from the producing module; swapping a conforming implementation changes no `evidence_id`.
26. One catalog belongs to exactly one run and competition; no claim resolves a reference across runs.
27. `EvidenceRecord` carries an explicit `domain`, and no claim uses source-domain evidence to prove an unmeasured dataset fact.
28. Every `quality="derived"` record names its parents and a registered derivation method.
29. `ModuleResult` carries typed limitations, and the complete/degraded/failed invariant is asserted by contract tests.
30. Execution severity and finding severity are separate types and are never assigned from one another.
31. Publication survives a crash at any step: no orphan bundle is visible, no catalog entry references an unpublished bundle, and the catalog can be rebuilt from published fragments alone.
32. Every module input is a named port with declared cardinality, optionality, and accepted versions.
33. An edge may escalate an optional dependency to blocking and may never downgrade a blocking one.
34. Critical constraints are reconciled structurally; no gate depends on judging whether prose contradicts a diagnostic.
35. Every whitelisted investigation capability has a typed parameter model and a mandatory expected evidence key.
36. No successor to the legacy pack contains two live fields for one concept; deprecated fields are written only by the legacy exporter.
37. Re-executing a node twice in one run leaves the catalog buildable: at most one attempt per node is active, and superseded attempts contribute no evidence.
38. Promoting an attempt is atomic; every successful committed node selects
exactly one attempt, while failed, skipped, and inactive nodes select none.
39. Publication never rewrites a file in place; every state transition is a rename.
40. The evidence ID algorithm is specified to the byte, and two independent implementations produce identical IDs for identical keys.
41. Every evidence namespace has exactly one declaring producer in a plan.
42. Preflight proves port compatibility from named contracts and versions alone, without executing modules.
43. Envelope, output-contract, and module-contract versions are independent fields and no code derives one from another.
44. Every claim declares its subject domain, and no gate infers the subject from prose.
45. Every serious or critical constraint has exactly one resolution, and a satisfied disposition is verified against a typed decision rather than narrative text.
46. Attempt promotion is committed per generation over the whole dependency closure; no committed generation mixes new upstream evidence with downstream results derived from earlier values.
47. A materialized catalog or export whose generation or digest disagrees with the committed manifest is rebuilt or refused, never served.
48. Nodes, not modules, are the unit of instantiation, configuration, and attempt identity; one module may appear in a plan more than once.
49. Every edge preflight-matched is deliverable at runtime; no port exists that the runner cannot connect.
50. Every constraint and upstream issue the gate requires is present in the same validated context that built the prompt.
51. Every constraint rule resolves through the registry to one decision field and one comparator.
52. Critical upstream diagnostics and limitations are reconciled structurally; no gate infers preservation from prose.
53. Exactly one canonical evidence catalog exists per run; source evidence is a domain within it, not a second store.
54. Artifact identity includes the attempt, so two attempts of one node are never one artifact.
55. Synthesis executes against a candidate snapshot and never requires its own output to already be committed; a first run with no committed generation completes normally.
56. No candidate generation is observable outside the run until commit, and an abandoned candidate leaves no trace in exports, presentation, or resume.
57. Every digest has a stated input; no digest is computed over a structure containing itself.
58. `NodeManifest` and `EvidenceFragment` are versioned contracts, and recovery reads them rather than inferring bundle state from filenames.
59. Reconciliation is keyed on instance identifiers, so two occurrences of one code are resolved independently.
60. Every `UpstreamIssue` severity is derived from a typed source field, never assigned by convention.
61. A dataset-domain fact or inference cites at least one dataset-domain evidence record.
62. Every input port satisfies the exact `one`/`many` and required/optional
cardinality matrix; edge and binding sources are never mixed on one port, and
multi-provider inputs arrive in deterministic order.
63. An investigation round commits inside one candidate generation over its
dependency closure and never appends evidence outside one.
64. Every candidate and committed generation contains one explicit
`NodeGenerationState` for every plan node; missing entries never mean inactive,
failed, or skipped.
65. `plan.json` is immutable after validation; `plan_id` and `plan_digest` match
every generation, manifest, node bundle, and evidence fragment in the run.
66. Every external input binding is a typed discriminated contract with an
explicit source contract and schema version; raw config and path strings are not
plan bindings.
67. Every evidence record names `producer_node_id`, which matches its enclosing
fragment and node manifest but is not an input to evidence identity.
68. Node output provenance is bound to a typed `input_snapshot_digest`; the
full candidate `snapshot_digest` may change after publication and is never used
as the Synthesis pre-publication comparison.
69. The Final Strategy publication gate is implementable as the ordered
structural checks in §23.1 and requires no interpretation of prose.
70. A failed recompute cannot reactivate or reuse the invalidated node's prior
attempt inside its candidate; the prior attempt remains visible only through
the previous committed generation.
71. Additional investigation changes node activation and generation state,
never the immutable plan body, and a round-1 request cannot recurse.
72. Every core hash, digest, fingerprint, and artifact ID uses the preimage and
algorithm registry in §10.4.1, with executable golden vectors.

---

## 36. Forbidden anti-patterns

```python
# Shared expanding result state
context["eda"]["validation"]["result"] = result

# Module reads and writes the same partially-built pack
# (assembly order silently becomes a contract)
pack["strategy_hints"] = build_hints(pack)

# Consumer reconstructs allowed evidence
allowed_refs = generate_allowed_evidence_refs(eda_pack)

# Synthesis addresses evidence by pack key instead of catalog ID
baseline = evidence_pack.get("baseline_ablation_evidence")

# Two live fields for one concept
pack.strategy_hints = hints
pack.eda_strategy_hints = hints

# Physical path used as a public ref
evidence_ref = "leakage_evidence[0].evidence.overlap_count"

# Evidence identity bound to the producing implementation
evidence_id = f"{producer_module}.{key}"

# Universal bulk-data JSON envelope
ArtifactEnvelope(payload=dataframe.to_dict())

# Runner contains domain policy
if metric.name == "gini_stability":
    validation = "temporal"

# Module imports a sibling's private implementation model
from dataset_intelligence.modules.schema_inferer import _InternalCandidate

# Warning replaces an invalid required contract
warnings.append("target is missing")

# LLM repairs deterministic data or schema mismatch
fixed_payload = await llm.repair_contract(payload)

# Derived export becomes an internal input
run_validation(load_json("eda_evidence_pack_v5.json"))

# Global config fingerprint invalidates every node
cache_key = hash_entire_pipeline_config(config)

# Missing evidence causes unlimited recursive analysis
while synthesis.requests_more_data():
    run_arbitrary_code(synthesis.request)
```

---

## 37. Definition of done for implementation tasks

Every task must state:

- files in scope;
- the public contract or invariant affected;
- acceptance criteria;
- focused unit tests, including at least one negative case;
- one offline verification command;
- confirmation that no second source of evidence truth was introduced.

Add the following only when applicable:

- fixture and golden updates for observable behavior changes;
- explicit version adapter for public schema changes;
- compatibility and migration notes;
- retention or resource-budget impact;
- documentation for user-visible CLI changes.

Unrelated refactoring is outside task scope.

---

## 38. Final architectural invariant

KaggleResearcher v6 is a modular monolith, not a distributed system simulated on local disk and not a loose collection of functions connected by convention.

Its balance is intentional:

- public module boundaries are strict and runtime-validated;
- private computation remains ordinary typed Python;
- same-run data flows in memory;
- expensive and externally consumed results are checkpointed;
- bulk data uses table/blob storage rather than JSON payloads;
- one catalog owns evidence identity;
- one small runner coordinates the graph;
- Final Synthesis can ask for limited additional evidence but cannot execute arbitrary analysis.

If a module replacement forces changes in Final Synthesis, the runner, presentation, or unrelated modules, the boundary is broken. If adding a module requires building new generic infrastructure before its domain value can be tested, the design is overextended.

One clarification on immutability, since `frozen=True` is easy to over-trust: Pydantic freezes attribute rebinding, not the interiors of nested containers, and `Mapping` is a typing-level promise that an implementation may satisfy with a plain `dict`. Neither gives deep immutability in Python, and this specification does not claim it does.

What is guaranteed instead is narrower and actually enforceable: a contract that has crossed a persistence boundary is **not mutated after validation**, and its canonical serialization is stable — the same logical content produces the same bytes and the same hash. Sequence fields on persisted contracts use `tuple[...]` because it costs nothing and removes the most common accidental mutation. Free-form fields such as `Diagnostic.details` and `EvidenceKey.dimensions` remain mapping-typed for ergonomics; they are covered by the no-mutation-after-validation rule and by canonical serialization, not by a false promise of frozen interiors. Contract tests assert stable re-serialization, which catches real mutation; they do not attempt to prove object graphs are immutable.

---

## Appendix A — Architectural decision log for revision 0.2

| Critique item | Decision in v0.2 | Result |
|---|---|---|
| Evidence and bulk data were mixed | Accepted | Separate JSON, Table, and Blob refs; Parquet/Arrow for tables |
| Universal determinism claim was untestable | Accepted | Three-level determinism budget and structural LLM goldens |
| Config fingerprint scope was undefined | Accepted | Module-specific typed config and targeted invalidation |
| Immutable outputs had no retention policy | Accepted | Retention classes, pinned runs, recovery cleanup, safe GC |
| Synthesis could not request missing evidence | Accepted with bounds | One whitelisted, budgeted additional-investigation round |
| Semantic IDs mixed names and dimensions | Accepted | `EvidenceKey(name, dimensions)` plus opaque catalog ID |
| SemVer conflicted with `extra="forbid"` | Accepted as explicit trade-off | Integer versions, no forward compatibility, coordinated producer/consumer updates, adapters only for declared cross-version reads (§9.2) |
| Runtime resembled a workflow platform | Accepted | Small sequential in-process DAG runner with five statuses |
| Dagster should be mandatory | Deferred | No framework dependency in MVP; reevaluate on operational triggers |
| Output hashes should be removed | Partially accepted | No CAS/Merkle identity; byte hash retained for integrity and resume |
| Pydantic should exist only at LLM boundaries | Rejected | Pydantic retained on public module boundaries and checkpoints |
| Artifacts should be replaced by shared state | Rejected | Typed in-memory handoffs plus selective checkpoints; no global result state |
| DoD was too heavy | Accepted | Six core requirements; compatibility/docs/fixtures only when applicable |
| Time-column rule was repeated | Accepted editorially | One canonical validation-policy rule |

Two of these accepted decisions carry a deferred cost that is named here so it does not resurface as a surprise. First, the sequential runner (§18.1) trades away intra-run parallelism: independent DAG branches such as `leakage` and `drift` run one after another even when nothing forces an order. This is acceptable at MVP scale and dataset sizes; it becomes a latency ceiling as data grows. Second, the framework decision is deferred rather than settled, and its reevaluation triggers are explicit, not vague: a demonstrated need for parallel execution of independent nodes, daemon scheduling, a multi-run debugging UI, or distributed execution. When any of those becomes a real operational requirement, adopting an established runner (for example Dagster) is preferable to growing the in-process runner into one.

Revision 0.2 therefore targets the middle ground deliberately: stronger boundaries than an informal function pipeline, substantially less infrastructure than the original v6 draft.

---

## Appendix B — Decision log for revision 0.2.2

> **Historical record.** This appendix documents what revision 0.2.2 decided and why. Several of its decisions were superseded by revision 0.2.3 — notably `acknowledged_diagnostic_codes`, which no longer exists. Implement from the numbered sections and Appendix C, not from this table.

Revision 0.2.2 reconciles two independent reviews of 0.2. Review **A** was an architecture and migration review grounded in the actual `kaggle_eda` branch; review **B** was a contract-kernel implementability review. They did not conflict: A established that the direction is right and priced the migration, B established that the contract kernel was not yet unambiguous enough to hand to implementation agents. Both verdicts are accepted.

| Finding | Source | Priority | Decision in v0.2.2 |
|---|---|---|---|
| Compatibility trade-off named as mechanism, not policy | A, B (independently) | P0 | Operational policy and behavior matrix (§9.2) |
| `evidence_id` derived from producer namespace contradicts module replaceability | B | P0 | Identity from evidence contract namespace; producer demoted to provenance (§11.3) |
| Catalog uniqueness scope was ambiguous | B | P0 | One catalog per `(run_id, competition_id)`; cross-run resolution forbidden (§11.4) |
| `degraded` required limitations that `ModuleResult` had no field for | B | P0 | `Limitation` type, `limitations` field, formal complete/degraded/failed invariant (§9) |
| One severity scale conflated execution failure and analytical finding | B | P0 | `DiagnosticSeverity` and `FindingSeverity` separated (§9) |
| Evidence provenance insufficient for the claim gates it promised | B | P0 | `domain`, `derived_from_refs`, `derivation_method`, registration rules (§11.2, §11.5) |
| Atomic publication promised across two resources without a protocol | B | P0 | Ordered staging → `prepared` → rename → `published` → recovery; catalog as materialized view (§10.5) |
| `requires_capabilities` too weak for multi-input modules | B | P1 | `InputPortSpec` with name, contract, versions, optionality, cardinality (§12) |
| Obligation modeled as a module property only | B | P1 | Edge-level override with escalate-only conflict rule (§12) |
| One version number serving three purposes | B | P1 | Contract / implementation / cache-fingerprint versions separated (§12, §12.1) |
| Contradiction gate required semantic judging | B | P1 | `StrategyConstraint` plus `acknowledged_diagnostic_codes`; structural reconciliation (§23, §23.1) |
| `EvidenceCatalogView` used but never defined for prompts | B | P1 | Typed projection with deterministic rendering rules (§22) |
| Investigation request could not yield its own dedup key | B | P1 | Mandatory `expected_key`, typed `CapabilityParameters`, `SynthesisAttemptResult` (§24) |
| Phase 5 hid the project's largest refactor in one step | A | P0 | Split into 5a indirection and 5b retirement, with grep exit conditions (§33) |
| Legacy dotted paths had no promotion rule | A | P1 | Declared mapping table; unmapped paths are blocking errors (§11.7) |
| Self-feeding partial pack not listed as an anti-pattern | A | P1 | Added, with the pack-key and duplicate-field cases observed in v5 (§36) |
| Duplicate representations of one concept in the legacy pack | A | P1 | Acceptance criterion 36 |
| Table backend priced as a new dependency | A | P2 | Noted that polars/pyarrow already ship in v5 (§10.2) |
| MVP still too large for a first usable slice | B | P1 | Five ordered slices; investigation loop and auto-GC deferred (§34) |
| `frozen=True` treated as deep immutability | B | P2 | `tuple`/`Mapping` required on persistence-crossing contracts (§38) |

Two review recommendations were **not** adopted as stated. The full input-port contract was proposed as a P0 blocker; it is adopted in form but treated as P1 in sequencing, because the current graph is nearly linear and multi-input modules are rare — building the complete port model before slice 1 would repeat the over-engineering this revision exists to remove. Deep immutability everywhere was likewise reduced to P2 and scoped to persistence boundaries, since in-process typed handoffs (§5.2) carry little mutation risk and blanket `tuple`/`Mapping` conversion is a real ergonomic tax on implementation agents.

The combined effect is narrow by design. Revision 0.2 fixed the architecture; 0.2.2 makes the contract kernel unambiguous enough to distribute as independent implementation tasks. No module boundary, package layout, or execution model changed. Phase 0 and Phase 1 can now begin without architectural risk, and the remaining open questions are implementation choices rather than specification gaps.

---

## Appendix C — Decision log for revision 0.2.3

> **Historical record.** Superseded in part by revision 0.2.4; implement from the numbered sections and Appendix D.

Revision 0.2.2 closed the architecture-level findings. A third review then checked whether the contract kernel was literally implementable, and found six blocking defects. Four were introduced by 0.2.2 itself, one was made sharper by it, and one was a residual inconsistency. All six are accepted without argument; a specification that cannot be implemented as written is not a specification, and the earlier claim that the kernel was "unambiguous enough to distribute" was premature.

| Finding | Origin | Decision in v0.2.3 |
|---|---|---|
| Node re-execution collides with catalog rebuild | Latent in 0.2, made fatal by 0.2.2 making the catalog a materialized view over all published bundles | `RunManifest.active_attempts`; catalog built only from active attempts; atomic promotion (§10.6) |
| Evidence contract namespace existed only in prose | Introduced by 0.2.2 | `EvidenceKey.name` **is** the namespace; `provides_evidence_namespaces` on `ModuleSpec` makes uniqueness checkable (§11.3) |
| ID derivation said "deterministic" without an algorithm | Introduced by 0.2.2 | Byte-level algorithm: NFC, name grammar, sorted dimensions, canonical JSON, blake2b, fixed suffix, explicit collision handling (§11.3) |
| Input ports typed, output ports not | Introduced by 0.2.2 | `OutputPortSpec`, `producer_port` on edges, `provides_capabilities` removed as a duplicate fact (§12) |
| Three version numbers could contradict each other | Introduced by 0.2.2 | Envelope / output-contract / module-contract versions separated with a table and a rule for which one preflight compares (§12) |
| Claim gate still required reading prose | Partially introduced by 0.2.2 | `Claim.subject_domain`; `ConstraintResolution` replacing two unlinked lists; typed `validation_decision` as the comparison target (§23, §23.1) |
| Manifest rewrite in place is not atomic | Introduced by 0.2.2 | Immutable `PUBLISHED` marker via temp-file rename; prepared manifest never mutated (§10.5) |
| `SynthesisAttemptResult` union was not constructible | Introduced by 0.2.2 | Explicit `StrategyAttempt` / `InvestigationAttempt` wrappers carrying the discriminator (§24) |
| "All public models have integer versions" was false | Pre-existing, worsened by new models | Rule scoped to persisted top-level contracts; nested types versioned by their container (§9) |
| Deep immutability was over-promised | Introduced by 0.2.2 | Promise narrowed to no-mutation-after-validation plus stable canonical serialization (§38) |
| Resume matched code version while caching used a fingerprint | Pre-existing | Resume keyed on `cache_fingerprint_version` only (§18.5) |
| §25.4 still described contradiction detection | Left stale by 0.2.2 | Rewritten to subject-domain and constraint reconciliation |

Two observations about this revision are worth recording, because they say something about how the document should be reviewed going forward.

First, most defects in 0.2.2 were of one kind: a concept was named in prose and never given a field, or a type was sketched without checking that it could be constructed. Architectural review does not catch this class — only reading the contracts as if compiling them does. Any future revision that introduces a new concept should be checked by asking, for each named thing, which field holds it and which test asserts it.

Second, the sharpest finding — attempt collision — was not a defect in any single section. Immutable bundles, contract-scoped evidence identity, catalog rebuild from published fragments, and rerun-creates-an-attempt are each individually correct and were introduced across three different revisions. They became contradictory only in combination. Section-local review cannot find that; it requires tracing one operation, in this case "run a node twice," end to end. The integration fixtures in §29.6 should include exactly that scenario.

Revision 0.2.3 changes no module boundary, package layout, or execution model. It makes the kernel constructible. Phase 0 and Phase 1 can now be distributed as independent implementation tasks, with the caveat that §29.6 must cover node re-execution before Phase 4 resume work begins.

---

## Appendix D — Decision log for revision 0.2.4

> **Historical record.** Superseded in part by revision 0.2.5; implement from the numbered sections and Appendix E.

A fourth review accepted the architecture as fixable-in-place and found six remaining blocking defects. Three were introduced by 0.2.3, two were long-standing gaps that earlier revisions had stepped around, and one was a literal bug in the ID algorithm that 0.2.3 introduced while claiming to remove ambiguity.

| Finding | Origin | Decision in v0.2.4 |
|---|---|---|
| Per-node promotion allows a run mixing new upstream and stale downstream | Introduced by 0.2.3 | `RunManifest.generation`; promotion commits the whole dependency closure in one atomic replacement (§10.6) |
| Materialized catalog and exports could outlive the manifest they were built from | Introduced by 0.2.3 | Generation and digest recorded and checked; stale views rebuilt, stale exports refused (§10.6) |
| "Exactly one active attempt" is false before a node's first run | Introduced by 0.2.3 | Restated as at most one active, exactly one per successfully published node in the committed generation (§10.6) |
| `PipelinePlan` and `NodeSpec` referenced but never defined | Long-standing | Both defined; edges connect nodes, configuration belongs to a node (§12) |
| Ports comparable at preflight but not deliverable at runtime | Introduced by 0.2.3 | One output port per node in the MVP; `PortBundle` named as the deferred extension; selectors rejected (§12) |
| Constraints demanded by the gate never reached Synthesis | Introduced by 0.2.3 | `constraints` and `upstream_issues` on `SynthesisContext`, under the same single-source rule as evidence (§22) |
| `StrategyConstraint.rule` had no defined meaning | Introduced by 0.2.3 | Rule registry binding rule to decision field and comparator (§23) |
| Diagnostics and limitations still checked by reading prose | Long-standing, half-fixed by 0.2.3 | `UpstreamIssue` and `UpstreamIssueResolution`; prose limitations kept for readers only (§23, §23.1) |
| ID grammar rejected the document's own example | Introduced by 0.2.3 | Separate name and dimension-key grammars, both allowing underscores in the first segment (§11.3) |
| Canonical JSON left `ensure_ascii` undefined | Introduced by 0.2.3 | Serialization settings stated field by field (§11.3) |
| 32-bit digest was needlessly narrow | Introduced by 0.2.3 | Widened to 48 bits (§11.3) |
| An undefined second catalog reappeared in §16 | Long-standing | Source evidence is `domain="source"` in the one canonical run catalog; no separate store (§16) |
| `artifact_id` omitted the attempt | Introduced by 0.2.3 | Attempt included in artifact identity (§10.4) |
| Synthesis attempt wrappers had no version | Introduced by 0.2.3 | `schema_version` added (§24) |
| Remaining persisted contracts still used `list` | Introduced by 0.2.3 | `Claim`, `FinalStrategy`, and `SynthesisContext` converted to tuples (§22, §23) |
| Header miscounted reviews | Editorial | Corrected |

Two structural changes came out of this round rather than out of any single finding.

`ConstraintResolution` moved from `Claim` to `FinalStrategy`. Per-claim resolutions would let one constraint be resolved differently in two sections, and would leave the gate with no single place to check — a duplicate representation of one fact, which acceptance criterion 36 forbids.

The integration fixtures in §29.6 gained six lifecycle scenarios. Every defect in the last two revisions lived between sections rather than inside one: publication, attempts, identity, and catalog rebuild were individually correct and jointly contradictory. Data fixtures cannot find that class; scenario tests that trace one operation end to end can. Scenarios 1 through 3 are prerequisites for Phase 4.

Revision 0.2.4 changes no module boundary, package layout, or execution model. With these closures the three remaining seams named by the fourth review — canonical snapshot, node/port runtime contract, and actual delivery of constraints and upstream issues — are closed at the contract level. The kernel may be frozen and Phase 0–2 tasks distributed, on one condition worth stating plainly: the lifecycle scenarios in §29.6 are what keep the next seam from being found in production instead of in review.

---

## Appendix E — Decision log for revision 0.2.5

> **Historical record.** Superseded in part by revision 0.2.6; implement from
> the numbered sections and Appendix F.

The fifth review confirmed the architecture as settled and found six remaining implementability gaps plus one blocker scheduled for a later phase. Five of the six were introduced by 0.2.4, and one of those is worth naming precisely: the generation protocol added in 0.2.4 to fix mixed-snapshot runs itself created a circular dependency, because Synthesis is a member of every closure it must read the results of. This is the second consecutive revision in which a correct fix produced the next defect, which is not an argument against the fixes but an argument for the scenario tests in §29.6.

| Finding | Origin | Decision in v0.2.5 |
|---|---|---|
| Generation commit was circular; Synthesis could not read its own closure | Introduced by 0.2.4 | Candidate and committed snapshots separated; closure nodes read the candidate, readers see the committed (§10.7) |
| First run had no committed generation to read | Introduced by 0.2.4 | Same split; a candidate exists from the first node onward |
| Pre-commit gate compared against the wrong manifest | Introduced by 0.2.4 | Gate compares `target_generation`; presentation compares the committed generation (§23.1, §10.7) |
| Synthesis was never told the generation it must return | Introduced by 0.2.4 | `target_generation` and `input_snapshot_digest` on `SynthesisContext` (§22) |
| `manifest_digest` hashed a structure containing itself | Introduced by 0.2.4 | Digest defined over the manifest minus the digest field, canonical JSON per §11.3 (§10.6) |
| `EvidenceCatalogView` could not be checked against run state | Introduced by 0.2.4 | View carries generation, snapshot status, and digest (§22) |
| `NodeManifest` and `EvidenceFragment` were undefined | Long-standing | Both defined as versioned contracts (§10.5) |
| Reconciliation keyed on codes, which repeat | Introduced by 0.2.4 | `constraint_id` and `issue_id` as instance keys; `code` remains classification (§23) |
| `Limitation` had no severity source for `UpstreamIssue` | Introduced by 0.2.4 | `Limitation.severity`; normative derivation rule (§9, §23) |
| Dataset claim rule was bypassable via source + system | Introduced by 0.2.3 | Restated positively: dataset claims require dataset-domain evidence (§23) |
| Input port runtime delivery undefined | Introduced by 0.2.4 | Normative port-to-field mapping, deterministic `many` ordering, `PlanInputBinding` for external inputs (§12) |
| Investigation loop contradicted the static plan | Long-standing, sharpened by the generation protocol | `activation="on_request"` nodes pre-declared in the plan; a granted request commits as an ordinary generation (§24) |
| `dataset run --plan` had undefined run scope | Long-standing | It continues the originating run with a new generation; it does not import fragments into a new run (§24) |

One pattern is now clear enough to state as guidance rather than as a finding. Every defect in revisions 0.2.2 through 0.2.5 fell into one of two classes: a concept named in prose without a field to hold it, or a rule that was locally correct but globally circular. The first class is caught by reading contracts as if compiling them. The second is caught only by tracing a complete operation — publish, recompute, commit, render — across every section that touches it. Neither is caught by reviewing sections in isolation, which is why §29.6 now carries nine lifecycle scenarios and why those scenarios, not the section text, are the real specification of the run lifecycle.

Revision 0.2.5 changes no module boundary, package layout, or execution model. The kernel is now constructible end to end: from the first node's `NodeManifest` through candidate snapshot, Synthesis, pre-commit gate, atomic commit, and presentation. Status may move to implementation-ready and Phase 0–2 tasks may be distributed independently, with §29.6 scenarios 1–3 and 7–8 as the gate on Phase 4 rather than as optional coverage.

---

## Appendix F — Decision log for revision 0.2.6

The v6 baseline audit compared revision 0.2.5 with the `kaggle_eda` codebase and
found that the architecture was coherent but still not safe to distribute as
independent code tasks. The remaining gaps were not new module boundaries; they
were missing terminal-state, identity, and lifecycle rules at the exact seams
the first implementation tasks would touch.

| Baseline finding | Decision in v0.2.6 |
|---|---|
| A generation did not represent every plan node, especially an unrequested investigation node | `NodeGenerationState` is total over the plan and distinguishes `pending`, `running`, `success`, `failed`, `skipped`, and `inactive` (§10.6, §18.2) |
| A failed recompute could be confused with reuse of the prior success | Invalidated attempts are removed from the candidate; an old success survives only in the previous committed generation, and failed-update status is explicit (§10.6, §18.4) |
| `plan_id` named a plan but did not prove its body or define its lifecycle | Immutable `plan.json`, normative `plan_digest`, propagation into every run artifact, and new-run requirement for a changed plan (§12.0.1) |
| Port cardinality prose handled `one` but not all `many` cases | Exact four-row cardinality matrix, no mixed edge/binding sources, and deterministic ordering (§12) |
| `PlanInputBinding.source` was an untyped enum | Discriminated external bindings with source contract and schema version; raw config bindings removed (§12) |
| Digests reused phrases such as "canonical JSON" without one shared byte contract | Canonical bytes and the complete core digest/preimage registry (§10.4.1) |
| Full candidate digest and Synthesis input digest were conflated, recreating self-reference when Synthesis published | Per-node `InputSnapshotPreimage`; `input_snapshot_digest` is distinct from the changing generation `snapshot_digest` (§10.7, §12.1, §22) |
| Evidence provenance named a module but not the configured plan node instance | Mandatory `producer_node_id` with fragment/manifest equality checks, excluded from evidence identity (§11.2, §11.5) |
| Publication requirements were still a prose checklist with non-structural recommendation and uncertainty rules | Ordered deterministic gate plus `validation_prerequisites` and `missing_observation` fields (§23, §23.1) |
| Additional investigation said "activate and continue" without a request decision contract or exact generation behavior | Typed decisions, stable plan digest, same-candidate round lifecycle, terminal round behavior, and optional/blocking failure semantics (§24) |

Revision 0.2.6 deliberately does **not** implement the runner, publication,
resume, or investigation loop. It freezes the contracts those later phases must
obey. The first code task remains the isolated evidence identity seam:

```text
EvidenceKey -> canonical evidence-key bytes -> evidence_id
```

That task implements only the special preimage in §11.3 and the relevant shared
UTF-8/NFC helpers from §10.4.1. It does not need `NodeGenerationState`,
`PipelinePlan`, publication, resume, or Synthesis. Its gate is lifecycle
scenario 6 plus unit golden vectors.

With this revision, the canonical document is
`docs/specs/KAGGLE_RESEARCHER_V6_SPEC.md`. Versioned copies may be retained for
review history, but implementation tasks cite this path and the exact numbered
sections they require.
