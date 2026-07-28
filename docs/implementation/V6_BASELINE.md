# KaggleResearcher v6 baseline audit

## 1. Audit identity and source status

| Item | Observed value |
|---|---|
| Branch | `kaggle_eda` |
| Commit | `67d350c5b1e6532ac042ea3cec8fd58991a27890` |
| Audit scope | Contract and integration baseline only; no production or test code changed |
| Rebase/merge state | No `.git/rebase-merge`, `.git/rebase-apply`, `MERGE_HEAD`, `CHERRY_PICK_HEAD`, or `REVERT_HEAD` was present |
| Canonical path requested by the task | `docs/specs/KAGGLE_RESEARCHER_V6_SPEC.md` — absent |
| Specification used by explicit user direction | `docs/specs/KAGGLE_RESEARCHER_V6_SPEC_v0_2_5 (1).md`, version `0.2.5`, draft dated 2026-07-24 |
| Assigned v6 task file | Not found; neither `v6/tasks` nor `docs/v6/tasks` exists |
| Similar task directory | `docs/v6/eda_engine_tasks_40_64/tasks` contains legacy v5 tasks 40–64 and is not a v6 task source under `AGENTS.md` |

The specification used for this audit is an untracked working-tree file. The same
filename is deleted at its former `docs/` location. Those pre-existing changes
belong to the user and were not modified by this audit.

The user's instruction to read the found specification is treated as permission
to complete this documentation-only audit against version 0.2.5. It does not
remove the implementation blocker created by the original minimum-version
requirement (`>=0.2.6`), the missing canonical filename, or the missing assigned
v6 task file.

Applicable v6 areas in the found specification are §§5–13, 16–18, 20, 22–25,
28–29, 32–33, 35, and 37–38. No file under `docs/archive/` was used.

## 2. Existing component map

### 2.1 Contract foundation and identifiers

| Component | Concrete implementation | Baseline finding |
|---|---|---|
| Shared Pydantic base | `kaggle_researcher/contracts/base.py:6`, `ContractModel` | `extra="forbid"` and assignment validation are useful. The model is not frozen, and v6 persistence-boundary immutability/stable reserialization is not expressed as a dedicated v6 base. |
| Logical ID types | `kaggle_researcher/contracts/ids.py:8`, `ContractId`; dynamically created `HypothesisId`, `EdaTaskId`, `ExperimentId`, `EvidenceId`, `StageId` | Runtime-distinct strings are reusable as a pattern. There are no `RunId`, `GenerationId`, `NodeId`, `AttemptId`, or `PlanId`; the current `EvidenceId` validates only non-emptiness and does not implement §11.3 derivation. |
| Version constants | `kaggle_researcher/contracts/versions.py:27`, `CURRENT_SCHEMA_VERSION`; `CURRENT_CONTRACT_VERSIONS` | Uses string versions such as `"1.0"`/`"2.0`; v6 top-level persisted contracts require positive integer `schema_version`. |
| Dispatch registry | `kaggle_researcher/contracts/registry.py:40`, `ContractHeader`; `ContractRegistry.resolve` | `ContractHeader.model_config` has `extra="ignore"` so it is an intentional permissive dispatch parser, not a v6 public contract. The registry is keyed by string versions and registers v5 contracts. |
| Contract definition checks | `kaggle_researcher/contracts/artifacts.py`, `validate_contract_definitions` | Useful characterization machinery, but its approved contract set and version assumptions are v5-specific. |

### 2.2 Canonical serialization and artifacts

| Component | Concrete implementation | Baseline finding |
|---|---|---|
| Generic canonical bytes | `kaggle_researcher/contracts/hashing.py:22`, `canonical_contract_bytes` | Deterministically normalizes supported values, sorts mapping keys, emits compact UTF-8 JSON, rejects non-finite numbers, and is reusable as a lower-level utility after v6 vector tests. It does not implement every v6 domain-specific byte layout. |
| Generic digest | `kaggle_researcher/contracts/hashing.py:64`, `sha256_contract` | Stable and tested, but v6 evidence IDs and manifests normatively use BLAKE2b with specified payloads/self-exclusion. SHA-256 cannot be substituted at those seams. |
| Atomic JSON write | `kaggle_researcher/contracts/artifacts.py:447`, `write_json_atomic` | Temp-file plus `os.replace` is reusable for pointer publication. It writes readable, sorted JSON rather than the exact canonical bytes used for identity. |
| EDA artifact writer | `kaggle_researcher/eda/io/artifact_writer.py:29`, `ArtifactWriter`; `create_run_dir`; `write_json` | Reusable filesystem mechanics, but it creates one mutable run directory and independent files. It has no immutable `node_id/attempt_id` bundle, prepared manifest, evidence fragment, or `PUBLISHED` marker. |

### 2.3 Evidence authority and identity

| Component | Concrete implementation | Baseline finding |
|---|---|---|
| Current EDA evidence registry | `kaggle_researcher/contracts/evidence.py:101`, `EvidenceRegistry`; `build_evidence_registry`; `resolve_evidence_path`; `generate_allowed_evidence_refs` | Registry IDs are dictionary paths, list indexes, and semantic aliases into `EdaEvidencePack`. This directly conflicts with v6 storage-independent `EvidenceKey` identity and the rule that consumers do not reconstruct an allow-list. |
| Published evidence manifest | `kaggle_researcher/contracts/evidence_manifest.py:56`, `EvidenceReferenceEntry`; `EvidenceReferenceManifest`; `PublishedEdaEvidenceBundle`; `build_evidence_reference_manifest`; `publish_eda_evidence_bundle`; `validate_published_eda_bundle` | Frozen validation, conflict diagnostics, hash checks, and producer-side construction are reusable concepts. The public identity remains `canonical_path`; the bundle embeds the entire legacy pack; schema versions and digests are v5. |
| Final-strategy reference catalog | `kaggle_researcher/contracts/reference_catalog.py:84`, `ReferenceCatalog`; `build_final_strategy_reference_catalog` | A second catalog is built downstream from the pack and manifest. It mixes evidence, hypothesis, risk, implication, and other namespaces, so it cannot be the single v6 `EvidenceCatalog` authority. |
| Final-strategy evidence preview | `kaggle_researcher/contracts/final_strategy.py:82`, `EvidenceCatalogEntry`; `FinalStrategyResult.evidence_catalog` | Stores a resolved value preview keyed by legacy refs, not v6 `EvidenceRecord`/`EvidenceCatalogView`. |
| Reasoning allow-list | `kaggle_researcher/reasoning/common.py:141`, `known_evidence_ids`; `validate_evidence_ids`; `kaggle_researcher/reasoning/experiment_planner.py:127`, `known_experiment_evidence_ids` | Derives another allowed-ID set from retrieved documents and constants. This must remain outside, or be adapted into, the one run-scoped catalog. |
| Synthesis selection context | `kaggle_researcher/reasoning/final_strategy_context.py:51`, `build_final_strategy_selection_context` | Iterates `FinalSynthesisContext.allowed_eda_result_refs` and resolves pack paths, coupling synthesis to storage layout. |

There is no implementation of the v6 `EvidenceKey`, byte-level evidence-ID
algorithm, `EvidenceRecord`, `EvidenceCatalog`, collision rejection, typed
`ArtifactRef`, run-scoped registration, or generation-scoped catalog view.

### 2.4 Research Scout → EDA → Final Synthesis handoff

| Boundary | Concrete implementation | Baseline finding |
|---|---|---|
| Scout draft models | `kaggle_researcher/research_scout/schemas.py:161`, `ResearchScoutOutput`; `ScoutHypothesis`; `EdaTaskPlanDraft` | Strict Scout-local models exist, but they are translated into v5 contracts and task/module vocabulary rather than v6 requested capabilities. |
| Research contracts | `kaggle_researcher/contracts/research.py:22`, `ResearchHypothesis`; `ResearchHypotheses` | Typed and validated, but use v5 string schema versions and embed EDA tasks. They do not produce a v6 `InvestigationPlan`/capability request. |
| Research → EDA validation | `kaggle_researcher/contracts/research_to_eda.py:824`, `require_valid_research_to_eda_contract` | Deterministic, early typed validation is reusable. The accepted contracts are `ResearchHypotheses` and `EdaTaskPlan`, not a typed `ResearchToDatasetAdapter` output. |
| EDA task plan | `kaggle_researcher/contracts/eda.py:47`, `EdaTaskPlan`; compatibility export `kaggle_researcher/contracts/eda_task_plan.py` | The plan exposes EDA module/task details and free-form parameter dictionaries. It is not the v6 static `PipelinePlan` with ports and typed external bindings. |
| EDA aggregate output | `kaggle_researcher/contracts/eda.py:179`, `EdaEvidencePack` | A growing global pack with deprecated dual fields (`experiment_candidates`, `eda_risk_register`, `eda_strategy_hints`). v6 forbids using a legacy pack as a core-module input. |
| EDA → synthesis contract | `kaggle_researcher/contracts/synthesis_context.py:64`, `FinalSynthesisContext`; `build_final_synthesis_context`; `adapt_legacy_eda_evidence_pack` | Embeds `PublishedEdaEvidenceBundle` and exposes the pack and path allow-list. Missing `RunIdentity`, target generation, input snapshot digest, `EvidenceCatalogView`, constraints, and upstream issues. |
| Final output | `kaggle_researcher/contracts/final_strategy.py:440`, `FinalStrategyResult` | Has substantial deterministic validation but no run/generation identity and no v6 typed claim/provenance model. |

The existing Research → EDA validator is a useful characterization boundary.
The EDA → Synthesis boundary must be replaced by a one-way adapter that creates a
v6 `SynthesisContext`; v6 core must not call `adapt_legacy_eda_evidence_pack`.

### 2.5 EDA orchestrator, checkpoints, and artifacts

| Component | Concrete implementation | Baseline finding |
|---|---|---|
| EDA configuration/result | `kaggle_researcher/eda/schemas.py:85`, `EdaRunConfig`; `EdaRunResult` | Both inherit plain `BaseModel`, so unknown fields are ignored by default. `EdaRunResult.module_statuses` is an untyped string dictionary and has no generation/node/attempt identity. |
| Procedural orchestrator | `kaggle_researcher/eda/orchestrator.py:92`, `run_eda`; `_run_blocking_module`; `_run_p1_modules` | Existing analyzers are coordinated in a fixed procedure while accumulating mutable dictionaries and one `EdaEvidencePack`. It is not a module registry plus DAG runner. |
| Module statuses | `kaggle_researcher/eda/orchestrator.py:690`, `_record_module_status` | Records success/failed/skipped-style states and diagnostics, but not a status for every plan node in a generation. Disabled optional modules are recorded as `skipped`, never `inactive`. |
| Partial checkpoint | `kaggle_researcher/eda/orchestrator.py:748`, `_write_partial_evidence_pack` | Rewrites a partial global pack. It is not an immutable node-attempt checkpoint and cannot support generation-level atomic selection. |
| Publication | `kaggle_researcher/eda/orchestrator.py`, call to `publish_eda_evidence_bundle` near the publication boundary | Produces pack/manifest/bundle files, but has no staging → prepared manifest → atomic rename → `PUBLISHED` protocol and no committed generation snapshot. |

The metric, schema, validation, leakage, drift, baseline, feature-probe, and
rendering modules can be retained behind adapters. `run_eda` itself should be
treated as a v5 compatibility workflow, not incrementally turned into the v6
runner.

### 2.6 Run manifest, resume, and rerun

| Component | Concrete implementation | Baseline finding |
|---|---|---|
| Run/stage state | `kaggle_researcher/contracts/manifest.py:53`, `RunStatus`; `StageStatus`; `StageManifestEntry`; `RunManifest` | Models one mutable stage table. `StageManifestEntry.attempt` is a counter, not immutable attempt identity. `RunManifest` has no `RunIdentity`, `plan_id`, generation, `active_attempts`, or `manifest_digest`. |
| Manifest update | `kaggle_researcher/contracts/manifest.py:286`, `write_run_manifest_atomic`; `mark_stage_running`; `mark_stage_failed` | Atomic replacement protects JSON syntax, but the selected truth is overwritten rather than advanced by a candidate/committed generation pointer. |
| Runner state | `kaggle_researcher/orchestration/state.py:27`, `FullRunConfig`; `FullRunState` | `FullRunState` stores analytical results directly. v6 runner context should hold execution/selection metadata while domain outputs remain module contracts/artifacts. |
| Static v5 stages | `kaggle_researcher/orchestration/full_run.py:106`, `StageDefinition`; `stage_registry` | Defines a useful deterministic order/dependency closure, but stages have no typed ports or declared external bindings. |
| Resume reuse | `kaggle_researcher/orchestration/full_run.py:772`, `_can_reuse_stage` | Checks configuration and artifact pointers/hashes. It lacks v6 module/version/config/input/dataset fingerprints, accepted output contract/adapters, published-bundle integrity, and plan/generation selection. |
| Forced rerun | `kaggle_researcher/orchestration/full_run.py:892`, `dependent_stage_ids`; `invalidated_stage_ids`; `run_full_research` | Computes downstream invalidation, but writes into the same run/stage locations. A failed recomputation cannot atomically keep the previous full closure selected. |

The current tests prove that some upstream stages can be reused and that a
manifest replacement failure preserves the prior JSON. They do not prove the v6
invariant: all nodes in the rerun closure are promoted together, while a failed
candidate leaves the previous committed generation and active attempts
unchanged.

### 2.7 Publication and validation gates

Current reusable deterministic gates include:

- `kaggle_researcher/contracts/evidence_manifest.py:366`,
  `validate_published_eda_bundle`, for pack/manifest/bundle hash consistency;
- `kaggle_researcher/contracts/research_to_eda.py:824`,
  `require_valid_research_to_eda_contract`, for producer/consumer handoff;
- validation methods on
  `kaggle_researcher/contracts/final_strategy.py:440`,
  `FinalStrategyResult`.

They are not the v6 contract/artifact/evidence/synthesis/publication gate stack.
In particular, publication currently permits the v5
`EvidenceConflictPolicy.DEGRADED` route in
`kaggle_researcher/contracts/evidence_manifest.py:303`,
`publish_eda_evidence_bundle`; v6 catalog identity collisions and invalid
registration are hard blockers, not degraded evidence.

### 2.8 CLI entry points

`kaggle_researcher/main.py:136` (`build_parser`) uses legacy positional/flag
dispatch. `kaggle_researcher/main.py:334` (`build_full_run_parser`) exposes
`--resume-run-dir` and `--force-rerun-stage`; `run` dispatches
`validate-contracts`, `full-run`, and alias `run-all` near line 2414.
`kaggle_researcher/eda/main.py:12` provides a separate EDA parser.

`pyproject.toml` has no `[project.scripts]` console entry. The v6 hierarchy
(`run`, `dataset run`, `synthesize`, `resume --run-id`, `runs inspect`, and
`runs gc`) is absent.

## 3. Requirement correspondence

| v6 requirement | Current correspondence | Result |
|---|---|---|
| Strict public contracts | `ContractModel` is strict; `EdaRunConfig`, `EdaRunResult`, and `ContractHeader` are permissive | Partial/conflicting |
| Integer top-level schema versions | Persisted contracts use string `"1.0"`/`"2.0"` | Missing |
| `RunIdentity` | Loose `run_id`/`competition_id` strings occur in `EdaEvidencePack`, `EdaRunResult`, and `RunManifest` | Missing |
| Typed run/generation/node/attempt/evidence IDs | Only generic v5 `ContractId` subclasses; no generation/node/attempt types; permissive `EvidenceId` | Missing |
| One `EvidenceCatalog` authority | `EvidenceRegistry`, `EvidenceReferenceManifest`, `ReferenceCatalog`, preview catalog, and reasoning allow-list coexist | Conflicting |
| Storage-independent evidence identity | Current refs resolve pack dictionary paths and list indexes | Conflicting |
| Immutable attempt bundles and publication marker | Atomic individual file writes only | Missing |
| Candidate/committed generations | Mutable stage manifest only | Missing |
| Static pipeline plan with typed ports | `StageDefinition` dependency list only | Missing |
| Typed external input binding | Filesystem/config values flow through `FullRunConfig`/`EdaRunConfig` | Missing |
| Deterministic `one`/`many` mapping | No port cardinality contract or producer-node ordering | Missing |
| Fingerprint-based resume | Coarse config and pointer/hash reuse | Partial |
| Additional investigation in same run/new generation | No request/result contracts or activation lifecycle | Missing |
| Synthesis context bound to generation/catalog view | Legacy bundle/pack path allow-list | Conflicting |
| Deterministic publication gates | Strong v5 validators exist, but not against v6 artifacts/catalog/snapshot | Partial |
| Required v6 CLI | Legacy `full-run`/EDA parsers | Missing |

## 4. Conflicting legacy models and compatibility exports

1. `kaggle_researcher/contracts/eda.py:179` `EdaEvidencePack` is both persisted
   output and an internal synthesis input. It must become a derived one-way v5
   export only.
2. `kaggle_researcher/contracts/evidence.py:101` `EvidenceRegistry` and
   `kaggle_researcher/contracts/evidence_manifest.py:88`
   `EvidenceReferenceManifest` expose physical paths as public evidence refs.
3. `kaggle_researcher/contracts/reference_catalog.py:84` `ReferenceCatalog`
   duplicates authority downstream and mixes object namespaces with evidence
   provenance.
4. `kaggle_researcher/contracts/final_strategy.py:247`
   `FinalStrategyAction.evidence_bindings` is accepted but excluded from output;
   `migrate_legacy_evidence_bindings_payload` performs compatibility inside the
   public model boundary. v6 compatibility must instead be an explicit adapter.
5. `kaggle_researcher/contracts/synthesis_context.py:239`
   `adapt_legacy_eda_evidence_pack` makes a legacy pack acceptable as a
   synthesis input. It must not be imported by v6 core.
6. `kaggle_researcher/contracts/eda_task_plan.py` and
   `kaggle_researcher/contracts/research_hypotheses.py` are compatibility
   re-exports. They can remain for v5 callers, but v6 modules must import only
   v6 contract modules.
7. `kaggle_eda_engine` is a package-level compatibility surface. It must not
   define or re-export alternate v6 contract identities.
8. `kaggle_researcher/contracts/manifest.py:152` `RunManifest` and
   `kaggle_researcher/orchestration/state.py:78` `FullRunState` encode the v5
   mutable-stage lifecycle and must not be silently widened to accept v6
   payloads.

## 5. Reusable elements

The following may be reused after targeted v6 conformance tests:

- the strict configuration pattern in
  `kaggle_researcher/contracts/base.py:6`, `ContractModel`;
- runtime-distinct string IDs as a design pattern from
  `kaggle_researcher/contracts/ids.py:8`, `ContractId`;
- scalar normalization and stable UTF-8 JSON mechanics from
  `kaggle_researcher/contracts/hashing.py:22`,
  `canonical_contract_bytes`, but only where the v6 contract delegates to that
  generic policy;
- atomic replace mechanics from
  `kaggle_researcher/contracts/artifacts.py:447`, `write_json_atomic`, and
  `kaggle_researcher/eda/io/artifact_writer.py:44`,
  `ArtifactWriter.write_json`;
- integrity/error structuring ideas from
  `validate_published_eda_bundle` and `EvidenceReferenceConflict`;
- early deterministic boundary validation from
  `require_valid_research_to_eda_contract`;
- transitive dependency calculation from
  `kaggle_researcher/orchestration/full_run.py:892`,
  `dependent_stage_ids`;
- offline analytical EDA implementations invoked by
  `kaggle_researcher/eda/orchestrator.py:92`, `run_eda`, after each is wrapped
  as a v6 module;
- current contract, evidence, resume, and publication tests as
  characterization tests for the v5 compatibility boundary.

Reuse does not include retaining v5 serialized shapes as implicit accepted
inputs to v6 contracts.

## 6. Required boundary adapters

1. **Scout to investigation adapter:** convert
   `ResearchScoutOutput`/`ResearchHypotheses` into v6 requested capabilities and
   typed constraints without exposing `EdaTaskPlan` module names.
2. **Plan external-input adapter:** bind competition, dataset, research, config,
   and optional user inputs into typed `PlanInputBinding` values before runner
   validation.
3. **EDA module adapters:** wrap each selected analyzer called by `run_eda` so it
   consumes its declared input contract and emits exactly one `ModuleResult`
   plus evidence records/fragments.
4. **Legacy evidence export adapter:** render a v5 `EdaEvidencePack` and
   `EvidenceReferenceManifest` from a committed v6 catalog/snapshot. Direction
   is v6 → v5 export only.
5. **Synthesis adapter:** build v6 `SynthesisContext` from committed generation,
   catalog view, constraints, and upstream issues. It must not expose
   `eda_evidence_pack` or `canonical_path`.
6. **Final strategy legacy renderer:** render current
   `FinalStrategyResult`/reports from a validated v6 strategy only where legacy
   consumers still require it.
7. **Manifest migration/inspection adapter:** inspect v5
   `RunManifest` explicitly; never deserialize it as a v6 run manifest through
   permissive defaults.
8. **CLI compatibility adapter:** retain `full-run`/`run-all` as explicit v5
   commands or aliases while new v6 commands target only the v6 application
   service.

## 7. Contract and architectural blockers

### B1 — specification and task identity

The canonical filename is absent, the found draft is `0.2.5` rather than the
required `>=0.2.6`, and no uniquely assigned v6 task file exists. Before
production changes, place an approved specification at
`docs/specs/KAGGLE_RESEARCHER_V6_SPEC.md` and assign a task such as
`docs/v6/tasks/V6-P0-001.md`. This is a hard process blocker.

### B2 — node status completeness

Neither `RunManifest.stages` nor `EdaRunResult.module_statuses` represents every
plan node inside a generation. `StageStatus` has no `inactive`; disabled EDA
modules are recorded as `skipped` by `_run_p1_modules`. The 0.2.5 draft lists
runner statuses without clearly making `inactive` a persisted node status while
also requiring predeclared on-request nodes to be inactive. Version 0.2.6/task
wording must state the persisted enum and transition rules for `success`,
`failed`, `skipped`, and `inactive`.

### B3 — failed recomputation and active attempt selection

`StageManifestEntry.attempt` is only an integer counter, and
`mark_stage_running`/`mark_stage_failed` mutate the selected stage entry.
`run_full_research` writes rerun artifacts under the same run tree. Therefore a
failed recomputation has no atomic mechanism to deactivate its candidate
attempts while retaining the entire previous successful dependency closure.
Candidate generation, committed generation, and `active_attempts` are missing.

### B4 — port cardinality

There is no implementation of `InputPortSpec`, `OutputPortSpec`,
`DependencyEdge`, or runtime `one`/`many` mapping. `StageDefinition` dependencies
do not identify ports. No code enforces exactly one producer for `one`, ordered
aggregation for `many`, or ordering by `producer_node_id` byte order.

### B5 — typed external bindings

`FullRunConfig`, `EdaRunConfig`, and path-based handoffs act as implicit external
inputs. There is no `PlanInputBinding`, no source discriminator, and no rule
that each required input has exactly one edge or external binding.

### B6 — plan lifecycle and digest

There is no v6 `PipelinePlan`, `plan_id`, activation state, or `plan_digest`.
The existing `StageDefinition` tuple is code configuration, while retrieval
`PlanData` in `kaggle_researcher/schemas.py` is an unrelated research plan.
The plan digest payload, self-exclusion rule if any, and whether activation
state is part of plan or generation state must be normative before runner work.

### B7 — normative digests and canonical serialization

`canonical_contract_bytes` plus `sha256_contract` defines one stable v5 policy,
but it does not satisfy:

- fixed-member-order `EvidenceKey` payload plus BLAKE2b-16;
- `RunManifest.manifest_digest` over a self-excluded payload;
- `GenerationSnapshot.snapshot_digest`;
- artifact/file hash semantics and directory manifest ordering.

These need shared byte-level vectors. Domain code must not independently call
`json.dumps` with near-equivalent settings.

### B8 — additional investigation continuation

No `AdditionalInvestigationRequest`, bounded iteration policy, predeclared
on-request node activation, or same-run/new-generation continuation exists.
`resume_run_dir` resumes v5 stages and is not this lifecycle.

### B9 — producer-node provenance

Current `EvidenceReferenceEntry.source_component` is not `producer_node_id`.
The 0.2.5 `EvidenceRecord` sketch identifies producer module/version, while
`EvidenceFragment` carries node identity. The task explicitly requires
unambiguous `producer_node_id` provenance. The canonical spec/task must state
where it is persisted and validated before `EvidenceRecord` or catalog
registration is implemented.

### B10 — multiple evidence authorities

`EvidenceRegistry`, `EvidenceReferenceManifest`, `ReferenceCatalog`,
`FinalStrategyResult.evidence_catalog`, and reasoning allow-lists can disagree.
No v6 consumer should be added until a single catalog authority and explicit
legacy export direction are established.

## 8. Current test baseline

Existing tests characterize v5 behavior:

- canonical SHA-256 serialization:
  `tests/contracts/test_contract_hashing_spec.py`, including
  `test_canonical_hash_ignores_dictionary_insertion_order` and
  `test_canonical_hash_is_repeatable_sha256_not_python_hash`;
- path-based evidence parity and conflict publication:
  `tests/contracts/test_evidence_manifest.py`,
  `tests/contracts/test_evidence_reference_parity.py`,
  `tests/contracts/test_published_eda_bundle_integrity.py`, and
  `tests/contracts/test_no_downstream_evidence_recomputation.py`;
- mixed final-strategy reference catalog:
  `tests/contracts/test_final_strategy_reference_catalog.py`;
- Research → EDA validation:
  `tests/contracts/test_research_to_eda_contract_matrix.py` and
  `tests/contracts/test_research_to_eda_bridge_integration.py`;
- manifest migration/integrity:
  `tests/contracts/test_manifest_migration.py`;
- stage resume/rerun:
  `tests/test_full_run_orchestration.py`, especially
  `test_full_run_tracks_canonical_stage_order_and_resume`,
  `test_resume_after_reasoning_failure_reuses_semantically_valid_scout_and_eda`,
  and `test_force_final_strategy_reuses_completed_upstream_contracts`;
- EDA publication and module status:
  `tests/eda/test_eda_orchestrator_mvp.py` and
  `tests/eda/test_eda_orchestrator_p1.py`.

Missing v6 coverage includes cross-implementation evidence-ID vectors, catalog
collision and cross-run rejection, port cardinality and input bindings, plan
digest, immutable bundle publication, every-node generation status, failed
candidate rollback, closure promotion, fingerprint invalidation, synthesis
snapshot binding, additional-investigation continuation, and v6 CLI commands.

## 9. Dependency-ordered backlog

Each item below should have its own task file and focused acceptance tests.

1. **V6-P0-000 — source normalization.** Approve version `>=0.2.6` at the
   canonical spec path; resolve `inactive`, `plan_digest`, and
   `producer_node_id`; add uniquely named task files.
2. **V6-P0-001 — evidence identity seam.** Add strict v6 `EvidenceKey` and the
   exact deterministic evidence-ID function with portable vectors. No catalog
   or legacy integration.
3. **V6-P0-002 — common identity/result kernel.** Add `RunIdentity`, typed
   run/plan/generation/node/attempt IDs, `Diagnostic`, `ModuleResult`, quality,
   and the resolved node-status enum.
4. **V6-P0-003 — artifact reference contracts.** Add typed JSON/table/blob refs,
   locators, hashes, and validation without writing bundles.
5. **V6-P0-004 — evidence record and catalog registration.** Add
   `EvidenceRecord`, one run-scoped `EvidenceCatalog`, collision rejection,
   producer-node provenance, and generation views.
6. **V6-P0-005 — port and external-binding contracts.** Add
   `InputPortSpec`, `OutputPortSpec`, edges, `PlanInputBinding`, and deterministic
   `one`/`many` validation.
7. **V6-P0-006 — static plan contract.** Add `PipelinePlan`, lifecycle,
   activation declarations, canonical serialization, and `plan_digest`.
8. **V6-P0-007 — immutable node bundle writer.** Add staging/prepared manifest,
   evidence fragment, atomic directory publication, and `PUBLISHED` validation.
9. **V6-P0-008 — generation/run manifest kernel.** Add candidate/committed
   snapshots, `active_attempts`, manifest/snapshot digests, and every-node
   statuses; no module execution yet.
10. **V6-P0-009 — minimal runner and promotion.** Execute a small offline DAG,
    map ports, publish attempts, and atomically promote or discard a full
    dependency closure.
11. **V6-P0-010 — fingerprint resume/rerun.** Implement cache fingerprints,
    accepted contract/adapters, dataset identity, published integrity, and
    invalidation tests.
12. **V6-P0-011 — Research → Dataset boundary.** Introduce requested
    capabilities and explicit Scout/v5 adapters.
13. **V6-P0-012 — EDA module adapters.** Wrap reusable analyzers incrementally;
    keep `run_eda` as the v5 compatibility path until parity is proven.
14. **V6-P0-013 — synthesis boundary and gates.** Add generation-bound
    `SynthesisContext`, typed final strategy/claims, evidence validation, and
    publication gate.
15. **V6-P0-014 — additional-investigation loop.** Activate only predeclared
    nodes, continue the same run in a new generation, enforce bounds, and
    preserve history.
16. **V6-P0-015 — CLI/application services.** Add the required command tree and
    offline inspect/gc behavior.
17. **V6-P0-016 — one-way compatibility exports.** Generate v5 packs/reports
    from committed v6 state and assert that v6 core imports no legacy pack,
    registry, or hidden compatibility path.

## 10. Exact scope proposed for `V6-P0-001`

### Contract seam

Implement only the mapping:

`EvidenceKey(name, dimensions)` → canonical bytes → deterministic `evidence_id`.

The task must not implement `EvidenceRecord`, `EvidenceCatalog`, artifact refs,
module results, plan/ports, generation runner, publication, resume, additional
investigation, synthesis, or orchestrator changes.

### Proposed production files (3)

1. `kaggle_researcher/contracts/v6/__init__.py` — explicit v6 exports only.
2. `kaggle_researcher/contracts/v6/base.py` — minimal strict v6 contract base
   (`extra="forbid"`, strict validation, persistence-boundary stable dump
   policy); no legacy fallback.
3. `kaggle_researcher/contracts/v6/evidence.py` — `EvidenceKey`, grammar/NFC
   validators, canonical evidence-key bytes, and evidence-ID derivation.

No existing v5 file should be edited or re-exported from
`kaggle_researcher/contracts/__init__.py` in this task.

### Proposed test files (2)

1. `tests/contracts/v6/test_evidence_identity.py` — strictness, normalization,
   determinism, invalid input, exact shape, and vector tests.
2. `tests/fixtures/contracts/v6/evidence_id_vectors.json` — language-neutral
   input, canonical UTF-8 payload hex/text, and expected ID vectors.

### Acceptance criteria

1. `EvidenceKey` rejects unknown fields, invalid dotted names, invalid dimension
   keys/values, non-contract types, and post-validation field assignment.
2. Names and dimension strings are normalized to Unicode NFC before identity.
3. Dimensions are ordered by normalized UTF-8 key byte order.
4. The canonical payload uses the exact member order `name`, `dimensions`;
   dimensions are serialized as the specified ordered sequence; output is
   compact UTF-8 JSON with `ensure_ascii=False`, no BOM, and no platform-dependent
   whitespace/newline.
5. Digest is BLAKE2b with `digest_size=16`; the public ID uses the normative
   `ev_` prefix, bounded slug, and first 12 lowercase hexadecimal digest
   characters.
6. Dictionary insertion order does not change bytes or ID. Meaningful name or
   dimension changes do.
7. Composed and decomposed equivalent Unicode inputs produce the same canonical
   bytes and ID.
8. Every portable fixture asserts both canonical bytes and final ID, preventing
   two implementations from sharing the same wrong serializer.
9. Legacy `kaggle_researcher.contracts.evidence`,
   `kaggle_researcher.contracts.ids.EvidenceId`, manifests, publication,
   synthesis, and orchestration behavior remain byte-for-byte/API unchanged.
10. Strict validation is not weakened and no compatibility coercion, alias, or
    hidden legacy input is added.

Before this task starts, its assigned task file must cite the approved
specification section and settle any difference between version 0.2.5 and the
required canonical version.

### Verification commands

Use the interpreter required by `docs/RUNBOOK.md`:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests\contracts\v6\test_evidence_identity.py -q
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests\contracts\test_contract_hashing_spec.py tests\contracts\test_evidence_reference_parity.py -q
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest tests\contracts -q
E:\wavebreaker\.venv-win\Scripts\python.exe -m ruff check kaggle_researcher\contracts\v6 tests\contracts\v6
E:\wavebreaker\.venv-win\Scripts\python.exe -m kaggle_researcher.main validate-contracts
```

The focused test and vector corpus are mandatory. The contract regression suite
and contract validator are mandatory before completion. If Ruff is not
installed in the documented environment, report that exact limitation rather
than changing dependencies.

## 11. Readiness decision

The codebase contains reusable analytical implementations and useful v5
characterization tests, so the proposed first seam is technically small and
isolated. The repository is **not yet authorized/ready to begin
`V6-P0-001`** because the approved `>=0.2.6` specification at the canonical path
and the uniquely assigned v6 task file are missing, and three normative points
(`inactive`, `plan_digest`, and `producer_node_id`) remain unresolved.

Once the source-normalization item is completed, `V6-P0-001` above is the
recommended first code task. It touches one contract seam, exactly three
production files and two test/fixture files, and deliberately excludes runner,
publication, resume, additional investigation, and orchestration.

## 12. V6-P0-002 offline characterization set

The executable index is
`tests/fixtures/offline_characterization_manifest.json`. It binds every current
`FULL_RUN_STAGES` value from
`kaggle_researcher/orchestration/full_run.py:114` to one or more existing pytest
nodes and stored fixture paths. `tests/integration/test_offline_characterization.py`
validates that none of those paths or node IDs silently drift.

The task description's starting count of 131 test files had already drifted:
the implementation checkout contained 172 Python files under `tests/` before
V6-P0-002. This task uses concrete test node IDs rather than relying on that
historical count.

### Dataset and boundary audit

| Characterization case | Fixture/input | Coverage | Disposition |
|---|---|---|---|
| Binary IID classification | `tests/fixtures/eda/iid_binary_tiny` | `tests/eda/test_eda_generic_fixture_matrix.py::test_iid_binary_selects_stratified_kfold_even_with_date_column` | Already covered |
| Regression | `tests/fixtures/eda/regression_outliers_tiny` | `tests/eda/test_eda_generic_fixture_matrix.py::test_regression_outliers_selects_kfold_and_profiles_outliers` | Already covered |
| Temporal/stability classification | `tests/fixtures/eda/home_credit_tiny` | `tests/eda/test_eda_generic_fixture_matrix.py::test_home_credit_fixture_still_runs` | Already covered |
| Grouped classification | `tests/fixtures/eda/grouped_binary_tiny` | `tests/eda/test_eda_generic_fixture_matrix.py::test_grouped_binary_selects_group_aware_validation` | Already covered |
| Panel entity × time | `tests/fixtures/eda/panel_entity_time_tiny` | `tests/integration/test_offline_characterization.py::test_panel_entity_time_fixture_runs_offline` | Newly added because no stored panel fixture existed |
| Ranking/query-grouped | `tests/fixtures/eda/ranking_tiny` | `tests/eda/test_eda_generic_fixture_matrix.py::test_ranking_metric_selects_query_group_validation` | Already covered |
| Multilabel or multi-output submission | `tests/fixtures/eda/multiclass_tiny` with three prediction columns | `tests/eda/test_eda_generic_fixture_matrix.py::test_multiclass_logloss_runs_and_uses_probabilistic_metric` | Existing multi-output submission coverage |
| Multi-table relational | `tests/fixtures/eda/home_credit_tiny` | `tests/eda/test_eda_integration_full_p1.py::test_home_credit_full_p1_offline_integration` | Already covered |
| Malformed data with safe partial inventory | `tests/fixtures/eda/malformed_inventory_tiny` | `tests/integration/test_offline_characterization.py::test_malformed_inventory_fixture_is_reported_without_crash` | Newly stored; earlier coverage used only `tmp_path` |
| Unknown/custom metric | `tests/fixtures/offline_characterization_unknown_metric.json` | `tests/integration/test_offline_characterization.py::test_unknown_metric_fixture_requires_custom_implementation` | Newly stored; existing behavior was covered only by inline objects |
| Full Research/EDA/Reasoning/Synthesis pipeline | `iid_binary_tiny`, `final_synthesis.py` | `tests/integration/test_full_pipeline_contract_smoke.py::test_real_internal_pipeline_boundaries_without_network_or_llm` | Already covered; promoted to the documented full-pipeline command |
| Research → EDA input validation | Stored valid contract pair under `tests/fixtures/contracts/research_to_eda/valid/iid_binary` | `tests/contracts/test_research_to_eda_bridge_integration.py::test_reasoner_files_load_and_validate_without_starting_eda_modules` | Already covered |
| Every current full-run stage represented | Manifest `stages` entries | `test_manifest_covers_full_run_stages_and_dataset_cases` | Newly added drift guard over existing stage tests |
| No outbound connection | Root `tests/conftest.py::block_external_network_for_offline_tests` | `test_external_network_attempt_is_rejected_by_offline_guard` | Newly added for all tests not marked `network` |
| Missing stored fixture | `require_offline_fixture` | `test_missing_fixture_failure_names_the_fixture` | Newly added; failure includes the exact missing path |
| Repository execution prerequisites | `AGENTS.md`, `docs/archive/` | `test_repository_prerequisites_for_offline_execution_exist` | Existing files, newly asserted |

### Offline guarantees and commands

The network guard is installed at the socket boundary for every test not marked
`network`. It allows loopback connections needed by local test doubles but
raises before any external `connect`, `connect_ex`, or `create_connection`.
Files under `tests/network/` are marked and excluded by the documented
`-m "not network"` commands.

The exact full-pipeline and stage-level commands, including removal of Kaggle and
LLM credentials, are maintained in `docs/RUNBOOK.md` under “Offline v5
characterization”. The machine-readable manifest is the durable mapping behind
those commands; it does not create a second product contract.
