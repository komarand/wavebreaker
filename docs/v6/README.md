# v6 implementation tasks

Task files are the durable, reviewable scope for v6 work. Chat prompts are not
requirements: a prompt only instructs an agent to execute one task file.

- Canonical specification: `docs/specs/KAGGLE_RESEARCHER_V6_SPEC.md` (v0.2.6)
- Execution prompt: `docs/tasks/v6/PROMPT_TEMPLATE.md`
- Definition of done for every task: specification §37

## Rules

1. One task file is one unit of work. Agents implement exactly one.
2. A task file may only cite the canonical specification. Documents under
   `docs/archive/` are historical and are never requirements.
3. `Allowed files` is a closed list. Touching anything outside it is a stop
   condition, not a judgement call.
4. Every acceptance criterion must be covered by a test, including the negative
   criteria.
5. If a task file, the specification, and existing code disagree, the agent
   stops and reports before changing anything.

## Index

| ID | Task | Depends on | Status |
|---|---|---|---|
| V6-P0-001 | Baseline audit | — | done |
| V6-P0-002 | Offline characterization fixtures | P0-001 | done |
| V6-P0-003 | Contract drift regressions | P0-002 | planned |
| V6-P0-004 | Reusable algorithm inventory | P0-001 | planned |
| V6-P1-001 | Evidence identity | Phase 0 exit | ready |
| V6-P1-002 | Common contract kernel | Phase 0 exit | ready |
| V6-P1-003 | Artifact refs and integrity | P1-002 | planned |
| V6-P1-004 | Evidence record and fragment | P1-001, P1-003 | planned |
| V6-P1-005 | Evidence catalog registration | P1-004 | planned |
| V6-P1-006 | Evidence catalog view | P1-005 | planned |
| V6-P1-007 | Module specs, ports, and plan | P1-002 | planned |
| V6-P1-008 | Publication contracts | P1-005, P1-007 | planned |

Phase 1 exit (specification §33): duplicate, broken, cross-scope, and
physical-path refs are rejected with no dataset or LLM dependency.

## Backlog

Detailed task files are written 1–3 at a time, immediately before
implementation. Paths and symbols change as earlier phases land, so writing
them now would produce stale `Allowed files` lists.

### Phase 2 — first vertical slice

`V6-P2-001_dataset_gateway_and_fingerprint`,
`V6-P2-002_inventory_module_adapter`,
`V6-P2-003_schema_module_adapter`,
`V6-P2-004_submission_module_adapter`,
`V6-P2-005_profile_module_adapter`,
`V6-P2-006_sequential_runner_core`,
`V6-P2-007_checkpoint_and_targeted_resume`,
`V6-P2-008_first_slice_integration`.

### Phase 3 — decision slice

`V6-P3-001_metric_module`,
`V6-P3-002_validation_policy_and_splits`,
`V6-P3-003_leakage_module`,
`V6-P3-004_hypothesis_evaluation`,
`V6-P3-005_decision_slice_integration`.

Note for P3-002: the v5 policy engine emits `GroupKFold`,
`StratifiedGroupKFold`, `StratifiedKFold`, and temporal holdout variants only.
It cannot currently produce `purged_group_time_series`, which the specification
uses as its canonical constraint example in §23. Closing that gap belongs to
this task, together with an explicit refusal path when the label horizon is
unknown.

### Phase 4 — Research boundary

`V6-P4-001_research_contracts_and_capabilities`,
`V6-P4-002_research_to_dataset_adapter`,
`V6-P4-003_research_preflight`,
`V6-P4-004_research_boundary_integration`.

Lifecycle scenarios 1–3 and 7–8 from §29.6 are a precondition for resume work
in this phase.

### Phase 5 — Synthesis boundary

Tasks are formed just-in-time because each consumer of the legacy
`EdaEvidencePack` migrates separately:

- parallel registration of v5 results into `EvidenceCatalog`;
- one task per consumer;
- prompt builder and validator switched to a single `EvidenceCatalogView`;
- removal of `generate_allowed_evidence_refs`;
- pack demoted to an export-only view;
- typed claims and `StrategyConstraint`;
- Final Strategy publication gate;
- Synthesis integration.

### Phases 6–8

Bounded additional investigation; optional EDA modules one at a time;
Markdown and DOCX presentation; v5/v6 comparison harness; default CLI switch;
retention and GC; removal of internal v5 dependencies.
