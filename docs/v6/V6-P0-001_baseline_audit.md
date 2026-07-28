# V6-P0-001 — Baseline audit

## Status
done

## Depends on
- none

## Goal
Record the factual starting state of the `kaggle_eda` branch so later tasks can
cite real paths and symbols instead of assumptions.

## Normative specification
- §33 Phase 0
- §36

## Outputs
`docs/V6_BASELINE.md`, containing at minimum:

- package inventory with line counts for `kaggle_researcher/eda/`,
  `kaggle_researcher/reasoning/`, `kaggle_researcher/contracts/`;
- the public entry points each stage exposes today;
- every symbol the v6 specification names as a forbidden anti-pattern, with
  file and line references;
- the existing test layout under `tests/`.

## Findings this audit must preserve
These were established during review and are load-bearing for Phase 1 scoping:

1. `kaggle_researcher/contracts/` already exists with roughly 35 modules,
   including `evidence.py`, `artifacts.py`, `ids.py`, `manifest.py`,
   `references.py`, `reference_catalog.py`, and `versions.py`. Phase 1 is not
   greenfield and every new contract module risks a name collision.
2. `kaggle_researcher/contracts/versions.py` uses string schema versions
   (`CURRENT_SCHEMA_VERSION = "1.0"`). Specification §9.1 mandates integer
   versions. This is a direct conflict, not a stylistic difference.
3. `run_manifest` already exists as a legacy contract family. Specification
   §10.6 defines a different `RunManifest`. The names collide.
4. `generate_allowed_evidence_refs` exists and is called from
   `kaggle_researcher/reasoning/final_synthesizer.py`. §36 forbids it.
5. `kaggle_researcher/eda/orchestrator.py` builds and mutates
   `evidence_pack_partial`, and at least one module both reads and writes that
   partially built mapping. §36 forbids both.
6. Evidence is addressed today by dotted dictionary paths, which
   `docs/contract_audit.md` records as the official address form. §5.5 forbids
   this and §11.7 defines the promotion path.
7. `polars` and `pyarrow` are already dependencies; `scan_parquet` is already
   used by the dataset reader. Table artifacts in §10.2 are not a new
   dependency.

## Acceptance criteria
1. Every claim in `docs/V6_BASELINE.md` cites a file path, and where relevant a
   symbol or line.
2. All seven findings above appear with current references.
3. No requirement or design proposal appears in the document; it records what
   exists.

## Stop conditions
- A cited symbol cannot be located in the branch.
