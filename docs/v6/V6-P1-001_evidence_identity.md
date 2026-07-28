# V6-P1-001 — Evidence identity

## Status
ready

## Depends on
- Phase 0 exit
- canonical specification 0.2.6

## Goal
Implement `EvidenceKey` and the deterministic derivation
`EvidenceKey → canonical bytes → evidence_id`, byte-for-byte as specified, with
no dependency on datasets, modules, or storage.

## Normative specification
- §9
- §10.4.1
- §11.1–11.5
- §29.1

## Package placement note
`kaggle_researcher/contracts/` already exists with legacy modules of the same
names. New v6 contracts land in `kaggle_researcher/contracts/v6/` and are never
merged into the legacy modules during Phase 1. Legacy code reaches v6 contracts
only through adapters (§8), never by import from a legacy contract module.

## Inputs and outputs
Input: `EvidenceKey(name, dimensions)`.
Output: a canonical byte string and an `evidence_id`.

Identity derives from the evidence contract namespace and dimensions only. The
producing module is never an input to identity; it is provenance.

## Allowed files
- `kaggle_researcher/contracts/v6/__init__.py`
- `kaggle_researcher/contracts/v6/evidence_key.py`
- `tests/contracts/v6/test_evidence_identity.py`
- `tests/contracts/v6/__init__.py`

## Forbidden scope
- No `EvidenceRecord`, no catalog, no registration, no artifact refs.
- No storage, filesystem access, or serialization to disk.
- No import from `kaggle_researcher/eda/` or `kaggle_researcher/reasoning/`.

## Acceptance criteria
1. `EvidenceKey.name` is the evidence contract namespace; no separate namespace
   field exists.
2. Name grammar `^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$` and dimension-key
   grammar `^[a-z][a-z0-9_]*$` are enforced, and both accept underscores in the
   first segment. `ablation_id` is a valid dimension key.
3. Canonical bytes are produced with fixed member order `name` then
   `dimensions`, dimensions as sorted `[key, value]` pairs, separators
   `(",", ":")`, `ensure_ascii=False`, NFC normalization, UTF-8 without BOM.
4. `evidence_id` is `"ev_" + slug(name) + "_" + blake2b(payload,
   digest_size=16).hexdigest()[:12]`, with `slug` replacing `.` with `_` and
   truncating to 48 characters.
5. Dimension order in the input does not affect the result.
6. Non-ASCII dimension values produce real UTF-8 bytes, never `\uXXXX` escapes,
   and a stored expected digest for a non-ASCII corpus matches.
7. Negative: a name or dimension key violating its grammar raises a typed error
   naming the offending token; it is never silently normalized.
8. Negative: no code path lengthens the digest suffix on collision. A genuine
   collision is a typed error naming both keys.
9. Negative: identity does not change when the producing module identifier
   changes.

## Verification
```bash
pytest tests/contracts/v6/test_evidence_identity.py -q
```
Include a stored key corpus with expected IDs, so a second implementation can
be checked against it (§29.6 scenario 6).

## Stop conditions
- The specified algorithm cannot reproduce a stored expected ID. Report the
  mismatch; do not adjust the algorithm to fit an implementation.
