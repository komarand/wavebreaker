# V6-P1-002 — Common contract kernel

## Status
ready

## Depends on
- Phase 0 exit

## Goal
Establish the shared contract base: identity, diagnostics, limitations, module
results, integer schema versions, strict validation, and typed errors.

## Normative specification
- §9, §9.1, §9.2
- §12.2
- §26
- §29.1

## Package placement note
`kaggle_researcher/contracts/` already exists with legacy modules of the same
names. New v6 contracts land in `kaggle_researcher/contracts/v6/` and are never
merged into the legacy modules during Phase 1. Legacy code reaches v6 contracts
only through adapters (§8), never by import from a legacy contract module.

## Known conflict to resolve
`kaggle_researcher/contracts/versions.py` defines string schema versions
(`CURRENT_SCHEMA_VERSION = "1.0"`). §9.1 mandates integer versions on persisted
top-level contracts. The legacy module is not modified by this task; v6 uses
its own integer versioning in `contracts/v6/`, and the two coexist until
Phase 8 cleanup.

## Inputs and outputs
Output contracts: `RunIdentity`, `Diagnostic`, `Limitation`, `ModuleResult[T]`,
`DiagnosticSeverity`, `FindingSeverity`, and the typed error hierarchy.

## Allowed files
- `kaggle_researcher/contracts/v6/common.py`
- `kaggle_researcher/contracts/v6/errors.py`
- `tests/contracts/v6/test_common_kernel.py`

## Forbidden scope
- No evidence, artifact, plan, or synthesis contracts.
- No modification of `kaggle_researcher/contracts/versions.py` or any other
  legacy contract module.

## Acceptance criteria
1. Every persisted top-level contract carries an integer `schema_version`;
   nested value types such as `Diagnostic` and `Limitation` do not.
2. All public models use `extra="forbid"` and `frozen=True`.
3. `DiagnosticSeverity` and `FindingSeverity` are distinct types, and no code
   converts one into the other.
4. `Limitation` carries `severity: FindingSeverity`.
5. `ModuleResult` carries `module_contract_version`,
   `module_implementation_version`, tuple-typed `diagnostics`, and tuple-typed
   `limitations`.
6. The degradation invariant is asserted directly: `complete` has no error or
   critical diagnostic and no module-scope limitation; `degraded` has at least
   one module-scope limitation or degradation diagnostic and a contract-valid
   required output; `failed` produces no `ModuleResult`.
7. Negative: constructing a `ModuleResult` with `quality="degraded"` and no
   limitation or degradation diagnostic raises a typed error.
8. Negative: an unknown field on any public model is rejected, and the error
   names the field.
9. Negative: sequence fields on persisted contracts reject `list` inputs that
   would leave a mutable interior, or normalize them to tuples at validation.

## Verification
```bash
pytest tests/contracts/v6/test_common_kernel.py -q
mypy kaggle_researcher/contracts/v6/
```

## Stop conditions
- A legacy import forces string versions into a v6 contract. Report rather than
  weakening §9.1.
