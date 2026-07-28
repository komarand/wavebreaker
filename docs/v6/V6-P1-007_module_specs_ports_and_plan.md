# V6-P1-007 — Module specs, ports, and plan

## Status
planned

## Depends on
- V6-P1-002

## Goal
Implement typed ports, plan input bindings, module specifications, the
immutable `PipelinePlan`, and plan identity.

## Normative specification
- §12, §12.0.1, §12.1
- §18.3 (preflight rules only)
- §29.1

## Package placement note
New v6 contracts land in `kaggle_researcher/contracts/v6/`. Legacy modules of
the same name are not modified during Phase 1.

## Allowed files
- `kaggle_researcher/contracts/v6/ports.py`
- `kaggle_researcher/contracts/v6/plan.py`
- `tests/contracts/v6/test_ports_and_plan.py`

## Forbidden scope
- No runner, no execution, no status machine.
- No dataset access.

## Acceptance criteria
1. `InputPortSpec` carries name, capability, contract, accepted schema
   versions, `required`, and `cardinality`.
2. `OutputPortSpec` carries name, capability, contract, and
   `contract_schema_version`, closing the triple capability → contract →
   version.
3. `ModuleSpec` carries `module_contract_version`,
   `module_implementation_version`, `cache_fingerprint_version`,
   `provides_evidence_namespaces`, and `default_failure_policy`. No
   `provides_capabilities` field exists.
4. `DependencyEdge` names `producer_node_id`, `producer_port`,
   `consumer_node_id`, and `consumer_port`.
5. `NodeSpec` carries `node_id`, `module_id`, `config`, and `activation`.
6. `PipelinePlan` is immutable, carries `input_bindings`, and computes a
   `plan_digest` whose input excludes the digest field itself.
7. Preflight proves port compatibility from named contracts and versions alone,
   with no module execution.
8. The MVP rule holds: a node has exactly one output port, and `OutputT` is
   that port's payload.
9. Negative: a required input port satisfied by both an edge and a binding, or
   by neither, is a plan error naming the port.
10. Negative: an edge override may escalate `optional` to `blocking` and may
    never downgrade a module's `blocking` default; the attempt raises a typed
    error.

## Verification
```bash
pytest tests/contracts/v6/test_ports_and_plan.py -q
mypy kaggle_researcher/contracts/v6/
```

## Stop conditions
- A planned Phase 2 module genuinely needs more than one output port. Report
  it; the `PortBundle` extension is deferred and must not be introduced here.
