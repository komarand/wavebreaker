# EDA architecture boundary

The pipeline remains `Research Scout -> EDA Engine -> reasoning / Final Synthesizer`.
No additional service is introduced.

## Central rule

EDA reports observations and diagnostic implications. The reasoning layer converts
them into experiments and strategy. The Final Synthesizer owns final priorities,
model plans, global risks, experiment order, submission guidance, and the final
do/do-not-do list.

## What EDA owns

EDA owns dataset resolution, inventory and schema evidence, profiles, target and
missingness diagnostics, leakage and relationship checks, train/test drift,
feature reliability, and bounded diagnostics based on fold-safe OOF predictions.
The lightweight baseline is a `diagnostic_sanity_floor`; ablations diagnose broad
feature blocks. Interactions and slices report evidence and uncertainty without
changing the final feature or model pipeline. Visual diagnostics only render
existing evidence.

Core EDA success is independent of optional model-assisted diagnostics, rendering,
and post-EDA source-claim validation. `module_classification` and `stage_status` in
the evidence pack make this distinction machine-readable.

## Output contract and origins

Canonical boundary outputs are `safety_constraints`, `validation_requirements`,
`eda_implications`, `eda_risks`, and `testable_hypotheses`. Testable hypotheses
are unresolved propositions from an explicit diagnostic allowlist, normally capped
at ten; they are not an experiment backlog. Each has stable IDs, evidence
references, controls, and an evidence origin. Supported origins are `dataset_measurement`,
`fold_safe_model_diagnostic`, `statistical_diagnostic`, `source_claim`,
`reasoning_inference`, and `final_strategy`. EDA never labels an output
`final_strategy`.

Source-claim validation is classified `post_eda_reasoning`: it compares Research
Scout claims with an assembled EDA evidence input and is not part of core EDA or
the EDA-local risk generator. Its deterministic implementation and CLI flag remain
available for compatibility.

## Downstream contract

The Final Synthesizer consumes EDA facts, implications, EDA-local risks,
experiment candidates, Research Scout evidence, and source validation. It keeps
origins distinguishable when turning them into final recommendations:

- `dataset_measurement`: “EDA found...”
- `fold_safe_model_diagnostic`: “A lightweight OOF diagnostic showed...”
- `source_claim`: “A source suggests...”
- `reasoning_inference`: “This implies...”
- `final_strategy`: “Recommended plan...”

## Compatibility

Legacy fields remain projections during migration:

- `strategy_hints` / `eda_strategy_hints` -> `eda_implications`
- `recommended_next_actions` -> `testable_hypotheses`
- `experiment_candidates` -> `testable_hypotheses`
- `eda_risk_register` -> `eda_risks`

`deprecated_outputs` documents replacements. New behavior must be implemented in
the canonical representation rather than creating a second generation pipeline.
