from __future__ import annotations

import json
from pathlib import Path

from kaggle_researcher.eda.modules.visual_diagnostics import (
    render_visual_diagnostics,
    select_visual_diagnostics,
    validate_visual_diagnostics,
)


def test_headless_evidence_selected_plot_artifacts_and_manifest(tmp_path: Path) -> None:
    result = render_visual_diagnostics(_evidence(), tmp_path, max_total_plots=30)

    assert result["status"] == "completed"
    assert result["generated_plots"]
    assert (tmp_path / "plots_manifest.json").is_file()
    assert validate_visual_diagnostics(result, tmp_path) == []
    manifest = json.loads((tmp_path / "plots_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_count"] == len(result["generated_plots"])
    assert all((tmp_path / item["artifact_path"]).is_file() for item in result["generated_plots"])
    assert all("customer/id" not in item["artifact_path"] for item in result["generated_plots"])


def test_plot_selection_is_bounded_and_target_role_is_not_an_ordinary_feature() -> None:
    specs, skipped = select_visual_diagnostics(_evidence(), {"max_total_plots": 2})

    assert len(specs) == 2
    assert skipped
    assert all(item["evidence_refs"] for item in specs)
    assert all(item["plot_id"] == f"plot_{index:03d}" for index, item in enumerate(specs, 1))


def test_no_target_skips_target_plot_but_keeps_overview(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence["inferred_schema"]["target_column"] = None
    specs, _ = select_visual_diagnostics(evidence, {"max_total_plots": 30})
    result = render_visual_diagnostics(evidence, tmp_path)

    assert not any(item["plot_type"] == "target_distribution" for item in specs)
    assert any(item["plot_type"] == "table_shape_overview" for item in result["generated_plots"])


def _evidence() -> dict:
    return {
        "table_profiles": [{"table_name": "train_base.csv", "n_rows": 100, "n_cols": 5}],
        "inferred_schema": {"target_column": "target", "primary_id_column": "customer/id", "prediction_column": "prediction"},
        "target_diagnostics": {"status": "completed"},
        "feature_diagnostics": {"missingness_diagnostics": {"columns": [{"column": "income", "missing_pct": 0.25}, {"column": "age", "missing_pct": 0.1}]}},
        "drift_evidence": {"numeric_psi": {"columns": [{"column": "income", "psi": 0.3}]}, "missingness_drift": {"columns": []}},
        "baseline_ablation_evidence": {"ablations": [{"ablation_id": "abl_001", "status": "completed", "metric_value": 0.7}, {"ablation_id": "abl_002", "status": "completed", "metric_value": 0.72, "ablation_kind": "composite_configuration"}]},
        "interaction_diagnostics": {"interaction_hypotheses": [{"columns": ["age", "income"], "materiality": "material"}]},
        "eda_risk_register": [{"severity": "high", "risk_type": "drift"}, {"severity": "low", "risk_type": "baseline"}],
        "validation_evidence": {"primary_validation": {"method": "stratified_kfold"}},
    }
