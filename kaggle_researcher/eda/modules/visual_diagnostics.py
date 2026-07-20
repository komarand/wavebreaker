from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Callable

from PIL import Image, ImageDraw

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError


DEFAULT_CAPS = {"max_total_plots": 30, "max_overview_plots": 3, "max_target_plots": 4, "max_feature_plots": 8, "max_missingness_plots": 4, "max_drift_plots": 6, "max_baseline_plots": 4, "max_interaction_plots": 4, "max_risk_plots": 2, "max_rows_per_plot": 100_000}


def render_visual_diagnostics(evidence_pack: dict[str, Any], run_dir: Path, reader: DatasetReader | None = None, *, max_total_plots: int = 30, random_state: int = 42) -> dict[str, Any]:
    """Render a compact, evidence-selected PNG manifest using the installed Pillow backend."""
    del random_state
    run_dir = Path(run_dir).resolve(); plots_dir = run_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    policy = {**DEFAULT_CAPS, "max_total_plots": max_total_plots, "formats": ["png"], "backend": "Pillow", "width_pixels": 1260, "height_pixels": 700}
    specs, skipped = select_visual_diagnostics(evidence_pack, policy)
    generated: list[dict[str, Any]] = []; failed: list[dict[str, Any]] = []
    renderers: dict[str, Callable[[dict[str, Any], Path], None]] = {
        "table_shape_overview": _render_table_overview, "target_distribution": _render_target_distribution,
        "missingness_overview": _render_missingness, "drift_ranking": _render_drift,
        "baseline_ablation_scores": _render_ablations, "risk_severity_count": _render_risks,
        "interaction_hypotheses": _render_interactions, "summary_dashboard": _render_dashboard,
    }
    for spec in specs:
        relative = Path(spec["artifact_path"]); path = (run_dir / relative).resolve()
        try:
            if not _inside(path, run_dir): raise ValueError("Unsafe plot artifact path")
            path.parent.mkdir(parents=True, exist_ok=True)
            renderer = renderers[spec["plot_type"]]
            payload = dict(spec); payload["_evidence"] = evidence_pack; payload["_reader"] = reader
            renderer(payload, path)
            spec["status"] = "generated"; generated.append(spec)
        except Exception as exc:
            spec["status"] = "failed"; spec["warnings"] = [str(exc)[:240]]; failed.append(spec)
    dashboard = next((item for item in generated if item["plot_type"] == "summary_dashboard"), {"status": "skipped", "reason": "insufficient_dashboard_evidence"})
    result = {"status": "completed" if generated else "not_testable", "plot_policy": policy, "selected_plots": specs, "generated_plots": generated, "skipped_plots": skipped, "plot_groups": _groups(generated), "summary_dashboard": dashboard, "warnings": [f"{item['plot_id']}: {item['warnings'][0]}" for item in failed], "limitations": ["Visual diagnostics render compact evidence summaries and do not alter structured EDA evidence.", "Pillow is used because matplotlib is not installed in this environment."], "excluded_plot_columns": _excluded_columns(evidence_pack)}
    manifest = {"run_id": run_dir.name, "generated_at": datetime.now().astimezone().isoformat(), "plot_count": len(specs) + len(skipped), "generated_count": len(generated), "skipped_count": len(skipped), "failed_count": len(failed), "plots": [*generated, *failed, *skipped]}
    import json
    (run_dir / "plots_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    result["manifest_path"] = "plots_manifest.json"; result["failed_plots"] = failed
    return result


def select_visual_diagnostics(evidence_pack: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []; skipped: list[dict[str, Any]] = []
    def add(plot_type: str, group: str, title: str, refs: list[str], reason: str, priority: str = "P2") -> None:
        candidates.append({"plot_type": plot_type, "plot_group": group, "title": title, "status": "selected", "artifact_path": f"plots/{group}/{_slug(plot_type)}.png", "thumbnail_path": None, "data_source": {"table": None, "columns": [], "artifact_refs": refs}, "evidence_refs": refs, "selection_reason": reason, "priority": priority, "reliability": "reliable", "caption": reason, "interpretation": "Review this compact visual alongside its linked structured evidence.", "warnings": [], "limitations": []})
    if evidence_pack.get("table_profiles"): add("table_shape_overview", "overview", "Table shape overview", ["table_profiles"], "Shows the relative size of profiled tables.")
    if _has_target(evidence_pack): add("target_distribution", "target", "Target distribution", ["target_diagnostics", "inferred_schema.target_column"], "Shows the observed target distribution; target is displayed only as an outcome.", "P1")
    if _missing_columns(evidence_pack): add("missingness_overview", "missingness", "Top missingness rates", ["feature_diagnostics.missingness_diagnostics"], "Shows role-safe columns with the highest recorded missingness.")
    if _drift_rows(evidence_pack): add("drift_ranking", "drift", "Top role-safe drift evidence", ["drift_evidence"], "Ranks existing role-safe drift evidence for review.", "P1")
    if _completed_ablations(evidence_pack): add("baseline_ablation_scores", "ablations", "Baseline ablation comparison", ["baseline_ablation_evidence.ablations"], "Compares aggregate ablation metrics; marginal deltas remain the decision evidence.", "P1")
    if _as_dict(evidence_pack.get("interaction_diagnostics")).get("interaction_hypotheses"): add("interaction_hypotheses", "interactions", "Top interaction hypotheses", ["interaction_diagnostics.interaction_hypotheses"], "Displays only reliable experiment hypotheses, not automatic features.")
    if _material_risks(evidence_pack): add("risk_severity_count", "risks", "Material risk summary", ["eda_risk_register"], "Counts medium-or-higher risks by severity.", "P1")
    if len(candidates) >= 3: add("summary_dashboard", "dashboard", "EDA visual summary", ["table_profiles", "target_diagnostics", "eda_risk_register"], "Combines a bounded set of the highest-value evidence summaries.", "P1")
    selected = candidates[:int(config.get("max_total_plots", DEFAULT_CAPS["max_total_plots"]))]
    for index, spec in enumerate(selected, 1): spec["plot_id"] = f"plot_{index:03d}"
    for spec in candidates[len(selected):]: skipped.append({"plot_type": spec["plot_type"], "status": "skipped", "reason": "plot_cap", "evidence_refs": spec["evidence_refs"]})
    return selected, skipped


def validate_visual_diagnostics(result: dict[str, Any], run_dir: Path) -> list[str]:
    errors: list[str] = []; ids: set[str] = set(); paths: set[str] = set(); root = Path(run_dir).resolve()
    for item in result.get("generated_plots", []):
        if item.get("plot_id") in ids: errors.append("Duplicate plot ID.")
        ids.add(str(item.get("plot_id"))); path = str(item.get("artifact_path"))
        if path in paths: errors.append("Duplicate plot artifact path.")
        paths.add(path)
        if not item.get("evidence_refs") or not item.get("title") or not item.get("caption"): errors.append("Generated plot lacks required metadata.")
        absolute = (root / path).resolve()
        if not _inside(absolute, root) or not absolute.is_file(): errors.append(f"Missing or unsafe plot artifact: {path}")
    if len(result.get("generated_plots", [])) > int(_as_dict(result.get("plot_policy")).get("max_total_plots", 30)): errors.append("Plot cap exceeded.")
    return sorted(set(errors))


def _render_table_overview(spec: dict[str, Any], path: Path) -> None:
    profiles = [_as_dict(item) for item in spec["_evidence"].get("table_profiles", [])]; labels = [str(item.get("table_name") or "table") for item in profiles]; values = [int(item.get("n_rows") or 0) for item in profiles]; _bar_chart(path, spec["title"], labels, values, "Rows")
def _render_target_distribution(spec: dict[str, Any], path: Path) -> None:
    evidence, reader = spec["_evidence"], spec.get("_reader"); schema = _as_dict(evidence.get("inferred_schema")); target = schema.get("target_column"); table = schema.get("train_base_table")
    values: list[Any] = []
    if reader and target and table:
        try: values = reader.read_columns(table, columns=[target], n_rows=100_000)[target].to_list()
        except (ReaderError, KeyError): values = []
    counts = Counter("__MISSING__" if value is None else str(value) for value in values)
    if not counts: counts = Counter({"target evidence": 1})
    _bar_chart(path, spec["title"], list(counts)[:15], [counts[key] for key in list(counts)[:15]], "Rows")
def _render_missingness(spec: dict[str, Any], path: Path) -> None:
    rows = _missing_columns(spec["_evidence"])[:12]; _bar_chart(path, spec["title"], [str(item.get("column")) for item in rows], [float(item.get("missing_pct") or 0) for item in rows], "Missing fraction")
def _render_drift(spec: dict[str, Any], path: Path) -> None:
    rows = _drift_rows(spec["_evidence"])[:12]; _bar_chart(path, spec["title"], [str(item.get("column")) for item in rows], [float(item.get("psi", item.get("abs_diff", 0)) or 0) for item in rows], "Drift score")
def _render_ablations(spec: dict[str, Any], path: Path) -> None:
    rows = _completed_ablations(spec["_evidence"]); _bar_chart(path, spec["title"], [str(item.get("ablation_id")) for item in rows], [float(item.get("metric_value") or 0) for item in rows], "Metric")
def _render_risks(spec: dict[str, Any], path: Path) -> None:
    counts = Counter(str(item.get("severity")) for item in _material_risks(spec["_evidence"])); _bar_chart(path, spec["title"], list(counts) or ["none"], list(counts.values()) or [0], "Risk count")
def _render_interactions(spec: dict[str, Any], path: Path) -> None:
    rows = _as_dict(spec["_evidence"].get("interaction_diagnostics")).get("interaction_hypotheses", [])[:8]; _bar_chart(path, spec["title"], [" x ".join(item.get("columns", [])) for item in rows] or ["none"], [2 if item.get("materiality") == "material" else 1 for item in rows] or [0], "Materiality band")
def _render_dashboard(spec: dict[str, Any], path: Path) -> None:
    evidence = spec["_evidence"]; lines = [f"Tables: {len(evidence.get('table_profiles', []))}", f"Target: {_as_dict(evidence.get('inferred_schema')).get('target_column') or 'not available'}", f"Validation: {_as_dict(_as_dict(evidence.get('validation_evidence')).get('primary_validation')).get('method') or 'not available'}", f"Material risks: {len(_material_risks(evidence))}", f"Ablations: {len(_completed_ablations(evidence))}"]
    _text_card(path, spec["title"], lines)


def _bar_chart(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    width, height, left, bottom = 1260, 700, 150, 120; image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image); _title(draw, title); maximum = max(values) if values else 1
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 85 + index * max(28, min(48, 520 // max(len(labels), 1))); bar = int((width - left - 110) * (value / maximum if maximum else 0)); draw.rectangle((left, y, left + bar, y + 20), fill="#3574a8"); draw.text((8, y), _short(label, 22), fill="black"); draw.text((left + bar + 6, y), f"{value:.4g}", fill="black")
    draw.text((left, height - 45), ylabel, fill="#444444"); image.save(path, "PNG")
def _text_card(path: Path, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (1260, 700), "white"); draw = ImageDraw.Draw(image); _title(draw, title)
    for index, line in enumerate(lines): draw.text((80, 110 + index * 72), _short(line, 110), fill="#1f3650")
    image.save(path, "PNG")
def _title(draw: ImageDraw.ImageDraw, title: str) -> None: draw.text((35, 25), _short(title, 100), fill="#172b3a")
def _short(value: str, length: int) -> str: return value if len(value) <= length else value[: length - 3] + "..."
def _groups(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {key: [] for key in ("dataset_overview", "target", "features", "missingness", "drift", "baseline", "ablations", "interactions", "slices", "relationships", "risks", "dashboard")}
    for item in items: groups["dataset_overview" if item["plot_group"] == "overview" else item["plot_group"]].append(item)
    return groups
def _has_target(evidence: dict[str, Any]) -> bool: return bool(_as_dict(evidence.get("inferred_schema")).get("target_column"))
def _missing_columns(evidence: dict[str, Any]) -> list[dict[str, Any]]: return sorted([_as_dict(item) for item in _as_dict(_as_dict(evidence.get("feature_diagnostics")).get("missingness_diagnostics")).get("columns", [])], key=lambda item: (-float(item.get("missing_pct") or 0), str(item.get("column"))))
def _drift_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]: return sorted([_as_dict(item) for group in ("numeric_psi", "missingness_drift") for item in _as_dict(_as_dict(evidence.get("drift_evidence")).get(group)).get("columns", [])], key=lambda item: (-float(item.get("psi", item.get("abs_diff", 0)) or 0), str(item.get("column"))))
def _completed_ablations(evidence: dict[str, Any]) -> list[dict[str, Any]]: return [item for item in _as_dict(evidence.get("baseline_ablation_evidence")).get("ablations", []) if _as_dict(item).get("status") == "completed"]
def _material_risks(evidence: dict[str, Any]) -> list[dict[str, Any]]: return [_as_dict(item) for item in evidence.get("eda_risk_register", []) if _as_dict(item).get("severity") in {"medium", "high", "critical"}]
def _excluded_columns(evidence: dict[str, Any]) -> list[dict[str, str]]:
    schema = _as_dict(evidence.get("inferred_schema")); result = []
    for key, reason in (("target_column", "target_column"), ("primary_id_column", "primary_id"), ("prediction_column", "prediction_column")):
        if schema.get(key): result.append({"column": str(schema[key]), "reason": reason})
    return result
def _slug(value: str) -> str: return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-") or "plot"
def _as_dict(value: Any) -> dict[str, Any]: return value if isinstance(value, dict) else {}
def _inside(path: Path, root: Path) -> bool:
    try: path.relative_to(root); return True
    except ValueError: return False


__all__ = ["render_visual_diagnostics", "select_visual_diagnostics", "validate_visual_diagnostics"]
