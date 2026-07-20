from __future__ import annotations

import json
from datetime import datetime

import pytest

from kaggle_researcher.eda.io.artifact_writer import ArtifactWriter, ArtifactWriterError
from kaggle_researcher.eda.schemas import RecommendedNextAction


def test_create_run_dir_creates_expected_structure(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    run_dir = writer.create_run_dir(
        "Fixture Competition!",
        timestamp=datetime(2026, 7, 6, 12, 30, 45),
    )

    assert run_dir == (tmp_path / "fixture-competition_20260706_123045").resolve()
    for relative_path in (
        "artifacts",
        "artifacts/plots",
        "artifacts/profiles",
        "artifacts/baseline",
        "artifacts/drift",
        "artifacts/validation",
        "artifacts/samples",
    ):
        assert (run_dir / relative_path).is_dir()


def test_write_json_pretty_prints_utf8_and_stable_keys(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    writer.create_run_dir("fixture", timestamp=datetime(2026, 7, 6, 12, 0, 0))

    output_path = writer.write_json(
        "metric_evidence.json",
        {
            "z_key": "значение",
            "a_key": RecommendedNextAction(
                priority="P0",
                action="Use ranks.",
                why="Metric evidence says ranking matters.",
                evidence_refs=["metric_evidence.rank_based"],
            ),
        },
    )

    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("{\n  \"a_key\"")
    assert "значение" in text
    parsed = json.loads(text)
    assert parsed["a_key"]["priority"] == "P0"
    assert parsed["z_key"] == "значение"


def test_write_markdown_writes_text(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    writer.create_run_dir("fixture", timestamp=datetime(2026, 7, 6, 12, 0, 0))

    output_path = writer.write_markdown("eda_summary.md", "# EDA Summary\n")

    assert output_path.name == "eda_summary.md"
    assert output_path.read_text(encoding="utf-8") == "# EDA Summary\n"


def test_copy_input_preserves_original_file_content(tmp_path) -> None:
    source = tmp_path / "input.json"
    source_bytes = b'{\n  "hello": "world"\n}\n'
    source.write_bytes(source_bytes)

    writer = ArtifactWriter(tmp_path / "runs")
    writer.create_run_dir("fixture", timestamp=datetime(2026, 7, 6, 12, 0, 0))

    copied_path = writer.copy_input(source, "input_research_hypotheses.json")

    assert copied_path.read_bytes() == source_bytes


def test_artifact_path_is_inside_current_run_dir(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    run_dir = writer.create_run_dir("fixture", timestamp=datetime(2026, 7, 6, 12, 0, 0))

    path = writer.artifact_path("plots", "target_by_week.png")

    assert path == run_dir / "artifacts" / "plots" / "target_by_week.png"


def test_writes_require_run_dir_and_reject_path_escape(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)

    with pytest.raises(ArtifactWriterError, match="create_run_dir"):
        writer.write_json("x.json", {})

    writer.create_run_dir("fixture", timestamp=datetime(2026, 7, 6, 12, 0, 0))

    with pytest.raises(ArtifactWriterError, match="escapes run directory"):
        writer.write_json("../outside.json", {})

    with pytest.raises(ArtifactWriterError, match="Input file does not exist"):
        writer.copy_input(tmp_path / "missing.json", "input.json")
