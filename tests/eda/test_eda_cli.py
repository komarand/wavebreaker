from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_eda_cli_runs_local_dataset_without_network(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train_base.csv").write_text(
        "id,target,feature\n1,0,10\n2,1,20\n",
        encoding="utf-8",
    )
    (dataset / "test_base.csv").write_text(
        "id,feature\n3,30\n",
        encoding="utf-8",
    )
    (dataset / "sample_submission.csv").write_text(
        "id,prediction\n3,0.2\n",
        encoding="utf-8",
    )
    hypotheses = tmp_path / "research_hypotheses.json"
    task_plan = tmp_path / "eda_task_plan.json"
    hypotheses.write_text(
        json.dumps(
            {
                "competition_id": "cli_fixture",
                "hypotheses": [
                        {
                            "hypothesis_id": "schema_001",
                            "category": "schema",
                            "claim": "Check whether schema roles can be inferred.",
                            "expected_eda_checks": ["schema_inferer.roles"],
                            "priority": "P0",
                            "confidence_before_eda": "medium",
                        },
                        {
                            "hypothesis_id": "metric_001",
                            "category": "metric",
                            "claim": "EDA should verify the AUC metric contract.",
                            "expected_eda_checks": ["metric_analyzer.resolve_metric"],
                            "priority": "P0",
                            "confidence_before_eda": "medium",
                        },
                        {
                            "hypothesis_id": "val_001",
                            "category": "validation",
                            "claim": "EDA should verify a primary validation policy.",
                            "expected_eda_checks": ["validation_analyzer.primary_policy"],
                            "priority": "P0",
                            "confidence_before_eda": "medium",
                        },
                        {
                            "hypothesis_id": "leak_001",
                            "category": "leakage",
                            "claim": "Check whether generic leakage exists.",
                            "expected_eda_checks": ["leakage_checker.basic"],
                            "priority": "P0",
                            "confidence_before_eda": "medium",
                        },
                ],
            }
        ),
        encoding="utf-8",
    )
    task_plan.write_text(
        json.dumps(
            {
                "competition_id": "cli_fixture",
                "task_type": "binary_classification",
                    "metric": {"name": "auc"},
                    "dataset": {"local_dataset_path": str(dataset)},
                    "eda_tasks": [
                        {"task_id": "inventory", "module": "file_inventory", "priority": "P0", "blocking": True, "related_hypothesis_ids": ["schema_001"]},
                        {"task_id": "schema", "module": "schema_inferer", "priority": "P0", "blocking": True, "related_hypothesis_ids": ["schema_001"]},
                        {"task_id": "metric", "module": "metric_analyzer", "priority": "P0", "related_hypothesis_ids": ["metric_001"]},
                        {"task_id": "validation", "module": "validation_analyzer", "priority": "P0", "blocking": True, "related_hypothesis_ids": ["val_001"]},
                        {"task_id": "leakage", "module": "leakage_checker", "priority": "P0", "blocking": True, "related_hypothesis_ids": ["leak_001"]},
                    ],
                    "hypothesis_index": {
                        "schema_001": ["inventory", "schema"],
                        "metric_001": ["metric"],
                        "val_001": ["validation"],
                        "leak_001": ["leakage"],
                    },
                    "recommended_module_sequence": [
                        "file_inventory", "schema_inferer", "metric_analyzer",
                        "validation_analyzer", "leakage_checker",
                    ],
                    "blocking_tasks": [
                        "file_inventory", "schema_inferer", "validation_analyzer", "leakage_checker",
                    ],
                }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle_eda_engine.main",
            "--competition-id",
            "cli_fixture",
            "--hypotheses-path",
            str(hypotheses),
            "--task-plan-path",
            str(task_plan),
            "--local-dataset-path",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--no-download-dataset",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "EDA evidence pack:" in result.stdout
    evidence_packs = list(output_dir.glob("*/eda_evidence_pack.json"))
    assert len(evidence_packs) == 1
