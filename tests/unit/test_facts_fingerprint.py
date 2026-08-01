from __future__ import annotations

import json
from pathlib import Path

from kaggle_researcher.facts.models import NotebookFacts
from kaggle_researcher.facts.notebook_ast import (
    assign_lineage_clusters,
    ast_fingerprint,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "facts"
FORK_A = FIXTURE_DIR / "notebook_fork_a.ipynb"
FORK_B = FIXTURE_DIR / "notebook_fork_b.ipynb"


def test_forks_differing_in_literals_random_state_and_comments_share_fingerprint() -> None:
    fingerprint_a = ast_fingerprint(FORK_A)
    fingerprint_b = ast_fingerprint(FORK_B)

    assert fingerprint_a == fingerprint_b
    assert len(fingerprint_a) == 64
    assert all(character in "0123456789abcdef" for character in fingerprint_a)


def test_notebook_with_different_splitter_has_different_fingerprint(tmp_path: Path) -> None:
    notebook = json.loads(FORK_A.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    notebook["cells"][1]["source"] = source.replace(
        "StratifiedKFold",
        "TimeSeriesSplit",
    )
    different_path = tmp_path / "different_splitter.ipynb"
    different_path.write_text(json.dumps(notebook), encoding="utf-8")

    assert ast_fingerprint(FORK_A) != ast_fingerprint(different_path)


def test_module_function_class_and_async_docstrings_do_not_affect_fingerprint(
    tmp_path: Path,
) -> None:
    source_a = '''"""Module documentation A."""
class Model:
    """Class documentation A."""
    def fit(self):
        """Function documentation A."""
        return 1

async def train():
    """Async documentation A."""
    return 2
'''
    source_b = '''"""Completely different module documentation."""
class Model:
    """Different class documentation."""
    def fit(self):
        """Different function documentation."""
        return 100

async def train():
    """Different async documentation."""
    return 200
'''
    path_a = _write_notebook(tmp_path / "a.ipynb", [source_a])
    path_b = _write_notebook(tmp_path / "b.ipynb", [source_b])

    assert ast_fingerprint(path_a) == ast_fingerprint(path_b)


def test_unparsed_cells_use_stable_marker(tmp_path: Path) -> None:
    path_a = _write_notebook(
        tmp_path / "broken_a.ipynb",
        ["if True print('a')", "value = 1"],
    )
    path_b = _write_notebook(
        tmp_path / "broken_b.ipynb",
        ["def incomplete(", "value = 999"],
    )

    assert ast_fingerprint(path_a) == ast_fingerprint(path_b)


def test_code_cell_order_affects_fingerprint(tmp_path: Path) -> None:
    forward = _write_notebook(
        tmp_path / "forward.ipynb",
        ["prepare()", "train()"],
    )
    reversed_order = _write_notebook(
        tmp_path / "reversed.ipynb",
        ["train()", "prepare()"],
    )

    assert ast_fingerprint(forward) != ast_fingerprint(reversed_order)


def test_assign_lineage_clusters_groups_forks_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    fork_fingerprint = ast_fingerprint(FORK_A)
    notebook = json.loads(FORK_A.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    notebook["cells"][1]["source"] = source.replace(
        "StratifiedKFold",
        "GroupKFold",
    )
    different_path = tmp_path / "different.ipynb"
    different_path.write_text(json.dumps(notebook), encoding="utf-8")
    different_fingerprint = ast_fingerprint(different_path)
    facts = [
        _notebook_fact("author/fork-a", fork_fingerprint, "pending-a"),
        _notebook_fact("author/fork-b", ast_fingerprint(FORK_B), "pending-b"),
        _notebook_fact("author/different", different_fingerprint, "pending-c"),
    ]

    clustered = assign_lineage_clusters(facts)

    expected_fork_cluster = f"lc_{fork_fingerprint[:12]}"
    assert clustered[0].lineage_cluster_id == expected_fork_cluster
    assert clustered[1].lineage_cluster_id == expected_fork_cluster
    assert clustered[2].lineage_cluster_id == f"lc_{different_fingerprint[:12]}"
    assert len({fact.lineage_cluster_id for fact in clustered}) < len(clustered)
    assert [fact.lineage_cluster_id for fact in facts] == [
        "pending-a",
        "pending-b",
        "pending-c",
    ]
    assert all(updated is not original for updated, original in zip(clustered, facts))


def test_fork_fixtures_have_valid_v4_structure() -> None:
    for path in (FORK_A, FORK_B):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] == 5
        for cell in notebook["cells"]:
            assert isinstance(cell["metadata"], dict)
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def _write_notebook(path: Path, code_cells: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": source,
                    }
                    for source in code_cells
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    return path


def _notebook_fact(ref: str, fingerprint: str, cluster_id: str) -> NotebookFacts:
    return NotebookFacts(
        ref=ref,
        title=ref,
        ast_fingerprint=fingerprint,
        lineage_cluster_id=cluster_id,
        splitters=[],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=[],
        parse_status="ok",
    )
