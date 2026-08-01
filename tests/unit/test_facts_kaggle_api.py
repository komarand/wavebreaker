from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kaggle_researcher.facts.competition as competition_module
import kaggle_researcher.facts.files as files_module
import kaggle_researcher.facts.notebooks as notebooks_module
from kaggle_researcher.facts.competition import fetch_competition_metadata
from kaggle_researcher.facts.files import fetch_file_manifest
from kaggle_researcher.facts.kaggle_api import (
    create_kaggle_api,
    extract_http_status,
    is_forbidden,
)
from kaggle_researcher.facts.notebooks import (
    list_competition_notebooks,
    pull_notebook,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_all_facts_modules_import_without_kaggle_credentials(tmp_path: Path) -> None:
    env = os.environ.copy()
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(name, None)
    env["KAGGLE_CONFIG_DIR"] = str(tmp_path / "missing-kaggle-config")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import importlib
import sys

modules = (
    "kaggle_researcher.facts",
    "kaggle_researcher.facts.competition",
    "kaggle_researcher.facts.files",
    "kaggle_researcher.facts.kaggle_api",
    "kaggle_researcher.facts.models",
    "kaggle_researcher.facts.notebooks",
)
for module in modules:
    importlib.import_module(module)
assert not any(name == "kaggle" or name.startswith("kaggle.") for name in sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_create_kaggle_api_imports_lazily_and_authenticates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"created": 0, "authenticated": 0}

    class FakeKaggleApi:
        def __init__(self) -> None:
            calls["created"] += 1

        def authenticate(self) -> None:
            calls["authenticated"] += 1

    fake_module = SimpleNamespace(KaggleApi=FakeKaggleApi)
    real_import = builtins.__import__

    def lazy_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "kaggle.api.kaggle_api_extended":
            return fake_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", lazy_import)

    api = create_kaggle_api()

    assert isinstance(api, FakeKaggleApi)
    assert calls == {"created": 1, "authenticated": 1}


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("plain failure"), None),
        (RuntimeError("HTTP 403 Forbidden"), 403),
        (RuntimeError("request failed with 429"), 429),
        (RuntimeError("upstream returned 503"), 503),
    ],
)
def test_extract_http_status_regex_fallback(
    exc: BaseException,
    expected: int | None,
) -> None:
    assert extract_http_status(exc) == expected


@pytest.mark.parametrize(
    "exc",
    [
        SimpleNamespace(status=403),
        SimpleNamespace(status=None, http_resp=SimpleNamespace(status=403)),
        SimpleNamespace(response=SimpleNamespace(status_code=403)),
        SimpleNamespace(response=SimpleNamespace(status=403)),
        RuntimeError("HTTP 403 Forbidden"),
    ],
)
def test_is_forbidden_recognizes_all_supported_status_shapes(exc: Any) -> None:
    assert is_forbidden(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("record id 1403 failed"),
        RuntimeError("record id 4030 failed"),
        RuntimeError("unrelated number 94039"),
        SimpleNamespace(status=404),
        RuntimeError("plain failure"),
    ],
)
def test_is_forbidden_does_not_match_unrelated_numbers(exc: Any) -> None:
    assert is_forbidden(exc) is False


def test_one_injected_api_is_reused_by_all_collectors_without_authentication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class SharedApi:
        def __init__(self) -> None:
            self.authenticate_calls = 0
            self.kernel_list_kwargs: dict[str, Any] | None = None

        def authenticate(self) -> None:
            self.authenticate_calls += 1

        def competitions_list(self, *, search: str) -> list[dict[str, Any]]:
            return [{"ref": search, "title": "Competition"}]

        def competition_list_files(self, slug: str) -> dict[str, Any]:
            return {
                "files": [
                    {
                        "name": "sample_submission.csv",
                        "totalBytes": 20,
                        "columns": ["id", "target"],
                    }
                ]
            }

        def kernels_list(self, **kwargs: Any) -> list[dict[str, Any]]:
            self.kernel_list_kwargs = kwargs
            return [
                {
                    "ref": "author/notebook",
                    "title": None,
                    "totalVotes": None,
                    "publicScore": "0.75",
                }
            ]

        def kernels_pull(
            self,
            kernel_ref: str,
            *,
            path: str,
            metadata: bool = False,
            quiet: bool = False,
        ) -> None:
            (Path(path) / "notebook.ipynb").write_text("{}", encoding="utf-8")

    def unexpected_factory() -> Any:
        raise AssertionError("create_kaggle_api must not run for an injected client")

    monkeypatch.setattr(competition_module, "create_kaggle_api", unexpected_factory)
    monkeypatch.setattr(files_module, "create_kaggle_api", unexpected_factory)
    monkeypatch.setattr(notebooks_module, "create_kaggle_api", unexpected_factory)
    api = SharedApi()

    metadata = fetch_competition_metadata("example", api=api)
    manifest = fetch_file_manifest("example", 100, api=api)
    notebooks = list_competition_notebooks("example", 10, api=api)
    notebook_path = pull_notebook("author/notebook", tmp_path, api=api)

    assert metadata.title == "Competition"
    assert manifest.sample_submission_source == "api"
    assert notebooks[0]["title"] == "author/notebook"
    assert notebooks[0]["votes"] == 0
    assert notebook_path == tmp_path / "notebook.ipynb"
    assert api.authenticate_calls == 0
    assert api.kernel_list_kwargs is not None
    assert api.kernel_list_kwargs["language"] == "python"
    assert api.kernel_list_kwargs["kernel_type"] == "notebook"
