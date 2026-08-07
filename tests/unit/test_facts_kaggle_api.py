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
import kaggle_researcher.facts.discussions as discussions_module
import kaggle_researcher.facts.files as files_module
import kaggle_researcher.facts.notebooks as notebooks_module
from kaggle_researcher.facts.competition import fetch_competition_metadata
from kaggle_researcher.facts.files import fetch_file_manifest
from kaggle_researcher.facts.kaggle_api import (
    GLOBAL_KAGGLE_POLICY,
    KaggleRequestPolicy,
    create_kaggle_api,
    extract_http_status,
    extract_request_attempt,
    extract_retry_after,
    is_forbidden,
    is_retryable_kaggle_error,
    unpack_list_response,
)
from kaggle_researcher.facts.notebooks import (
    list_competition_notebooks,
    pull_notebook,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_facts_kaggle_consumers_share_one_process_wide_policy() -> None:
    assert GLOBAL_KAGGLE_POLICY.max_attempts == 6
    assert GLOBAL_KAGGLE_POLICY.min_interval_seconds == pytest.approx(1.5)
    assert GLOBAL_KAGGLE_POLICY.jitter_fraction == pytest.approx(0.2)
    assert notebooks_module._NOTEBOOK_REQUEST_POLICY is GLOBAL_KAGGLE_POLICY
    assert files_module._FILE_REQUEST_POLICY is GLOBAL_KAGGLE_POLICY
    assert discussions_module._COMPETITION_REQUEST_POLICY is GLOBAL_KAGGLE_POLICY


def test_unpack_list_response_accepts_kaggle_1_direct_list() -> None:
    items = [{"ref": "one"}, {"ref": "two"}]

    unpacked = unpack_list_response(items, "competitions")

    assert unpacked.items == items
    assert unpacked.next_page_token is None
    assert unpacked.wrapped is False


@pytest.mark.parametrize("collection_name", ["competitions", "files", "kernels"])
def test_unpack_list_response_accepts_kaggle_2_envelopes(
    collection_name: str,
) -> None:
    item = {"ref": collection_name}
    response = SimpleNamespace(**{collection_name: [item], "next_page_token": "next-token"})

    unpacked = unpack_list_response(response, collection_name)

    assert unpacked.items == [item]
    assert unpacked.next_page_token == "next-token"
    assert unpacked.wrapped is True


def test_unpack_list_response_accepts_mapping_and_camel_case_token() -> None:
    unpacked = unpack_list_response(
        {"files": [{"name": "train.csv"}], "nextPageToken": "page-2"},
        "files",
    )

    assert unpacked.items == [{"name": "train.csv"}]
    assert unpacked.next_page_token == "page-2"
    assert unpacked.wrapped is True


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
        (
            RuntimeError(
                "HTTPSConnectionPool(host='www.kaggle.com', port=443): " "Max retries exceeded"
            ),
            None,
        ),
    ],
)
def test_extract_http_status_regex_fallback(
    exc: BaseException,
    expected: int | None,
) -> None:
    assert extract_http_status(exc) == expected


def test_extract_http_status_accepts_urllib_code_shape() -> None:
    assert extract_http_status(SimpleNamespace(code=429)) == 429


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_retryable_kaggle_statuses_are_explicit(status: int) -> None:
    assert is_retryable_kaggle_error(SimpleNamespace(status=status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501])
def test_non_retryable_kaggle_statuses_are_not_retried(status: int) -> None:
    assert is_retryable_kaggle_error(SimpleNamespace(status=status)) is False


def test_retry_after_is_read_without_exposing_response_body() -> None:
    exc = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "12.5"}))

    assert extract_retry_after(exc) == pytest.approx(12.5)


def test_request_policy_retries_with_bounded_exponential_backoff() -> None:
    clock = _FakeClock()
    attempts = 0

    class RateLimited(RuntimeError):
        status = 429

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RateLimited("secret response body")
        return "ok"

    policy = KaggleRequestPolicy(
        max_attempts=3,
        base_delay_seconds=1,
        max_delay_seconds=5,
        min_interval_seconds=0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert policy.call(operation) == "ok"
    assert attempts == 3
    assert clock.sleeps == [1, 2]


def test_request_policy_can_add_bounded_deterministic_jitter() -> None:
    clock = _FakeClock()
    attempts = 0

    class RateLimited(RuntimeError):
        status = 429

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimited("rate limited")
        return "ok"

    policy = KaggleRequestPolicy(
        max_attempts=2,
        base_delay_seconds=2,
        max_delay_seconds=10,
        min_interval_seconds=0,
        jitter_fraction=0.25,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        random_value=lambda: 0.5,
    )

    assert policy.call(operation) == "ok"
    assert clock.sleeps == [2.25]


def test_request_policy_caps_retry_after_and_attempt_count() -> None:
    clock = _FakeClock()
    attempts = 0

    class RateLimited(RuntimeError):
        status = 429
        response = SimpleNamespace(headers={"retry-after": "120"})

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise RateLimited("rate limited")

    policy = KaggleRequestPolicy(
        max_attempts=3,
        max_delay_seconds=7,
        min_interval_seconds=0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(RateLimited) as exc_info:
        policy.call(operation)

    assert attempts == 3
    assert clock.sleeps == [7, 7]
    assert extract_request_attempt(exc_info.value) == (3, 3)


def test_request_policy_rate_limits_successive_calls() -> None:
    clock = _FakeClock()
    policy = KaggleRequestPolicy(
        min_interval_seconds=0.5,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    policy.call(lambda: None)
    policy.call(lambda: None)

    assert clock.sleeps == [0.5]


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
    monkeypatch.setattr(
        competition_module,
        "_fetch_evaluation_metric",
        lambda slug: None,
    )
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


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
