from __future__ import annotations

import json

import pytest

from kaggle_researcher.contracts.research_to_eda import (
    ContractIssue,
    ResearchToEdaContractError,
    ResearchToEdaContractValidationResult,
)


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


@pytest.mark.parametrize(
    "secret",
    [
        "DEEPSEEK_API_KEY=super-secret-value",
        "KAGGLE_KEY=super-secret-value",
        "GITHUB_TOKEN=super-secret-value",
        "postgresql://user:password@example.invalid/database",
        "sk-abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz",
        r"C:\Users\alice\private\artifact.json",
        "/home/alice/private/artifact.json",
    ],
)
def test_structured_issues_and_exception_diagnostics_redact_secrets(secret: str) -> None:
    issue = ContractIssue(
        code="invalid_contract",
        severity="error",
        path=secret,
        message=f"Bad value: {secret}",
        related_ids=[secret],
    )
    result = ResearchToEdaContractValidationResult(valid=False, errors=[issue], warnings=[])
    error = ResearchToEdaContractError(result)
    diagnostic = json.dumps(error.as_manifest_error())
    assert secret not in str(error)
    assert secret not in diagnostic
    assert "super-secret-value" not in diagnostic
    assert "password@example" not in diagnostic


def test_error_contract_is_bounded_and_machine_readable() -> None:
    issue = ContractIssue(
        code="known_code",
        severity="error",
        path="hypotheses[0].claim",
        message="x" * 10_000,
        related_ids=[str(index) for index in range(100)],
    )
    assert len(issue.message) == 500
    assert len(issue.related_ids) == 16
    error = ResearchToEdaContractError(
        ResearchToEdaContractValidationResult(valid=False, errors=[issue], warnings=[])
    )
    manifest = error.as_manifest_error()
    assert manifest["issues"][0]["code"] == "known_code"
    assert manifest["issues"][0]["severity"] == "error"
    assert manifest["recoverable"] is False

