from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ContractIssue:
    field_path: str
    value: object
    expected: str
    reason: str
    actual_namespace: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContractError(ValueError):
    """Base error carrying bounded text and machine-readable issues."""

    def __init__(
        self,
        message: str,
        *,
        issues: Iterable[ContractIssue] = (),
        stage: str = "contract_boundary",
        contract: str | None = None,
        recoverable: bool = False,
    ) -> None:
        self.issues = tuple(issues)
        self.stage = stage
        self.contract = contract
        self.recoverable = recoverable
        self.field_paths = tuple(issue.field_path for issue in self.issues)
        self.invalid_ids = tuple(
            str(issue.value)
            for issue in self.issues
            if issue.actual_namespace is not None or "reference" in issue.reason
        )
        bounded = "; ".join(
            f"{issue.field_path}: {issue.reason}" for issue in self.issues[:8]
        )
        super().__init__(f"{message}: {bounded}" if bounded else message)

    def as_manifest_error(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "stage": self.stage,
            "contract": self.contract,
            "recoverable": self.recoverable,
            "issues": [issue.as_dict() for issue in self.issues],
        }


class ContractValidationError(ContractError):
    pass


class UnsupportedSchemaVersionError(ContractError):
    pass


class ContractMigrationError(ContractError):
    pass


class CrossArtifactReferenceError(ContractError):
    pass


class CrossNamespaceReferenceError(CrossArtifactReferenceError):
    pass


class UnknownReferenceError(CrossArtifactReferenceError):
    pass


class AmbiguousReferenceError(CrossArtifactReferenceError):
    pass


class ArtifactContractError(ContractError):
    pass


class BoundaryRepairError(ContractError):
    pass

