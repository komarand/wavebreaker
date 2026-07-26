from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Literal


@dataclass(frozen=True)
class ContractIssue:
    field_path: str = ""
    value: object = None
    expected: str = ""
    reason: str = ""
    actual_namespace: str | None = None
    stage: str = "contract_boundary"
    issue_type: str = "validation_error"
    message: str | None = None
    received_type: str | None = None
    reference: str | None = None
    severity: Literal["warning", "error"] = "error"

    def __post_init__(self) -> None:
        if self.message is None:
            object.__setattr__(self, "message", self.reason)
        elif not self.reason:
            object.__setattr__(self, "reason", self.message)
        if self.received_type is None:
            object.__setattr__(self, "received_type", type(self.value).__name__)

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
        self.issues = tuple(
            replace(issue, stage=stage)
            if issue.stage == "contract_boundary" and stage != "contract_boundary"
            else issue
            for issue in issues
        )
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


class UnknownContractFamilyError(ContractError):
    pass


class DuplicateContractRegistrationError(ContractError):
    pass


class ContractMigrationError(ContractError):
    pass


class InternalContractValidationError(ContractValidationError):
    """A deterministic producer emitted an invalid contract and cannot be repaired."""

    classification = "internal_contract_error"


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


class ContractCanonicalizationError(ContractError):
    """A value cannot be represented by the versioned canonical hash policy."""


class EvidenceManifestBuildError(ContractError):
    """The evidence reference manifest could not be built deterministically."""


class EvidenceManifestConflictError(EvidenceManifestBuildError):
    """Strict publication rejected error-severity evidence conflicts."""

    def __init__(self, message: str, *, manifest: Any, issues: Iterable[ContractIssue]) -> None:
        self.manifest = manifest
        self.conflicts = tuple(getattr(manifest, "conflicts", ()))
        super().__init__(
            message,
            issues=issues,
            stage="eda_publication_boundary",
            contract="evidence_reference_manifest",
        )


class EvidenceManifestPackMismatchError(EvidenceManifestBuildError):
    """A published manifest is paired with a different evidence-pack snapshot."""

    def __init__(
        self,
        *,
        expected_hash: str,
        actual_hash: str,
        manifest_hash: str | None = None,
        bundle_hash: str | None = None,
        manifest_schema_version: str | None = None,
        bundle_schema_version: str | None = None,
    ) -> None:
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.expected_pack_hash = expected_hash
        self.actual_pack_hash = actual_hash
        self.manifest_hash = manifest_hash
        self.bundle_hash = bundle_hash
        self.schema_versions = {
            "evidence_reference_manifest": manifest_schema_version,
            "published_eda_evidence_bundle": bundle_schema_version,
        }
        self.classification = "internal_contract_error"
        super().__init__(
            "Published evidence bundle pack hash mismatch",
            issues=(ContractIssue(
                "evidence_pack",
                {
                    "expected_pack_hash": expected_hash,
                    "actual_pack_hash": actual_hash,
                    "manifest_hash": manifest_hash,
                    "bundle_hash": bundle_hash,
                    "schema_versions": self.schema_versions,
                },
                "pack matching the published evidence manifest",
                "evidence manifest and pack identify different snapshots",
            ),),
            stage="published_bundle_validation",
            contract="published_eda_evidence_bundle",
        )
