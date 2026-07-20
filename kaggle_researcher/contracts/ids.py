from __future__ import annotations

from typing import Any, ClassVar

from pydantic_core import PydanticCustomError, core_schema


class ContractId(str):
    """String-compatible, runtime-distinct logical identifier namespace."""

    namespace: ClassVar[str] = "contract"

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.str_schema(strip_whitespace=True, min_length=1),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def _validate(cls, value: str) -> "ContractId":
        if not value:
            raise PydanticCustomError(
                f"{cls.namespace}_id",
                f"expected a non-empty {cls.namespace} ID",
            )
        return cls(value)


def _id_type(name: str, namespace: str) -> type[ContractId]:
    return type(name, (ContractId,), {"namespace": namespace, "__module__": __name__})


HypothesisId = _id_type("HypothesisId", "hypothesis")
EdaTaskId = _id_type("EdaTaskId", "eda_task")
ExperimentId = _id_type("ExperimentId", "experiment")
EvidenceId = _id_type("EvidenceId", "evidence")
SourceClaimId = _id_type("SourceClaimId", "source_claim")
RiskId = _id_type("RiskId", "risk")
ImplicationId = _id_type("ImplicationId", "implication")
ReviewIssueId = _id_type("ReviewIssueId", "review_issue")
StageId = _id_type("StageId", "stage")
ValidationRequirementId = _id_type("ValidationRequirementId", "validation_requirement")
SafetyConstraintId = _id_type("SafetyConstraintId", "safety_constraint")


__all__ = [
    "ContractId",
    "EdaTaskId",
    "EvidenceId",
    "ExperimentId",
    "HypothesisId",
    "ImplicationId",
    "ReviewIssueId",
    "RiskId",
    "SafetyConstraintId",
    "SourceClaimId",
    "StageId",
    "ValidationRequirementId",
]
