from __future__ import annotations

from dataclasses import dataclass

from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    PlanData,
    RetrievedDocument,
    ReviewResult,
    SourceDocument,
    ValidationResult,
)


COMPETITION_ID = "offline-credit-risk"
COMPETITION_URL = f"https://www.kaggle.com/competitions/{COMPETITION_ID}"
COMPETITION_DESC = (
    "Predict credit default risk from anonymized tabular borrower features. "
    "The public leaderboard is a sampled holdout, so validation stability matters."
)


@dataclass(frozen=True)
class OfflineReasoningOutputs:
    metric: MetricResult
    validation: ValidationResult
    leakage: LeakageRiskResult
    leaderboard: LeaderboardAuditResult
    experiments: list[ExperimentItem]
    review: ReviewResult

    def as_quality_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "validation": self.validation,
            "leakage": self.leakage,
            "leaderboard": self.leaderboard,
            "experiments": self.experiments,
            "review": self.review,
        }


def competition_desc() -> str:
    return COMPETITION_DESC


def plan_data() -> PlanData:
    return PlanData(
        task_type="binary classification",
        metric="auc",
        domain="credit risk",
        kaggle_queries=["credit risk auc notebook", "default prediction lightgbm"],
        arxiv_queries=["credit scoring tabular validation", "auc imbalanced classification"],
        github_queries=["credit risk lightgbm kaggle"],
        key_techniques=["LightGBM", "CatBoost", "target encoding", "rank averaging"],
        similar_competitions=["Home Credit Default Risk"],
    )


def kaggle_documents() -> list[SourceDocument]:
    return [
        _source_document("kaggle-1", "kaggle", "Public notebook baseline with LightGBM folds."),
        _source_document("kaggle-2", "kaggle", "Notebook discussion of categorical encodings."),
        _source_document("kaggle-3", "kaggle", "Leaderboard notes about stable cross-validation."),
    ]


def arxiv_documents() -> list[SourceDocument]:
    return [
        _source_document("arxiv-1", "arxiv", "Paper on credit scoring validation and calibration."),
        _source_document("arxiv-2", "arxiv", "Paper on robust AUC optimization for tabular data."),
    ]


def github_documents() -> list[SourceDocument]:
    return [
        _source_document("github-1", "github", "Repository README for reproducible LightGBM pipeline.")
    ]


def source_documents() -> list[SourceDocument]:
    return [*kaggle_documents(), *arxiv_documents(), *github_documents()]


def mock_embeddings() -> list[list[float]]:
    return [[float(index), float(index) / 10.0] for index, _ in enumerate(source_documents(), start=1)]


def retrieved_documents() -> list[RetrievedDocument]:
    return [
        _retrieved_document("retrieved-kaggle-1", "kaggle", 0.31),
        _retrieved_document("retrieved-arxiv-1", "arxiv", 0.24),
        _retrieved_document("retrieved-github-1", "github", 0.17),
    ]


def reasoning_outputs() -> OfflineReasoningOutputs:
    evidence_ids = [document.id for document in retrieved_documents()]
    return OfflineReasoningOutputs(
        metric=MetricResult(
            confidence="medium",
            evidence_ids=evidence_ids,
            metric_explanation="AUC rewards ranking quality and is stable for imbalanced labels.",
            needs_calibration=False,
            rank_averaging_useful=True,
            threshold_search_needed=False,
            surrogate_loss_suggestion="binary logloss with AUC monitoring",
        ),
        validation=ValidationResult(
            confidence="medium",
            evidence_ids=evidence_ids,
            recommended_cv="StratifiedGroupKFold with temporal holdout if date-like fields exist",
            validation_risk="medium",
            likely_split="stratified holdout",
            failure_modes=["target leakage through post-application aggregates"],
            reasoning="Use stable folds and reserve a final untouched holdout.",
            primary_validation={"method": "stratified_group_kfold"},
        ),
        leakage=LeakageRiskResult(
            confidence="medium",
            evidence_ids=evidence_ids,
            risk_level="medium",
            possible_issues=["post-outcome features", "duplicate applicants across folds"],
            recommended_checks=["feature availability audit", "duplicate key scan"],
        ),
        leaderboard=LeaderboardAuditResult(
            confidence="medium",
            evidence_ids=evidence_ids,
            shake_up_risk="medium",
            submission_selection_rule="choose models by CV mean and variance before public LB",
            public_lb_trust="medium",
            warnings=["Public LB may be noisy for small positive class slices."],
        ),
        experiments=[
            ExperimentItem(
                priority="P0",
                experiment="LightGBM baseline with stable categorical handling",
                why="Establish a calibrated AUC benchmark before larger ensembles.",
                cost="low",
                expected_gain="medium",
                risk="low",
                evidence_ids=evidence_ids,
            ),
            ExperimentItem(
                priority="P1",
                experiment="CatBoost comparison with ordered target statistics",
                why="Categorical handling can improve credit-risk tabular tasks.",
                cost="medium",
                expected_gain="medium",
                risk="medium",
                evidence_ids=evidence_ids,
            ),
        ],
        review=ReviewResult(
            confidence="medium",
            evidence_ids=evidence_ids,
            unsupported_claims=[],
            too_generic=[],
            unnecessary_experiments=[],
        ),
    )


def _source_document(doc_id: str, source: str, content: str) -> SourceDocument:
    return SourceDocument(
        id=doc_id,
        competition_id=COMPETITION_ID,
        source=source,
        title=f"{source} fixture {doc_id}",
        url=f"https://example.com/{doc_id}",
        content=content,
        metadata={"offline_fixture": True},
    )


def _retrieved_document(doc_id: str, source: str, rrf_score: float) -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id=COMPETITION_ID,
        source=source,
        title=f"Retrieved fixture {doc_id}",
        url=f"https://example.com/{doc_id}",
        content="Offline retrieved evidence with provenance-friendly guidance.",
        score=0.9,
        rrf_score=rrf_score,
        metadata={"offline_fixture": True},
    )
