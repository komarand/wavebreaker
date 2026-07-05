from __future__ import annotations

from kaggle_researcher.retrieval.source_quality import (
    rerank_by_source_quality,
    score_source_quality,
)
from kaggle_researcher.schemas import RetrievedDocument


def doc(source: str, title: str, content: str = "", rrf: float = 0.02) -> RetrievedDocument:
    return RetrievedDocument(
        id=f"{source}-{title}",
        competition_id="home-credit-credit-risk-model-stability",
        source=source,
        title=title,
        url="https://example.com/doc",
        content=content,
        score=1.0,
        rrf_score=rrf,
    )


def test_competition_specific_kaggle_notebook_gets_high_score() -> None:
    result = score_source_quality(
        {
            "source": "kaggle",
            "title": "Home Credit Credit Risk Model Stability starter",
            "content": "competition notebook",
        },
        "home-credit-credit-risk-model-stability",
    )

    assert result["quality_score"] > 1.5
    assert result["specificity"] == "competition_specific"


def test_exact_competition_github_repo_gets_high_score() -> None:
    result = score_source_quality(
        {
            "source": "github",
            "title": "home-credit-credit-risk-model-stability solution",
            "content": "repo readme",
        },
        "home-credit-credit-risk-model-stability",
    )

    assert result["quality_score"] > 1.4
    assert result["specificity"] == "competition_specific"


def test_generic_credit_risk_github_repo_gets_lower_score() -> None:
    result = score_source_quality(
        {"source": "github", "title": "credit risk tutorial", "content": "generic toy repo"},
        "home-credit-credit-risk-model-stability",
    )

    assert result["quality_score"] < 1.0
    assert result["specificity"] == "domain_specific"


def test_off_topic_hf_paper_gets_low_score() -> None:
    result = score_source_quality(
        {"source": "huggingface_papers", "title": "Hamiltonian Neural Networks for NLP fairness"},
        "home-credit-credit-risk-model-stability",
    )

    assert result["quality_score"] < 0.5


def test_academic_credit_scoring_paper_gets_medium_high_score() -> None:
    result = score_source_quality(
        {"source": "arxiv", "title": "Credit scoring calibration with AUC and Gini"},
        "home-credit-credit-risk-model-stability",
    )

    assert result["quality_score"] >= 1.3
    assert result["evidence_type"] == "academic_paper"


def test_reranking_moves_competition_specific_kaggle_above_generic_sources() -> None:
    generic = doc("github", "generic credit risk repo", rrf=0.04)
    specific = doc("kaggle", "Home Credit Credit Risk Model Stability notebook", rrf=0.02)

    results = rerank_by_source_quality(
        [generic, specific],
        competition_id="home-credit-credit-risk-model-stability",
    )

    assert results[0].source == "kaggle"
    assert results[0].metadata["final_score"] > results[1].metadata["final_score"]


def test_generic_source_cap_limits_context() -> None:
    docs = [doc("github", f"generic repo {index}", rrf=0.02) for index in range(6)]

    results = rerank_by_source_quality(
        docs,
        competition_id="home-credit-credit-risk-model-stability",
    )

    assert len(results) == 3
