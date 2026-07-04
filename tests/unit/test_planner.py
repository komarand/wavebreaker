from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kaggle_researcher.planner import fallback_plan, plan
from kaggle_researcher.schemas import PlanData


def run(coro):
    return asyncio.run(coro)


class FakeDeepSeekClient:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: int = 90,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "timeout": timeout,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_mock_llm_response_becomes_plan_data() -> None:
    client = FakeDeepSeekClient(
        {
            "task_type": "binary_classification",
            "metric": "auc",
            "domain": "credit_risk_tabular",
            "kaggle_queries": ["credit default auc kaggle"],
            "arxiv_queries": ["credit scoring gradient boosting"],
            "github_queries": ["credit default kaggle solution"],
            "key_techniques": ["stratified cv", "rank averaging"],
            "similar_competitions": ["home credit default risk"],
        }
    )

    result = run(plan("Predict credit default. Metric: AUC.", client, "deepseek-v4-pro"))

    assert isinstance(result, PlanData)
    assert result.task_type == "binary_classification"
    assert result.metric == "auc"
    assert result.kaggle_queries == ["credit default auc kaggle"]
    assert client.calls[0]["model"] == "deepseek-v4-pro"
    assert client.calls[0]["timeout"] == 90
    assert "Return JSON only" in client.calls[0]["system_prompt"]
    assert "Predict credit default" in client.calls[0]["user_prompt"]


def test_missing_optional_lists_become_empty_lists() -> None:
    client = FakeDeepSeekClient(
        {
            "task_type": "regression",
            "metric": "rmse",
            "domain": "time_series",
        }
    )

    result = run(plan("Forecast demand with RMSE.", client, "deepseek-v4-pro"))

    assert result.kaggle_queries == []
    assert result.arxiv_queries == []
    assert result.github_queries == []
    assert result.key_techniques == []
    assert result.similar_competitions == []


def test_fallback_plan_returns_useful_non_empty_query_lists() -> None:
    result = fallback_plan("Classify loan default risk using ROC AUC on tabular credit data.")

    assert result.task_type == "classification"
    assert result.metric in {"auc", "roc_auc"}
    assert result.domain == "credit_risk_tabular"
    assert result.kaggle_queries
    assert result.arxiv_queries
    assert result.github_queries
    assert result.key_techniques
    assert result.similar_competitions


def test_plan_does_not_silently_swallow_llm_failure() -> None:
    client = FakeDeepSeekClient(RuntimeError("llm unavailable"))

    with pytest.raises(RuntimeError, match="llm unavailable"):
        run(plan("Any competition description", client, "deepseek-v4-pro"))


def test_malformed_plan_response_fails_validation() -> None:
    client = FakeDeepSeekClient(
        {
            "task_type": "classification",
            "metric": "auc",
        }
    )

    with pytest.raises(Exception, match="domain"):
        run(plan("Classify tabular examples.", client, "deepseek-v4-pro"))
