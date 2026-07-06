from __future__ import annotations

import asyncio

from kaggle_researcher.research_scout import build_research_hypotheses


async def main() -> None:
    payload = await build_research_hypotheses(
        competition_id="home-credit-credit-risk-model-stability",
        competition_url="https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability",
        competition_desc=(
            "Predict credit default risk. Metric: Gini Stability. "
            "Tabular credit data with temporal stability concerns."
        ),
        plan_data={
            "task_type": "binary_classification",
            "metric": "gini_stability",
            "domain": "tabular_credit_risk",
        },
        retrieved_documents=[
            {
                "id": "kaggle:test",
                "source": "kaggle",
                "title": "Home Credit notebook with WEEK_NUM validation",
                "content": "Uses WEEK_NUM and discusses Gini stability and time validation.",
                "url": "https://www.kaggle.com/code/example",
                "rrf_score": 0.03,
                "metadata": {
                    "specificity": "competition_specific",
                    "quality_score": 1.7,
                },
            }
        ],
        source_quality_summary=None,
        domain_patterns=None,
    )

    print("num_hypotheses:", len(payload["hypotheses"]))
    print("num_eda_tasks:", len(payload["eda_tasks"]))
    for hypothesis in payload["hypotheses"][:5]:
        print(
            hypothesis["id"],
            hypothesis["category"],
            hypothesis["priority"],
            hypothesis["claim"],
        )


if __name__ == "__main__":
    asyncio.run(main())
