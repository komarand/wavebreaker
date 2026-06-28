from __future__ import annotations

from typing import Any

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.schemas import PlanData


OPTIONAL_LIST_FIELDS = (
    "kaggle_queries",
    "arxiv_queries",
    "github_queries",
    "key_techniques",
    "similar_competitions",
)


SYSTEM_PROMPT = """You are KaggleResearcher's competition planning module.

Decompose a Kaggle competition description into search queries for a research-only
pipeline. Return JSON only with these keys:
- task_type: concise task family, e.g. binary_classification, regression, ranking
- metric: competition metric if known, otherwise "unknown"
- domain: domain/family, e.g. tabular_credit_risk, medical_imaging, nlp
- kaggle_queries: queries for Kaggle notebooks/writeups
- arxiv_queries: queries for papers and methods
- github_queries: queries for repositories
- key_techniques: likely technique names to search for
- similar_competitions: known similar Kaggle competitions if any

Do not claim train/test data was analyzed. Do not propose executing notebooks,
running EDA, adversarial validation, or leakage detection. Produce search terms,
not final modeling conclusions."""


async def plan(description: str, client: DeepSeekClient, model: str) -> PlanData:
    response = await client.chat_json(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(description),
        timeout=90,
    )
    return PlanData.model_validate(_with_optional_list_defaults(response))


def fallback_plan(description: str) -> PlanData:
    normalized = " ".join(description.lower().split())
    task_type = _infer_task_type(normalized)
    metric = _infer_metric(normalized)
    domain = _infer_domain(normalized)
    keywords = _extract_keywords(normalized)
    base_query = " ".join(keywords[:6]) if keywords else f"{domain} {task_type}"

    key_techniques = _fallback_techniques(task_type, metric, domain)

    return PlanData(
        task_type=task_type,
        metric=metric,
        domain=domain,
        kaggle_queries=[
            f"{base_query} kaggle solution",
            f"{domain} {metric} kaggle notebook",
            f"{task_type} feature engineering kaggle",
        ],
        arxiv_queries=[
            f"{domain} {task_type} {metric}",
            f"{domain} machine learning benchmark",
            f"{metric} optimization {task_type}",
        ],
        github_queries=[
            f"{domain} {task_type} kaggle",
            f"{metric} {task_type} solution",
            f"{domain} machine learning baseline",
        ],
        key_techniques=key_techniques,
        similar_competitions=[
            f"{domain} kaggle competitions",
            f"{task_type} kaggle benchmark",
        ],
    )


def _with_optional_list_defaults(response: dict[str, Any]) -> dict[str, Any]:
    data = dict(response)
    for field in OPTIONAL_LIST_FIELDS:
        data.setdefault(field, [])
    return data


def _build_user_prompt(description: str) -> str:
    return (
        "Competition description:\n"
        f"{description.strip()}\n\n"
        "Return the planning JSON object now."
    )


def _infer_task_type(normalized_description: str) -> str:
    if any(token in normalized_description for token in ("rank", "ndcg", "map@", "recommend")):
        return "ranking"
    if any(token in normalized_description for token in ("rmse", "rmsle", "mae", "regression", "predict price")):
        return "regression"
    if any(token in normalized_description for token in ("segmentation", "dice", "mask")):
        return "segmentation"
    if any(token in normalized_description for token in ("classify", "classification", "auc", "logloss", "f1")):
        return "classification"
    return "tabular_prediction"


def _infer_metric(normalized_description: str) -> str:
    metric_patterns = (
        ("roc auc", "roc_auc"),
        ("auc", "auc"),
        ("log loss", "log_loss"),
        ("logloss", "log_loss"),
        ("rmse", "rmse"),
        ("rmsle", "rmsle"),
        ("mae", "mae"),
        ("f1", "f1"),
        ("dice", "dice"),
        ("map@", "map@k"),
        ("ndcg", "ndcg"),
    )
    for pattern, metric in metric_patterns:
        if pattern in normalized_description:
            return metric
    return "unknown"


def _infer_domain(normalized_description: str) -> str:
    domain_patterns = (
        (("credit", "loan", "default", "bank"), "credit_risk_tabular"),
        (("medical", "image", "xray", "mri", "ct scan"), "medical_imaging"),
        (("recommend", "user", "item", "click"), "recommender_system"),
        (("time series", "forecast", "sales", "demand"), "time_series"),
        (("text", "nlp", "language", "review"), "nlp"),
        (("image", "vision", "classification"), "computer_vision"),
    )
    for terms, domain in domain_patterns:
        if any(term in normalized_description for term in terms):
            return domain
    return "general_kaggle"


def _fallback_techniques(task_type: str, metric: str, domain: str) -> list[str]:
    techniques = ["cross-validation", "baseline model"]

    if "tabular" in domain or task_type in {"classification", "regression", "tabular_prediction"}:
        techniques.extend(["gradient boosting", "feature engineering"])
    if metric in {"auc", "roc_auc"}:
        techniques.extend(["rank averaging", "stratified validation"])
    if metric == "log_loss":
        techniques.extend(["probability calibration", "prediction clipping"])
    if task_type == "ranking":
        techniques.extend(["candidate generation", "learning to rank"])
    if task_type == "segmentation":
        techniques.extend(["dice optimization", "augmentation"])

    return list(dict.fromkeys(techniques))


def _extract_keywords(normalized_description: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "predict",
        "the",
        "to",
        "with",
    }
    words = [
        word.strip(".,:;()[]{}\"'")
        for word in normalized_description.split()
        if len(word.strip(".,:;()[]{}\"'")) > 2
    ]
    unique_words = [word for word in dict.fromkeys(words) if word not in stopwords]
    return unique_words[:12]
