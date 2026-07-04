from __future__ import annotations

from types import SimpleNamespace

import pytest

from kaggle_researcher import embedder


class FakeSentenceTransformer:
    instances: list[FakeSentenceTransformer] = []

    def __init__(self, model_name: str, device: str, model_kwargs: dict | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.encode_calls: list[dict] = []
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        self.encode_calls.append({"texts": texts, "kwargs": kwargs})
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


@pytest.fixture(autouse=True)
def reset_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(embedder, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        embedder,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
        ),
    )
    monkeypatch.setattr(embedder, "_model", None)
    monkeypatch.setattr(embedder, "_embedding_dim", None)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("MAX_EMBED_BATCH_SIZE", raising=False)


def test_embed_texts_preserves_order_and_batches() -> None:
    result = embedder.embed_texts(["a", "bb", "ccc"], batch_size=2)

    assert result == [[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]]
    assert all(isinstance(row, list) for row in result)
    assert all(isinstance(value, float) for row in result for value in row)
    model = FakeSentenceTransformer.instances[0]
    assert [call["texts"] for call in model.encode_calls] == [["a", "bb"], ["ccc"]]
    assert all(call["kwargs"]["normalize_embeddings"] is True for call in model.encode_calls)


def test_embed_one_returns_single_embedding() -> None:
    result = embedder.embed_one("query")

    assert result == [0.0, 5.0]


def test_empty_list_returns_empty_list() -> None:
    assert embedder.embed_texts([]) == []
    assert FakeSentenceTransformer.instances == []


def test_get_embedding_dim_returns_detected_dimension() -> None:
    assert embedder.get_embedding_dim() == 2
    assert embedder.get_embedding_dim() == 2
    assert len(FakeSentenceTransformer.instances[0].encode_calls) == 1


def test_model_loads_lazily_once_and_reuses_default_qwen_model() -> None:
    assert FakeSentenceTransformer.instances == []

    first = embedder.embed_one("first")
    second = embedder.embed_one("second")

    assert first == [0.0, 5.0]
    assert second == [0.0, 6.0]
    assert len(FakeSentenceTransformer.instances) == 1
    assert FakeSentenceTransformer.instances[0].model_name == "Qwen/Qwen3-Embedding-0.6B"


def test_env_batch_size_controls_default_batching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_EMBED_BATCH_SIZE", "2")

    embedder.embed_texts(["a", "bb", "ccc"])

    model = FakeSentenceTransformer.instances[0]
    assert [call["texts"] for call in model.encode_calls] == [["a", "bb"], ["ccc"]]


def test_invalid_batch_size_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_EMBED_BATCH_SIZE", "not-an-int")

    with pytest.raises(embedder.EmbedderError, match="MAX_EMBED_BATCH_SIZE"):
        embedder.embed_texts(["a"])


def test_explicit_non_positive_batch_size_raises_value_error() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        embedder.embed_texts(["a"], batch_size=0)


def test_model_uses_cuda_bf16_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embedder,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
        ),
    )

    embedder.embed_one("query")

    model = FakeSentenceTransformer.instances[0]
    assert model.device == "cuda"
    assert model.model_kwargs == {"torch_dtype": "bf16"}


def test_cuda_without_bf16_support_falls_back_to_plain_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embedder,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False),
        ),
    )

    embedder.embed_one("query")

    model = FakeSentenceTransformer.instances[0]
    assert model.device == "cuda"
    assert model.model_kwargs == {}
