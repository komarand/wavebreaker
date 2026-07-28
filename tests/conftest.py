from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.schemas import RetrievedDocument, SourceDocument


FAKE_EMBEDDING_DIM = 1024
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class FakeSentenceTransformer:
    instances: list["FakeSentenceTransformer"] = []

    def __init__(self, model_name: str, device: str = "cpu", model_kwargs: dict | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.encode_calls: list[dict[str, Any]] = []
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_calls.append({"texts": list(texts), "kwargs": kwargs})
        return [fake_embedding_vector(index=index, dim=FAKE_EMBEDDING_DIM) for index, _ in enumerate(texts)]


class FakeHTTPResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def fake_embedding_vector(index: int = 0, dim: int = FAKE_EMBEDDING_DIM) -> list[float]:
    return [float((index + offset) % 17) / 17.0 for offset in range(dim)]


@pytest.fixture
def fake_embedding_dim() -> int:
    return FAKE_EMBEDDING_DIM


@pytest.fixture
def fake_embedding() -> list[float]:
    return fake_embedding_vector()


@pytest.fixture
def fake_sentence_transformer() -> type[FakeSentenceTransformer]:
    FakeSentenceTransformer.instances = []
    return FakeSentenceTransformer


@pytest.fixture
def fake_source_document() -> SourceDocument:
    return SourceDocument(
        id="doc-1",
        competition_id="comp-1",
        source="kaggle",
        title="Useful notebook",
        url="https://example.com/notebook",
        content="A deterministic source document for offline tests.",
        metadata={"votes": 10},
    )


@pytest.fixture
def fake_retrieved_document() -> RetrievedDocument:
    return RetrievedDocument(
        id="doc-1",
        competition_id="comp-1",
        source="kaggle",
        title="Useful notebook",
        url="https://example.com/notebook",
        content="A deterministic retrieved document for offline tests.",
        score=0.9,
        rrf_score=0.1,
        metadata={"votes": 10},
    )


@pytest.fixture
def fake_deepseek_json_response() -> dict[str, Any]:
    return {
        "task_type": "classification",
        "metric": "auc",
        "domain": "tabular",
        "kaggle_queries": ["tabular auc kaggle"],
        "arxiv_queries": ["tabular auc papers"],
        "github_queries": ["tabular auc github"],
    }


@pytest.fixture
def fake_http_response() -> type[FakeHTTPResponse]:
    return FakeHTTPResponse


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def pg_dsn() -> str:
    return os.getenv(
        "PG_DSN",
        "postgresql://researcher:researcher@localhost:5432/kaggle_research",
    )


@pytest.fixture(autouse=True)
def block_external_network_for_offline_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail non-network tests at the socket boundary before an outbound connection."""

    if request.node.get_closest_marker("network") is not None:
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def reject_external(address: object) -> None:
        if not isinstance(address, tuple) or not address:
            return
        if str(address[0]).lower() in LOOPBACK_HOSTS:
            return
        raise AssertionError(
            f"external network access is forbidden for offline test "
            f"{request.node.nodeid}: {address!r}"
        )

    def guarded_connect(
        sock: socket.socket,
        address: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        reject_external(address)
        original_connect(sock, address, *args, **kwargs)

    def guarded_connect_ex(
        sock: socket.socket,
        address: object,
        *args: object,
        **kwargs: object,
    ) -> int:
        reject_external(address)
        return original_connect_ex(sock, address, *args, **kwargs)

    def guarded_create_connection(
        address: tuple[str, int],
        *args: object,
        **kwargs: object,
    ) -> socket.socket:
        reject_external(address)
        return original_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_pg = os.getenv("RUN_PG_INTEGRATION") == "1"
    run_network = os.getenv("RUN_NETWORK_TESTS") == "1"
    run_real_embedding = os.getenv("RUN_REAL_EMBEDDING_TEST") == "1"

    skip_integration = pytest.mark.skip(reason="set RUN_PG_INTEGRATION=1 to run PostgreSQL tests")
    skip_network = pytest.mark.skip(reason="set RUN_NETWORK_TESTS=1 to run network tests")
    skip_real_embedding = pytest.mark.skip(
        reason="set RUN_NETWORK_TESTS=1 and RUN_REAL_EMBEDDING_TEST=1 to download/use a real model"
    )

    for item in items:
        path = Path(str(item.fspath)).as_posix()
        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        if "/tests/smoke/" in path:
            item.add_marker(pytest.mark.unit)
        if "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
            if not run_pg and "pipeline_smoke" not in item.keywords:
                item.add_marker(skip_integration)
        if "/tests/network/" in path:
            item.add_marker(pytest.mark.network)
            if not run_network:
                item.add_marker(skip_network)
        if "real_embedding" in item.name and not (run_network and run_real_embedding):
            item.add_marker(skip_real_embedding)
