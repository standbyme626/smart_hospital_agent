import os
import sys
from unittest.mock import Mock

import requests

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )
)

from app.services import embedding as embedding_module
from app.services.embedding import EmbeddingService


def _build_service(*, local_enabled: bool, remote_enabled: bool = True) -> EmbeddingService:
    service = object.__new__(EmbeddingService)
    service.remote_enabled = remote_enabled
    service.remote_url = "http://100.90.236.32:11434"
    service.remote_model = "qwen3-embedding:0.6b"
    service.remote_timeout_s = 10
    service.local_enabled = local_enabled
    service.degraded_mode = False
    service.dim = 4
    return service


def test_get_embedding_remote_success(monkeypatch):
    service = _build_service(local_enabled=False)
    monkeypatch.setattr(embedding_module.vram_manager, "mark_used", lambda *_args, **_kwargs: None)

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}
    post_mock = Mock(return_value=response)
    monkeypatch.setattr(embedding_module.requests, "post", post_mock)

    result = service.get_embedding("头痛发热")

    assert result == [0.1, 0.2, 0.3, 0.4]
    assert service.degraded_mode is False
    post_mock.assert_called_once_with(
        "http://100.90.236.32:11434/api/embed",
        json={"model": "qwen3-embedding:0.6b", "input": ["头痛发热"]},
        timeout=10,
    )


def test_get_embedding_remote_failure_local_disabled_enters_degraded(monkeypatch):
    service = _build_service(local_enabled=False)
    monkeypatch.setattr(embedding_module.vram_manager, "mark_used", lambda *_args, **_kwargs: None)

    post_mock = Mock(side_effect=requests.RequestException("remote unavailable"))
    monkeypatch.setattr(embedding_module.requests, "post", post_mock)

    result = service.get_embedding("咳嗽")

    assert result == [0.001, 0.001, 0.001, 0.001]
    assert service.degraded_mode is True


def test_batch_get_embeddings_remote_failure_local_enabled_fallback(monkeypatch):
    service = _build_service(local_enabled=True)
    monkeypatch.setattr(embedding_module.vram_manager, "mark_used", lambda *_args, **_kwargs: None)

    post_mock = Mock(side_effect=requests.Timeout("timeout"))
    monkeypatch.setattr(embedding_module.requests, "post", post_mock)
    local_fallback = Mock(return_value=[[0.9, 0.8], [0.7, 0.6]])
    monkeypatch.setattr(service, "_batch_get_embeddings_local", local_fallback)

    result = service.batch_get_embeddings(["a", "b"], batch_size=2)

    assert result == [[0.9, 0.8], [0.7, 0.6]]
    local_fallback.assert_called_once_with(["a", "b"], batch_size=2)
