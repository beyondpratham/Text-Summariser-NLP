import pytest
from fastapi.testclient import TestClient

from textsumm.api import ModelRegistry, app, get_registry


class FakeSummarizer:
    def summarize(self, texts, batch_size=8):
        return [f"सारांश {i}" for i, _ in enumerate(texts)]


@pytest.fixture
def client():
    registry = ModelRegistry(checkpoint="fake")
    registry._model = FakeSummarizer()
    app.dependency_overrides[get_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_configured": True, "model_loaded": True}


def test_summarize_with_lead(client, article):
    response = client.post("/summarize", json={"texts": [article], "method": "lead", "num_sentences": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["summaries"] == ["दिल्ली में आज भारी बारिश हुई।"]
    assert body["latency_ms"] >= 0


def test_summarize_batch_with_model(client, article):
    response = client.post("/summarize", json={"texts": [article, article]})
    assert response.status_code == 200
    assert response.json()["summaries"] == ["सारांश 0", "सारांश 1"]


@pytest.mark.parametrize(
    "payload",
    [{"texts": []}, {"texts": ["   "]}, {"texts": ["क" * 20_001]}, {"texts": ["ok"], "num_sentences": 0}],
)
def test_summarize_rejects_invalid_input(client, payload):
    assert client.post("/summarize", json=payload).status_code == 422


def test_model_request_without_checkpoint_returns_503(article):
    app.dependency_overrides[get_registry] = lambda: ModelRegistry(checkpoint=None)
    try:
        response = TestClient(app).post("/summarize", json={"texts": [article]})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
