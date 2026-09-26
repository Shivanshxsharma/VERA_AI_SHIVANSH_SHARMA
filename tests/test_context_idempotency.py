"""
Test POST /v1/context idempotency: 409 on stale, 200 on new/higher version.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.store import store


@pytest.fixture(autouse=True)
def clean_store():
    store.wipe()
    yield
    store.wipe()


client = TestClient(app)


def _ctx(scope="merchant", context_id="m1", version=1, payload=None):
    return {
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload or {"identity": {"name": "Test Shop"}},
        "delivered_at": "2025-01-01T00:00:00Z",
    }


def test_first_insert_returns_200():
    r = client.post("/v1/context", json=_ctx(version=1))
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert "ack_id" in body
    assert "stored_at" in body


def test_same_version_returns_409():
    client.post("/v1/context", json=_ctx(version=1))
    r = client.post("/v1/context", json=_ctx(version=1))
    assert r.status_code == 409
    body = r.json()
    assert body["accepted"] is False
    assert body["reason"] == "stale_version"
    assert body["current_version"] == 1


def test_lower_version_returns_409():
    client.post("/v1/context", json=_ctx(version=3))
    r = client.post("/v1/context", json=_ctx(version=2))
    assert r.status_code == 409
    body = r.json()
    assert body["current_version"] == 3


def test_higher_version_returns_200():
    client.post("/v1/context", json=_ctx(version=1))
    r = client.post("/v1/context", json=_ctx(version=2))
    assert r.status_code == 200
    assert r.json()["accepted"] is True


def test_invalid_scope_returns_400():
    r = client.post("/v1/context", json=_ctx(scope="invalid_scope"))
    assert r.status_code == 400
    body = r.json()
    assert body["reason"] == "invalid_scope"


def test_payload_too_large_returns_413():
    big_payload = {"data": "x" * 600_000}
    r = client.post("/v1/context", json=_ctx(payload=big_payload))
    assert r.status_code == 413
