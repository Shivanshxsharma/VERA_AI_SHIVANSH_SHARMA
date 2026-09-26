"""
Test POST /v1/tick response shape including template_name / template_params.
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


def _seed():
    """Seed a category, merchant, and trigger."""
    client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dental",
        "version": 1,
        "payload": {
            "slug": "dental",
            "display_name": "Dental",
            "voice": {"tone": "professional", "vocab_allowed": ["health", "care"]},
        },
        "delivered_at": "2025-01-01T00:00:00Z",
    })
    client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m1",
        "version": 1,
        "payload": {
            "identity": {
                "name": "Bright Smile Dental",
                "category_slug": "dental",
                "city": "Delhi",
                "languages": ["en"],
            },
            "services": [{"name": "Dental Cleaning", "price": "299"}],
        },
        "delivered_at": "2025-01-01T00:00:00Z",
    })
    client.post("/v1/context", json={
        "scope": "trigger",
        "context_id": "t1",
        "version": 1,
        "payload": {
            "kind": "perf_spike",
            "merchant_id": "m1",
            "headline": "Bookings up 40% this week",
            "urgency": 8,
            "expires_at": "2099-12-31T23:59:59Z",
        },
        "delivered_at": "2025-01-01T00:00:00Z",
    })


def test_tick_returns_actions_with_required_fields():
    _seed()
    r = client.post("/v1/tick", json={
        "now": "2025-06-01T10:00:00Z",
        "available_triggers": ["t1"],
    })
    assert r.status_code == 200
    body = r.json()
    assert "actions" in body
    assert len(body["actions"]) >= 1

    action = body["actions"][0]
    required_keys = {
        "conversation_id", "merchant_id", "customer_id",
        "send_as", "trigger_id", "template_name", "template_params",
        "body", "cta", "suppression_key", "rationale",
    }
    assert required_keys.issubset(set(action.keys())), (
        f"Missing keys: {required_keys - set(action.keys())}"
    )
    assert action["merchant_id"] == "m1"
    assert action["send_as"] == "vera"  # no customer_id -> vera
    assert action["template_name"]  # must not be empty
    assert isinstance(action["template_params"], list)
    assert action["body"]  # must not be empty


def test_tick_empty_triggers():
    r = client.post("/v1/tick", json={
        "now": "2025-06-01T10:00:00Z",
        "available_triggers": [],
    })
    assert r.status_code == 200
    assert r.json()["actions"] == []


def test_tick_unknown_trigger_skipped():
    r = client.post("/v1/tick", json={
        "now": "2025-06-01T10:00:00Z",
        "available_triggers": ["nonexistent"],
    })
    assert r.status_code == 200
    assert r.json()["actions"] == []
