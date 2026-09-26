"""
Test /v1/reply intent transition detection.
"Ok lets do it. Whats next?" -> action-mode vocabulary, no qualifying questions.
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

CONV_ID = "conv-intent-test"
ACTION_WORDS = {"done", "sending", "draft", "here", "confirm", "proceed", "next", "booking", "details", "ready"}
QUALIFYING_WORDS = {"would you", "do you", "can you tell", "what if", "how about"}


def _reply(msg: str, turn: int = 1):
    return client.post("/v1/reply", json={
        "conversation_id": CONV_ID,
        "merchant_id": "m1",
        "customer_id": None,
        "from_role": "customer",
        "message": msg,
        "received_at": "2025-06-01T10:00:00Z",
        "turn_number": turn,
    })


def test_explicit_intent_triggers_action_mode():
    r = _reply("Ok lets do it. Whats next?")
    body = r.json()
    assert body["action"] == "send", f"Expected 'send', got {body}"
    response_body = body["body"].lower()

    # Must contain at least one action-mode word
    has_action = any(w in response_body for w in ACTION_WORDS)
    assert has_action, f"Response should use action-mode vocabulary: {body['body']}"

    # Must NOT contain qualifying questions
    for q in QUALIFYING_WORDS:
        assert q not in response_body, (
            f"Response should NOT contain qualifying question '{q}': {body['body']}"
        )


def test_hindi_positive_intent():
    r = _reply("Haan chaliye, shuru karte hain")
    body = r.json()
    assert body["action"] == "send"


def test_simple_yes():
    r = _reply("Yes")
    body = r.json()
    assert body["action"] == "send"
