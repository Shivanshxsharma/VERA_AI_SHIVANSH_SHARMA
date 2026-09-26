"""
Test /v1/reply hostility detection.
"Stop messaging me. This is useless spam." -> brief acknowledgment or end.
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

CONV_ID = "conv-hostile-test"


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


def test_hostile_message_acknowledged():
    r = _reply("Stop messaging me. This is useless spam.")
    body = r.json()
    # Should either send a brief acknowledgment or end
    assert body["action"] in ("send", "end"), f"Expected send or end, got {body}"
    if body["action"] == "send":
        resp = body["body"].lower()
        # Should contain apologetic/disengaging language
        has_disengage = any(w in resp for w in ["sorry", "no problem", "won't message", "apologize", "won't bother", "take care"])
        assert has_disengage, f"Hostile response should be brief & disengaging: {body['body']}"
        # Should NOT re-pitch
        repitch_words = ["offer", "discount", "deal", "check out", "try our"]
        for w in repitch_words:
            assert w not in resp, f"Should NOT re-pitch with '{w}': {body['body']}"


def test_not_interested():
    r = _reply("Not interested, please stop")
    body = r.json()
    assert body["action"] in ("send", "end")


def test_profanity():
    r = _reply("This is fucking annoying, stop it")
    body = r.json()
    assert body["action"] in ("send", "end")
