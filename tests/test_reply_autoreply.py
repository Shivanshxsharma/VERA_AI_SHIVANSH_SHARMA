"""
Test /v1/reply auto-reply detection.
Reproduces judge_simulator auto_reply_hell: same literal string sent 4x.

CRITICAL: The judge sends each auto-reply in a SEPARATE conversation
(conv_auto_1, conv_auto_2, etc.), so pattern detection must work
cross-conversation, not just within a single conversation.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.store import store
from app.graph.conversation_graph import reset_global_auto_reply_tracker


@pytest.fixture(autouse=True)
def clean_store():
    store.wipe()
    reset_global_auto_reply_tracker()
    yield
    store.wipe()
    reset_global_auto_reply_tracker()


client = TestClient(app)

AUTO_REPLY_MSG = "Thank you for contacting us! Our team will respond shortly."
MERCHANT_ID = "m1"


def _reply(conv_id: str, turn: int, msg: str = AUTO_REPLY_MSG):
    return client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": MERCHANT_ID,
        "customer_id": None,
        "from_role": "merchant",
        "message": msg,
        "received_at": "2025-06-01T10:00:00Z",
        "turn_number": turn,
    })


def test_first_auto_reply_tries_to_route_around():
    """First auto-reply -> send (try to route around)."""
    r = _reply("conv_auto_1", 2)
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "send", f"Expected 'send' on first auto-reply, got {body}"
    assert body.get("body"), "Body should not be empty on first auto-reply"


def test_second_auto_reply_ends():
    """Second auto-reply (different conversation) -> end."""
    _reply("conv_auto_1", 2)  # first -> send
    r = _reply("conv_auto_2", 2)  # second (different conv!) -> end
    body = r.json()
    assert body["action"] == "end", f"Expected 'end' on second auto-reply, got {body}"


def test_judge_auto_reply_hell_four_turns():
    """
    Mirrors judge_simulator._auto_reply exactly:
    4 turns, each in a separate conversation (conv_auto_1..4),
    same auto-reply string. The bot should end at some point.
    """
    ended = False
    for i in range(1, 5):
        r = _reply(f"conv_auto_{i}", i + 1)
        body = r.json()
        if body["action"] == "end":
            ended = True
            # All subsequent should also end
            break
        elif i == 1:
            assert body["action"] == "send", (
                f"Turn 1: expected 'send' but got {body['action']}"
            )

    assert ended, "Bot should have ended after seeing repeated auto-reply pattern"


def test_same_conversation_repeat_ends():
    """
    Within a SINGLE conversation, same message repeated -> end on 2nd.
    """
    r1 = _reply("conv_single", 1, "Some custom repeated message")
    assert r1.json()["action"] != "end"  # first time, not auto-reply pattern

    # Record the message in history manually for the test
    store.append_conversation_turn("conv_single", {
        "from_role": "merchant",
        "message": "Some custom repeated message",
        "turn_number": 1,
    })

    r2 = _reply("conv_single", 2, "Some custom repeated message")
    body = r2.json()
    assert body["action"] == "end", f"Expected 'end' on repeat, got {body}"
