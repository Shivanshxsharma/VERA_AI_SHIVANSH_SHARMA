"""
Versioned in-memory context store.
Key = (scope, context_id) -> {"version": int, "payload": dict, "delivered_at": str}

Behind a thin repository interface so it can be swapped for Redis later.
"""
from __future__ import annotations
import threading
from typing import Any, Dict, List, Optional, Tuple

_Scope = str   # "category" | "merchant" | "customer" | "trigger"
_Key = Tuple[_Scope, str]

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


class ContextStore:
    """Thread-safe, versioned, in-memory context store."""

    def __init__(self) -> None:
        self._data: Dict[_Key, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Conversation history for /v1/reply anti-repetition
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}
        # Suppression ledger: suppression_key -> expiry iso8601
        self._suppression: Dict[str, str] = {}

    # ── context CRUD ──

    def get(self, scope: str, context_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get((scope, context_id))

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        entry = self._data.get((scope, context_id))
        return entry["version"] if entry else None

    def upsert(self, scope: str, context_id: str, version: int,
               payload: dict, delivered_at: str) -> bool:
        """Store atomically. Returns True if stored, False if stale (caller handles 409)."""
        with self._lock:
            current = self._data.get((scope, context_id))
            if current and version <= current["version"]:
                return False
            self._data[(scope, context_id)] = {
                "version": version,
                "payload": payload,
                "delivered_at": delivered_at,
            }
            return True

    def count_by_scope(self, scope: str) -> int:
        return sum(1 for (s, _) in self._data if s == scope)

    def all_of_scope(self, scope: str) -> List[Dict[str, Any]]:
        return [
            {"context_id": cid, **entry}
            for (s, cid), entry in self._data.items()
            if s == scope
        ]

    # ── conversation history ──

    def append_conversation_turn(self, conversation_id: str, turn: Dict[str, Any]) -> None:
        with self._lock:
            self._conversations.setdefault(conversation_id, []).append(turn)

    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self._conversations.get(conversation_id, [])

    # ── suppression ledger ──

    def is_suppressed(self, key: str, now_iso: str) -> bool:
        expiry = self._suppression.get(key)
        if expiry is None:
            return False
        try:
            from datetime import datetime
            now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
            exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            return now_dt < exp_dt
        except Exception:
            return now_iso < expiry

    def suppress(self, key: str, until_iso: str) -> None:
        with self._lock:
            self._suppression[key] = until_iso

    # ── teardown ──

    def wipe(self) -> None:
        with self._lock:
            self._data.clear()
            self._conversations.clear()
            self._suppression.clear()


# Module-level singleton
store = ContextStore()
