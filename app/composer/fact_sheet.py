"""
Build a deterministic FACT SHEET from stored context.
Pure Python — no LLM. This is the ONLY source of truth the LLM composer
may reference.

Handles both nested (identity.name) and flat (name at top level) formats
from the seed data.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def _safe_get(d: Optional[Dict], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def build_fact_sheet(
    trigger_payload: Dict[str, Any],
    merchant_payload: Optional[Dict[str, Any]],
    category_payload: Optional[Dict[str, Any]],
    customer_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return a flat-ish dict of ONLY verifiable facts present in stored context.
    Keys not present in the source data are omitted (not set to None).
    """
    facts: Dict[str, Any] = {}

    # ── trigger facts ──
    # Handle both flat and nested trigger formats
    facts["trigger_kind"] = trigger_payload.get("kind", "unknown")
    for field in ("trigger_id", "id", "headline", "body", "expires_at",
                  "urgency", "merchant_id", "customer_id", "scope", "source"):
        val = trigger_payload.get(field)
        if val is not None:
            key = f"trigger_{field}" if field not in ("merchant_id", "customer_id") else field
            facts[key] = val

    # Nested trigger payload (trigger-specific data from seeds)
    nested_payload = trigger_payload.get("payload", {})
    if isinstance(nested_payload, dict):
        facts["trigger_data"] = nested_payload
        # Extract common nested fields for easy access
        for field in ("service_due", "due_date", "metric", "delta_pct",
                      "festival", "days_remaining", "plan", "renewal_amount",
                      "headline"):
            if field in nested_payload:
                facts[f"trigger_{field}"] = nested_payload[field]

    # ── merchant facts ──
    if merchant_payload:
        # Handle both nested (identity.name) and flat (name at top level)
        identity = merchant_payload.get("identity", {})
        if isinstance(identity, dict):
            for key in ("name", "locality", "city", "languages",
                        "category_slug", "owner_first_name", "established_year"):
                if key in identity:
                    facts[f"merchant_{key}"] = identity[key]

        # Top-level category_slug (seed format)
        if "category_slug" in merchant_payload and "merchant_category_slug" not in facts:
            facts["merchant_category_slug"] = merchant_payload["category_slug"]

        # Services / offers
        services = merchant_payload.get("services")
        if services:
            facts["merchant_services"] = services

        offers = merchant_payload.get("offers")
        if offers:
            active_offers = [o for o in offers
                           if isinstance(o, dict) and o.get("status") == "active"]
            if active_offers:
                facts["merchant_active_offers"] = active_offers

        # Performance
        perf = merchant_payload.get("performance")
        if perf:
            facts["merchant_performance"] = perf

        # Hours
        hours = merchant_payload.get("hours")
        if hours:
            facts["merchant_hours"] = hours

        # Signals
        signals = merchant_payload.get("signals")
        if signals:
            facts["merchant_signals"] = signals

        # Subscription
        sub = merchant_payload.get("subscription")
        if sub:
            facts["merchant_subscription"] = sub

        # Review themes
        reviews = merchant_payload.get("review_themes")
        if reviews:
            facts["merchant_review_themes"] = reviews

        # Customer aggregate
        cust_agg = merchant_payload.get("customer_aggregate")
        if cust_agg:
            facts["merchant_customer_aggregate"] = cust_agg

    # ── category facts ──
    if category_payload:
        voice = category_payload.get("voice", {})
        if isinstance(voice, dict):
            facts["voice_tone"] = voice.get("tone", "friendly")
            if "register" in voice:
                facts["voice_register"] = voice["register"]
            if "code_mix" in voice:
                facts["voice_code_mix"] = voice["code_mix"]
            if "vocab_allowed" in voice:
                facts["vocab_allowed"] = voice["vocab_allowed"]
            if "salutation_examples" in voice:
                facts["salutation_examples"] = voice["salutation_examples"]
            # Union both taboo field name variants
            taboos = set()
            for key in ("taboos", "vocab_taboo"):
                val = voice.get(key)
                if isinstance(val, list):
                    taboos.update(str(w) for w in val)
            if taboos:
                facts["taboo_words"] = sorted(taboos)

        if "slug" in category_payload:
            facts["category_slug"] = category_payload["slug"]
        if "display_name" in category_payload:
            facts["category_name"] = category_payload["display_name"]

        # Offer catalog
        catalog = category_payload.get("offer_catalog")
        if catalog:
            facts["category_offer_catalog"] = catalog

        # Peer stats
        peer = category_payload.get("peer_stats")
        if peer:
            facts["category_peer_stats"] = peer

        # Digest items
        digest = category_payload.get("digest")
        if digest:
            facts["category_digest"] = digest

    # ── customer facts ──
    if customer_payload:
        cust_identity = customer_payload.get("identity", {})
        if isinstance(cust_identity, dict):
            for key in ("name", "language_pref", "age_band"):
                if key in cust_identity:
                    facts[f"customer_{key}"] = cust_identity[key]

        relationship = customer_payload.get("relationship")
        if relationship:
            facts["customer_relationship"] = relationship

        state = customer_payload.get("state")
        if state:
            facts["customer_state"] = state

        preferences = customer_payload.get("preferences")
        if preferences:
            facts["customer_preferences"] = preferences

        history = customer_payload.get("history")
        if history:
            facts["customer_history"] = history

    return facts


def fact_sheet_to_text(facts: Dict[str, Any]) -> str:
    """Render the fact sheet as a human-readable block for the LLM system prompt."""
    lines: List[str] = ["=== FACT SHEET (only source of truth) ==="]
    for k, v in facts.items():
        lines.append(f"- {k}: {v}")
    lines.append("=== END FACT SHEET ===")
    return "\n".join(lines)
