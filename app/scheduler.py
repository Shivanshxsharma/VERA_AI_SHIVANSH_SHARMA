"""
Tick eligibility engine: suppression check, expiry, priority ranking, 20-action cap.

Handles both nested and flat trigger payload formats:
- Flat: trigger fields (kind, merchant_id, urgency, etc.) at top level of payload
- Nested: trigger fields under payload.payload (from seed data)
"""
from __future__ import annotations
import concurrent.futures
import time
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import TICK_ACTION_CAP, SUPPRESSION_TTL_SECONDS, GEMINI_API_KEY1, GEMINI_API_KEY2, MISTRAL_API_KEY, MISTRAL_MODEL
from app.store import store
from app.composer.fact_sheet import build_fact_sheet
from app.composer.rule_composer import compose_rule_based

logger = logging.getLogger("vera.scheduler")


def _parse_iso(s: str) -> datetime:
    """Parse ISO8601 string to datetime, tolerating various formats."""
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)


def _make_suppression_key(trigger_kind: str, scope: str, now: datetime) -> str:
    """Encode trigger kind + scope + stable time bucket (1-hour buckets)."""
    bucket = now.strftime("%Y-%m-%dT%H")
    raw = f"{trigger_kind}:{scope}:{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_trigger_field(payload: Dict, field: str, default: Any = None) -> Any:
    """
    Get a field from the trigger payload, checking both flat and nested formats.
    The seed data has triggers with fields at top level (kind, merchant_id, etc.)
    but also has a nested 'payload' dict with trigger-specific data.
    """
    # Check top level first
    if field in payload:
        return payload[field]
    # Check nested payload
    nested = payload.get("payload", {})
    if isinstance(nested, dict) and field in nested:
        return nested[field]
    return default


def process_tick(
    now_iso: str,
    available_trigger_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Main tick processing. Returns list of action dicts matching TickAction schema.
    """
    now_dt = _parse_iso(now_iso)
    actions: List[Dict[str, Any]] = []
    seen_merchants: set = set()  # one action per merchant
    llm_count = 0

    # ── Step 1: resolve triggers and gather eligible ones ──
    eligible: List[Tuple[float, str, str, Dict, Dict, Optional[Dict], Optional[Dict]]] = []

    for trigger_context_id in available_trigger_ids:
        trigger_entry = store.get("trigger", trigger_context_id)
        if not trigger_entry:
            logger.warning(f"Trigger {trigger_context_id} not found in store; skipping.")
            continue

        trigger_payload = trigger_entry["payload"]

        # Expiry check — check both top-level and nested
        expires_at = _get_trigger_field(trigger_payload, "expires_at")
        if expires_at:
            try:
                if _parse_iso(expires_at) < now_dt:
                    logger.info(f"Trigger {trigger_context_id} expired; skipping.")
                    continue
            except Exception:
                pass

        # Resolve merchant — check trigger_payload for merchant_id
        merchant_id = _get_trigger_field(trigger_payload, "merchant_id", "")
        merchant_entry = store.get("merchant", merchant_id) if merchant_id else None
        merchant_payload = merchant_entry["payload"] if merchant_entry else {}

        # Resolve category via merchant.category_slug or merchant.identity.category_slug
        category_slug = ""
        if merchant_payload:
            # Try top-level category_slug first (seed format)
            category_slug = merchant_payload.get("category_slug", "")
            if not category_slug:
                identity = merchant_payload.get("identity", {})
                if isinstance(identity, dict):
                    category_slug = identity.get("category_slug", "")
        category_entry = store.get("category", category_slug) if category_slug else None
        category_payload = category_entry["payload"] if category_entry else None

        # Resolve customer (optional)
        customer_id = _get_trigger_field(trigger_payload, "customer_id")
        customer_entry = store.get("customer", customer_id) if customer_id else None
        customer_payload = customer_entry["payload"] if customer_entry else None

        # Suppression check
        trigger_kind = _get_trigger_field(trigger_payload, "kind", "unknown")

        # Use the trigger's own suppression_key if provided, else generate one
        supp_key = _get_trigger_field(trigger_payload, "suppression_key")
        if not supp_key:
            supp_key = _make_suppression_key(
                trigger_kind,
                f"merchant:{merchant_id}" + (f":customer:{customer_id}" if customer_id else ""),
                now_dt,
            )

        if store.is_suppressed(supp_key, now_iso):
            logger.info(f"Trigger {trigger_context_id} suppressed ({supp_key}); skipping.")
            continue

        # Extract urgency for ranking
        urgency = _get_trigger_field(trigger_payload, "urgency", 0)
        if isinstance(urgency, str):
            urgency_map = {"low": 1, "medium": 5, "high": 8, "critical": 10}
            urgency = urgency_map.get(urgency.lower(), 0)
        elif not isinstance(urgency, (int, float)):
            urgency = 0

        expires_sort = expires_at or "9999-12-31T23:59:59Z"

        eligible.append((
            urgency, expires_sort, trigger_context_id,
            trigger_payload, merchant_payload, category_payload, customer_payload,
        ))

    # ── Step 2: rank — urgency desc, expiry asc, id asc ──
    eligible.sort(key=lambda x: (-x[0], x[1], x[2]))

    # ── Step 3: Gather candidates up to cap (one action per merchant) ──
    candidates: List[Tuple[float, str, str, Dict, Dict, Optional[Dict], Optional[Dict]]] = []
    for item in eligible:
        if len(candidates) >= TICK_ACTION_CAP:
            break
        m_id = _get_trigger_field(item[3], "merchant_id", "")
        if m_id in seen_merchants:
            continue
        seen_merchants.add(m_id)
        candidates.append(item)

    def _process_candidate(cand_item) -> Dict[str, Any]:
        (
            urgency_val, _, trigger_ctx_id,
            trg_p, merch_p, cat_p, cust_p,
        ) = cand_item
        m_id = _get_trigger_field(trg_p, "merchant_id", "")
        c_id = _get_trigger_field(trg_p, "customer_id")
        conv_id = str(uuid.uuid4())
        trg_k = _get_trigger_field(trg_p, "kind", "unknown")

        facts = build_fact_sheet(trg_p, merch_p, cat_p, cust_p)
        b_val, c_val, r_val = None, None, None
        t_name, t_params = None, None

        has_llm = bool(GEMINI_API_KEY1 or GEMINI_API_KEY2 or MISTRAL_API_KEY)
        if has_llm:
            try:
                from app.composer.llm_composer import _call_llm_sync
                from app.composer.validator import validate_composed_body
                llm_b, llm_c, llm_r, p_name = _call_llm_sync(facts, trg_k)
                validate_composed_body(llm_b, llm_c, facts, prior_bodies=[])
                b_val, c_val, r_val = llm_b, llm_c, llm_r
                t_name = f"llm_{p_name.split()[0]}"
                t_params = [p_name, trg_k]
            except Exception as exc:
                logger.warning("LLM composition error for %s (%s); falling back to rule composer.", trg_k, exc)

        if not b_val:
            b_val, c_val, r_val, t_name, t_params = compose_rule_based(facts, trg_k)

        s_as = "merchant_on_behalf" if c_id else "vera"
        s_key = _get_trigger_field(trg_p, "suppression_key")
        if not s_key:
            s_key = _make_suppression_key(
                trg_k,
                f"merchant:{m_id}" + (f":customer:{c_id}" if c_id else ""),
                now_dt,
            )

        return {
            "conversation_id": conv_id,
            "merchant_id": m_id,
            "customer_id": c_id,
            "send_as": s_as,
            "trigger_id": trigger_ctx_id,
            "template_name": t_name,
            "template_params": t_params,
            "body": b_val,
            "cta": c_val,
            "suppression_key": s_key,
            "rationale": r_val,
        }

    # ── Step 4: Process candidates concurrently with single tick-level deadline ──
    from app.config import TICK_TIMEOUT_SECONDS
    deadline = time.time() + TICK_TIMEOUT_SECONDS
    max_workers = min(6, len(candidates)) if candidates else 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cand = {executor.submit(_process_candidate, cand): cand for cand in candidates}
        for future in concurrent.futures.as_completed(future_to_cand):
            remaining = max(0.1, deadline - time.time())
            cand = future_to_cand[future]
            try:
                action = future.result(timeout=remaining)
            except Exception as exc:
                logger.warning("Action processing error/timeout (%s); generating rule fallback.", exc)
                action = _process_candidate(cand)

            actions.append(action)
            supp_until = (now_dt + timedelta(seconds=SUPPRESSION_TTL_SECONDS)).isoformat()
            store.suppress(action["suppression_key"], supp_until)

    return actions
