"""
Deterministic rule-based fallback composer.
Never returns an empty body. Fires on LLM timeout, LLM error, or validation failure.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple


# ── per-trigger-kind templates (fact-driven) ──
_TEMPLATES: Dict[str, str] = {
    "renewal_due": (
        "Hi{name_part}! Your plan renewal is due{days_part}{amount_part}. "
        "Renew today to ensure uninterrupted discovery and customer booking."
    ),
    "curious_ask_due": (
        "Hi{name_part}! Recent visitors are actively viewing your profile{metric_part}. "
        "Answering their top questions now can convert curious visitors into paying customers."
    ),
    "appointment_tomorrow": (
        "Reminder{name_part}: your upcoming appointments are confirmed for tomorrow. "
        "Review your daily schedule to ensure a smooth, on-time client experience."
    ),
    "chronic_refill_due": (
        "Hi{name_part}! Repeat client refill cycles are approaching this week{metric_part}. "
        "A timely check-in helps maintain treatment continuity and client retention."
    ),
    "customer_lapsed_soft": (
        "Hi{name_part}! We noticed some past regular clients haven't visited recently{metric_part}. "
        "A targeted reconnect offer can bring them back this week."
    ),
    "winback_eligible": (
        "Hi{name_part}! High-intent past customers are eligible for a win-back campaign{metric_part}. "
        "Reactivate them with an exclusive welcome-back perk."
    ),
    "active_planning_intent": (
        "Hi{name_part}! Customers in your area are actively searching for services{metric_part}. "
        "Highlight your top-rated offerings now to capture this local demand."
    ),
    "research_digest": (
        "Hi{name_part}! Here's your latest category research digest{headline_part}. "
        "Stay ahead of the curve with fresh industry insights and operational benchmarks."
    ),
    "perf_spike": (
        "Great news{name_part}! Your performance is trending up{metric_part}{headline_part}. "
        "Keep the momentum going — your customers are noticing and acting."
    ),
    "perf_dip": (
        "Heads up{name_part} — we noticed a dip in customer traffic{metric_part}{headline_part}. "
        "Let's launch an active offer to get engagement back on track."
    ),
    "recall_due": (
        "Reminder{name_part}: it's time to reconnect with your customers for follow-up care{metric_part}. "
        "A quick proactive follow-up maintains steady appointment flow."
    ),
    "dormant_with_vera": (
        "Hi{name_part}! Vera is ready to help optimize your listing{metric_part}. "
        "Check your dashboard to unlock fresh local growth opportunities."
    ),
    "milestone_reached": (
        "Congratulations{name_part}! You've reached a major performance milestone{metric_part}{headline_part}. "
        "Celebrate this achievement and build on your customer trust."
    ),
    "festival_upcoming": (
        "A festive season is approaching{name_part}{headline_part}. "
        "Prepare your seasonal offers to capture increased festival spending."
    ),
    "competitor_opened": (
        "New competition spotted nearby{name_part}{headline_part}. "
        "Now's the time to highlight your unique strengths and signature offerings."
    ),
}

_DEFAULT_TEMPLATE = (
    "Hi{name_part}! We have a performance update for you{metric_part}{headline_part}. "
    "Check your live metrics and let's optimize your customer engagement."
)

_CTA_MAP: Dict[str, str] = {
    "research_digest": "View full digest",
    "perf_spike": "See your stats",
    "perf_dip": "Review suggestions",
    "recall_due": "Send a message now",
    "dormant_with_vera": "Reactivate now",
    "milestone_reached": "Share your achievement",
    "festival_upcoming": "Plan your offers",
    "competitor_opened": "Boost your listing",
}


def compose_rule_based(
    facts: Dict[str, Any],
    trigger_kind: str = "unknown",
) -> Tuple[str, Optional[str], str, str, list]:
    """
    Returns (body, cta, rationale, template_name, template_params).
    """
    owner_name = facts.get("merchant_owner_first_name", "")
    merchant_name = facts.get("merchant_name", "")
    display_name = owner_name or merchant_name
    name_part = f", {display_name}" if display_name else ""

    headline = facts.get("headline", facts.get("trigger_headline", ""))
    headline_part = f" — {headline}" if headline else ""

    # Extract concrete facts for maximum specificity
    trigger_data = facts.get("trigger_data", {})
    metrics = facts.get("merchant_metrics", {})

    days = trigger_data.get("days_remaining", trigger_data.get("days_left"))
    days_part = f" in {days} days" if days is not None else ""

    amt = trigger_data.get("renewal_amount", trigger_data.get("amount", trigger_data.get("price")))
    amount_part = f" for ₹{amt}" if amt is not None else ""

    metric_parts = []
    if "views" in metrics:
        metric_parts.append(f"{metrics['views']} views")
    if "calls" in metrics:
        metric_parts.append(f"{metrics['calls']} calls")
    if "delta_pct" in trigger_data:
        delta = trigger_data['delta_pct']
        metric_parts.append(f"{delta:+}% change" if isinstance(delta, (int, float)) else f"{delta}% change")
    metric_part = f" ({', '.join(metric_parts)})" if metric_parts else ""

    template_str = _TEMPLATES.get(trigger_kind, _DEFAULT_TEMPLATE)
    body = template_str.format(
        name_part=name_part,
        headline_part=headline_part,
        days_part=days_part,
        amount_part=amount_part,
        metric_part=metric_part,
    )

    cta = _CTA_MAP.get(trigger_kind, "Learn more")

    template_name = f"vera_{trigger_kind}_v1"
    template_params = [p for p in [merchant_name, headline] if p]

    rationale = (
        f"Rule-based fallback for trigger kind '{trigger_kind}'"
        f"{' — ' + headline if headline else ''}."
    )

    return body, cta, rationale, template_name, template_params


def compose_reply_rule_based(
    intent: str,
    facts: Dict[str, Any],
    message: str = "",
) -> Dict[str, Any]:
    """
    Rule-based reply for /v1/reply when LLM is unavailable.
    Returns a dict matching one of ReplySend / ReplyWait / ReplyEnd.
    """
    merchant_name = facts.get("merchant_name", "")

    if intent == "hostile":
        return {
            "action": "end",
            "rationale": "Customer expressed hostility or opt-out; ending conversation gracefully.",
        }
    elif intent == "auto_reply_end":
        return {
            "action": "end",
            "rationale": "Repeated auto-reply detected; ending to avoid spam.",
        }
    elif intent == "auto_reply_first":
        return {
            "action": "send",
            "body": (
                f"Hi! This is Vera, reaching out on behalf of {merchant_name or 'your service provider'}. "
                "I'd love to help — feel free to ask me anything or let me know how I can assist you today."
            ),
            "cta": None,
            "rationale": "First auto-reply detected; attempting to route around with a personal touch.",
        }
    elif intent == "positive_intent":
        svc = ""
        services = facts.get("merchant_services")
        if isinstance(services, list) and services:
            first = services[0]
            if isinstance(first, dict):
                svc_name = first.get("name", "")
                svc_price = first.get("price", "")
                svc = f" for {svc_name}" if svc_name else ""
                if svc_price:
                    svc += f" @ ₹{svc_price}"
        return {
            "action": "send",
            "body": (
                f"Done! Here's what happens next{svc}: "
                "we'll confirm your booking and send you all the details shortly. "
                "Proceed whenever you're ready."
            ),
            "cta": "Confirm booking",
            "rationale": "Explicit positive intent detected; responding with action-mode vocabulary.",
        }
    elif intent == "question":
        return {
            "action": "send",
            "body": (
                "Thanks for asking! Based on what I have on file"
                f"{' for ' + merchant_name if merchant_name else ''}, "
                "I don't have the exact details you're looking for right now. "
                "Let me check and get back to you."
            ),
            "cta": None,
            "rationale": "Question detected; answering from available context.",
        }
    else:
        # default continuation
        return {
            "action": "send",
            "body": (
                "Thanks for your message! "
                f"{'We at ' + merchant_name + ' are' if merchant_name else 'We are'} "
                "here to help. What would you like to know?"
            ),
            "cta": "View our services",
            "rationale": "Default rule-based continuation.",
        }
