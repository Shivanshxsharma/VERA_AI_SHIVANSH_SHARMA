"""
LLM composer using a Multi-Tier Auto-Rotating Provider Architecture:
- Tier 1: Gemini 2.5 Flash (Primary Key 1)
- Tier 2: Gemini 2.5 Flash (Backup Key 2, auto-rotated on HTTP 429 rate limits / quota)
- Tier 3: Mistral AI (3rd Backup Failover)
- Tier 4: Deterministic rule_composer fallback

Constrained strictly to the fact sheet. Anti-hallucination verified.
"""
from __future__ import annotations
import asyncio
import functools
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.config import (
    GEMINI_API_KEY1,
    GEMINI_API_KEY2,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    MISTRAL_TIMEOUT_SECONDS,
)
from app.composer.fact_sheet import fact_sheet_to_text
from app.composer.validator import (
    validate_composed_body,
    validate_reply_facts,
    ValidationError,
    SAFE_QUESTION_FALLBACK,
)
from app.composer.rule_composer import compose_rule_based

logger = logging.getLogger("vera.llm_composer")

# Exact required anti-hallucination instruction
ANTI_HALLUCINATION_INSTRUCTION = (
    "You may ONLY state facts, numbers, names, policies, or claims that appear verbatim in the "
    "FACT SHEET below. If the merchant asks about something not present in the fact sheet, "
    "say honestly that you don't have that information and offer to check, rather than answering from "
    "general knowledge. Never invent numbers, discounts, policies, or product features."
)


def _build_system_prompt(facts: Dict[str, Any]) -> str:
    tone = facts.get("voice_tone", "friendly")
    register = facts.get("voice_register", "")
    vocab = facts.get("vocab_allowed", [])
    taboos = facts.get("taboo_words", [])
    salutations = facts.get("salutation_examples", [])
    merchant_name = facts.get("merchant_name", "")
    owner_name = facts.get("merchant_owner_first_name", "")
    trigger_kind = facts.get("trigger_kind", "unknown")

    return f"""You are Vera, an AI assistant for local merchants on magicpin.
Your tone is: {tone}. {f'Register: {register}.' if register else ''}
{f'Use salutations like: {", ".join(salutations)}' if salutations else ''}
{f'Preferred vocabulary: {", ".join(vocab[:15])}.' if vocab else ''}
{f'FORBIDDEN words (never use): {", ".join(taboos)}.' if taboos else ''}
{f'Merchant name: {merchant_name}' if merchant_name else ''}
{f'Owner first name: {owner_name}' if owner_name else ''}
Trigger type: {trigger_kind}

CRITICAL ANTI-HALLUCINATION INSTRUCTION:
{ANTI_HALLUCINATION_INSTRUCTION}

RULES:
- ONLY use facts from the FACT SHEET below. NEVER invent numbers, dates, names, prices, or claims not in the fact sheet.
- MANDATORY SPECIFICITY: Every message MUST contain at least TWO concrete, verifiable facts from the FACT SHEET below:
  * Exact numbers: cite exact statistics, counts, percentages (e.g., view/call counts, CTR %, footfall %, or delta metrics), days remaining, or prices with currency symbol (e.g., ₹499).
  * Exact names: cite exact service, treatment, product combo, or plan names present verbatim in the fact sheet.
  * Exact timeframes/dates: cite exact deadlines, days, or time intervals from the trigger or merchant data.
  * PROHIBITED: NEVER use generic assertions like 'we noticed a dip in activity', 'trending up', 'we have an update', or 'reach out' without immediately citing the specific numbers and metrics from the fact sheet.
- Include exactly ONE call-to-action (CTA) if the trigger is action-oriented; zero if purely informational.
- Prefer specific service+price ("Dental Cleaning @ ₹299") over generic discount framing.
- Include at least one compulsion lever: specificity, loss aversion, social proof, effort externalization, curiosity, reciprocity, or asking-the-merchant-a-question.
- Keep the message concise (2-4 sentences max). Sound like a knowledgeable peer, not a marketing bot.
- Output ONLY the message body text. No JSON, no labels, no "Body:" prefix, no quotes around the message.
- After the message body, on a new line starting with "CTA:", write ONLY the CTA text (or "NONE" if no CTA).
- After the CTA, on a new line starting with "RATIONALE:", write a one-sentence explanation of why this message matters now.

{fact_sheet_to_text(facts)}
"""


def _build_user_prompt(trigger_kind: str, facts: Dict[str, Any]) -> str:
    merchant = facts.get("merchant_name", "the merchant")
    headline = facts.get("headline", facts.get("trigger_headline", ""))
    trigger_data = facts.get("trigger_data", {})
    metrics = facts.get("merchant_metrics", {})
    offers = facts.get("merchant_offers", [])
    services = facts.get("merchant_services", [])

    return f"""Compose a message for trigger kind "{trigger_kind}" to {merchant}.
{f'Headline context: {headline}' if headline else ''}
{f'Trigger payload details: {trigger_data}' if trigger_data else ''}
{f'Merchant metrics: {metrics}' if metrics else ''}
{f'Available offers/services: {offers or services[:3]}' if (offers or services) else ''}
Remember: MAXIMIZE SPECIFICITY. You MUST include concrete numbers, percentages, prices (₹), or days from the fact sheet. Use ONLY facts from the fact sheet. Be specific, compelling, and strictly factual."""


def _call_gemini_raw(api_key: str, model: str, system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
    """Synchronous REST call to Gemini 2.5 Flash."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned empty candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ValueError("Gemini returned empty parts")
        return parts[0].get("text", "").strip()


def _call_mistral_raw(api_key: str, model: str, system: str, user: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
    """Synchronous REST call to Mistral AI backup."""
    url = "https://api.mistral.ai/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=MISTRAL_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


class ModelRotator:
    """
    Manages LLM providers with automatic key rotation and multi-tier failover:
    - Tier 1: Gemini Key 1 (Primary)
    - Tier 2: Gemini Key 2 (Auto-rotated on HTTP 429 / quota limit)
    - Tier 3: Mistral Backup (Failover if both Gemini keys exhausted)
    """
    def __init__(self):
        self.gemini_keys = [k for k in [GEMINI_API_KEY1, GEMINI_API_KEY2] if k]
        self.gemini_model = GEMINI_MODEL
        self.mistral_key = MISTRAL_API_KEY
        self.mistral_model = MISTRAL_MODEL
        self.current_gemini_idx = 0
        self.cooldowns: Dict[int, float] = {}
        self._lock = threading.Lock()

    def generate(self, system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> Tuple[str, str]:
        now = time.time()

        # ── 1. Try Gemini Keys with auto-rotation ──
        if self.gemini_keys:
            for attempt in range(len(self.gemini_keys)):
                with self._lock:
                    idx = (self.current_gemini_idx + attempt) % len(self.gemini_keys)
                    if self.cooldowns.get(idx, 0) > now:
                        continue  # Key is in rate-limit cooldown
                    key = self.gemini_keys[idx]

                try:
                    text = _call_gemini_raw(key, self.gemini_model, system, user, max_tokens, temperature)
                    if text:
                        with self._lock:
                            self.current_gemini_idx = idx  # Keep using working key
                        logger.info("Generated completion via Gemini Key #%d (%s)", idx + 1, self.gemini_model)
                        return text, f"gemini (key #{idx+1})"
                except urllib.error.HTTPError as e:
                    err_body = ""
                    try:
                        err_body = e.read().decode("utf-8")
                    except Exception:
                        pass
                    if e.code == 429 or "RESOURCE_EXHAUSTED" in err_body or "quota" in err_body.lower():
                        logger.warning(
                            "Gemini Key #%d hit HTTP 429 / quota limit! Backing off for 60s and auto-rotating to next key.",
                            idx + 1
                        )
                        with self._lock:
                            self.cooldowns[idx] = time.time() + 60.0
                            self.current_gemini_idx = (idx + 1) % len(self.gemini_keys)
                        continue
                    else:
                        logger.warning("Gemini Key #%d returned HTTP %d: %s; rotating to next key.", idx + 1, e.code, err_body[:100])
                        continue
                except Exception as e:
                    logger.warning("Gemini Key #%d request failed (%s); trying next key.", idx + 1, e)
                    continue

        # ── 2. Tier 3: Mistral Backup Failover ──
        if self.mistral_key:
            try:
                logger.info("All Gemini keys busy or in cooldown; routing to Tier 3 backup: Mistral (%s)", self.mistral_model)
                text = _call_mistral_raw(self.mistral_key, self.mistral_model, system, user, max_tokens=400, temperature=temperature)
                if text:
                    logger.info("Generated completion via Mistral Backup (%s)", self.mistral_model)
                    return text, f"mistral ({self.mistral_model})"
            except Exception as e:
                logger.warning("Mistral backup call failed (%s)", e)

        raise RuntimeError("All LLM providers (Gemini Key 1, Gemini Key 2, Mistral) failed or rate-limited.")


# Global thread-safe rotator singleton
rotator = ModelRotator()


def _parse_llm_response(raw: str, trigger_kind: str) -> Tuple[str, Optional[str], str]:
    """Parse LLM output into (body, cta, rationale)."""
    lines = raw.strip().splitlines()
    body_lines = []
    cta = None
    rationale = None

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("CTA:"):
            val = stripped[4:].strip()
            cta = None if val.upper() == "NONE" else val
        elif stripped.upper().startswith("RATIONALE:"):
            rationale = stripped[10:].strip()
        else:
            if cta is None and rationale is None:
                body_lines.append(line)

    body = "\n".join(body_lines).strip()
    if body.startswith('"') and body.endswith('"'):
        body = body[1:-1].strip()

    if not rationale:
        rationale = f"LLM-composed message for {trigger_kind} trigger."

    if not body:
        raise ValueError("LLM returned empty body")

    return body, cta, rationale


def _call_llm_sync(facts: Dict[str, Any], trigger_kind: str = "unknown") -> Tuple[str, Optional[str], str, str]:
    """
    Synchronous LLM call via the rotator.
    Returns (body, cta, rationale, provider_name).
    """
    system_prompt = _build_system_prompt(facts)
    user_prompt = _build_user_prompt(trigger_kind, facts)

    raw, provider = rotator.generate(system_prompt, user_prompt, max_tokens=1000, temperature=0.0)
    logger.info("LLM raw response from %s: %s", provider, raw[:200])

    body, cta, rationale = _parse_llm_response(raw, trigger_kind)
    return body, cta, rationale, provider


# Backward compatibility alias
_call_mistral_sync = lambda facts, trigger_kind="unknown": _call_llm_sync(facts, trigger_kind)[:3]


async def compose_with_llm(
    facts: Dict[str, Any],
    trigger_kind: str = "unknown",
    prior_bodies: Optional[List[str]] = None,
) -> Tuple[str, Optional[str], str]:
    """
    Asynchronous entry point for composition with multi-tier failover.
    Returns (body, cta, rationale).
    On any failure, falls back to rule_composer.
    """
    loop = asyncio.get_event_loop()
    try:
        body, cta, rationale, provider = await loop.run_in_executor(
            None,
            functools.partial(_call_llm_sync, facts, trigger_kind)
        )
        validate_composed_body(body, cta, facts, prior_bodies=prior_bodies or [])
        logger.info("Composition succeeded via %s for trigger_kind=%s", provider, trigger_kind)
        return body, cta, rationale
    except Exception as e:
        logger.warning("LLM composition error (%s); falling back to rule composer.", e)
        body, cta, rationale, _, _ = compose_rule_based(facts, trigger_kind)
        return body, cta, rationale


def _call_llm_reply_sync(
    facts: Dict[str, Any],
    message: str,
    intent: str,
    conversation_history: list,
) -> Dict[str, Any]:
    """
    Synchronous multi-tier reply call for /v1/reply.
    """
    tone = facts.get("voice_tone", "friendly")
    merchant = facts.get("merchant_name", "")
    taboos = facts.get("taboo_words", [])

    history_text = ""
    if conversation_history:
        recent = conversation_history[-6:]
        history_text = "\nRecent conversation:\n" + "\n".join(
            f"  {t.get('from_role', '?')}: {t.get('message', '')[:100]}"
            for t in recent
        )

    system = f"""You are Vera, a helpful AI assistant for local merchants on magicpin.
Tone: {tone}. Be concise, specific, and grounded in facts.
{f'Merchant: {merchant}' if merchant else ''}
{f'FORBIDDEN words: {", ".join(taboos)}' if taboos else ''}
{history_text}

CRITICAL ANTI-HALLUCINATION INSTRUCTION:
{ANTI_HALLUCINATION_INSTRUCTION}

Reply to the merchant's message. Output ONLY your reply text. No JSON, no labels."""

    user = f"Merchant says: \"{message}\"\n\n{fact_sheet_to_text(facts)}"

    try:
        raw_reply, provider = rotator.generate(system, user, max_tokens=400, temperature=0.0)
        body = raw_reply.strip()
        if not body:
            raise ValueError("Empty reply from LLM")

        validate_reply_facts(body, facts, user_message=message)
        logger.info("Reply generated via %s for intent=%s", provider, intent)

        return {
            "action": "send",
            "body": body,
            "cta": None,
            "rationale": f"Composed reply to {intent} intent via {provider} (fact-verified).",
        }
    except Exception as e:
        logger.warning("LLM reply error (%s); using safe deterministic fallback.", e)
        return {
            "action": "send",
            "body": SAFE_QUESTION_FALLBACK,
            "cta": None,
            "rationale": f"Safe fallback after LLM error ({e}).",
        }


# Backward compatibility alias
_call_mistral_reply_sync = _call_llm_reply_sync


async def compose_reply_with_llm(
    facts: Dict[str, Any],
    message: str,
    intent: str,
    conversation_history: list,
) -> Dict[str, Any]:
    """Asynchronous entry point for reply composition."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(_call_llm_reply_sync, facts, message, intent, conversation_history)
    )
