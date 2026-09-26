"""
Validation layer for composed messages.
Rejects -> caller falls back to rule_composer or safe deterministic fallback.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set

SAFE_QUESTION_FALLBACK = (
    "That's a good question — I don't have the specifics on that in front of me "
    "right now, but I can check and follow up."
)


class ValidationError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def extract_fact_sheet_tokens(facts: Dict[str, Any]) -> Set[str]:
    """
    Extract all numbers, words, prices, percentages, and terms present in the fact sheet.
    This creates an exact whitelist of grounded numerical and factual entities.
    """
    tokens: Set[str] = set()

    def _collect(val: Any):
        if val is None:
            return
        if isinstance(val, (int, float)):
            tokens.add(str(val))
            if isinstance(val, int):
                tokens.add(f"{val:,}")
        elif isinstance(val, str):
            val_lower = val.lower()
            tokens.add(val_lower)
            # extract isolated numbers
            for num in re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", val):
                tokens.add(num.replace(",", ""))
                tokens.add(num)
            # extract percentages
            for pct in re.findall(r"\d+(?:\.\d+)?%", val):
                tokens.add(pct)
            # extract currency amounts
            for curr in re.findall(r"[₹$]\s*[\d,]+(?:\.\d+)?", val):
                tokens.add(curr.replace(" ", ""))
                tokens.add(re.sub(r"[₹$\s,]", "", curr))
        elif isinstance(val, dict):
            for k, v in val.items():
                tokens.add(str(k).lower())
                _collect(v)
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                _collect(item)

    _collect(facts)
    return tokens


def validate_composed_body(
    body: str,
    cta: Optional[str],
    facts: Dict[str, Any],
    prior_bodies: Optional[List[str]] = None,
    language_pref: Optional[str] = None,
) -> None:
    """
    Validation for /v1/tick composed messages.
    Raises ValidationError if any check fails.
    Checks:
      1. Empty body
      2. Taboo words
      3. Near-duplicate of prior body
      4. Fabricated currency / prices not in fact sheet
      5. Fabricated percentages not in fact sheet
      6. Fabricated substantive numbers not in fact sheet
    """
    if not body or not body.strip():
        raise ValidationError("empty_body")

    # 1. Taboo words
    taboo_words: List[str] = facts.get("taboo_words", [])
    body_lower = body.lower()
    for word in taboo_words:
        if word.lower() in body_lower:
            raise ValidationError(f"taboo_word_found: {word}")

    # 2. Anti-repetition
    if prior_bodies:
        for prior in prior_bodies:
            similarity = _simple_similarity(body, prior)
            if similarity > 0.85:
                raise ValidationError("near_duplicate_body")

    # 3. Grounding checks against fact sheet
    fact_tokens = extract_fact_sheet_tokens(facts)

    # 3a. Currencies (₹, $, Rs, INR)
    currencies_in_body = re.findall(r"(?:[₹$]|(?:rs\.?\s*)|(?:inr\s*))(\d+(?:,\d+)*(?:\.\d+)?)", body, re.IGNORECASE)
    for c_val in currencies_in_body:
        clean = c_val.replace(",", "")
        if clean not in fact_tokens and c_val not in fact_tokens:
            raise ValidationError(f"fabricated_price: {c_val}")

    # 3b. Percentages
    pcts_in_body = re.findall(r"(\d+(?:\.\d+)?)%", body)
    for p_val in pcts_in_body:
        if f"{p_val}%" not in fact_tokens and p_val not in fact_tokens:
            raise ValidationError(f"fabricated_percentage: {p_val}%")

    # 3c. Substantive numbers (exclude single digit counters like 1, 2, 3 in text)
    cleaned = re.sub(r"(?m)^\s*\d+[\.\)]\s*", "", body)
    numbers_in_body = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", cleaned)
    for num in numbers_in_body:
        clean = num.replace(",", "")
        if clean in ("1", "2", "3") and len(num) == 1:
            continue
        if clean not in fact_tokens and num not in fact_tokens:
            raise ValidationError(f"fabricated_number: {num}")


def validate_reply_facts(
    body: str,
    facts: Dict[str, Any],
    user_message: str = "",
) -> None:
    """
    Post-generation fact check specifically for /v1/reply question-answering.
    Extracts every number, percentage, currency amount, and named policy/feature term
    from the LLM's draft body, and confirms each appears in the fact sheet.

    Raises ValidationError if any unverified fact, number, or invented policy term is found.
    """
    if not body or not body.strip():
        raise ValidationError("empty_body")

    # Check for graceful decline phrases (always allowed if the bot doesn't know)
    decline_phrases = [
        "don't have that information",
        "do not have that information",
        "don't have the specifics",
        "do not have the specifics",
        "don't have details",
        "do not have details",
        "can check and follow up",
        "will check and follow up",
        "can't find details",
        "not available in my records",
    ]
    body_lower = body.lower()
    is_declining = any(p in body_lower for p in decline_phrases)

    # 1. Taboo words
    taboo_words: List[str] = facts.get("taboo_words", [])
    for word in taboo_words:
        if word.lower() in body_lower:
            raise ValidationError(f"taboo_word_found: {word}")

    # 2. Extract grounded fact tokens
    fact_tokens = extract_fact_sheet_tokens(facts)
    facts_str = str(facts).lower()

    # 3. Currency check: every currency figure in body must be in facts
    currencies = re.findall(r"(?:[₹$]|(?:rs\.?\s*)|(?:inr\s*))([\d,]+(?:\.\d+)?)", body, re.IGNORECASE)
    for c_val in currencies:
        clean = c_val.replace(",", "")
        if clean not in fact_tokens and c_val not in fact_tokens:
            raise ValidationError(f"unverified_currency: {c_val}")

    # 4. Percentage check: every percentage in body must be in facts
    percentages = re.findall(r"(\d+(?:\.\d+)?)%", body)
    for p_val in percentages:
        if f"{p_val}%" not in fact_tokens and p_val not in fact_tokens:
            raise ValidationError(f"unverified_percentage: {p_val}%")

    # 5. Number check: all substantive numbers must be in facts
    cleaned = re.sub(r"(?m)^\s*\d+[\.\)]\s*", "", body)
    numbers = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", cleaned)
    for num in numbers:
        clean = num.replace(",", "")
        if clean in ("1", "2") and len(num) == 1:
            continue
        if clean not in fact_tokens and num not in fact_tokens:
            raise ValidationError(f"unverified_number: {num}")

    # 6. Policy / Feature hallucination check
    unverified_policy_terms = [
        "billing discount",
        "refund policy",
        "cancellation policy",
        "return policy",
        "price match",
        "auto-checkout",
        "instant checkout",
        "cashback guarantee",
        "tiered model",
        "magicpin merchant portal",
    ]
    for term in unverified_policy_terms:
        if term in body_lower and term not in facts_str and not is_declining:
            raise ValidationError(f"unverified_policy_term: {term}")


def _simple_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity on word tokens."""
    if not a or not b:
        return 0.0
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def get_taboo_words(category_payload: Optional[Dict[str, Any]]) -> Set[str]:
    """Extract taboo words from both field name variants."""
    taboos: Set[str] = set()
    if not category_payload:
        return taboos
    voice = category_payload.get("voice", {})
    if not isinstance(voice, dict):
        return taboos
    for key in ("taboos", "vocab_taboo"):
        val = voice.get(key)
        if isinstance(val, list):
            taboos.update(str(w).lower() for w in val)
    return taboos
