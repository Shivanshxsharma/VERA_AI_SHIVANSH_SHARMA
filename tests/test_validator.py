"""
Test the validation layer.
"""
import pytest
from app.composer.validator import validate_composed_body, ValidationError


def test_empty_body_rejected():
    with pytest.raises(ValidationError, match="empty_body"):
        validate_composed_body("", None, {}, [])


def test_whitespace_body_rejected():
    with pytest.raises(ValidationError, match="empty_body"):
        validate_composed_body("   ", None, {}, [])


def test_taboo_word_rejected():
    facts = {"taboo_words": ["guarantee", "cure"]}
    with pytest.raises(ValidationError, match="taboo_word_found"):
        validate_composed_body(
            "We guarantee the best results!", None, facts, [],
        )


def test_near_duplicate_rejected():
    prior = ["Hello! Check out our dental cleaning service today."]
    with pytest.raises(ValidationError, match="near_duplicate_body"):
        validate_composed_body(
            "Hello! Check out our dental cleaning service today.",
            None, {}, prior,
        )


def test_fabricated_price_rejected():
    facts = {}  # no prices in facts
    with pytest.raises(ValidationError, match="fabricated_price"):
        validate_composed_body(
            "Get dental cleaning at ₹199 today!", None, facts, [],
        )


def test_valid_body_passes():
    facts = {"taboo_words": ["guarantee"]}
    # Should not raise
    validate_composed_body(
        "Hi! We have a great service for you.", None, facts, [],
    )
