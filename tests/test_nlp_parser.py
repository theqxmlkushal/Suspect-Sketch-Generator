"""
tests/test_nlp_parser.py
=========================
Unit tests for the NLP attribute parser.
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from nlp.nlp_parser import (
    extract_attributes_rule_based,
    _parse_age,
    _validate_attrs,
    ALLOWED,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Age parsing
# ─────────────────────────────────────────────────────────────────────────────
class TestParseAge:
    def test_bare_number(self):
        assert _parse_age("male, 40 years old") == 40

    def test_decade_s(self):
        assert _parse_age("man in his 40s") == 45

    def test_early_decade(self):
        assert _parse_age("early 30s") == 31

    def test_mid_decade(self):
        assert _parse_age("mid 40s") == 45

    def test_late_decade(self):
        assert _parse_age("late 50s") == 58

    def test_around_keyword(self):
        assert _parse_age("around 38") == 38

    def test_approximately(self):
        assert _parse_age("approximately 55 years old") == 55

    def test_none_when_missing(self):
        assert _parse_age("male with brown hair") is None


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-based parser
# ─────────────────────────────────────────────────────────────────────────────
class TestRuleBasedParser:
    def test_gender_male(self):
        a = extract_attributes_rule_based("White male, 40 years old")
        assert a["gender"] == "male"

    def test_gender_female(self):
        a = extract_attributes_rule_based("Young woman with blonde hair")
        assert a["gender"] == "female"

    def test_gender_unknown(self):
        a = extract_attributes_rule_based("Person with brown hair")
        assert a["gender"] is None   # FIX: was "unknown" string before

    def test_hair_color_brown(self):
        a = extract_attributes_rule_based("Short brown hair")
        assert a["hair_color"] == "brown"

    def test_hair_color_blonde(self):
        a = extract_attributes_rule_based("Blonde female, early 20s")
        assert a["hair_color"] == "blonde"

    def test_bald(self):
        a = extract_attributes_rule_based("Bald man, thick beard")
        assert a["hair_style"] == "bald"
        assert a["hair_color"] == "bald"

    def test_square_jaw(self):
        a = extract_attributes_rule_based("Square jaw, brown hair")
        assert a["jaw_shape"] == "square"

    def test_scar_distinguishing(self):
        a = extract_attributes_rule_based("Male, scar on left cheek")
        assert any("scar" in f for f in a["distinguishing_features"])

    def test_glasses(self):
        a = extract_attributes_rule_based("Wearing glasses, male")
        assert a["glasses"] == "regular"

    def test_beard(self):
        a = extract_attributes_rule_based("Full beard, bald, 50s")
        assert a["facial_hair"] == "full_beard"

    def test_stubble(self):
        a = extract_attributes_rule_based("Light stubble, square jaw")
        assert a["facial_hair"] == "stubble"

    def test_skin_tone_light(self):
        a = extract_attributes_rule_based("Pale skin, blue eyes")
        assert a["skin_tone"] == "light"

    def test_build_stocky(self):
        a = extract_attributes_rule_based("Stocky male, 40s")
        assert a["build"] == "stocky"

    def test_no_hallucination_for_missing_fields(self):
        """Fields not mentioned should be None, not filled with defaults."""
        a = extract_attributes_rule_based("Male with a scar on his chin")
        # Unmentioned features must be None
        assert a.get("hair_color") is None
        assert a.get("jaw_shape") is None
        assert a.get("eye_color") is None

    def test_returns_valid_schema(self):
        """All returned field values must be in ALLOWED sets or None."""
        a = extract_attributes_rule_based(
            "Hispanic woman, late 30s, curly black hair, full lips, oval face, athletic build"
        )
        for field, allowed in ALLOWED.items():
            val = a.get(field)
            if val is not None:
                assert val in allowed, f"{field}={val!r} not in allowed set"


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────
class TestValidateAttrs:
    def test_age_clamp(self):
        assert _validate_attrs({"age": 200})["age"] == 95
        assert _validate_attrs({"age": 5})["age"] == 10

    def test_invalid_enum_becomes_none(self):
        a = _validate_attrs({"gender": "helicopter"})
        assert a["gender"] is None

    def test_empty_distinguishing_stays_empty(self):
        a = _validate_attrs({"distinguishing_features": []})
        assert a["distinguishing_features"] == []

    def test_none_strings_stripped(self):
        a = _validate_attrs({"distinguishing_features": ["none", "N/A", "scar on chin"]})
        assert a["distinguishing_features"] == ["scar on chin"]

    def test_missing_keys_filled_with_none(self):
        a = _validate_attrs({"gender": "male"})
        assert "hair_color" in a
        assert a["hair_color"] is None
