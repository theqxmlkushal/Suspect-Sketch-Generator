"""
tests/test_prompt_engineer.py
==============================
Unit tests for the prompt builder.
Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pipeline.prompt_engineer import build_forensic_prompt, build_prompt_parts, STYLE_PRESETS


FULL_ATTRS = {
    "age": 42, "gender": "male", "jaw_shape": "square",
    "hair_color": "brown", "hair_style": "short",
    "eye_color": "blue", "eyebrows": "bushy",
    "nose_shape": "large", "facial_hair": "stubble",
    "distinguishing_features": ["scar on left cheek"],
    "expression": "stern", "skin_tone": "light",
}

SPARSE_ATTRS = {
    "gender": "female",
    "distinguishing_features": ["tattoo on neck"],
}

NULL_ATTRS = {k: None for k in FULL_ATTRS}
NULL_ATTRS["distinguishing_features"] = []


class TestBuildPromptParts:
    def test_distinguishing_features_first(self):
        parts = build_prompt_parts(FULL_ATTRS)
        # First part should be the scar (emphasis syntax)
        assert parts[0].startswith("(scar on left cheek")

    def test_emphasis_weight_applied(self):
        parts = build_prompt_parts(FULL_ATTRS)
        assert any(":1.4)" in p for p in parts)

    def test_null_fields_skipped(self):
        parts = build_prompt_parts(NULL_ATTRS)
        # Should still have some parts (at minimum gender/subject)
        assert len(parts) >= 1
        # Should NOT contain "None" or empty fragments
        combined = " ".join(parts)
        assert "None" not in combined
        assert "null" not in combined

    def test_bald_handled(self):
        a = {"hair_style": "bald", "hair_color": "bald"}
        parts = build_prompt_parts(a)
        combined = " ".join(parts)
        assert "bald" in combined
        # Should NOT produce "bald bald hair" redundancy
        assert "bald bald" not in combined

    def test_medium_values_skipped(self):
        """Fields equal to 'medium' or 'normal' should not appear in prompt."""
        a = {"nose_shape": "medium", "lip_shape": "medium",
             "forehead": "normal", "cheekbones": "normal"}
        parts = build_prompt_parts(a)
        combined = " ".join(parts)
        assert "medium nose" not in combined
        assert "normal forehead" not in combined

    def test_sparse_attrs_no_crash(self):
        """Sparse attrs (mostly None) must not raise."""
        parts = build_prompt_parts(SPARSE_ATTRS)
        assert isinstance(parts, list)


class TestBuildForensicPrompt:
    def test_returns_tuple(self):
        result = build_forensic_prompt(FULL_ATTRS)
        assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_is_non_empty_string(self):
        p, n = build_forensic_prompt(FULL_ATTRS)
        assert isinstance(p, str) and len(p) > 20
        assert isinstance(n, str) and len(n) > 10

    def test_all_styles_work(self):
        for style in STYLE_PRESETS:
            p, n = build_forensic_prompt(FULL_ATTRS, style=style)
            assert isinstance(p, str) and len(p) > 20

    def test_prefix_is_anchor(self):
        """Prompt must start with the portrait anchor (face centered)."""
        p, _ = build_forensic_prompt(FULL_ATTRS, style="forensic_sketch")
        assert "face centered" in p or "close-up" in p

    def test_distinguishing_feature_in_prompt(self):
        p, _ = build_forensic_prompt(FULL_ATTRS)
        assert "scar on left cheek" in p

    def test_suffix_is_short(self):
        """Style suffix must be ≤20 words."""
        for style in STYLE_PRESETS:
            suffix = STYLE_PRESETS[style]["suffix"]
            word_count = len(suffix.split())
            assert word_count <= 20, f"Style '{style}' suffix too long: {word_count} words"

    def test_sparse_attrs_no_filler(self):
        """Sparse attrs must not produce 'medium' or 'normal' in prompt."""
        p, _ = build_forensic_prompt(SPARSE_ATTRS)
        assert "medium nose" not in p
        assert "normal forehead" not in p
        assert "None" not in p
