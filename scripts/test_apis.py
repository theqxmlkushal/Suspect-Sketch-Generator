"""
scripts/test_apis.py
=====================
Test all image generation backends and Groq parser.
Run from project root: python scripts/test_apis.py

MOVED from root to scripts/ — keep project root clean.
"""

import os
import sys
import io
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from PIL import Image


def _print_result(name: str, success: bool, detail: str = ""):
    icon = "PASS" if success else "FAIL"
    print(f"  [{icon}] {name}" + (f" — {detail}" if detail else ""))


# ── 1. NLP Parser ─────────────────────────────────────────────────────────────
print("\n=== NLP Parser ===")

from nlp.nlp_parser import extract_attributes_rule_based, extract_attributes_groq

test_desc = "White male, early 40s, square jaw, brown hair, scar on left cheek, stubble"

attrs_rule = extract_attributes_rule_based(test_desc)
non_null = {k: v for k, v in attrs_rule.items() if v not in (None, [], "unknown")}
_print_result("Rule-based parser", len(non_null) >= 4,
              f"{len(non_null)} non-null fields: {non_null}")

if os.getenv("GROQ_API_KEY"):
    try:
        attrs_llm = extract_attributes_groq(test_desc)
        non_null_llm = {k: v for k, v in attrs_llm.items() if v not in (None, [], "unknown")}
        _print_result("Groq / Llama-3", len(non_null_llm) >= 6,
                      f"{len(non_null_llm)} non-null fields")
    except Exception as e:
        _print_result("Groq / Llama-3", False, str(e))
else:
    print("  [SKIP] Groq — GROQ_API_KEY not set")


# ── 2. Prompt engineer ────────────────────────────────────────────────────────
print("\n=== Prompt builder ===")

from pipeline.prompt_engineer import build_forensic_prompt, STYLE_PRESETS

for style in STYLE_PRESETS:
    p, n = build_forensic_prompt(attrs_rule, style=style)
    suffix_words = len(STYLE_PRESETS[style]["suffix"].split())
    ok = (
        "scar on left cheek" in p and
        "face centered" in p and
        suffix_words <= 20 and
        "None" not in p
    )
    _print_result(f"Style: {style}", ok,
                  f"prompt length: {len(p)} chars | suffix: {suffix_words} words")


# ── 3. Image backends ─────────────────────────────────────────────────────────
print("\n=== Image backends ===")

test_prompt = (
    "close-up forensic pencil sketch portrait, face centered, white paper, "
    "40 year old male, square jawline, brown short hair, "
    "(scar on left cheek:1.4), graphite portrait, sharp pencil lines"
)

# HuggingFace
hf_token = os.getenv("HF_TOKEN", "").strip()
if hf_token:
    try:
        from pipeline.generation_pipeline import _generate_hf
        t0 = time.time()
        img = _generate_hf(test_prompt, 512, 512, seed=42, token=hf_token)
        elapsed = time.time() - t0
        assert isinstance(img, Image.Image)
        _print_result("HuggingFace FLUX", True, f"{img.size} in {elapsed:.1f}s")
    except Exception as e:
        _print_result("HuggingFace FLUX", False, str(e)[:80])
else:
    print("  [SKIP] HuggingFace — HF_TOKEN not set")

# Together AI
together_key = os.getenv("TOGETHER_API_KEY", "").strip()
if together_key:
    try:
        from pipeline.generation_pipeline import _generate_together
        t0 = time.time()
        img = _generate_together(test_prompt, 512, 512, seed=42, key=together_key)
        elapsed = time.time() - t0
        assert isinstance(img, Image.Image)
        _print_result("Together AI FLUX", True, f"{img.size} in {elapsed:.1f}s")
    except Exception as e:
        _print_result("Together AI FLUX", False, str(e)[:80])
else:
    print("  [SKIP] Together AI — TOGETHER_API_KEY not set")

# Pollinations (always tested — no key)
try:
    from pipeline.generation_pipeline import _generate_pollinations
    t0 = time.time()
    img = _generate_pollinations(test_prompt, 512, 512, seed=42)
    elapsed = time.time() - t0
    assert isinstance(img, Image.Image)
    _print_result("Pollinations.ai", True, f"{img.size} in {elapsed:.1f}s")
except Exception as e:
    _print_result("Pollinations.ai", False, str(e)[:80])


# ── 4. Face validation ────────────────────────────────────────────────────────
print("\n=== Face validation ===")
from pipeline.generation_pipeline import _has_face, FACE_VALIDATION_AVAILABLE

if FACE_VALIDATION_AVAILABLE:
    # Should pass on a real face image (we'll just test with a blank canvas)
    blank = Image.new("RGB", (512, 512), color=(200, 180, 160))
    result = _has_face(blank)
    _print_result("Face detection available", True,
                  f"blank image → face detected: {result}")
else:
    print("  [SKIP] facenet-pytorch not installed  (pip install facenet-pytorch)")


# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
keys = {
    "GROQ_API_KEY":      os.getenv("GROQ_API_KEY"),
    "HF_TOKEN":          os.getenv("HF_TOKEN"),
    "TOGETHER_API_KEY":  os.getenv("TOGETHER_API_KEY"),
}
for name, val in keys.items():
    status = "SET" if val else "NOT SET"
    print(f"  {name}: {status}")
print()
print("Run 'streamlit run ui/app.py' to start the UI.")
print("Run 'uvicorn api.api:app --reload' to start the API.")
