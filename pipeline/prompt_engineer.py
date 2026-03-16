"""
pipeline/prompt_engineer.py
============================
Converts structured attribute dict → SDXL/FLUX-optimized prompt.

FIXES APPLIED vs original:
  1. Null-aware: skips every field that is None (no "medium nose" filler)
  2. Priority order: distinguishing marks → strong structure → hair/eyes → expression
  3. Short suffix: ≤12 words (was 40+) — prevents style tokens drowning face tokens
  4. Emphasis weighting: (scar on cheek:1.4) boosts key distinguishing marks
  5. Portrait anchor: prepended prefix locks subject framing
  6. build_prompt_parts() is testable independently of the full pipeline
"""

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Style presets  — SHORT suffixes only
# ─────────────────────────────────────────────────────────────────────────────
STYLE_PRESETS = {
    "forensic_sketch": {
        "prefix":   "close-up forensic pencil sketch portrait, face centered, white paper",
        "suffix":   "graphite portrait, sharp pencil lines, law enforcement composite",
        "negative": (
            "color, photograph, painting, cartoon, anime, blurry, distorted, "
            "multiple people, watermark, text, frame, nsfw, extra limbs"
        ),
    },
    "photorealistic": {
        "prefix":   "close-up photorealistic mugshot portrait, face centered, plain gray background",
        "suffix":   "8k uhd, studio lighting, sharp focus, realistic skin",
        "negative": (
            "drawing, sketch, cartoon, anime, blurry, distorted, deformed, "
            "multiple people, watermark, text, oversaturated, nsfw"
        ),
    },
    "composite": {
        "prefix":   "close-up digital forensic composite portrait, face centered, neutral background",
        "suffix":   "clean digital render, even lighting, FBI composite style",
        "negative": (
            "artistic, abstract, cartoon, anime, blurry, distorted, "
            "multiple people, watermark, text, nsfw"
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Priority-ordered prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt_parts(attrs: dict) -> list[str]:
    """
    Convert attribute dict → ordered list of descriptive phrases.

    Priority (high → low):
      1. Distinguishing marks  — scars, tattoos, moles (most identity-critical)
      2. Core demographics     — gender, age, ethnicity
      3. Face structure        — jaw, face shape, forehead, cheekbones, chin
      4. Hair                  — color + style
      5. Eyes + eyebrows       — eye color, shape, brow weight
      6. Nose + lips
      7. Facial hair + glasses
      8. Expression
      9. Build / complexion    — least identity-critical, dropped if null

    FIX: null values are silently skipped — no "average nose" filler.
    FIX: distinguishing marks get (emphasis:1.4) syntax for SDXL weight boost.
    """
    parts = []

    # ── 1. Distinguishing marks  (HIGHEST priority) ───────────────────────────
    for feat in attrs.get("distinguishing_features") or []:
        feat = feat.strip()
        if feat:
            # Emphasis weight for SDXL — marks with colon syntax
            parts.append(f"({feat}:1.4)")

    # ── 2. Demographics ───────────────────────────────────────────────────────
    gender = attrs.get("gender")
    age    = attrs.get("age")

    subject = "person"
    if gender in ("male", "female"):
        subject = gender

    ethnicity = attrs.get("ethnicity")
    if ethnicity and ethnicity not in ("unknown", None):
        subject = f"{ethnicity.replace('_', ' ')} {subject}"

    age_str = f"{age} year old" if age else None
    if age_str:
        parts.append(f"{age_str} {subject}")
    else:
        parts.append(subject)

    # ── 3. Face structure ─────────────────────────────────────────────────────
    _add(parts, attrs, "jaw_shape",   "{} jawline")
    _add(parts, attrs, "face_shape",  "{} face shape")
    _add(parts, attrs, "forehead",    "{} forehead",  skip={"normal"})
    _add(parts, attrs, "cheekbones",  "{} cheekbones", skip={"normal"})
    _add(parts, attrs, "chin",        "{} chin",      skip={"normal"})

    # ── 4. Hair ───────────────────────────────────────────────────────────────
    hair_color = attrs.get("hair_color")
    hair_style = attrs.get("hair_style")

    if hair_style == "bald" or hair_color == "bald":
        parts.append("bald head, completely shaved")
    elif hair_color and hair_style:
        tex = attrs.get("hair_texture")
        if tex and tex != "straight":
            parts.append(f"{tex} {hair_color} {hair_style} hair")
        else:
            parts.append(f"{hair_color} {hair_style} hair")
    elif hair_color:
        parts.append(f"{hair_color} hair")
    elif hair_style:
        parts.append(f"{hair_style} hair")

    if attrs.get("hair_style") == "receding":
        parts.append("receding hairline")

    # ── 5. Eyes & eyebrows ────────────────────────────────────────────────────
    eye_color = attrs.get("eye_color")
    eye_shape = attrs.get("eye_shape")
    if eye_color and eye_shape:
        parts.append(f"{eye_shape} {eye_color} eyes")
    elif eye_color:
        parts.append(f"{eye_color} eyes")
    elif eye_shape:
        parts.append(f"{eye_shape} eyes")

    _add(parts, attrs, "eyebrows",  "{} eyebrows",  skip={"thick"})

    # ── 6. Nose & lips ────────────────────────────────────────────────────────
    _add(parts, attrs, "nose_shape", "{} nose",  skip={"medium", "straight"})
    _add(parts, attrs, "lip_shape",  "{} lips",  skip={"medium"})

    # ── 7. Facial hair & glasses ──────────────────────────────────────────────
    fhair = attrs.get("facial_hair")
    if fhair and fhair != "none":
        parts.append(fhair.replace("_", " "))

    glasses = attrs.get("glasses")
    if glasses and glasses != "none":
        parts.append(f"wearing {glasses} glasses")

    # ── 8. Skin tone ──────────────────────────────────────────────────────────
    _add(parts, attrs, "skin_tone", "{} skin tone", skip={"medium"})

    # ── 9. Expression ─────────────────────────────────────────────────────────
    _add(parts, attrs, "expression", "{} expression", skip={"neutral"})

    # ── 10. Build / complexion  (low priority) ────────────────────────────────
    _add(parts, attrs, "build",      "{} build",      skip={"average"})
    _add(parts, attrs, "complexion", "{} complexion", skip={"clear"})

    return parts


def _add(parts: list, attrs: dict, field: str, template: str,
         skip: set = None) -> None:
    """Helper: add field to parts list if not None and not in skip set."""
    val = attrs.get(field)
    if val is None:
        return
    if skip and val in skip:
        return
    parts.append(template.format(val.replace("_", " ")))


# ─────────────────────────────────────────────────────────────────────────────
#  Public interface
# ─────────────────────────────────────────────────────────────────────────────
def build_forensic_prompt(
    attrs: dict,
    style: str = "forensic_sketch",
) -> tuple[str, str]:
    """
    Convert attribute dict → (prompt, negative_prompt) ready for SDXL/FLUX.

    Args:
        attrs : validated dict from nlp_parser.extract_attributes()
        style : "forensic_sketch" | "photorealistic" | "composite"

    Returns:
        (prompt, negative_prompt)

    Example:
        >>> attrs = {'gender':'male', 'age':40, 'jaw_shape':'square',
        ...           'distinguishing_features':['scar on left cheek']}
        >>> p, n = build_forensic_prompt(attrs)
        >>> # p starts with "(scar on left cheek:1.4), 40 year old male, square jawline, ..."
    """
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["forensic_sketch"])
    parts  = build_prompt_parts(attrs)

    face_desc = ", ".join(p for p in parts if p.strip())
    prompt    = f"{preset['prefix']}, {face_desc}, {preset['suffix']}"
    return prompt, preset["negative"]


# ─────────────────────────────────────────────────────────────────────────────
#  CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {
            # Full attrs — from Groq
            "age": 42, "gender": "male", "jaw_shape": "square",
            "hair_color": "brown", "hair_style": "short",
            "eye_color": "blue", "eyebrows": "bushy",
            "nose_shape": "large", "facial_hair": "stubble",
            "distinguishing_features": ["scar on left cheek", "broken nose"],
            "expression": "stern", "skin_tone": "light",
        },
        {
            # Sparse attrs — only what was mentioned
            "gender": "female", "age": 28,
            "hair_color": "black", "hair_style": "long",
            "distinguishing_features": [],
        },
        {
            # Bald with beard
            "gender": "male", "age": 55,
            "hair_style": "bald", "hair_color": "bald",
            "facial_hair": "full_beard",
            "glasses": "regular",
            "distinguishing_features": ["tattoo on neck"],
        },
    ]

    for i, attrs in enumerate(test_cases, 1):
        for style in STYLE_PRESETS:
            p, n = build_forensic_prompt(attrs, style=style)
            print(f"\nCase {i} | {style}")
            print(f"  PROMPT   : {p}")
            print(f"  NEGATIVE : {n[:60]}...")
