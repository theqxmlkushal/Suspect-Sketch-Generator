"""
nlp/nlp_parser.py
==================
NLP Attribute Extractor — AI Suspect Sketch Generator.

FIXES APPLIED vs original:
  1. Groq parser: unmentioned fields return null (not hallucinated defaults)
  2. Age regex: now handles "40s", "early 40s", "late 30s" correctly
  3. Rule-based: skin_tone parses "white", "black", "asian", "hispanic" correctly
  4. Unified schema: both parsers return identical field names
  5. Validation: _validate_attrs() sanitises both outputs to known enum values
  6. description_to_cgan_tensor: backward-compatible Phase 2 bridge kept
"""

import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

# ── backward compat for Phase 2 cGAN path ────────────────────────────────────
try:
    from data.dataset import dict_to_celeba_tensor
except ImportError:
    dict_to_celeba_tensor = None


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical schema — every field name and allowed value
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_DEFAULTS = {
    "age":                    None,
    "gender":                 None,   # male | female | unknown
    "ethnicity":              None,
    "skin_tone":              None,   # very_light | light | medium | olive | brown | dark
    "hair_color":             None,   # black | brown | blonde | red | gray | white | bald
    "hair_style":             None,   # short | medium | long | curly | straight | wavy | buzz_cut | bald | receding
    "hair_texture":           None,
    "eye_color":              None,
    "eye_shape":              None,
    "eyebrows":               None,
    "nose_shape":             None,
    "lip_shape":              None,
    "jaw_shape":              None,
    "face_shape":             None,
    "facial_hair":            None,   # none | stubble | mustache | full_beard | goatee | sideburns
    "glasses":                None,   # none | regular | sunglasses | reading
    "distinguishing_features":[],     # list of free strings: ["scar on left cheek", ...]
    "expression":             None,
    "build":                  None,
    "forehead":               None,
    "cheekbones":             None,
    "chin":                   None,
    "ears":                   None,
    "neck":                   None,
    "complexion":             None,
}

# Values allowed per field — anything else gets set to None
ALLOWED = {
    "gender":       {"male", "female", "unknown"},
    "skin_tone":    {"very_light", "light", "medium", "olive", "brown", "dark"},
    "hair_color":   {"black", "brown", "blonde", "red", "gray", "white", "bald"},
    "hair_style":   {"short", "medium", "long", "curly", "straight", "wavy",
                     "buzz_cut", "bald", "receding"},
    "hair_texture": {"straight", "wavy", "curly", "coily"},
    "eye_color":    {"brown", "blue", "green", "hazel", "gray", "black"},
    "eye_shape":    {"narrow", "wide", "almond", "round", "hooded", "deep_set"},
    "eyebrows":     {"thick", "thin", "arched", "bushy", "sparse", "straight"},
    "nose_shape":   {"small", "medium", "large", "wide", "narrow",
                     "pointed", "flat", "hooked", "straight"},
    "lip_shape":    {"thin", "medium", "full", "wide"},
    "jaw_shape":    {"square", "round", "oval", "pointed", "strong", "soft"},
    "face_shape":   {"oval", "round", "square", "heart", "oblong", "diamond", "triangle"},
    "facial_hair":  {"none", "stubble", "mustache", "full_beard", "goatee", "sideburns"},
    "glasses":      {"none", "regular", "sunglasses", "reading"},
    "expression":   {"neutral", "stern", "smiling", "frowning", "angry"},
    "build":        {"thin", "slim", "average", "athletic", "stocky", "heavy"},
    "forehead":     {"normal", "high", "broad", "narrow"},
    "cheekbones":   {"normal", "high", "prominent", "flat"},
    "chin":         {"normal", "pointed", "cleft", "double", "receding", "prominent"},
    "ears":         {"normal", "large", "small", "protruding"},
    "neck":         {"normal", "thick", "thin", "long", "short"},
    "complexion":   {"clear", "freckled", "acne", "wrinkled", "weathered", "smooth"},
}


def _validate_attrs(raw: dict) -> dict:
    """
    Sanitise a raw attribute dict:
      - Fill missing keys with None / []
      - Set any value not in ALLOWED to None
      - Ensure distinguishing_features is always a list of non-empty strings
      - Clamp age to [10, 95]
    """
    out = dict(SCHEMA_DEFAULTS)           # start from canonical defaults
    out.update(raw)                        # overlay raw values

    for field, allowed in ALLOWED.items():
        val = out.get(field)
        if val is not None and str(val).lower() not in allowed:
            out[field] = None

    # Age
    if out["age"] is not None:
        try:
            out["age"] = max(10, min(95, int(out["age"])))
        except (TypeError, ValueError):
            out["age"] = None

    # Distinguishing features: keep only non-empty strings
    feats = out.get("distinguishing_features") or []
    out["distinguishing_features"] = [
        str(f).strip() for f in feats
        if f and str(f).strip().lower() not in ("none", "n/a", "")
    ]

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Groq LLM Parser  (Primary)
# ─────────────────────────────────────────────────────────────────────────────

# FIX: "use null for anything not explicitly mentioned" — prevents hallucination
SYSTEM_PROMPT = """You are a forensic facial attribute extractor for law enforcement.
Given a verbal suspect description, return ONLY valid JSON.

CRITICAL RULES:
- If a feature is NOT explicitly mentioned in the description, set it to null.
- Do NOT invent or assume features based on ethnicity, age, or gender stereotypes.
- Age must be a single integer. For ranges like "40s" return 45, "early 30s" return 32.
- distinguishing_features must be a JSON array of strings. Empty array [] if none.
- Return ONLY the JSON object. No markdown, no explanation, no extra text.

JSON schema (all fields required, use null for unmentioned):
{
  "age": <int or null>,
  "gender": "male"|"female"|"unknown"|null,
  "ethnicity": "caucasian"|"african"|"asian"|"hispanic"|"middle_eastern"|"south_asian"|null,
  "skin_tone": "very_light"|"light"|"medium"|"olive"|"brown"|"dark"|null,
  "hair_color": "black"|"brown"|"blonde"|"red"|"gray"|"white"|"bald"|null,
  "hair_style": "short"|"medium"|"long"|"curly"|"straight"|"wavy"|"buzz_cut"|"bald"|"receding"|null,
  "hair_texture": "straight"|"wavy"|"curly"|"coily"|null,
  "eye_color": "brown"|"blue"|"green"|"hazel"|"gray"|"black"|null,
  "eye_shape": "narrow"|"wide"|"almond"|"round"|"hooded"|"deep_set"|null,
  "eyebrows": "thick"|"thin"|"arched"|"bushy"|"sparse"|"straight"|null,
  "nose_shape": "small"|"medium"|"large"|"wide"|"narrow"|"pointed"|"flat"|"hooked"|"straight"|null,
  "lip_shape": "thin"|"medium"|"full"|"wide"|null,
  "jaw_shape": "square"|"round"|"oval"|"pointed"|"strong"|"soft"|null,
  "face_shape": "oval"|"round"|"square"|"heart"|"oblong"|"diamond"|"triangle"|null,
  "facial_hair": "none"|"stubble"|"mustache"|"full_beard"|"goatee"|"sideburns"|null,
  "glasses": "none"|"regular"|"sunglasses"|"reading"|null,
  "distinguishing_features": [],
  "expression": "neutral"|"stern"|"smiling"|"frowning"|"angry"|null,
  "build": "thin"|"slim"|"average"|"athletic"|"stocky"|"heavy"|null,
  "forehead": "normal"|"high"|"broad"|"narrow"|null,
  "cheekbones": "normal"|"high"|"prominent"|"flat"|null,
  "chin": "normal"|"pointed"|"cleft"|"double"|"receding"|"prominent"|null,
  "ears": "normal"|"large"|"small"|"protruding"|null,
  "neck": "normal"|"thick"|"thin"|"long"|"short"|null,
  "complexion": "clear"|"freckled"|"acne"|"wrinkled"|"weathered"|"smooth"|null
}"""


def extract_attributes_groq(description: str) -> dict:
    """
    Parse description via Groq API (Llama-3).
    Falls back to rule-based parser if API unavailable.
    """
    try:
        from groq import Groq
    except ImportError:
        print("[Groq] package not installed. pip install groq  → using rule-based")
        return extract_attributes_rule_based(description)

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[Groq] GROQ_API_KEY missing in .env → using rule-based")
        return extract_attributes_rule_based(description)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": description},
            ],
            temperature=0.05,   # very low — we want deterministic structured output
            max_tokens=1024,
        )
        raw_text = response.choices[0].message.content.strip()
        # Strip markdown fences if the model wraps them anyway
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
        parsed = json.loads(raw_text)
        return _validate_attrs(parsed)

    except json.JSONDecodeError as e:
        print(f"[Groq] JSON parse error: {e} → using rule-based")
        return extract_attributes_rule_based(description)
    except Exception as e:
        print(f"[Groq] API error: {e} → using rule-based")
        return extract_attributes_rule_based(description)


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-Based Parser  (Fallback — no API needed)
# ─────────────────────────────────────────────────────────────────────────────
MALE_WORDS   = {"male", "man", "boy", "gentleman", "he", "his", "guy", "fellow", "sir"}
FEMALE_WORDS = {"female", "woman", "girl", "lady", "she", "her", "gal", "ma'am"}

HAIR_COLOR_KW = {
    "black":  ["black hair", "dark hair", "jet black"],
    "blonde": ["blond", "blonde", "fair hair", "golden hair", "light hair"],
    "brown":  ["brown hair", "brunette", "dark brown hair", "chestnut hair"],
    "gray":   ["gray hair", "grey hair", "silver hair", "salt and pepper"],
    "white":  ["white hair", "white-haired"],
    "red":    ["red hair", "ginger", "auburn", "redhead"],
}

JAW_KW    = ["square", "round", "oval", "pointed", "strong", "soft"]
NOSE_KW   = {"large": ["big nose", "large nose"], "wide": ["wide nose"],
              "small": ["small nose", "tiny nose"], "flat": ["flat nose"],
              "pointed": ["pointed nose", "sharp nose"], "hooked": ["hooked nose"]}
FACIAL_HAIR_KW = {
    "full_beard": ["full beard", "thick beard", "heavy beard", "long beard"],
    "stubble":    ["stubble", "five o'clock shadow", "5 o'clock", "light beard"],
    "mustache":   ["mustache", "moustache"],
    "goatee":     ["goatee"],
    "sideburns":  ["sideburns", "mutton chops"],
}
BUILD_KW = {
    "thin":     ["thin", "slim", "slender", "skinny", "lean"],
    "athletic": ["athletic", "muscular", "fit", "toned"],
    "stocky":   ["stocky", "heavyset", "heavy set", "broad-shouldered"],
    "heavy":    ["heavy", "overweight", "obese", "large build", "heavy build"],
}
SKIN_KW = {
    "very_light": ["very pale", "very light skin", "extremely fair"],
    "light":      ["white", "pale", "fair skin", "light skin", "light-skinned"],
    "olive":      ["olive skin", "olive complexion", "olive-skinned"],
    "brown":      ["brown skin", "tan skin", "tanned", "light brown skin"],
    "dark":       ["dark skin", "dark complexion", "dark-skinned", "deep brown"],
}


def _parse_age(text: str) -> int | None:
    """
    FIX: handles all common age expressions:
      "40"  → 40
      "40s" → 45   (mid-decade)
      "early 40s" → 41
      "mid 40s" → 45
      "late 40s" → 48
      "around 40" → 40
      "approximately 35" → 35
    """
    t = text.lower()

    # "early/mid/late Xs"
    m = re.search(r'\b(early|mid|late)\s+(\d0)s\b', t)
    if m:
        qualifier, decade = m.group(1), int(m.group(2))
        return {"early": decade + 1, "mid": decade + 5, "late": decade + 8}[qualifier]

    # "Xs"  e.g. "40s", "30s"
    m = re.search(r'\b(\d0)s\b', t)
    if m:
        return int(m.group(1)) + 5

    # "around/approximately/about N"
    m = re.search(r'\b(?:around|approximately|about|age|aged)\s+(\d{2})\b', t)
    if m:
        return int(m.group(1))

    # bare two-digit number
    m = re.search(r'\b(\d{2})\b', t)
    if m:
        return int(m.group(1))

    return None


def extract_attributes_rule_based(description: str) -> dict:
    """
    Parse a verbal suspect description with keyword matching.
    Returns None for every undetected feature (no hallucinated defaults).
    """
    t = description.lower()
    words = set(re.findall(r"\b\w+\b", t))
    attrs: dict = {}

    # Gender
    if words & MALE_WORDS:
        attrs["gender"] = "male"
    elif words & FEMALE_WORDS:
        attrs["gender"] = "female"

    # Age
    age = _parse_age(t)
    if age:
        attrs["age"] = age

    # Skin tone — check ethnicity keywords first, then explicit skin descriptors
    for tone, kws in SKIN_KW.items():
        if any(kw in t for kw in kws):
            attrs["skin_tone"] = tone
            break
    # Ethnicity-derived skin hints (only if no explicit skin tone found)
    if "skin_tone" not in attrs:
        if any(w in t for w in ["black", "african", "afro"]):
            attrs["skin_tone"] = "dark"
        elif any(w in t for w in ["asian", "chinese", "japanese", "korean"]):
            attrs["skin_tone"] = "light"
        elif any(w in t for w in ["hispanic", "latino", "latina"]):
            attrs["skin_tone"] = "medium"

    # Hair color
    for color, kws in HAIR_COLOR_KW.items():
        if any(kw in t for kw in kws):
            attrs["hair_color"] = color
            break

    # Hair style
    if any(kw in t for kw in ["bald", "shaved head", "no hair"]):
        attrs["hair_style"] = "bald"
        attrs["hair_color"] = "bald"
    elif any(kw in t for kw in ["buzz cut", "buzz-cut", "crew cut"]):
        attrs["hair_style"] = "buzz_cut"
    elif any(kw in t for kw in ["receding", "receding hairline", "thinning hair"]):
        attrs["hair_style"] = "receding"
    elif any(kw in t for kw in ["curly hair", "curly", "afro"]):
        attrs["hair_style"] = "curly"
    elif any(kw in t for kw in ["wavy hair", "wavy"]):
        attrs["hair_style"] = "wavy"
    elif "long hair" in t or "long" in words:
        attrs["hair_style"] = "long"
    elif "short hair" in t or "short" in words:
        attrs["hair_style"] = "short"

    # Jaw
    for jaw in JAW_KW:
        if f"{jaw} jaw" in t or f"{jaw} jawline" in t:
            attrs["jaw_shape"] = jaw
            break

    # Nose
    for shape, kws in NOSE_KW.items():
        if any(kw in t for kw in kws):
            attrs["nose_shape"] = shape
            break

    # Facial hair
    for style, kws in FACIAL_HAIR_KW.items():
        if any(kw in t for kw in kws):
            attrs["facial_hair"] = style
            break
    if "facial_hair" not in attrs and "clean shaven" in t:
        attrs["facial_hair"] = "none"

    # Glasses
    if any(kw in t for kw in ["glasses", "eyeglasses", "spectacles"]):
        attrs["glasses"] = "regular"
    elif "sunglasses" in t:
        attrs["glasses"] = "sunglasses"

    # Eyebrows
    if any(kw in t for kw in ["bushy eyebrows", "thick eyebrows", "heavy brows"]):
        attrs["eyebrows"] = "bushy"
    elif any(kw in t for kw in ["thin eyebrows", "thin brows"]):
        attrs["eyebrows"] = "thin"
    elif any(kw in t for kw in ["arched eyebrows", "arched brows"]):
        attrs["eyebrows"] = "arched"

    # Build
    for build, kws in BUILD_KW.items():
        if any(kw in t for kw in kws):
            attrs["build"] = build
            break

    # Distinguishing features
    feats = []
    for marker in ["scar", "tattoo", "mole", "birthmark", "wrinkles", "freckles"]:
        m = re.search(rf"\b{marker}\b\s*(?:on\s+)?([\w\s]+?)(?:,|and|\.|$)", t)
        if m:
            loc = m.group(1).strip()
            feats.append(f"{marker} on {loc}" if loc else marker)
    attrs["distinguishing_features"] = feats

    # Expression
    if any(kw in t for kw in ["smiling", "smile", "grinning"]):
        attrs["expression"] = "smiling"
    elif any(kw in t for kw in ["stern", "serious", "severe"]):
        attrs["expression"] = "stern"
    elif any(kw in t for kw in ["angry", "menacing", "hostile"]):
        attrs["expression"] = "angry"

    return _validate_attrs(attrs)


# ─────────────────────────────────────────────────────────────────────────────
#  Unified entry point
# ─────────────────────────────────────────────────────────────────────────────
def extract_attributes(description: str, use_llm: bool = True) -> dict:
    """
    Extract facial attributes from a verbal description.
    Tries Groq LLM first (if use_llm=True), falls back to rule-based.

    Returns:
        Validated attribute dict.  All undetected fields = None (never hallucinated).
    """
    if not description or not description.strip():
        raise ValueError("Description cannot be empty")

    if use_llm:
        return extract_attributes_groq(description)
    return extract_attributes_rule_based(description)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 backward-compat bridge
# ─────────────────────────────────────────────────────────────────────────────
def description_to_cgan_tensor(description: str, use_llm: bool = False):
    """Convert description → 40-dim CelebA tensor for cGAN (legacy Phase 2)."""
    import torch  # noqa: local import
    if dict_to_celeba_tensor is None:
        raise ImportError("data.dataset not available")
    attrs = extract_attributes(description, use_llm=use_llm)
    binary = {
        "male":            attrs.get("gender") == "male",
        "young":           (attrs.get("age") or 35) < 40,
        "bald":            attrs.get("hair_style") == "bald",
        "eyeglasses":      attrs.get("glasses", "none") not in (None, "none"),
        "black_hair":      attrs.get("hair_color") == "black",
        "blond_hair":      attrs.get("hair_color") == "blonde",
        "brown_hair":      attrs.get("hair_color") == "brown",
        "bushy_eyebrows":  attrs.get("eyebrows") in ("thick", "bushy"),
        "sideburns":       attrs.get("facial_hair") == "sideburns",
        "smiling":         attrs.get("expression") == "smiling",
        "pale_skin":       attrs.get("skin_tone") in ("very_light", "light"),
    }
    return dict_to_celeba_tensor(binary)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        "White male around 40 years old, square jaw, brown hair, scar on left cheek",
        "Young Asian woman, straight black hair, narrow eyes, no glasses",
        "Heavyset bald man, late 50s, thick grey beard, wearing glasses",
        "Hispanic female, early 30s, wavy hair, full lips",
        "Black male, mid 40s, short hair, goatee, heavy build",
    ]
    print("=" * 60)
    for desc in cases:
        attrs = extract_attributes_rule_based(desc)
        non_null = {k: v for k, v in attrs.items()
                    if v not in (None, [], "unknown")}
        print(f"\nInput : {desc}")
        print(f"Parsed: {non_null}")
    print("=" * 60)
