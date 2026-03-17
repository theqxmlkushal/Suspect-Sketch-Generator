# 🕵️ Suspect Sketch Generator — Project Architecture

> **A forensic face portrait pipeline powered entirely by SDXL + LCM Scheduler.**
> Text description in → validated JSON attributes → SDXL image out.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Tech Stack](#2-tech-stack)
3. [Directory Structure](#3-directory-structure)
4. [Component Breakdown](#4-component-breakdown)
5. [End-to-End Execution Flow](#5-end-to-end-execution-flow)
6. [Data Flow Diagram](#6-data-flow-diagram)
7. [SDXL + LCM Pipeline Deep Dive](#7-sdxl--lcm-pipeline-deep-dive)
8. [API Contract](#8-api-contract)
9. [State Management (Streamlit)](#9-state-management-streamlit)
10. [Error Handling Strategy](#10-error-handling-strategy)
11. [Testing Architecture](#11-testing-architecture)

---

## 1. System Overview

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     SUSPECT SKETCH GENERATOR v2.1                        ║
║                                                                            ║
║  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────────┐    ║
║  │   Streamlit  │    │  FastAPI     │    │   SDXL + LCM Scheduler    │    ║
║  │   Frontend   │───▶│  REST Layer  │───▶│   Local Inference Engine  │    ║
║  │  (ui/app.py) │    │ (api/api.py) │    │ (pipeline/generation_     │    ║
║  └─────────────┘    └──────────────┘    │  pipeline.py)             │    ║
║         │                  │            └───────────────────────────┘    ║
║         │                  │                          │                   ║
║         ▼                  ▼                          ▼                   ║
║  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────────┐    ║
║  │  NLP Parser  │    │  Pydantic    │    │  Face Validation          │    ║
║  │  + Groq LLM  │    │  Schemas     │    │  (MTCNN Detection)        │    ║
║  │(nlp/nlp_     │    │(api/models.  │    └───────────────────────────┘    ║
║  │ parser.py)   │    │ py)          │                                      ║
║  └─────────────┘    └──────────────┘                                      ║
║         │                                                                  ║
║         ▼                                                                  ║
║  ┌─────────────────────────┐                                               ║
║  │   Prompt Engineer        │                                               ║
║  │ (pipeline/prompt_        │                                               ║
║  │  engineer.py)            │                                               ║
║  └─────────────────────────┘                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive UI, session state, image display |
| **API** | FastAPI + Uvicorn | REST endpoints, request validation |
| **NLP** | Groq (Llama-3) + rule-based | Convert text description → structured attributes |
| **Image Generation** | SDXL Base 1.0 + LCM Scheduler | Local diffusion-based face generation |
| **Face Validation** | facenet-pytorch (MTCNN) | Verify output contains a human face |
| **Schemas** | Pydantic v2 | Typed request/response models |
| **Testing** | pytest | Unit + integration tests |
| **Task Runner** | Makefile | Dev workflow automation |

---

## 3. Directory Structure

```
suspect-sketch-generator/
│
├── .env                        ← Runtime secrets (never commit)
├── .env.example                ← Key template
├── .gitignore
├── requirements.txt            ← All Python dependencies
├── Makefile                    ← Dev commands (install, run, test)
├── README.md
├── PROJECT_ARCHITECTURE.md     ← This file
│
├── nlp/
│   ├── __init__.py
│   └── nlp_parser.py           ← Groq LLM + regex fallback → JSON attributes
│
├── pipeline/
│   ├── __init__.py
│   ├── prompt_engineer.py      ← Attributes → SDXL prompt string
│   └── generation_pipeline.py ← SDXL + LCM inference, retry logic
│
├── api/
│   ├── __init__.py
│   ├── api.py                  ← FastAPI app: /health /parse /generate
│   └── models.py               ← Pydantic request/response schemas
│
├── ui/
│   ├── __init__.py
│   └── app.py                  ← Streamlit frontend, seed persistence
│
├── scripts/
│   └── test_apis.py            ← Smoke test (run before full app)
│
└── tests/
    ├── test_nlp_parser.py
    ├── test_prompt_engineer.py
    ├── test_bugfix_validate_faces.py
    └── test_preservation_properties.py
```

---

## 4. Component Breakdown

### 4.1 NLP Parser (`nlp/nlp_parser.py`)

Converts a raw English description into a validated JSON attribute object.

```
Input:  "White male, early 40s, square jaw, scar on left cheek"
          │
          ├─ Path A: Groq LLM (Llama-3-8b-8192)
          │    └─ System prompt enforces: null for every unmentioned feature
          │
          └─ Path B: Rule-based regex fallback (no API key required)
               ├─ Age parser: "40s" → 45, "early 30s" → 31, "mid 50s" → 53
               ├─ Gender: male/female/non-binary
               ├─ Physical attributes: jaw, hair, eyes, nose, lips, etc.
               └─ Distinguishing features: scars, tattoos, piercings, etc.

Output: {
  "age": 41,
  "gender": "male",
  "ethnicity": "white",
  "jaw_shape": "square",
  "hair_color": null,       ← null = NOT mentioned, will not appear in prompt
  "eye_color": null,
  "distinguishing_features": ["scar on left cheek"],
  ...
}
```

**Key design rule:** Features not explicitly mentioned are set to `null`. They are never guessed from demographics or stereotypes. This prevents hallucinated traits from polluting the SDXL prompt.

---

### 4.2 Prompt Engineer (`pipeline/prompt_engineer.py`)

Converts the attribute JSON into a structured, priority-ordered SDXL prompt.

```
Attribute priority order (highest → lowest identity impact):
  1. Distinguishing marks    → (scar on left cheek:1.4)  [emphasis weight]
  2. Demographics            → 41 year old male
  3. Ethnicity               → white
  4. Face structure          → square jawline
  5. Hair                    → brown short hair
  6. Eyes                    → blue eyes
  7. Nose / lips             → only if non-default
  8. Facial hair / glasses
  9. Skin tone
 10. Expression / build      → only if non-neutral/average

Prefix (always):
  "close-up forensic pencil sketch portrait, face centered, white paper,"

Suffix (≤12 words):
  "graphite portrait, sharp pencil lines, law enforcement composite"

Negative prompt:
  "color, photograph, blurry, body, landscape, nsfw, cartoon"
```

**Example output:**

```
POSITIVE:
"close-up forensic pencil sketch portrait, face centered, white paper,
 (scar on left cheek:1.4), 41 year old white male, square jawline,
 graphite portrait, sharp pencil lines, law enforcement composite"

NEGATIVE:
"color, photograph, blurry, body, landscape, nsfw, cartoon, text, watermark"
```

---

### 4.3 Generation Pipeline (`pipeline/generation_pipeline.py`)

Runs the SDXL + LCM Scheduler locally to produce face images.

```
┌─────────────────────────────────────────────────────────────┐
│                  SDXL + LCM SCHEDULER PIPELINE               │
│                                                               │
│  Model load (first run only, ~5–8 GB VRAM or CPU offload):   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  stabilityai/stable-diffusion-xl-base-1.0             │   │
│  │  Scheduler: LCMScheduler (latent consistency model)    │   │
│  │  LoRA weights: latent-consistency/lcm-lora-sdxl        │   │
│  └───────────────────────────────────────────────────────┘   │
│                         │                                     │
│                    Inference call                             │
│                         │                                     │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  Steps:       4–8  (LCM = ultra-fast, 4 steps ok)     │   │
│  │  Guidance:    1.5–2.0 (LCM requires low CFG)          │   │
│  │  Resolution:  1024×1024 (SDXL native)                 │   │
│  │  Seed:        user-controlled (session_state)         │   │
│  └───────────────────────────────────────────────────────┘   │
│                         │                                     │
│              Face validation (MTCNN)                         │
│                         │                                     │
│          ┌──────────────┴──────────────┐                     │
│          │ face detected (≥0.85 conf)  │ no face detected    │
│          ▼                             ▼                     │
│       return image               retry with new seed         │
│                                  (max 2 retries)             │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.4 FastAPI Layer (`api/api.py` + `api/models.py`)

```
Endpoints:

  GET  /health    → system status, model loaded flag, face validation available
  POST /parse     → run NLP parser only, return attribute JSON
  POST /generate  → full pipeline: parse → prompt → SDXL → validate → return images

Error codes:
  200 → success
  422 → invalid input (Pydantic validation failed)
  503 → generation failed after all retries
  500 → unexpected server error
```

---

### 4.5 Streamlit Frontend (`ui/app.py`)

```
Page layout:
  ┌──────────────────────────────────────────────────────┐
  │  🕵️ AI Suspect Sketch Generator                       │
  │                                                        │
  │  [Text area: Enter suspect description...]            │
  │                                                        │
  │  Style: [forensic_sketch ▼]   Images: [2 ▼]          │
  │                                                        │
  │  [Parse only]          [Generate sketch]              │
  │                                                        │
  │  ┌───────────┐  ┌───────────┐                        │
  │  │  Image 1  │  │  Image 2  │                        │
  │  └───────────┘  └───────────┘                        │
  │                                                        │
  │  Seed: 12345     [New variation]                      │
  │                                                        │
  │  Parsed attributes:                                    │
  │  age=41, gender=male, jaw_shape=square, ...           │
  └──────────────────────────────────────────────────────┘

Session state keys:
  seed        → integer, persists across reruns
  pipe_loaded → bool, True after SDXL model is in memory
  last_attrs  → dict, last parsed attribute set
```

---

## 5. End-to-End Execution Flow

### 5.1 First-Ever Run (Cold Start)

```
$ git clone https://github.com/theqxmlkushal/Suspect-Sketch-Generator.git
$ cd Suspect-Sketch-Generator
$ python -m venv venv && source venv/bin/activate
$ pip install -r requirements.txt
$ pip install facenet-pytorch
$ cp .env.example .env         ← add your GROQ_API_KEY here
$ streamlit run ui/app.py
```

```
BOOT SEQUENCE
═════════════

[1] Python process starts
     └─ Streamlit loads ui/app.py
          └─ load_dotenv() reads .env → GROQ_API_KEY into environment
          └─ session_state initialised:
               seed        = random.randint(1, 99_999)
               pipe_loaded = False

[2] SDXL model load (runs once, ~30–90 seconds on first launch)
     └─ from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
     └─ load_lora_weights("latent-consistency/lcm-lora-sdxl")
     └─ scheduler = LCMScheduler.from_config(pipe.scheduler.config)
     └─ pipe.to("cuda")  [or "cpu" if no GPU]
     └─ session_state.pipe_loaded = True

[3] UI renders — user sees "Ready" status
```

### 5.2 Typical Generation Request

```
USER TYPES: "White male, early 40s, square jaw, scar on left cheek"
USER CLICKS: "Generate sketch"

Step 1 — NLP PARSING
─────────────────────
  ui/app.py calls nlp_parser.extract_attributes(description, use_llm=True)
       │
       ├─ POST to Groq API (Llama-3-8b-8192)
       │    └─ System prompt: "Return ONLY JSON. Set null for unmentioned features."
       │    └─ Response parsed and validated against attribute schema
       │
       └─ If Groq unavailable → rule_based_fallback(description)
            └─ Regex extracts: age="early 40s"→41, gender="male", etc.

  Result: {age:41, gender:"male", ethnicity:"white",
           jaw_shape:"square", distinguishing_features:["scar on left cheek"],
           hair_color:null, eye_color:null, ...}

Step 2 — PROMPT ENGINEERING
─────────────────────────────
  build_forensic_prompt(attrs, style="forensic_sketch")
       │
       └─ Filter: skip all null + default-value attributes
       └─ Sort by priority: distinguishing marks first, demographics second
       └─ Apply emphasis: "(scar on left cheek:1.4)"
       └─ Prepend anchor: "close-up forensic pencil sketch portrait..."
       └─ Append ≤12-word style suffix

  Positive: "close-up forensic pencil sketch portrait, face centered, white paper,
             (scar on left cheek:1.4), 41 year old white male, square jawline,
             graphite portrait, sharp pencil lines, law enforcement composite"

  Negative: "color, photograph, blurry, body, landscape, nsfw, cartoon, watermark"

Step 3 — SDXL GENERATION
─────────────────────────
  generate_images(prompt, negative_prompt, seed=12345, num_images=2)
       │
       ├─ torch.manual_seed(12345)
       ├─ pipe(
       │    prompt=...,
       │    negative_prompt=...,
       │    num_inference_steps=4,      ← LCM: 4 steps is sufficient
       │    guidance_scale=1.5,         ← LCM: must be low (1.0–2.0)
       │    width=1024, height=1024,
       │    num_images_per_prompt=2,
       │    generator=torch.Generator().manual_seed(12345)
       │  )
       └─ Returns list of PIL Images

Step 4 — FACE VALIDATION
──────────────────────────
  For each image:
       └─ MTCNN(image) → boxes, probabilities
            ├─ prob ≥ 0.85 → ✅ keep image
            └─ prob < 0.85 or no face → ❌ retry with seed+1

Step 5 — DISPLAY
──────────────────
  Streamlit renders validated images side-by-side
  Parsed attributes shown below in expandable section
  Seed displayed with "New variation" button
```

### 5.3 "New Variation" Click

```
User clicks [New variation]
     └─ session_state.seed = random.randint(1, 99_999)
     └─ st.rerun()
     └─ Same description + same parsed attributes
     └─ Different seed → different face sampling path
     └─ Steps 3–5 repeat with new seed
```

---

## 6. Data Flow Diagram

```
                        ┌──────────────────────────┐
                        │     USER INPUT            │
                        │  "White male, 40s,        │
                        │   square jaw, scar..."    │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   NLP PARSER              │
                        │   nlp/nlp_parser.py       │
                        │                           │
                        │  Groq LLM (primary)       │
                        │       ↓                   │
                        │  Rule-based (fallback)    │
                        └────────────┬─────────────┘
                                     │
                          Validated JSON attributes
                          {age, gender, jaw_shape,
                           distinguishing_features...}
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   PROMPT ENGINEER         │
                        │  pipeline/prompt_         │
                        │  engineer.py              │
                        │                           │
                        │  Priority sort            │
                        │  Null filtering           │
                        │  Emphasis weighting       │
                        └────────────┬─────────────┘
                                     │
                           SDXL prompt string
                           + negative prompt
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   SDXL + LCM PIPELINE     │
                        │  pipeline/generation_     │
                        │  pipeline.py              │
                        │                           │
                        │  SDXL Base 1.0            │
                        │  + LCM LoRA               │
                        │  LCMScheduler             │
                        │  4 steps / CFG 1.5        │
                        └────────────┬─────────────┘
                                     │
                                PIL Image(s)
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   FACE VALIDATION         │
                        │  facenet-pytorch MTCNN    │
                        │                           │
                        │  conf ≥ 0.85 → pass       │
                        │  conf < 0.85 → retry      │
                        └────────────┬─────────────┘
                                     │
                          Validated face image(s)
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   STREAMLIT UI            │
                        │   ui/app.py               │
                        │                           │
                        │  Display images           │
                        │  Show attributes          │
                        │  Seed control             │
                        └──────────────────────────┘
```

---

## 7. SDXL + LCM Pipeline Deep Dive

### Why LCM?

Standard SDXL requires 25–50 denoising steps and 5–10 seconds per image on a good GPU.

LCM (Latent Consistency Model) + LCM-LoRA reduces this to **4 steps** with comparable quality — roughly **4–10× faster**. This makes iterative forensic use practical (witness can give real-time feedback).

### How LCM-LoRA integrates with SDXL

```
┌─────────────────────────────────────────────────────────────────┐
│  Standard SDXL                   SDXL + LCM-LoRA               │
│  ────────────                    ──────────────                 │
│  Scheduler: DDIM/DPM++           Scheduler: LCMScheduler       │
│  Steps:     25–50                Steps:     4–8                 │
│  CFG scale: 7.0–9.0              CFG scale: 1.0–2.0             │
│  Time/img:  8–15s (GPU)          Time/img:  1–3s (GPU)          │
│                                                                   │
│  LoRA adds consistency distillation weights on top of the       │
│  base SDXL UNet. No separate model download needed.             │
└─────────────────────────────────────────────────────────────────┘
```

### Model initialization code pattern

```python
from diffusers import StableDiffusionXLPipeline, LCMScheduler
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")   # or "mps" on Apple Silicon, "cpu" as last resort
```

### Inference call pattern

```python
generator = torch.Generator(device="cuda").manual_seed(seed)

result = pipe(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=4,      # LCM works best at 4-8
    guidance_scale=1.5,         # Must be low for LCM
    width=1024,
    height=1024,
    num_images_per_prompt=num_images,
    generator=generator,
)
images = result.images   # List[PIL.Image]
```

### LCM Step-by-Step Denoising (4 steps visualised)

```
Random noise (t=1000)
       │
       ▼  [Step 1 — LCM leap]  ~75% of denoising done here
  Rough face shape visible
       │
       ▼  [Step 2]  Facial structure solidified
  Jaw line, brow, nose visible
       │
       ▼  [Step 3]  Fine features appear
  Eyes, lips, skin texture
       │
       ▼  [Step 4]  Final polish
  Sharp edges, shading complete
       │
       ▼
  Final 1024×1024 image
```

---

## 8. API Contract

### `GET /health`

```json
{
  "status": "ok",
  "groq_key_set": true,
  "model_loaded": true,
  "face_validation_available": true,
  "styles_available": ["forensic_sketch", "photorealistic", "composite"]
}
```

### `POST /parse`

**Request:**
```json
{
  "description": "White male, early 40s, square jaw, scar on left cheek",
  "use_llm": true
}
```

**Response:**
```json
{
  "attributes": {
    "age": 41,
    "gender": "male",
    "ethnicity": "white",
    "jaw_shape": "square",
    "distinguishing_features": ["scar on left cheek"],
    "hair_color": null,
    "eye_color": null
  },
  "parser_used": "groq_llama3",
  "non_null_count": 5
}
```

### `POST /generate`

**Request:**
```json
{
  "description": "White male, early 40s, square jaw, scar on left cheek",
  "style": "forensic_sketch",
  "num_images": 2,
  "seed": 42,
  "use_llm": true,
  "validate_faces": true
}
```

**Response:**
```json
{
  "images": ["base64...", "base64..."],
  "images_generated": 2,
  "attributes": { "...": "..." },
  "prompt": "close-up forensic pencil sketch portrait...",
  "negative_prompt": "color, photograph, blurry...",
  "generation_time_seconds": 3.2,
  "seed_used": 42
}
```

**Error codes:**

| Code | Meaning |
|---|---|
| 200 | Success |
| 422 | Invalid input (Pydantic validation) |
| 503 | SDXL generation failed after retries |
| 500 | Unexpected server error |

---

## 9. State Management (Streamlit)

Streamlit reruns the entire script on every user interaction. Without explicit state management, the seed would change on every click, making iterative refinement impossible.

```
session_state lifecycle:

  App boot
    └─ st.session_state.setdefault("seed", random.randint(1, 99_999))
    └─ st.session_state.setdefault("pipe_loaded", False)
    └─ st.session_state.setdefault("last_attrs", {})

  User clicks "Generate"
    └─ seed read from session_state  ← same as last run
    └─ images generated with that seed

  User clicks "New variation"
    └─ session_state.seed = random.randint(1, 99_999)  ← only now does seed change
    └─ st.rerun()

  User changes description
    └─ session_state.seed unchanged  ← same face structure, different features
```

---

## 10. Error Handling Strategy

```
Layer           Error type              Handler
──────────────────────────────────────────────────────────
NLP Parser      Groq API timeout        → fallback to rule-based regex
NLP Parser      JSON parse failure      → fallback to rule-based regex
NLP Parser      All-null result         → raise ValueError("no attributes found")

Prompt Eng.     All attrs null          → return minimal anchor prompt only

Generation      SDXL OOM (GPU)          → switch to CPU offload, retry
Generation      No face detected        → increment seed by 1, retry (max 2×)
Generation      Generation timeout      → retry same call (max 2×, 1.5s delay)
Generation      All retries exhausted   → raise RuntimeError → API returns 503

FastAPI         Pydantic mismatch       → 422 with field-level error detail
FastAPI         Unhandled exception     → 500 with sanitised message (no stack trace)

Streamlit       pipe not loaded yet     → spinner + "Initialising model..."
Streamlit       generation error        → st.error() with human-readable message
```

---

## 11. Testing Architecture

```
tests/
├── test_nlp_parser.py
│    ├── test_age_decade_parsing()       "40s" → 45, "early 30s" → 31
│    ├── test_null_for_unmentioned()     hair_color=null when not in input
│    ├── test_distinguishing_features()  scar/tattoo/piercing extraction
│    └── test_groq_vs_rulebased_parity() both paths return same schema shape
│
├── test_prompt_engineer.py
│    ├── test_priority_ordering()        distinguishing marks first
│    ├── test_null_skipped()             null attrs absent from prompt
│    ├── test_emphasis_weights()         (scar:1.4) format correct
│    └── test_suffix_length()           style suffix ≤ 12 words
│
├── test_bugfix_validate_faces.py
│    ├── test_mtcnn_accepts_real_face()
│    └── test_mtcnn_rejects_landscape()
│
└── test_preservation_properties.py
     ├── test_seed_reproducibility()    same seed → same image hash
     └── test_variation_differs()       different seed → different output
```

Run all tests:
```bash
make test
# or
pytest tests/ -v
```

---

## Quick Reference — Makefile Commands

```
make install       pip install -r requirements.txt
make run-ui        streamlit run ui/app.py → http://localhost:8501
make run-api       uvicorn api.api:app --reload → http://localhost:8000/docs
make test          pytest tests/ -v
make smoke         quick end-to-end check (no API keys needed)
make lint          syntax check all Python files
make clean         remove __pycache__ directories
```

---

## Environment Variables

```bash
# .env.example

# Required — NLP parsing
GROQ_API_KEY=gsk_...         # Free at console.groq.com

# Optional — Override SDXL model path (defaults to HuggingFace cache)
SDXL_MODEL_PATH=./models/sdxl-base-1.0

# Optional — Generation defaults
DEFAULT_NUM_STEPS=4
DEFAULT_GUIDANCE_SCALE=1.5
DEFAULT_RESOLUTION=1024
```

> **Note:** Image generation runs entirely locally via SDXL + LCM Scheduler. No external image API keys are required or used.

---

*Architecture document for Suspect Sketch Generator v2.1 — SDXL + LCM Scheduler edition.*
