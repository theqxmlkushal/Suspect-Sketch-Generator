# 🕵️ AI Suspect Sketch Generator v2.1

> **Text description → forensic face portrait in seconds.**
> Pipeline: `Groq/Llama-3 → validated JSON → SDXL/FLUX → face image`

---

## What changed from v2.0 (and why your output was wrong)

| File | Root cause fixed |
|------|-----------------|
| `nlp_parser.py` | Groq was **hallucinating** traits for unmentioned features. Fixed: `null` for everything not explicitly said. |
| `nlp_parser.py` | Age regex missed `"40s"`, `"early 30s"`. Fixed: full decade-range parser. |
| `prompt_engineer.py` | 40+ style tokens drowned the face description. Fixed: ≤12-word suffix, priority ordering, emphasis weights. |
| `prompt_engineer.py` | `"medium nose"` and `"normal forehead"` appeared as filler. Fixed: skip all null and default values. |
| `generation_pipeline.py` | No retries — one timeout = blank result. Fixed: 2 retries per backend. |
| `generation_pipeline.py` | No face validation — Pollinations sometimes returns abstract art. Fixed: MTCNN check. |
| `ui/app.py` | Seed was random on every rerun — couldn't iterate. Fixed: seed in `session_state`. |
| `api/api.py` | All errors returned 500. Fixed: 422 for bad input, 503 for backend failure. |

---

## Project structure

```
suspect-sketch-ai/
├── .env                    ← your API keys (never commit this)
├── .env.example            ← template
├── .gitignore
├── requirements.txt
├── Makefile
│
├── nlp/
│   ├── __init__.py
│   └── nlp_parser.py       ← Groq + rule-based parser, validated schema
│
├── pipeline/
│   ├── __init__.py
│   ├── prompt_engineer.py  ← attrs → SDXL prompt, priority-ordered
│   └── generation_pipeline.py  ← HF / Together / Pollinations backends
│
├── api/
│   ├── __init__.py
│   ├── api.py              ← FastAPI, /health /parse /generate
│   └── models.py           ← Pydantic request/response models
│
├── ui/
│   ├── __init__.py
│   └── app.py              ← Streamlit frontend with seed persistence
│
├── tests/
│   ├── test_nlp_parser.py
│   └── test_prompt_engineer.py
│
└── scripts/
    └── test_apis.py        ← end-to-end backend smoke test
```

---

## Quick start (5 steps)

### Step 1 — Clone and create virtual environment

```bash
git clone https://github.com/yourname/suspect-sketch-ai.git
cd suspect-sketch-ai

# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

For face validation (strongly recommended):
```bash
pip install facenet-pytorch
```

### Step 3 — Set up API keys

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_...        # free at console.groq.com
HF_TOKEN=hf_...             # optional, free at huggingface.co/settings/tokens
TOGETHER_API_KEY=...        # optional, free at api.together.ai
```

Only `GROQ_API_KEY` is needed for NLP parsing. Image generation falls back to
Pollinations.ai (completely free, no key) if no image API keys are set.

### Step 4 — Run the smoke test

```bash
python scripts/test_apis.py
```

You should see all configured backends pass before running the full app.

### Step 5 — Start the UI

```bash
streamlit run ui/app.py
# → http://localhost:8501
```

Or start the API:
```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs
```

---

## Step-by-step improvement guide

### Step 1 — Fix Groq hallucinations (done in v2.1)

**Problem:** Original system prompt said *"use reasonable defaults for unmentioned features"*.
Llama-3 would invent `"high cheekbones"`, `"deep-set eyes"`, `"prominent brow"` even when the
description only said `"male, 40s, square jaw"`. These invented details became part of the
SDXL prompt and produced a generic Llama-imagined face, not the described face.

**Fix applied:**
```python
# OLD system prompt (wrong)
"If a feature is not mentioned, use reasonable defaults based on demographics."

# NEW system prompt (correct)
"If a feature is NOT explicitly mentioned, set it to null.
 Do NOT invent or assume features based on ethnicity, age, or gender stereotypes."
```

**Verify this works:**
```python
from nlp.nlp_parser import extract_attributes_rule_based
a = extract_attributes_rule_based("Male with a scar on his chin")
assert a.get("hair_color") is None   # was "brown" before fix
assert a.get("eye_color")  is None   # was "brown" before fix
```

---

### Step 2 — Fix age range parsing (done in v2.1)

**Problem:** Regex `\b(\d{2})\b` only matched bare numbers. `"40s"` → no match → age defaults to 35.

**Fix applied in `_parse_age()`:**
```python
# Handles: "40s" → 45, "early 30s" → 31, "mid 40s" → 45, "late 50s" → 58
# "around 38" → 38, "approximately 55" → 55
```

---

### Step 3 — Shorten and prioritize the SDXL prompt (done in v2.1)

**Problem:** Original suffix was 40+ tokens. SDXL weights all tokens equally, so
style tokens crowded out face tokens. A description with 8 face features + 40 style
tokens means only ~17% of the model's attention went to the actual face.

**Fix applied:** Suffix trimmed to ≤12 words. Features ordered by identity importance:

```
1. Distinguishing marks  ← (scar on cheek:1.4) — emphasis weight
2. Demographics          ← 42 year old male
3. Face structure        ← square jawline
4. Hair                  ← brown short hair
5. Eyes                  ← blue eyes
6. Nose / lips           ← only if non-medium
7. Facial hair / glasses
8. Skin tone
9. Expression / build    ← only if non-neutral/average
```

**What good vs bad prompts look like:**
```
# BAD (v2.0) — style drowns out face
"42 year old male, light skin tone, brown short hair, bushy eyebrows,
 forensic police sketch, pencil drawing on white paper, detailed graphite portrait,
 law enforcement composite sketch, professional forensic artist rendering,
 charcoal shading, realistic facial proportions, high detail, sharp lines"

# GOOD (v2.1) — face first, tight suffix
"close-up forensic pencil sketch portrait, face centered, white paper,
 (scar on left cheek:1.4), 42 year old male, square jawline, brown short hair,
 bushy eyebrows, stubble, light skin tone,
 graphite portrait, sharp pencil lines, law enforcement composite"
```

---

### Step 4 — Add seed persistence to Streamlit (done in v2.1)

**Problem:** Every widget interaction triggered a Streamlit rerun, which called
`random.randint()` for the seed → different face every time.

**Fix applied in `ui/app.py`:**
```python
# Initialise once — survives reruns
st.session_state.setdefault("seed", random.randint(1, 99_999))

# Use stored seed in generate call
imgs = generate_images(prompt=prompt, seed=st.session_state.seed, ...)

# Only change seed when user explicitly requests it
if st.button("New variation"):
    st.session_state.seed = random.randint(1, 99_999)
    st.rerun()
```

---

### Step 5 — Add retry logic to generation backends (done in v2.1)

**Problem:** One network timeout = blank output. No retry.

**Fix applied:**
```python
def _with_retry(fn, name, retries=2):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[{name}] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(1.5)
    return None
```

---

### Step 6 — Add face validation (done in v2.1)

**Problem:** Pollinations.ai occasionally returns abstract/landscape images when
busy, especially with unusual prompts.

**Fix applied:**
```python
from facenet_pytorch import MTCNN
detector = MTCNN(keep_all=False, device="cpu", post_process=False)

def _has_face(img, min_confidence=0.85):
    boxes, probs = detector.detect(img.convert("RGB"))
    return boxes is not None and float(probs[0]) >= min_confidence
```

Images without a detected face are silently retried with a different seed.

---

### Step 7 — Next improvements to make (not yet done)

#### 7a. Add ControlNet for face structure guidance

Use a reference face sketch as a structural guide so FLUX respects
the jaw/face shape more precisely:

```python
# pipeline/controlnet_pipeline.py  (Phase 3)
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import cv2

def sketch_to_canny(sketch_img):
    gray  = cv2.cvtColor(np.array(sketch_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
```

#### 7b. Add iterative refinement endpoint

Let the user adjust one attribute at a time and regenerate:

```python
# api/api.py — new endpoint
@app.post("/refine")
async def refine(req: RefineRequest):
    """
    Take existing attributes + one change → new image.
    req.base_attributes = last parsed attrs
    req.change = {"jaw_shape": "round"}  ← override one field
    """
    attrs = {**req.base_attributes, **req.change}
    prompt, neg = build_forensic_prompt(attrs, style=req.style)
    images = generate_images(prompt=prompt, seed=req.seed, num_images=1)
    ...
```

#### 7c. Add CUFS forensic sketch fine-tuning (Phase 3)

Fine-tune SDXL with LoRA on the CUFS dataset (606 photo↔sketch pairs)
for authentic pencil-sketch style:

```bash
# Download CUFS dataset
# http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html

# Fine-tune with LoRA (requires GPU with 16GB+ VRAM)
accelerate launch scripts/train_lora.py \
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data_dir ./data/cufs/sketches \
  --output_dir ./checkpoints/cufs_lora \
  --num_train_epochs 50 \
  --lora_rank 16
```

#### 7d. Add ArcFace identity similarity scoring

Measure how similar the generated face is to a reference photo (useful for
evaluating how well the description was followed):

```python
# evaluation/arcface_eval.py
from insightface.app import FaceAnalysis

def face_similarity(img1: Image, img2: Image) -> float:
    """Cosine similarity between ArcFace embeddings. 1.0 = identical."""
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))
    e1 = app.get(np.array(img1))[0].normed_embedding
    e2 = app.get(np.array(img2))[0].normed_embedding
    return float(np.dot(e1, e2))
```

---

## API reference

### `GET /health`

```json
{
  "status": "ok",
  "groq_key_set": true,
  "hf_token_set": false,
  "together_key_set": false,
  "face_validation_available": true,
  "styles_available": ["forensic_sketch", "photorealistic", "composite"]
}
```

### `POST /parse`

```json
// Request
{ "description": "White male, early 40s, square jaw, scar on left cheek", "use_llm": true }

// Response
{
  "attributes": { "age": 41, "gender": "male", "jaw_shape": "square",
                  "distinguishing_features": ["scar on left cheek"], ... },
  "parser_used": "groq_llama3",
  "non_null_count": 7
}
```

### `POST /generate`

```json
// Request
{
  "description": "White male, early 40s, square jaw, scar on left cheek",
  "style": "forensic_sketch",
  "num_images": 2,
  "seed": 42,
  "use_llm": true,
  "validate_faces": true
}

// Response
{
  "images": ["base64...", "base64..."],
  "images_generated": 2,
  "attributes": { ... },
  "prompt": "close-up forensic pencil sketch...",
  "negative_prompt": "color, photograph...",
  "generation_time_seconds": 18.4,
  "backends_tried": ["HuggingFace", "Together AI", "Pollinations"]
}
```

---

## Common problems and fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Generic face, ignores description | Groq hallucinating defaults | Upgrade to v2.1 nlp_parser.py |
| Face not in portrait frame | Missing portrait anchor | Upgrade to v2.1 prompt_engineer.py |
| Different face every run | Seed not persisted | Upgrade to v2.1 app.py |
| `"40s"` parsed as age 35 | Broken regex | Upgrade to v2.1 nlp_parser.py |
| Abstract/landscape image | Pollinations busy | Face validation now rejects these |
| `500` error on bad input | Missing validation | Now returns `422` |
| Timeout kills generation | No retry | Now retries 2× per backend |
| `pyvenv.cfg` in git | Missing gitignore | Updated .gitignore |

---

## Makefile commands

```bash
make install       # pip install -r requirements.txt
make run-ui        # streamlit run ui/app.py
make run-api       # uvicorn api.api:app --reload
make test          # pytest tests/ -v
make smoke         # quick end-to-end test (no API keys needed)
make lint          # syntax check all Python files
make clean         # remove __pycache__
```
