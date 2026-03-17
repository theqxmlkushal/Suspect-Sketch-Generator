# 🏗️ Suspect Sketch Generator - Project Architecture

## System Overview

The Suspect Sketch Generator is a three-tier system that converts text descriptions of suspects into forensic sketch portraits using AI. The pipeline combines NLP parsing, prompt engineering, and image generation.

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit UI (ui/app.py)                                        │
│  ├─ Text input: suspect description                             │
│  ├─ Parse button: extract attributes                            │
│  ├─ Generate button: create sketch                              │
│  └─ Display: show generated images with metadata                │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   APPLICATION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Backend (api/api.py)                                    │
│  ├─ /health: system status check                                │
│  ├─ /parse: extract attributes from description                 │
│  └─ /generate: create forensic sketch                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   PROCESSING LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ NLP Parser (nlp/nlp_parser.py)                           │   │
│  │ ├─ Rule-based parsing: extract age, gender, features    │   │
│  │ ├─ Groq/Llama-3 LLM: structured attribute extraction    │   │
│  │ └─ Validation: ensure no hallucinated defaults          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Prompt Engineer (pipeline/prompt_engineer.py)            │   │
│  │ ├─ Priority ordering: identity features first           │   │
│  │ ├─ Emphasis weights: (scar:1.4) for key features        │   │
│  │ ├─ Style tokens: forensic sketch aesthetic              │   │
│  │ └─ Negative prompt: exclude unwanted elements           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Generation Pipeline (pipeline/generation_pipeline.py)    │   │
│  │ ├─ Backend selection: HF → Together → Pollinations      │   │
│  │ ├─ Retry logic: 2 retries per backend on failure        │   │
│  │ ├─ Face validation: MTCNN detection (optional)          │   │
│  │ └─ Image processing: format & return results            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   EXTERNAL SERVICES                              │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Groq API: LLM for attribute extraction                      │
│  ├─ HuggingFace: SDXL image generation (optional)               │
│  ├─ Together AI: FLUX image generation (optional)               │
│  ├─ Pollinations.ai: Free fallback image generation             │
│  └─ facenet-pytorch: Face detection & validation (optional)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. NLP Parser (`nlp/nlp_parser.py`)

**Purpose:** Extract structured attributes from unstructured text descriptions.

**Key Functions:**
- `extract_attributes_rule_based()`: Fast regex-based parsing
- `extract_attributes_groq()`: LLM-powered parsing with Groq/Llama-3
- `extract_attributes()`: Wrapper that chooses parser based on `use_llm` flag

**Attributes Extracted:**
```python
{
  "age": int,                              # 18-100
  "gender": str,                           # "male", "female", "other"
  "ethnicity": str,                        # "white", "black", "asian", etc.
  "hair_color": str,                       # "brown", "black", "blonde", etc.
  "hair_style": str,                       # "short", "long", "curly", etc.
  "eye_color": str,                        # "blue", "brown", "green", etc.
  "face_shape": str,                       # "round", "square", "oval", etc.
  "nose_shape": str,                       # "straight", "hooked", "bulbous", etc.
  "build": str,                            # "slim", "average", "muscular", etc.
  "skin_tone": str,                        # "light", "medium", "dark", etc.
  "facial_hair": str,                      # "clean", "stubble", "beard", etc.
  "distinguishing_features": list[str],    # ["scar on left cheek", "tattoo on neck"]
  "glasses": bool,                         # true/false
  "expression": str                        # "neutral", "angry", "sad", etc.
}
```

**Design Decisions:**
- `null` for unmentioned features (no hallucination)
- Age range parsing: "40s" → 45, "early 30s" → 31
- Distinguishing features prioritized for identity

---

### 2. Prompt Engineer (`pipeline/prompt_engineer.py`)

**Purpose:** Convert structured attributes into optimized SDXL/FLUX prompts.

**Key Functions:**
- `build_forensic_prompt()`: Create positive + negative prompts
- `_build_positive_prompt()`: Feature-ordered prompt with emphasis weights
- `_build_negative_prompt()`: Exclude unwanted elements

**Prompt Structure:**
```
[Style anchor] [Distinguishing marks] [Demographics] [Face structure] 
[Hair] [Eyes] [Nose/Lips] [Facial hair] [Skin tone] [Expression/Build]
[Style suffix]
```

**Example Output:**
```
Positive:
"close-up forensic pencil sketch portrait, face centered, white paper,
 (scar on left cheek:1.4), 42 year old male, square jawline, brown short hair,
 bushy eyebrows, stubble, light skin tone,
 graphite portrait, sharp pencil lines, law enforcement composite"

Negative:
"color, photograph, blurry, abstract, landscape, multiple faces, 
 cartoon, painting, watercolor, low quality"
```

**Design Decisions:**
- Emphasis weights: (feature:1.4) for distinguishing marks
- Feature ordering: identity-critical features first
- Suffix ≤12 words: prevents style tokens from drowning face details
- Skip null/default values: no filler tokens

---

### 3. Generation Pipeline (`pipeline/generation_pipeline.py`)

**Purpose:** Generate images using multiple backends with fallback and validation.

**Key Functions:**
- `generate_images()`: Main entry point
- `_generate_pollinations()`: Free fallback backend
- `_generate_hf()`: HuggingFace SDXL (requires HF_TOKEN)
- `_generate_together()`: Together AI FLUX (requires TOGETHER_API_KEY)
- `_has_face()`: MTCNN face detection for validation
- `_with_retry()`: Retry wrapper for fault tolerance

**Backend Selection Logic:**
```
1. Try HuggingFace (if HF_TOKEN set)
   └─ Retry 2× on failure
2. Try Together AI (if TOGETHER_API_KEY set)
   └─ Retry 2× on failure
3. Fall back to Pollinations.ai (always available)
   └─ Retry 2× on failure
4. If all fail, return None
```

**Face Validation:**
- Uses MTCNN (Multi-task Cascaded Convolutional Networks)
- Confidence threshold: 0.85
- If face not detected: retry with different seed
- If facenet-pytorch not installed: skip validation (accept all images)

**Design Decisions:**
- Retry logic: 2 retries per backend (handles transient failures)
- Fallback chain: ensures generation always succeeds if any backend works
- Face validation: optional but recommended for quality assurance
- Seed control: enables reproducible results

---

### 4. Streamlit UI (`ui/app.py`)

**Purpose:** User-friendly interface for description input and image display.

**Key Features:**
- **Parse Only**: Extract attributes without generating images
- **Generate**: Create forensic sketches with optional face validation
- **Seed Control**: Reproducible results with "New variation" button
- **Session State**: Persistent seed across reruns
- **Status Display**: Show which API keys are configured

**UI Flow:**
```
1. User enters suspect description
2. Click "Parse only" → Display extracted attributes
3. Click "Generate sketch" → Create images
4. Click "New variation" → Generate with new seed
5. Display images with metadata (seed, generation time, etc.)
```

**Design Decisions:**
- Session state for seed persistence (survives reruns)
- Spinner feedback during generation
- Face validation toggle (optional)
- Error handling with user-friendly messages

---

### 5. FastAPI Backend (`api/api.py`)

**Purpose:** REST API for programmatic access to parsing and generation.

**Endpoints:**

#### `GET /health`
Returns system status and available backends.

#### `POST /parse`
Extract attributes from description.
```json
{
  "description": "White male, early 40s, square jaw",
  "use_llm": true
}
```

#### `POST /generate`
Generate forensic sketch.
```json
{
  "description": "White male, early 40s, square jaw",
  "style": "forensic_sketch",
  "num_images": 2,
  "seed": 42,
  "use_llm": true,
  "validate_faces": true
}
```

**Design Decisions:**
- Pydantic models for request/response validation
- Proper HTTP status codes (422 for bad input, 503 for backend failure)
- Base64 encoding for image responses
- Metadata included (generation time, backends tried, etc.)

---

## Data Flow

### Parsing Flow
```
User Description
    ↓
[NLP Parser]
├─ Rule-based: Fast, no API calls
└─ LLM-based: Slower, more accurate
    ↓
Structured Attributes (JSON)
    ↓
[Validation]
├─ Check required fields
└─ Ensure no hallucinated defaults
    ↓
Validated Attributes
```

### Generation Flow
```
Validated Attributes
    ↓
[Prompt Engineer]
├─ Build positive prompt (features + style)
└─ Build negative prompt (exclusions)
    ↓
Prompts (positive + negative)
    ↓
[Generation Pipeline]
├─ Try HuggingFace
├─ Try Together AI
└─ Fall back to Pollinations
    ↓
Generated Image
    ↓
[Face Validation] (optional)
├─ Detect face with MTCNN
└─ Retry if no face found
    ↓
Final Image
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Streamlit | Web interface |
| **API** | FastAPI | REST backend |
| **NLP** | Groq/Llama-3 | LLM parsing |
| **Image Gen** | SDXL, FLUX | Diffusion models |
| **Face Detection** | facenet-pytorch (MTCNN) | Validation |
| **Testing** | pytest, Hypothesis | Quality assurance |

---

## Configuration

### Environment Variables (`.env`)
```
GROQ_API_KEY=gsk_...              # Required for LLM parsing
HF_TOKEN=hf_...                   # Optional for HuggingFace backend
TOGETHER_API_KEY=...              # Optional for Together AI backend
```

### Feature Flags
- `use_llm`: Enable LLM-based parsing (slower, more accurate)
- `validate_faces`: Enable face validation (requires facenet-pytorch)
- `num_images`: Number of images to generate (1-4)

---

## Error Handling

| Error | Cause | Resolution |
|-------|-------|-----------|
| `NameError: validate_faces` | Bug in UI generation | Fixed in v2.1 |
| `GROQ_API_KEY not set` | Missing API key | Set in `.env` |
| `Generation timeout` | Backend slow/down | Retry logic handles this |
| `No face detected` | Image doesn't contain face | Retry with different seed |
| `Abstract/landscape image` | Pollinations busy | Face validation rejects these |

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Rule-based parsing | <100ms | No API calls |
| LLM parsing | 1-3s | Groq API call |
| Image generation | 10-30s | Depends on backend |
| Face validation | 1-2s | Per image |
| **Total (UI)** | 15-40s | Parse + generate + validate |

---

## Future Improvements

1. **ControlNet Integration**: Use reference sketches for structure guidance
2. **Iterative Refinement**: Adjust one attribute at a time
3. **CUFS Fine-tuning**: LoRA training on forensic sketch dataset
4. **ArcFace Scoring**: Measure similarity to reference photos
5. **Batch Processing**: Generate multiple suspects in parallel
6. **Caching**: Cache parsed attributes and generated images

---

## Testing Strategy

### Unit Tests
- `tests/test_nlp_parser.py`: Attribute extraction
- `tests/test_prompt_engineer.py`: Prompt generation
- `tests/test_bugfix_validate_faces.py`: Bug condition verification

### Integration Tests
- `scripts/test_apis.py`: End-to-end backend smoke test

### Property-Based Tests
- `tests/test_preservation_properties.py`: Regression prevention

---

## Deployment

### Local Development
```bash
streamlit run ui/app.py
```

### Production API
```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000
```

### Docker (Future)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "ui/app.py"]
```

---

## Maintenance

### Adding a New Backend
1. Create function in `pipeline/generation_pipeline.py`
2. Add to backend selection logic
3. Update `/health` endpoint
4. Add tests in `tests/`

### Adding a New Attribute
1. Update `nlp/nlp_parser.py` extraction logic
2. Update `api/models.py` Pydantic schema
3. Update `pipeline/prompt_engineer.py` prompt building
4. Add tests

### Updating Prompts
1. Modify `pipeline/prompt_engineer.py`
2. Test with `scripts/test_apis.py`
3. Verify face validation still works

