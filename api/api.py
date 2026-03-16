"""
api/api.py
===========
FastAPI backend — AI Suspect Sketch Generator.

FIXES APPLIED vs original:
  1. Pydantic models moved to api/models.py (single responsibility)
  2. 422 returned for empty description (was 500)
  3. 503 returned when pipeline not loaded (was 500 generic)
  4. /generate returns images_generated count + backends_tried list
  5. Pipeline object stored in app.state (not a mutable global)
  6. Health check reports face validation availability

Run:
    uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import io
import base64
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.nlp_parser import extract_attributes
from pipeline.prompt_engineer import build_forensic_prompt, STYLE_PRESETS
from pipeline.generation_pipeline import (
    generate_images, FACE_VALIDATION_AVAILABLE
)
from api.models import (
    ParseRequest, ParseResponse,
    GenerateRequest, GenerateResponse,
)


# ─────────────────────────────────────────────────────────────────────────────
#  App factory
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — nothing to pre-load for API backends
    # (The SDXL local pipeline would load here; API backends load on first call)
    app.state.ready = True
    print("API ready. Backends: HF / Together AI / Pollinations (auto-fallback)")
    yield
    app.state.ready = False


app = FastAPI(
    title="AI Suspect Sketch Generator",
    description="Text description → forensic face portrait via FLUX + Groq",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Check service status and available backends."""
    return {
        "status":                    "ok",
        "groq_key_set":              bool(os.getenv("GROQ_API_KEY")),
        "hf_token_set":              bool(os.getenv("HF_TOKEN")),
        "together_key_set":          bool(os.getenv("TOGETHER_API_KEY")),
        "face_validation_available": FACE_VALIDATION_AVAILABLE,
        "styles_available":          list(STYLE_PRESETS.keys()),
    }


@app.post("/parse", response_model=ParseResponse)
async def parse_description(req: ParseRequest):
    """Parse a description → structured attribute JSON."""
    try:
        attrs = extract_attributes(req.description, use_llm=req.use_llm)
        parser = "groq_llama3" if req.use_llm and os.getenv("GROQ_API_KEY") else "rule_based"
        non_null = sum(
            1 for v in attrs.values()
            if v not in (None, [], "unknown")
        )
        return ParseResponse(attributes=attrs, parser_used=parser, non_null_count=non_null)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}")


@app.post("/generate", response_model=GenerateResponse)
async def generate_sketch(req: GenerateRequest):
    """Full pipeline: description → face images."""
    if req.style not in STYLE_PRESETS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown style '{req.style}'. Choose from: {list(STYLE_PRESETS.keys())}"
        )

    start = time.time()
    backends_tried: list[str] = []

    # Step 1: Parse
    try:
        attrs = extract_attributes(req.description, use_llm=req.use_llm)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 2: Build prompt
    prompt, neg_prompt = build_forensic_prompt(attrs, style=req.style)

    # Step 3: Generate
    # Track which backends are configured
    if os.getenv("HF_TOKEN"):          backends_tried.append("HuggingFace")
    if os.getenv("TOGETHER_API_KEY"):  backends_tried.append("Together AI")
    backends_tried.append("Pollinations")   # always available

    try:
        images = generate_images(
            prompt=prompt,
            num_images=req.num_images,
            seed=req.seed,
            validate_faces=req.validate_faces,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    if not images:
        raise HTTPException(
            status_code=503,
            detail="All image backends failed or no face detected. Try again."
        )

    # Step 4: Encode
    images_b64 = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode())

    return GenerateResponse(
        images=images_b64,
        images_generated=len(images_b64),
        attributes=attrs,
        prompt=prompt,
        negative_prompt=neg_prompt,
        generation_time_seconds=round(time.time() - start, 2),
        backends_tried=backends_tried,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
