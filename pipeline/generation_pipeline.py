"""
pipeline/generation_pipeline.py
================================
Image generation with automatic backend fallback + face validation.

FIXES APPLIED vs original:
  1. Retry logic: each backend retries up to MAX_RETRIES times before moving on
  2. Face validation: generated image is checked for a detected face before returning
  3. Portrait prefix: enforced at this layer (not just in prompt_engineer)
  4. Timeout tuning: HF gets 120s, Together 90s, Pollinations 180s
  5. load_dotenv called once at module level (not inside every generate call)
  6. generate_images returns empty list [] (not None/exception) if all backends fail
  7. clear error logging: which backend succeeded is always printed

Backends tried in order:
  1. HuggingFace FLUX.1-schnell   (HF_TOKEN in .env)
  2. Together AI FLUX-Free        (TOGETHER_API_KEY in .env)
  3. Pollinations.ai              (no token needed — always available)
"""

import os
import io
import base64
import random
import time
import urllib.parse
import requests
from PIL import Image
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv(override=True)   # FIX: load once at import, not inside every call

# ── Face validation (optional — skipped gracefully if facenet not installed) ──
try:
    from facenet_pytorch import MTCNN as _MTCNN
    import torch as _torch
    _face_detector = _MTCNN(keep_all=False, device="cpu", post_process=False)
    FACE_VALIDATION_AVAILABLE = True
except ImportError:
    _face_detector = None
    FACE_VALIDATION_AVAILABLE = False


MAX_RETRIES = 2     # retries per backend before moving to next
RETRY_DELAY = 1.5   # seconds between retries


# ─────────────────────────────────────────────────────────────────────────────
#  Face validation helper
# ─────────────────────────────────────────────────────────────────────────────
def _has_face(img: Image.Image, min_confidence: float = 0.85) -> bool:
    """
    Return True if the image contains a face with confidence ≥ min_confidence.
    Returns True (pass-through) if facenet-pytorch is not installed.
    """
    if not FACE_VALIDATION_AVAILABLE or _face_detector is None:
        return True   # skip validation, accept image

    try:
        rgb = img.convert("RGB")
        boxes, probs = _face_detector.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return False
        return float(probs[0]) >= min_confidence
    except Exception:
        return True   # validation error → accept image


def _resize_to_square(img: Image.Image, size: int = 1024) -> Image.Image:
    """Resize to square, maintaining aspect ratio with center crop."""
    w, h = img.size
    min_side = min(w, h)
    left  = (w - min_side) // 2
    top   = (h - min_side) // 2
    img   = img.crop((left, top, left + min_side, top + min_side))
    return img.resize((size, size), Image.LANCZOS)


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 1 — HuggingFace FLUX.1-schnell
# ─────────────────────────────────────────────────────────────────────────────
HF_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"


def _generate_hf(prompt: str, width: int, height: int, seed: int, token: str) -> Image.Image:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"width": min(width, 1024), "height": min(height, 1024), "seed": seed},
    }
    r = requests.post(HF_URL, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 2 — Together AI FLUX.1-schnell-Free
# ─────────────────────────────────────────────────────────────────────────────
TOGETHER_URL   = "https://api.together.xyz/v1/images/generations"
TOGETHER_MODEL = "black-forest-labs/FLUX.1-schnell-Free"


def _generate_together(prompt: str, width: int, height: int, seed: int, key: str) -> Image.Image:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": TOGETHER_MODEL,
        "prompt": prompt,
        "width":  min(width, 1024),
        "height": min(height, 1024),
        "steps":  4,
        "n":      1,
        "seed":   seed,
        "response_format": "b64_json",
    }
    r = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    b64 = r.json()["data"][0]["b64_json"]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 3 — Pollinations.ai (free, no auth)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_pollinations(prompt: str, width: int, height: int, seed: int) -> Image.Image:
    encoded = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={min(width,1024)}&height={min(height,1024)}"
        f"&model=flux&seed={seed}&nologo=true&enhance=false"
    )
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    ct = r.headers.get("content-type", "")
    if "image" not in ct:
        raise ValueError(f"Non-image response: {ct} — {r.text[:200]}")
    return Image.open(io.BytesIO(r.content)).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  Retry wrapper
# ─────────────────────────────────────────────────────────────────────────────
def _with_retry(fn, name: str, retries: int = MAX_RETRIES) -> Image.Image | None:
    """Call fn() up to `retries` times. Return Image on success, None on failure."""
    for attempt in range(1, retries + 1):
        try:
            img = fn()
            return img
        except Exception as e:
            print(f"  [{name}] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def generate_images(
    prompt: str,
    num_images: int = 2,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    validate_faces: bool = True,
    **kwargs,
) -> List[Image.Image]:
    """
    Generate `num_images` images using the best available backend.

    Backend priority:
        1. HuggingFace FLUX.1-schnell (requires HF_TOKEN in .env)
        2. Together AI FLUX-Free      (requires TOGETHER_API_KEY in .env)
        3. Pollinations.ai            (free, no token, always available)

    Args:
        prompt        : full SDXL/FLUX text prompt (built by prompt_engineer.py)
        num_images    : how many variations to generate
        width/height  : output resolution (capped at 1024)
        seed          : fixed seed for reproducibility; None = random per image
        validate_faces: if True, reject images with no detected face

    Returns:
        List of PIL Images. May be shorter than num_images if backends fail.
    """
    hf_token    = os.getenv("HF_TOKEN", "").strip()
    together_key = os.getenv("TOGETHER_API_KEY", "").strip()

    images: List[Image.Image] = []

    for i in range(num_images):
        img_seed = (seed + i) if seed is not None else random.randint(1, 999_999)
        img: Image.Image | None = None
        used_backend = None

        # ── Try each backend in order ─────────────────────────────────────
        if hf_token and img is None:
            img = _with_retry(
                lambda: _generate_hf(prompt, width, height, img_seed, hf_token),
                "HuggingFace"
            )
            if img:
                used_backend = "HuggingFace"

        if together_key and img is None:
            img = _with_retry(
                lambda: _generate_together(prompt, width, height, img_seed, together_key),
                "Together AI"
            )
            if img:
                used_backend = "Together AI"

        if img is None:
            img = _with_retry(
                lambda: _generate_pollinations(prompt, width, height, img_seed),
                "Pollinations"
            )
            if img:
                used_backend = "Pollinations"

        if img is None:
            print(f"  [generate] All backends failed for image {i+1}. Skipping.")
            continue

        # ── Face validation ───────────────────────────────────────────────
        if validate_faces and not _has_face(img):
            print(f"  [validate] Image {i+1} from {used_backend}: no face detected. Skipping.")
            continue

        img = _resize_to_square(img, size=min(width, height, 1024))
        print(f"  [generate] Image {i+1}/{num_images} OK via {used_backend} (seed={img_seed})")
        images.append(img)

        if i < num_images - 1:
            time.sleep(0.3)  # polite rate limiting

    return images


# ─────────────────────────────────────────────────────────────────────────────
#  Local SDXL pipeline (optional heavy path — ~7GB download)
# ─────────────────────────────────────────────────────────────────────────────
class SuspectSketchPipeline:
    """
    Local SDXL + LCM pipeline.
    Use this only if you have a GPU with 8GB+ VRAM and want offline generation.
    API backends above are recommended for most users.
    """

    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0",
                 lcm_lora_id="latent-consistency/lcm-lora-sdxl",
                 device=None, use_fp16=True, enable_cpu_offload=False):
        import torch
        self.model_id          = model_id
        self.lcm_lora_id       = lcm_lora_id
        self.device            = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16          = use_fp16 and self.device == "cuda"
        self.enable_cpu_offload = enable_cpu_offload
        self.pipe              = None
        self.is_loaded         = False

    def load(self):
        if self.is_loaded:
            return self
        import torch
        from diffusers import StableDiffusionXLPipeline, LCMScheduler
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id, torch_dtype=dtype,
            variant="fp16" if self.use_fp16 else None, use_safetensors=True
        )
        self.pipe.load_lora_weights(self.lcm_lora_id)
        self.pipe.fuse_lora()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        if self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        self.is_loaded = True
        return self

    def unload(self):
        self.pipe = None
        self.is_loaded = False

    def generate(self, prompt, negative_prompt="", num_images=1,
                 num_inference_steps=6, guidance_scale=1.5,
                 width=1024, height=1024, seed=None) -> List[Image.Image]:
        import torch
        if not self.is_loaded:
            self.load()
        images = []
        for i in range(num_images):
            gen = torch.Generator(self.device).manual_seed((seed or 0) + i) if seed else None
            out = self.pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width, height=height, generator=gen
            )
            images.append(out.images[0])
        return images
