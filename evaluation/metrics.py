"""
evaluation/metrics.py
======================
Quantitative metrics for comparing image generation models.

Metrics implemented:
  1. CLIP Score  — text-to-image alignment (higher = better prompt match)
  2. SSIM        — structural similarity between two images (0–1)
  3. Face Conf.  — MTCNN face-detection confidence (0–100%)

All metrics run on CPU. CLIP model is lazy-loaded on first call (~600 MB download).
"""

import numpy as np
from PIL import Image
from typing import Optional

# ── Lazy-loaded CLIP globals ─────────────────────────────────────────────────
_clip_model = None
_clip_processor = None
CLIP_AVAILABLE = False


def _load_clip():
    """Load CLIP model once. Returns (model, processor) or raises ImportError."""
    global _clip_model, _clip_processor, CLIP_AVAILABLE
    if _clip_model is not None:
        return _clip_model, _clip_processor

    import torch
    from transformers import CLIPModel, CLIPProcessor

    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
    CLIP_AVAILABLE = True
    return _clip_model, _clip_processor


# ─────────────────────────────────────────────────────────────────────────────
#  1. CLIP Score — how well does the image match the text prompt?
# ─────────────────────────────────────────────────────────────────────────────
def clip_score(image: Image.Image, text: str) -> float:
    """
    Compute CLIP cosine similarity between an image and a text prompt.

    Returns:
        Float score, typically 15–40. Higher means the image better
        matches the text description.  Scaled by ×100 for readability.
    """
    import torch

    model, processor = _load_clip()
    inputs = processor(
        text=[text], images=[image.convert("RGB")],
        return_tensors="pt", padding=True, truncation=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    img_emb = outputs.image_embeds
    txt_emb = outputs.text_embeds

    # L2-normalise
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    score = (img_emb @ txt_emb.T).item() * 100.0
    return round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  2. SSIM — structural similarity (pure numpy, no scikit-image needed)
# ─────────────────────────────────────────────────────────────────────────────
def ssim_score(
    image1: Image.Image,
    image2: Image.Image,
    size: int = 256,
) -> float:
    """
    Compute SSIM between two PIL images (converted to grayscale).

    Both images are resized to `size × size` for fair comparison.
    Uses the standard Wang et al. 2004 formula with default constants.

    Returns:
        Float in [-1, 1]. Typical range [0, 1] where 1 = identical.
    """
    g1 = np.array(image1.convert("L").resize((size, size)), dtype=np.float64)
    g2 = np.array(image2.convert("L").resize((size, size)), dtype=np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = _uniform_filter(g1, 11)
    mu2 = _uniform_filter(g2, 11)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _uniform_filter(g1 ** 2, 11) - mu1_sq
    sigma2_sq = _uniform_filter(g2 ** 2, 11) - mu2_sq
    sigma12 = _uniform_filter(g1 * g2, 11) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    return round(float(ssim_map.mean()), 4)


def _uniform_filter(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Simple box filter via cumulative sum (avoids scipy dependency)."""
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode="reflect")
    # Cumulative sum along rows then columns
    cs = np.cumsum(padded, axis=0)
    cs = cs[kernel_size:] - cs[:-kernel_size]
    cs = np.cumsum(cs, axis=1)
    cs = cs[:, kernel_size:] - cs[:, :-kernel_size]
    return cs / (kernel_size * kernel_size)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Face detection confidence — reuses project's MTCNN
# ─────────────────────────────────────────────────────────────────────────────
def face_confidence(image: Image.Image) -> Optional[float]:
    """
    Return MTCNN face-detection confidence (0–100) for the image.
    Returns None if facenet-pytorch is not installed.
    """
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        return None

    detector = MTCNN(keep_all=False, device="cpu", post_process=False)
    try:
        boxes, probs = detector.detect(image.convert("RGB"))
        if boxes is None or len(boxes) == 0:
            return 0.0
        return round(float(probs[0]) * 100.0, 1)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: compute all metrics at once
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(
    diffusion_img: Image.Image,
    cgan_img: Image.Image,
    text_prompt: str,
) -> dict:
    """
    Run every metric on both images.

    Returns dict with keys:
        clip_diffusion, clip_cgan,
        ssim,
        face_conf_diffusion, face_conf_cgan
    """
    results = {}

    # CLIP
    try:
        results["clip_diffusion"] = clip_score(diffusion_img, text_prompt)
        results["clip_cgan"] = clip_score(cgan_img, text_prompt)
    except Exception as e:
        print(f"[metrics] CLIP score failed: {e}")
        results["clip_diffusion"] = None
        results["clip_cgan"] = None

    # SSIM between the two outputs
    try:
        results["ssim"] = ssim_score(diffusion_img, cgan_img)
    except Exception as e:
        print(f"[metrics] SSIM failed: {e}")
        results["ssim"] = None

    # Face confidence
    results["face_conf_diffusion"] = face_confidence(diffusion_img)
    results["face_conf_cgan"] = face_confidence(cgan_img)

    return results
