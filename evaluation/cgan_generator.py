"""
evaluation/cgan_generator.py
==============================
Lightweight Conditional GAN (cGAN) baseline generator.

Architecture:
    Input : 100-dim noise vector + 25-dim condition vector  →  125-dim
    Layers: 5 transposed-conv blocks with BatchNorm + ReLU
    Output: 64×64 RGB image  (upscaled to 256×256 for display)

This generator uses Kaiming-initialised weights (no pre-training).
The output will NOT look like a realistic face — that is the entire
point of the comparison.  It demonstrates that:

  - GANs require extensive training (200k+ images, hours of GPU time)
  - Even with training, 64x64 output is blurry compared to Diffusion 1024x1024
  - Binary condition vectors lose the nuance of natural-language prompts

ALL heavy imports (torch, torch.nn) are lazy-loaded so the app starts
instantly even if PyTorch is not installed.
"""

import numpy as np
from PIL import Image
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Condition vector: parsed attributes → 25-dim binary tensor
# ─────────────────────────────────────────────────────────────────────────────
CONDITION_MAPPING = [
    ("gender",     "male"),
    ("gender",     "female"),
    ("_age_young",  None),
    ("_age_mid",    None),
    ("_age_old",    None),
    ("hair_color", "black"),
    ("hair_color", "brown"),
    ("hair_color", "blonde"),
    ("hair_color", "gray"),
    ("hair_style", "bald"),
    ("hair_style", "short"),
    ("eye_color",  "brown"),
    ("eye_color",  "blue"),
    ("eye_color",  "green"),
    ("jaw_shape",  "square"),
    ("jaw_shape",  "round"),
    ("face_shape", "oval"),
    ("facial_hair", "_any"),
    ("glasses",     "_any"),
    ("_has_marks",  None),
    ("expression",  "stern"),
    ("expression",  "smiling"),
    ("skin_tone",  "light"),
    ("skin_tone",  "medium"),
    ("skin_tone",  "dark"),
]

NOISE_DIM = 100
COND_DIM = len(CONDITION_MAPPING)   # 25
FEATURE_MAPS = 64
NATIVE_SIZE = 64
DISPLAY_SIZE = 256


def attrs_to_condition(attrs: dict):
    """Convert a parsed attribute dict to a 25-dim binary tensor."""
    import torch

    vec = []
    age = attrs.get("age") or 35

    for field, match_val in CONDITION_MAPPING:
        if field == "_age_young":
            vec.append(1.0 if age < 35 else 0.0)
        elif field == "_age_mid":
            vec.append(1.0 if 35 <= age < 55 else 0.0)
        elif field == "_age_old":
            vec.append(1.0 if age >= 55 else 0.0)
        elif field == "_has_marks":
            feats = attrs.get("distinguishing_features") or []
            vec.append(1.0 if len(feats) > 0 else 0.0)
        elif match_val == "_any":
            val = attrs.get(field)
            vec.append(1.0 if val and val != "none" else 0.0)
        else:
            val = attrs.get(field)
            vec.append(1.0 if val == match_val else 0.0)

    return torch.tensor(vec, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Generator network — built lazily to avoid top-level torch import
# ─────────────────────────────────────────────────────────────────────────────
def _build_generator():
    """Construct and return a Kaiming-initialised DCGAN generator."""
    import torch
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            in_ch = NOISE_DIM + COND_DIM

            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_ch, FEATURE_MAPS * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(FEATURE_MAPS * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(FEATURE_MAPS * 8, FEATURE_MAPS * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(FEATURE_MAPS * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(FEATURE_MAPS * 4, FEATURE_MAPS * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(FEATURE_MAPS * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(FEATURE_MAPS * 2, FEATURE_MAPS, 4, 2, 1, bias=False),
                nn.BatchNorm2d(FEATURE_MAPS),
                nn.ReLU(True),

                nn.ConvTranspose2d(FEATURE_MAPS, 3, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

        def forward(self, noise, condition):
            x = torch.cat([noise, condition], dim=1)
            x = x.unsqueeze(-1).unsqueeze(-1)
            return self.main(x)

    net = _Net()
    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────────────────
#  Public interface
# ─────────────────────────────────────────────────────────────────────────────
class CGANGenerator:
    """
    Conditional GAN face generator for model-comparison demos.

    Usage:
        gen = CGANGenerator()
        img = gen.generate(attrs_dict, seed=42)   # → PIL Image 256×256
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self._net = None
        self._checkpoint_path = checkpoint_path

    def _ensure_loaded(self):
        """Lazy-load the network on first generate() call."""
        if self._net is not None:
            return
        import torch

        self._net = _build_generator()

        if self._checkpoint_path:
            try:
                state = torch.load(self._checkpoint_path, map_location="cpu",
                                   weights_only=True)
                self._net.load_state_dict(state)
                print(f"[cGAN] Loaded checkpoint: {self._checkpoint_path}")
            except Exception as e:
                print(f"[cGAN] Could not load checkpoint ({e}), using init weights")

    def generate(
        self,
        attrs: dict,
        seed: int = 42,
        upscale: int = DISPLAY_SIZE,
    ) -> Image.Image:
        """
        Generate a face image from parsed attributes.

        Returns:
            PIL.Image.Image (RGB, upscale × upscale)
        """
        import torch

        self._ensure_loaded()
        torch.manual_seed(seed)

        noise = torch.randn(1, NOISE_DIM)
        cond = attrs_to_condition(attrs).unsqueeze(0)

        with torch.no_grad():
            out = self._net(noise, cond)

        img_np = out.squeeze(0).permute(1, 2, 0).numpy()
        img_np = ((img_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np, "RGB")

        if upscale and upscale != NATIVE_SIZE:
            img = img.resize((upscale, upscale), Image.BILINEAR)

        return img

    def get_condition_vector_display(self, attrs: dict) -> dict:
        """Return a human-readable dict of the condition vector."""
        import torch

        cond = attrs_to_condition(attrs)
        labels = []
        for field, val in CONDITION_MAPPING:
            if field.startswith("_"):
                label = field.lstrip("_")
            elif val == "_any":
                label = f"has_{field}"
            else:
                label = f"{field}={val}"
            labels.append(label)

        return {labels[i]: int(cond[i].item()) for i in range(len(labels))}
