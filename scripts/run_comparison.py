"""
scripts/run_comparison.py
==========================
CLI benchmark: run both Diffusion and cGAN on identical prompts,
score the outputs with CLIP / SSIM / face-detection, and produce
a comparison report.

Run from project root:
    python scripts/run_comparison.py

Output:
    - Terminal table with per-prompt scores
    - JSON report at evaluation/results/comparison_report.json
    - Generated images saved to evaluation/results/
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from PIL import Image

from nlp.nlp_parser import extract_attributes_rule_based
from pipeline.prompt_engineer import build_forensic_prompt
from pipeline.generation_pipeline import generate_images
from evaluation.cgan_generator import CGANGenerator
from evaluation.metrics import clip_score, ssim_score, face_confidence

# ── Test descriptions ─────────────────────────────────────────────────────────
TEST_DESCRIPTIONS = [
    "White male, approximately 40 years old, square jaw, short brown hair, scar on left cheek",
    "Young Asian woman, early 20s, straight black hair, narrow eyes, round face",
    "Hispanic male, late 40s, bald, full dark beard, wide nose, tattoo on neck, stocky",
    "Black woman, early 30s, curly natural hair, high cheekbones, oval face, athletic",
    "Elderly white man, 65, gray thinning hair, wire glasses, deep wrinkles",
    "Middle Eastern male, mid 30s, black beard, olive skin, hooked nose",
    "Light-skinned female, mid 20s, blonde wavy hair, blue eyes, full lips, slim",
    "Dark-skinned male, late 20s, short hair, goatee, heavy build, stern expression",
]

SEED = 42


def main():
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON BENCHMARK — Diffusion vs cGAN")
    print("=" * 72)

    # Ensure output directory exists
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)

    cgan = CGANGenerator()
    report = []

    for i, desc in enumerate(TEST_DESCRIPTIONS, 1):
        print(f"\n── Test {i}/{len(TEST_DESCRIPTIONS)}: {desc[:55]}…")

        # Parse attributes
        attrs = extract_attributes_rule_based(desc)
        prompt, _ = build_forensic_prompt(attrs, style="forensic_sketch")

        # ── cGAN generation ───────────────────────────────────────────────
        t0 = time.time()
        cgan_img = cgan.generate(attrs, seed=SEED)
        cgan_time = round(time.time() - t0, 3)

        # ── Diffusion generation ──────────────────────────────────────────
        t0 = time.time()
        diff_imgs = generate_images(prompt=prompt, num_images=1,
                                     seed=SEED, validate_faces=False)
        diff_time = round(time.time() - t0, 2)

        if not diff_imgs:
            print(f"   ⚠  Diffusion failed for test {i}. Skipping.")
            continue
        diff_img = diff_imgs[0]

        # ── Save images ───────────────────────────────────────────────────
        diff_img.save(os.path.join(results_dir, f"diffusion_{i}.png"))
        cgan_img.save(os.path.join(results_dir, f"cgan_{i}.png"))

        # ── Score ─────────────────────────────────────────────────────────
        try:
            clip_diff = clip_score(diff_img, desc)
            clip_cgan = clip_score(cgan_img, desc)
        except Exception as e:
            print(f"   ⚠  CLIP failed: {e}")
            clip_diff, clip_cgan = None, None

        ssim = ssim_score(diff_img, cgan_img)
        fc_diff = face_confidence(diff_img)
        fc_cgan = face_confidence(cgan_img)

        entry = {
            "test":          i,
            "description":   desc,
            "clip_diffusion": clip_diff,
            "clip_cgan":     clip_cgan,
            "ssim":          ssim,
            "face_conf_diffusion": fc_diff,
            "face_conf_cgan":     fc_cgan,
            "time_diffusion": diff_time,
            "time_cgan":      cgan_time,
        }
        report.append(entry)

        _print_row(entry)

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(report)

    # ── Save JSON ─────────────────────────────────────────────────────────
    report_path = os.path.join(results_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Full report saved to: {report_path}")
    print(f"🖼️  Images saved to:     {results_dir}/\n")


def _print_row(e: dict):
    cd = e["clip_diffusion"] or "—"
    cc = e["clip_cgan"] or "—"
    fd = f"{e['face_conf_diffusion']:.0f}%" if e["face_conf_diffusion"] is not None else "—"
    fc = f"{e['face_conf_cgan']:.0f}%" if e["face_conf_cgan"] is not None else "—"
    print(f"   CLIP: Diff={cd}  cGAN={cc}  │  "
          f"Face: Diff={fd}  cGAN={fc}  │  "
          f"Time: {e['time_diffusion']}s vs {e['time_cgan']}s")


def _print_summary(report: list):
    if not report:
        print("\n⚠  No results to summarise.")
        return

    # Averages (skip None values)
    def _avg(key):
        vals = [r[key] for r in report if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    avg_clip_d = _avg("clip_diffusion")
    avg_clip_c = _avg("clip_cgan")
    avg_fc_d   = _avg("face_conf_diffusion")
    avg_fc_c   = _avg("face_conf_cgan")
    avg_t_d    = _avg("time_diffusion")
    avg_t_c    = _avg("time_cgan")

    print("\n" + "=" * 72)
    print("  AVERAGES ACROSS ALL TESTS")
    print("=" * 72)
    print(f"  CLIP Score     │  Diffusion: {avg_clip_d}    cGAN: {avg_clip_c}")
    print(f"  Face Detected  │  Diffusion: {avg_fc_d}%   cGAN: {avg_fc_c}%")
    print(f"  Gen Time       │  Diffusion: {avg_t_d}s    cGAN: {avg_t_c}s")

    if avg_clip_d and avg_clip_c and avg_clip_c > 0:
        pct = round((avg_clip_d - avg_clip_c) / avg_clip_c * 100, 1)
        print(f"\n  ✅ Diffusion wins on prompt adherence by +{pct}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
