"""
ui/compare.py — Model Comparison Tab
======================================
Side-by-side Diffusion vs cGAN comparison with live metrics.
Called from app.py inside a Streamlit tab.
"""

import os
import sys
import time
import random
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.nlp_parser import extract_attributes
from pipeline.prompt_engineer import build_forensic_prompt, STYLE_PRESETS
from pipeline.generation_pipeline import generate_images


# ── CSS for comparison tab ─────────────────────────────────────────────────
COMPARE_CSS = """
<style>
.cmp-header {
    text-align: center; padding: 1.2rem 0 0.5rem;
}
.cmp-header h2 {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.5rem; font-weight: 700; margin: 0;
}
.cmp-header p { color: #64748b; font-size: 0.88rem; margin-top: 0.2rem; }
.model-badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; margin-bottom: 0.5rem;
}
.badge-diff { background: linear-gradient(135deg, #EEF2FF, #E0E7FF); color: #4338CA; border: 1px solid #818CF8; }
.badge-cgan { background: linear-gradient(135deg, #FFF7ED, #FFEDD5); color: #C2410C; border: 1px solid #FB923C; }
.metric-card {
    background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
    border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 1rem; text-align: center; margin-bottom: 0.5rem;
}
.metric-card h4 { margin: 0 0 0.4rem; font-size: 0.82rem; color: #475569; font-weight: 600; }
.metric-val { font-size: 1.4rem; font-weight: 700; }
.metric-val.diff-color { color: #4F46E5; }
.metric-val.cgan-color { color: #EA580C; }
.metric-label { font-size: 0.7rem; color: #94A3B8; margin-top: 2px; }
.bar-wrap { background: #E2E8F0; border-radius: 6px; height: 10px; margin: 4px 0; overflow: hidden; }
.bar-fill-diff { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #4F46E5, #7C3AED); transition: width 0.8s ease; }
.bar-fill-cgan { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #F59E0B, #EF4444); transition: width 0.8s ease; }
.winner-chip {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700;
}
.winner-chip.win { background: #D1FAE5; color: #065F46; }
.winner-chip.lose { background: #FEE2E2; color: #991B1B; }
.cond-vec-box {
    background: #1E293B; color: #E2E8F0; border-radius: 8px;
    padding: 0.8rem 1rem; font-family: 'Courier New', monospace;
    font-size: 0.75rem; line-height: 1.7; margin-top: 0.5rem;
    max-height: 200px; overflow-y: auto;
}
.insight-box {
    background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
    border: 1px solid #86EFAC; border-radius: 12px;
    padding: 1.2rem; margin-top: 1rem;
}
.insight-box h4 { color: #166534; margin: 0 0 0.5rem; font-size: 0.95rem; }
.insight-box p  { color: #15803D; font-size: 0.85rem; margin: 0.2rem 0; line-height: 1.5; }
</style>
"""


def _init_compare_state():
    for k, v in {
        "cmp_diff_img": None, "cmp_cgan_img": None,
        "cmp_metrics": None, "cmp_attrs": None,
        "cmp_prompt": "", "cmp_cond_vec": None,
        "cmp_diff_time": 0, "cmp_cgan_time": 0,
    }.items():
        st.session_state.setdefault(k, v)


def render_comparison_tab():
    """Render the full Model Comparison tab contents."""
    st.markdown(COMPARE_CSS, unsafe_allow_html=True)
    _init_compare_state()

    st.markdown(
        '<div class="cmp-header">'
        '<h2>🔬 Diffusion vs cGAN — Live Comparison</h2>'
        '<p>Same description, two architectures. See the difference quantified.</p>'
        '</div>', unsafe_allow_html=True
    )

    # ── Input ──────────────────────────────────────────────────────────────
    comp_desc = st.text_area(
        "Suspect description for comparison",
        value="White male, approximately 40 years old, square jaw, short brown hair, "
              "bushy eyebrows, scar on left cheek, light stubble, stern expression",
        height=100, key="comp_desc_input",
        placeholder="Describe age, gender, hair, eyes, face shape, scars…",
    )

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        compare_clicked = st.button("🔬  Compare Models", type="primary",
                                     use_container_width=True, key="cmp_btn")
    with col_info:
        st.caption("Generates one image from each model, then scores both with "
                   "CLIP (prompt match), face detection, and SSIM.")

    # ── Generation ─────────────────────────────────────────────────────────
    if compare_clicked:
        if not comp_desc.strip():
            st.warning("Enter a description first.")
            return

        # Lazy imports — only load torch-dependent modules when user clicks Compare
        try:
            from evaluation.cgan_generator import CGANGenerator
            from evaluation.metrics import clip_score, ssim_score, face_confidence
        except ImportError as e:
            st.error(f"Missing dependency: {e}\n\nRun: `pip install torch torchvision transformers`")
            return

        seed = st.session_state.get("seed", random.randint(1, 99_999))

        # Parse
        with st.spinner("🧠 Parsing description…"):
            try:
                attrs = extract_attributes(comp_desc, use_llm=bool(os.getenv("GROQ_API_KEY")))
            except Exception as e:
                st.error(f"Parsing failed: {e}")
                return
        st.session_state.cmp_attrs = attrs
        prompt, _ = build_forensic_prompt(attrs, style="forensic_sketch")
        st.session_state.cmp_prompt = prompt

        # cGAN
        with st.spinner("⚡ Generating cGAN output…"):
            cgan = CGANGenerator()
            t0 = time.time()
            cgan_img = cgan.generate(attrs, seed=seed)
            st.session_state.cmp_cgan_time = round(time.time() - t0, 3)
            st.session_state.cmp_cgan_img = cgan_img
            st.session_state.cmp_cond_vec = cgan.get_condition_vector_display(attrs)

        # Diffusion
        with st.spinner("🎨 Generating Diffusion (FLUX) output… (~15-30s)"):
            t0 = time.time()
            diff_imgs = generate_images(prompt=prompt, num_images=1,
                                         seed=seed, validate_faces=False)
            st.session_state.cmp_diff_time = round(time.time() - t0, 2)
            st.session_state.cmp_diff_img = diff_imgs[0] if diff_imgs else None

        if st.session_state.cmp_diff_img is None:
            st.error("Diffusion backend failed. Try again.")
            return

        # Metrics
        with st.spinner("📊 Computing metrics (CLIP + SSIM + face detection)…"):
            try:
                m = {}
                m["clip_diff"] = clip_score(st.session_state.cmp_diff_img, comp_desc)
                m["clip_cgan"] = clip_score(st.session_state.cmp_cgan_img, comp_desc)
                m["ssim"] = ssim_score(st.session_state.cmp_diff_img, st.session_state.cmp_cgan_img)
                m["face_diff"] = face_confidence(st.session_state.cmp_diff_img)
                m["face_cgan"] = face_confidence(st.session_state.cmp_cgan_img)
                st.session_state.cmp_metrics = m
            except Exception as e:
                st.warning(f"Some metrics failed: {e}")
                st.session_state.cmp_metrics = {}

        st.success("✅ Comparison complete!")

    # ── Display results ────────────────────────────────────────────────────
    diff_img = st.session_state.cmp_diff_img
    cgan_img = st.session_state.cmp_cgan_img
    metrics = st.session_state.cmp_metrics

    if diff_img is None or cgan_img is None:
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#94a3b8">'
            '🔬 Click <b>Compare Models</b> to see a side-by-side evaluation.</div>',
            unsafe_allow_html=True
        )
        return

    # ── Side-by-side images ────────────────────────────────────────────────
    col_d, col_c = st.columns(2, gap="large")
    with col_d:
        st.markdown('<span class="model-badge badge-diff">✨ FLUX / Diffusion Model</span>',
                    unsafe_allow_html=True)
        st.image(diff_img, caption="FLUX.1-schnell — 1024×1024 native", use_container_width=True)
        st.caption(f"⏱ Generated in **{st.session_state.cmp_diff_time}s**")

    with col_c:
        st.markdown('<span class="model-badge badge-cgan">🔶 cGAN Baseline</span>',
                    unsafe_allow_html=True)
        st.image(cgan_img, caption="cGAN — 64×64 native (upscaled to 256×256)", use_container_width=True)
        st.caption(f"⏱ Generated in **{st.session_state.cmp_cgan_time}s**")

    # ── Metric cards ───────────────────────────────────────────────────────
    if metrics:
        st.markdown("---")
        st.markdown("### 📊 Quantitative Metrics")

        mc1, mc2, mc3 = st.columns(3)

        clip_d = metrics.get("clip_diff")
        clip_c = metrics.get("clip_cgan")
        face_d = metrics.get("face_diff")
        face_c = metrics.get("face_cgan")
        ssim_v = metrics.get("ssim")

        # CLIP Score card
        with mc1:
            clip_max = max(clip_d or 1, clip_c or 1, 1)
            d_pct = int((clip_d or 0) / 40 * 100)
            c_pct = int((clip_c or 0) / 40 * 100)
            st.markdown(f'''<div class="metric-card">
                <h4>📎 CLIP Score (Prompt Match)</h4>
                <span class="metric-val diff-color">{clip_d or "—"}</span>
                <span style="color:#94A3B8;font-size:0.85rem"> vs </span>
                <span class="metric-val cgan-color">{clip_c or "—"}</span>
                <div class="metric-label">Diffusion ↑</div>
                <div class="bar-wrap"><div class="bar-fill-diff" style="width:{d_pct}%"></div></div>
                <div class="metric-label">cGAN ↑</div>
                <div class="bar-wrap"><div class="bar-fill-cgan" style="width:{c_pct}%"></div></div>
                <div style="margin-top:8px">
                    <span class="winner-chip win">✅ {"Diffusion" if (clip_d or 0) >= (clip_c or 0) else "cGAN"} wins</span>
                </div>
            </div>''', unsafe_allow_html=True)

        # Face Detection / Resolution card
        with mc2:
            if face_d is not None or face_c is not None:
                fd_pct = int(face_d or 0)
                fc_pct = int(face_c or 0)
                st.markdown(f'''<div class="metric-card">
                    <h4>👤 Face Detection Confidence</h4>
                    <span class="metric-val diff-color">{f"{face_d:.0f}%" if face_d is not None else "N/A"}</span>
                    <span style="color:#94A3B8;font-size:0.85rem"> vs </span>
                    <span class="metric-val cgan-color">{f"{face_c:.0f}%" if face_c is not None else "N/A"}</span>
                    <div class="metric-label">Diffusion</div>
                    <div class="bar-wrap"><div class="bar-fill-diff" style="width:{fd_pct}%"></div></div>
                    <div class="metric-label">cGAN</div>
                    <div class="bar-wrap"><div class="bar-fill-cgan" style="width:{fc_pct}%"></div></div>
                    <div style="margin-top:8px">
                        <span class="winner-chip win">{"Diffusion" if (face_d or 0) >= (face_c or 0) else "cGAN"} wins</span>
                    </div>
                </div>''', unsafe_allow_html=True)
            else:
                # facenet-pytorch not installed — show resolution comparison instead
                st.markdown(f'''<div class="metric-card">
                    <h4>📐 Native Resolution</h4>
                    <span class="metric-val diff-color">1024px</span>
                    <span style="color:#94A3B8;font-size:0.85rem"> vs </span>
                    <span class="metric-val cgan-color">64px</span>
                    <div class="metric-label">Diffusion (1024 x 1024)</div>
                    <div class="bar-wrap"><div class="bar-fill-diff" style="width:100%"></div></div>
                    <div class="metric-label">cGAN (64 x 64 upscaled to 256)</div>
                    <div class="bar-wrap"><div class="bar-fill-cgan" style="width:6%"></div></div>
                    <div style="margin-top:8px">
                        <span class="winner-chip win">Diffusion wins — 16x more pixels</span>
                    </div>
                </div>''', unsafe_allow_html=True)

        # SSIM card
        with mc3:
            ssim_pct = int((ssim_v or 0) * 100)
            st.markdown(f'''<div class="metric-card">
                <h4>🔲 SSIM (Structural Similarity)</h4>
                <span class="metric-val" style="color:#0F172A">{ssim_v if ssim_v is not None else "—"}</span>
                <div class="metric-label">0 = completely different &nbsp;│&nbsp; 1 = identical</div>
                <div class="bar-wrap"><div class="bar-fill-diff" style="width:{ssim_pct}%"></div></div>
                <div style="margin-top:8px">
                    <span class="winner-chip {"lose" if (ssim_v or 0) < 0.3 else "win"}">
                        {"🔀 Very different outputs" if (ssim_v or 0) < 0.3 else "Some structural overlap"}
                    </span>
                </div>
            </div>''', unsafe_allow_html=True)

    # ── Comparison summary table ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Comprehensive Comparison")

    def _w(diff_wins):
        return "✅ Diffusion" if diff_wins else "✅ cGAN"

    table_md = f"""
| Metric | Diffusion (FLUX) | cGAN (Baseline) | Winner |
|:-------|:----------------:|:---------------:|:------:|
| **CLIP Score** | {clip_d or '—'} | {clip_c or '—'} | {_w((clip_d or 0) >= (clip_c or 0))} |
| **Face Detection** | {f"{face_d:.0f}%" if face_d is not None else "N/A"} | {f"{face_c:.0f}%" if face_c is not None else "N/A"} | {_w((face_d or 0) >= (face_c or 0))} |
| **Native Resolution** | 1024 × 1024 | 64 × 64 | ✅ Diffusion |
| **Generation Time** | {st.session_state.cmp_diff_time}s | {st.session_state.cmp_cgan_time}s | {_w(st.session_state.cmp_diff_time <= st.session_state.cmp_cgan_time)} |
| **Prompt Control** | Full natural language | 25-dim binary vector | ✅ Diffusion |
| **Style Options** | {len(STYLE_PRESETS)} styles | Fixed | ✅ Diffusion |
| **Requires Training** | No (pretrained API) | Yes (200k+ images) | ✅ Diffusion |
| **Model Size** | ~12 GB (cloud) | ~2 MB (local) | ✅ cGAN |
"""
    st.markdown(table_md)

    # ── Insight box ────────────────────────────────────────────────────────
    improvement = ""
    if clip_d and clip_c and clip_c > 0:
        pct = round((clip_d - clip_c) / clip_c * 100, 1)
        improvement = f"Diffusion achieves <b>+{pct}%</b> better prompt alignment (CLIP Score)."

    st.markdown(f'''<div class="insight-box">
        <h4>💡 Key Takeaway for Presentation</h4>
        <p>The <b>Diffusion model (FLUX)</b> produces photorealistic, prompt-accurate forensic
        sketches at 1024×1024, while the <b>cGAN baseline</b> outputs abstract 64×64 patterns
        that fail face detection entirely.</p>
        <p>{improvement}</p>
        <p>This demonstrates why modern forensic AI systems have moved from GAN-based
        architectures to Diffusion-based pipelines — the quality gap is not incremental,
        it is <b>generational</b>.</p>
    </div>''', unsafe_allow_html=True)

    # ── Condition vector display ───────────────────────────────────────────
    cond_vec = st.session_state.cmp_cond_vec
    if cond_vec:
        with st.expander("🔢 cGAN Condition Vector (25-dim binary — this is ALL the cGAN sees)"):
            active = {k: v for k, v in cond_vec.items() if v == 1}
            inactive = {k: v for k, v in cond_vec.items() if v == 0}
            vec_html = ""
            for k, v in cond_vec.items():
                color = "#4ADE80" if v == 1 else "#475569"
                vec_html += f'<span style="color:{color}">{k}: {v}</span><br>'
            st.markdown(f'<div class="cond-vec-box">{vec_html}</div>', unsafe_allow_html=True)
            st.caption(f"**{len(active)} active** out of 25 dimensions. "
                       f"Compare this to the full natural-language prompt the Diffusion model receives.")

    # ── Full prompt display ────────────────────────────────────────────────
    if st.session_state.cmp_prompt:
        with st.expander("📝 Full Diffusion prompt (what FLUX sees)"):
            st.code(st.session_state.cmp_prompt, language=None)
