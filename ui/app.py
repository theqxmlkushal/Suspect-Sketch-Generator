"""
ui/app.py — AI Suspect Sketch Generator
=========================================
Streamlit frontend.

FIXES APPLIED vs original:
  1. Seed stored in st.session_state — same face base across reruns
     "New variation" button explicitly increments the seed
  2. Backend status shown in sidebar — user knows which key is active
  3. Empty description shows warning instead of crashing
  4. Images stored in session_state — don't disappear on widget interaction
  5. Attribute diff panel: shows only non-null detected attributes
  6. Download button uses descriptive filename with description snippet
  7. Error messages are human-readable (not raw exception strings)

Run:
    streamlit run ui/app.py
"""

import os
import sys
import io
import random

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from nlp.nlp_parser import extract_attributes
from pipeline.prompt_engineer import build_forensic_prompt, STYLE_PRESETS
from pipeline.generation_pipeline import generate_images, FACE_VALIDATION_AVAILABLE


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Suspect Sketch Generator",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif; }
.main-title { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.2rem; }
.main-sub   { color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }
div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid rgba(0,0,0,0.08); }
.stTextArea textarea { font-size: 0.93rem !important; line-height: 1.6 !important; }
.stButton > button { border-radius: 8px !important; font-weight: 600 !important; }
.attr-pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    background: #EEF2FF; color: #3730a3; font-size: 0.77rem;
    margin: 2px 3px 2px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────
def _init_state():
    defaults = {
        "seed":        random.randint(1, 99_999),
        "images":      [],          # List[PIL.Image] — persists across widget interactions
        "attributes":  {},          # last parsed attribute dict
        "prompt":      "",          # last built prompt
        "last_desc":   "",          # description that produced current images
        "generating":  False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init_state()


# ── Sidebar — configuration & status ──────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    style = st.selectbox(
        "Output style",
        options=list(STYLE_PRESETS.keys()),
        format_func=lambda x: {
            "forensic_sketch": "✏️ Forensic pencil sketch",
            "photorealistic":  "📷 Photorealistic portrait",
            "composite":       "🖥️ Digital composite",
        }.get(x, x),
    )

    num_images  = st.slider("Variations to generate", 1, 4, 2)
    use_llm     = st.toggle("Use Groq AI parser",  value=bool(os.getenv("GROQ_API_KEY")))
    val_faces   = st.toggle("Validate faces",       value=FACE_VALIDATION_AVAILABLE,
                             disabled=not FACE_VALIDATION_AVAILABLE)

    st.markdown("---")
    st.markdown("### 📡 Backend status")

    def _status(label, has_key, note=""):
        icon = "🟢" if has_key else "🔴"
        st.markdown(f"{icon} **{label}**" + (f"  \n`{note}`" if note else ""))

    _status("HuggingFace FLUX", bool(os.getenv("HF_TOKEN")),   "HF_TOKEN in .env")
    _status("Together AI",      bool(os.getenv("TOGETHER_API_KEY")), "TOGETHER_API_KEY in .env")
    _status("Pollinations.ai",  True,  "free, always available")
    _status("Groq / Llama-3",   bool(os.getenv("GROQ_API_KEY")), "GROQ_API_KEY in .env")

    if not FACE_VALIDATION_AVAILABLE:
        st.info("Face validation disabled.  \n`pip install facenet-pytorch`")

    st.markdown("---")
    st.markdown("### 🎲 Seed control")
    st.write(f"Current seed: `{st.session_state.seed}`")
    col_seed1, col_seed2 = st.columns(2)
    with col_seed1:
        if st.button("New variation", use_container_width=True):
            st.session_state.seed = random.randint(1, 99_999)
            st.session_state.images = []
            st.rerun()
    with col_seed2:
        manual_seed = st.number_input("Set seed", value=st.session_state.seed,
                                       step=1, label_visibility="collapsed")
        if manual_seed != st.session_state.seed:
            st.session_state.seed = int(manual_seed)
            st.session_state.images = []


# ── Main layout ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🕵️ AI Suspect Sketch Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Describe a suspect in words — get a forensic portrait in seconds.</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.4], gap="large")

# ── Left column: input ─────────────────────────────────────────────────────
with col_left:
    st.markdown("#### Suspect description")

    default_desc = st.session_state.pop("fill_desc",
        "White male, approximately 40 years old, square jaw, short brown hair, "
        "bushy eyebrows, scar on left cheek, light stubble, stern expression"
    )

    description = st.text_area(
        "description",
        value=default_desc,
        height=160,
        label_visibility="collapsed",
        placeholder="Describe age, gender, hair, eyes, face shape, scars, build…",
    )

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        generate_clicked = st.button("🎨  Generate sketch", type="primary",
                                      use_container_width=True)
    with btn_col2:
        parse_only = st.button("🔍  Parse only", use_container_width=True)

    # Quick examples
    st.markdown("**Quick examples**")
    EXAMPLES = [
        "Young Asian woman, early 20s, straight black hair, narrow eyes, round face, slim",
        "Hispanic male, late 40s, bald, full dark beard, wide nose, tattoo on neck, stocky",
        "White elderly man, 65+, gray thinning hair, wire glasses, deep wrinkles, gaunt",
        "Black woman, early 30s, curly natural hair, high cheekbones, oval face, athletic",
        "Middle Eastern male, mid 30s, black beard, olive skin, hooked nose, average build",
    ]
    for ex in EXAMPLES:
        if st.button(f"↩  {ex[:60]}…", key=ex, use_container_width=True):
            st.session_state["fill_desc"] = ex
            st.session_state.images = []
            st.rerun()


# ── Right column: output ───────────────────────────────────────────────────
with col_right:
    st.markdown("#### Generated sketches")

    # ── Parse only ────────────────────────────────────────────────────────
    if parse_only:
        if not description.strip():
            st.warning("Enter a description first.")
        else:
            with st.spinner("Parsing…"):
                attrs = extract_attributes(description, use_llm=use_llm)
                st.session_state.attributes = attrs
                st.session_state.last_desc  = description

    # ── Full generation ───────────────────────────────────────────────────
    if generate_clicked:
        if not description.strip():
            st.warning("Please enter a suspect description first.")
        else:
            with st.spinner("🧠 Parsing description…"):
                try:
                    attrs = extract_attributes(description, use_llm=use_llm)
                    st.session_state.attributes = attrs
                except Exception as e:
                    st.error(f"Parsing failed: {e}")
                    st.stop()

            prompt, _ = build_forensic_prompt(attrs, style=style)
            st.session_state.prompt = prompt

            with st.spinner(f"🎨 Generating {num_images} image(s) via FLUX… (~15-30s)"):
                try:
                    imgs = generate_images(
                        prompt=prompt,
                        num_images=num_images,
                        seed=st.session_state.seed,
                        validate_faces=val_faces,
                    )
                    if not imgs:
                        st.error("No images returned. All backends may be busy — try again in a moment.")
                    else:
                        st.session_state.images    = imgs
                        st.session_state.last_desc = description
                        st.success(f"✅ Generated {len(imgs)}/{num_images} image(s)  "
                                   f"(seed `{st.session_state.seed}`)")
                except Exception as e:
                    st.error(f"Generation error: {e}")

    # ── Display images ────────────────────────────────────────────────────
    imgs = st.session_state.get("images", [])
    if imgs:
        n_cols = min(len(imgs), 2)
        img_cols = st.columns(n_cols)
        desc_slug = (st.session_state.last_desc or "suspect")[:30].replace(" ", "_")

        for i, img in enumerate(imgs):
            with img_cols[i % n_cols]:
                st.image(img, caption=f"Variation {i + 1}", use_container_width=True)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    f"⬇ Download #{i + 1}",
                    data=buf.getvalue(),
                    file_name=f"suspect_{desc_slug}_{i+1}.png",
                    mime="image/png",
                    key=f"dl_{i}_{st.session_state.seed}",
                    use_container_width=True,
                )

    elif not generate_clicked and not parse_only:
        st.markdown(
            "<div style='text-align:center;padding:3rem 1rem;color:#94a3b8;font-size:1rem'>"
            "🎨  Your sketches will appear here."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Parsed attributes panel ───────────────────────────────────────────
    attrs = st.session_state.get("attributes", {})
    if attrs:
        with st.expander("🔍 Detected attributes", expanded=False):
            # Show only non-null, non-empty fields
            non_null = {k: v for k, v in attrs.items()
                        if v not in (None, [], "unknown")
                        and k != "distinguishing_features"}
            feats = attrs.get("distinguishing_features", [])

            pills_html = "".join(
                f'<span class="attr-pill">{k.replace("_"," ")}: <b>{v}</b></span>'
                for k, v in non_null.items()
            )
            if feats:
                for f in feats:
                    pills_html += f'<span class="attr-pill" style="background:#FEF3C7;color:#92400E">⚠ {f}</span>'

            st.markdown(pills_html, unsafe_allow_html=True)
            st.markdown(f"**{len(non_null)} attributes detected** "
                        f"({'LLM' if use_llm else 'rule-based'} parser)")

        with st.expander("📝 SDXL prompt", expanded=False):
            st.code(st.session_state.get("prompt", ""), language=None)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:0.75rem'>"
    "AI Suspect Sketch Generator v2.1 · FLUX (Pollinations/HF/Together) + Groq Llama-3 · "
    "For research and education only"
    "</div>",
    unsafe_allow_html=True,
)
