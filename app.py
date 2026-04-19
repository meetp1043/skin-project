# =============================================================================
# STREAMLIT WEB APP — Skin Disease Detection
# =============================================================================
# Run: streamlit run app.py
# =============================================================================

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import cv2
import time
from utils.gradcam import get_gradcam
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="🩺 Skin Disease Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Lazy TF import (avoids slow startup message) ─────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    MODEL_PATH = "outputs/models/best_model.h5"
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# =============================================================================
# CONSTANTS
# =============================================================================

IMG_SIZE = 224

DISEASES = {
    0: {
        "name"       : "Melanoma",
        "code"       : "mel",
        "severity"   : "High",
        "color"      : "#E74C3C",
        "description": (
            "A serious form of skin cancer that begins in cells known as melanocytes. "
            "While it is less common than basal cell or squamous cell carcinoma, "
            "melanoma is more dangerous because it's much more likely to spread."
        ),
        "advice"     : "Seek immediate dermatologist consultation. Early detection is critical."
    },
    1: {
        "name"       : "Melanocytic Nevi",
        "code"       : "nv",
        "severity"   : "Low",
        "color"      : "#3498DB",
        "description": (
            "Commonly known as moles, melanocytic nevi are benign proliferations of "
            "melanocytes. Most are harmless; however, atypical moles may require monitoring."
        ),
        "advice"     : "Generally benign. Monitor for changes in size, shape, or color."
    },
    2: {
        "name"       : "Basal Cell Carcinoma",
        "code"       : "bcc",
        "severity"   : "High",
        "color"      : "#E67E22",
        "description": (
            "The most common type of skin cancer. Basal cell carcinoma develops in basal "
            "cells and rarely spreads to other parts of the body. Still requires prompt treatment."
        ),
        "advice"     : "Consult a dermatologist for excision or topical therapy. Avoid prolonged sun exposure."
    },
    3: {
        "name"       : "Actinic Keratosis",
        "code"       : "akiec",
        "severity"   : "Medium",
        "color"      : "#9B59B6",
        "description": (
            "A pre-cancerous lesion caused by UV damage. Rough, scaly patches on sun-damaged skin. "
            "About 5–10% of actinic keratoses can develop into squamous cell carcinoma."
        ),
        "advice"     : "Treat promptly to prevent progression. Cryotherapy or topical creams recommended."
    },
    4: {
        "name"       : "Benign Keratosis",
        "code"       : "bkl",
        "severity"   : "Low",
        "color"      : "#27AE60",
        "description": (
            "Also known as seborrheic keratosis. A common non-cancerous skin growth that often "
            "appears as a waxy, scaly, slightly raised growth. Purely cosmetic concern."
        ),
        "advice"     : "No treatment necessary unless irritated. Monitor for changes."
    },
    5: {
        "name"       : "Dermatofibroma",
        "code"       : "df",
        "severity"   : "Low",
        "color"      : "#F39C12",
        "description": (
            "A common benign fibrous nodule of the skin. Typically small, firm bumps "
            "that appear on the legs. They are harmless and usually asymptomatic."
        ),
        "advice"     : "Usually harmless. Surgical removal if causing discomfort."
    },
    6: {
        "name"       : "Vascular Lesions",
        "code"       : "vasc",
        "severity"   : "Medium",
        "color"      : "#1ABC9C",
        "description": (
            "Include cherry angiomas, angiokeratomas, pyogenic granulomas, and hemorrhage. "
            "Most vascular lesions are benign but some may require treatment."
        ),
        "advice"     : "Consult a dermatologist for accurate diagnosis. Laser therapy may be effective."
    }
}

SEVERITY_BADGE = {
    "High"  : ("🔴", "#E74C3C"),
    "Medium": ("🟡", "#F39C12"),
    "Low"   : ("🟢", "#27AE60"),
}

# =============================================================================
# CSS STYLING
# =============================================================================

def inject_css():
    st.markdown("""
    <style>
        /* Main background */
        .stApp { background: #0F1117; color: #ECEFF4; }

        /* Header */
        .hero-header {
            background: linear-gradient(135deg, #1A1D27 0%, #1E2235 100%);
            border: 1px solid #2D3248;
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        .hero-header h1 { font-size: 2.4rem; color: #E2E8F0; margin: 0 0 0.4rem 0; }
        .hero-header p  { color: #94A3B8; font-size: 1rem; margin: 0; }

        /* Upload zone */
        .upload-card {
            background: #1A1D27;
            border: 2px dashed #2D3248;
            border-radius: 14px;
            padding: 1.5rem;
            text-align: center;
        }

        /* Result card */
        .result-card {
            background: #1A1D27;
            border-radius: 14px;
            padding: 1.8rem;
            border-left: 5px solid;
            margin-bottom: 1rem;
        }
        .result-disease { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
        .result-confidence { font-size: 1rem; color: #94A3B8; }

        /* Confidence bar */
        .conf-bar-outer {
            background: #2D3248;
            border-radius: 20px;
            height: 10px;
            margin: 0.5rem 0;
            overflow: hidden;
        }
        .conf-bar-inner {
            height: 100%;
            border-radius: 20px;
            transition: width 1s ease;
        }

        /* Disease info card */
        .info-card {
            background: #1E2235;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-top: 1rem;
            border: 1px solid #2D3248;
        }
        .info-card h4 { color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;
                        letter-spacing: 0.1em; margin-bottom: 0.4rem; }
        .info-card p  { color: #CBD5E1; font-size: 0.93rem; line-height: 1.55; margin: 0; }

        /* Disclaimer */
        .disclaimer {
            background: #1E2235;
            border: 1px solid #3D4461;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            color: #94A3B8;
            font-size: 0.83rem;
            margin-top: 1.5rem;
        }

        /* Sidebar */
        .sidebar-section { margin-bottom: 1.5rem; }
        .sidebar-section h3 { color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;
                               letter-spacing: 0.08em; margin-bottom: 0.6rem; }

        /* Class probability badges */
        .prob-row { display: flex; align-items: center; margin-bottom: 0.4rem; }
        .prob-label { width: 160px; font-size: 0.85rem; color: #CBD5E1; flex-shrink: 0; }
        .prob-bar { flex: 1; background: #2D3248; border-radius: 20px;
                    height: 8px; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 20px; }
        .prob-val { width: 48px; text-align: right; font-size: 0.82rem;
                    color: #94A3B8; flex-shrink: 0; margin-left: 8px; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background: #1A1D27; border-radius: 8px 8px 0 0;
            color: #94A3B8; border: 1px solid #2D3248;
        }
        .stTabs [aria-selected="true"] { background: #252840 !important; color: #E2E8F0 !important; }

        /* Metric boxes */
        .metric-box {
            background: #1A1D27;
            border: 1px solid #2D3248;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .metric-val  { font-size: 1.8rem; font-weight: 700; color: #60A5FA; }
        .metric-label{ font-size: 0.8rem; color: #64748B; margin-top: 0.2rem; }

        /* Hide default streamlit elements */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔬 Skin Disease Detector")
        st.markdown("---")

        st.markdown("**Model Info**")
        st.info("Architecture: **EfficientNetB3**\nInput: 224×224 RGB\nClasses: **7 diseases**")

        st.markdown("---")
        st.markdown("**Disease Classes**")
        for idx, d in DISEASES.items():
            emoji, _ = SEVERITY_BADGE[d["severity"]]
            st.markdown(f"{emoji} {d['name']}")

        st.markdown("---")
        show_topk = st.checkbox("📊 Show all class probabilities", value=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.78rem; color:#64748B;'>
        ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only.
        It is NOT a substitute for professional medical advice.
        Always consult a licensed dermatologist for diagnosis.
        </div>
        """, unsafe_allow_html=True)

    return show_topk

# =============================================================================
# PREDICTION
# =============================================================================

def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32")
    arr = preprocess_input(arr)   
    return np.expand_dims(arr, axis=0), img

def predict(model, img_array):
    probs = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return probs, top_idx

# =============================================================================
# GRAD-CAM (inline, no external file dependency)
# =============================================================================


def make_gradcam_overlay(heatmap, orig_img_array):
    orig = np.array(orig_img_array)
    h, w = orig.shape[:2]
    hmap = cv2.resize(heatmap, (w, h))
    hmap_uint8  = np.uint8(255 * hmap)
    hmap_color  = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)
    hmap_rgb    = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(orig, 0.55, hmap_rgb, 0.45, 0)
    return overlay

# =============================================================================
# RESULT DISPLAY
# =============================================================================

def render_result_card(probs, top_idx):
    disease   = DISEASES[top_idx]
    conf      = probs[top_idx] * 100
    sev_emoji, sev_color = SEVERITY_BADGE[disease["severity"]]

    st.markdown(f"""
    <div class="result-card" style="border-color: {disease['color']};">
        <div class="result-disease" style="color: {disease['color']};">
            {disease['name']}
        </div>
        <div class="result-confidence">
            Confidence: <b>{conf:.1f}%</b> &nbsp;|&nbsp;
            Severity: <b style="color:{sev_color};">{sev_emoji} {disease['severity']}</b>
        </div>
        <div class="conf-bar-outer">
            <div class="conf-bar-inner"
                 style="width:{conf:.1f}%; background:{disease['color']};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
        <h4>About this condition</h4>
        <p>{disease['description']}</p>
    </div>
    <div class="info-card">
        <h4>💊 Clinical Advice</h4>
        <p>{disease['advice']}</p>
    </div>
    """, unsafe_allow_html=True)


def render_probabilities(probs, top_idx):
    st.markdown("#### All Class Probabilities")
    sorted_idx = np.argsort(probs)[::-1]

    prob_html = ""
    for idx in sorted_idx:
        d       = DISEASES[idx]
        pct     = probs[idx] * 100
        bold    = "font-weight:700;" if idx == top_idx else ""
        opacity = "1.0" if idx == top_idx else "0.65"
        prob_html += f"""
        <div class="prob-row" style="opacity:{opacity};">
            <div class="prob-label" style="{bold}">{d['name']}</div>
            <div class="prob-bar">
                <div class="prob-fill" style="width:{pct:.1f}%; background:{d['color']};"></div>
            </div>
            <div class="prob-val">{pct:.1f}%</div>
        </div>
        """
    st.markdown(prob_html, unsafe_allow_html=True)

# =============================================================================
# ABOUT PAGE
# =============================================================================

def render_about():
    st.markdown("## 📖 About This Project")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["7", "10,000+", "EfficientNetB3", "~88%"],
        ["Disease Classes", "Training Images", "Architecture", "Accuracy"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🔬 How It Works

    1. **Upload** a clear photo of the skin lesion (JPG/PNG)
    2. The image is **preprocessed** — resized to 224×224 and normalized
    3. **EfficientNetB3** (pre-trained on ImageNet, fine-tuned on HAM10000) performs inference
    4. **Softmax** outputs probability scores for all 7 disease classes
    5. **Grad-CAM** highlights the skin region that most influenced the prediction

    ### 🧠 Model Architecture
    - **Base**: EfficientNetB3 (ImageNet weights)
    - **Head**: GAP → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Softmax(7)
    - **Training**: Two-phase (frozen base → fine-tune top 30 layers)
    - **Augmentation**: Rotation, flips, zoom, brightness, shear
    - **Imbalance handling**: Class weights (sklearn balanced)

    ### 📊 Dataset
    - **HAM10000** — Human Against Machine with 10000 training images (Kaggle / ISIC)
    - 7 classes: mel, nv, bcc, akiec, bkl, df, vasc

    ### ⚠️ Ethical Note
    > This application is intended for **educational and research purposes only**.
    > It is **not** a certified medical device and should **never** replace professional
    > dermatological evaluation. Always consult a licensed healthcare provider.
    """)

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    inject_css()
    show_topk = render_sidebar()

    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1>🩺 Skin Disease Detector</h1>
        <p>Upload a skin lesion image for instant AI-powered classification across 7 disease categories</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation tabs
    tab_detect, tab_about = st.tabs(["🔬 Detect Disease", "📖 About"])

    # ── Detection Tab ────────────────────────────────────────────────────────
    with tab_detect:
        col_upload, col_result = st.columns([1, 1.4], gap="large")

        with col_upload:
            st.markdown("#### 📤 Upload Skin Image")
            uploaded = st.file_uploader(
                label="",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear close-up photo of the skin lesion"
            )

            if uploaded:
                pil_img = Image.open(io.BytesIO(uploaded.read()))
                st.image(pil_img, caption="Uploaded Image", use_container_width=True)
                st.caption(f"Size: {pil_img.size[0]}×{pil_img.size[1]} px | "
                           f"Mode: {pil_img.mode}")

                # Tips
                with st.expander("📌 Tips for best results"):
                    st.markdown("""
                    - Use **good lighting** — natural daylight preferred
                    - **Center the lesion** in the frame
                    - Avoid **blurry or overexposed** images
                    - Minimum recommended resolution: **224×224 px**
                    - This model was trained on **dermatoscopic images**
                    """)

        with col_result:
            if uploaded is not None:
                # Load model
                with st.spinner("Loading model..."):
                    model = load_model()

                if model is None:
                    st.error(
                        "❌ Model file not found at `outputs/models/final_model.h5`.\n\n"
                        "Run `python train_model.py` first to train and save the model."
                    )
                else:
                    st.markdown("#### 🧠 Analysis Results")

                    with st.spinner("Analyzing image..."):
                        pil_img = Image.open(io.BytesIO(uploaded.getvalue()))
                        img_array, img_resized = preprocess_image(pil_img)
                        time.sleep(0.4)   # slight delay for UX feel
                        probs, top_idx = predict(model, img_array)

                    # Main result card
                    render_result_card(probs, top_idx)

                    # All probabilities
                    if show_topk:
                        st.markdown("---")
                        render_probabilities(probs, top_idx)

                    

                    # Disclaimer
                    st.markdown("""
                    <div class="disclaimer">
                    ⚠️ <b>Medical Disclaimer:</b> This prediction is generated by an AI model
                    and is intended for educational purposes only. It is NOT a medical diagnosis.
                    Consult a licensed dermatologist for proper evaluation and treatment.
                    </div>
                    """, unsafe_allow_html=True)

            else:
                # Placeholder
                st.markdown("""
                <div style='background:#1A1D27; border:2px dashed #2D3248; border-radius:14px;
                            padding:3rem; text-align:center; color:#64748B;'>
                    <div style='font-size:3rem;'>📷</div>
                    <div style='font-size:1.1rem; margin-top:0.8rem;'>
                        Upload a skin image to begin analysis
                    </div>
                    <div style='font-size:0.85rem; margin-top:0.5rem;'>
                        Supports JPG, JPEG, PNG
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── About Tab ────────────────────────────────────────────────────────────
    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
