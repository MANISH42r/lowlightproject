import os
import sys
import time
import glob
import torch
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to Python search path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from enhancement.inference import ZeroDCEEnhancer
from detection.inference import YOLOObjectDetector

st.set_page_config(
    page_title="Low-Light Robust Perception",
    page_icon="🌙",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #0b0f17;
        --bg-card: rgba(18, 24, 38, 0.75);
        --bg-sidebar: #0e131f;
        --border-color: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(0, 229, 153, 0.25);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent-emerald: #00e599;
        --accent-cyan: #38bdf8;
        --accent-purple: #a855f7;
    }

    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(0, 229, 153, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(56, 189, 248, 0.05) 0%, transparent 40%),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px;
        color: var(--text-primary);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-primary) !important;
    }

    h1, h2, h3, h4, h5, h6, [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', 'DM Sans', sans-serif;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em;
    }

    p, span, label {
        color: var(--text-primary);
    }

    .main .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1.75rem 2rem;
        background: linear-gradient(135deg, rgba(18, 24, 38, 0.8), rgba(15, 23, 42, 0.6));
        border: 1px solid var(--border-color);
        border-radius: 12px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
        backdrop-filter: blur(12px);
    }
    .hero-kicker {
        color: var(--accent-emerald);
        font: 700 0.72rem 'DM Sans', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-kicker::before {
        content: '';
        display: inline-block;
        width: 6px;
        height: 6px;
        background-color: var(--accent-emerald);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--accent-emerald);
    }
    .hero h1 {
        font: 700 2.5rem/1.1 'Space Grotesk', sans-serif;
        margin: 0;
        background: linear-gradient(180deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: var(--text-secondary);
        font-size: 1.05rem;
        margin: 0.6rem 0 0;
    }
    .hero-stamp {
        background: rgba(0, 229, 153, 0.1);
        border: 1px solid rgba(0, 229, 153, 0.3);
        color: var(--accent-emerald);
        padding: 0.6rem 1rem;
        font: 700 0.72rem 'Space Grotesk', sans-serif;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        white-space: nowrap;
        border-radius: 6px;
        box-shadow: 0 0 15px rgba(0, 229, 153, 0.15);
    }

    .section-label {
        color: var(--accent-cyan);
        font: 700 0.75rem 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin: 1.25rem 0 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(56, 189, 248, 0.25) !important;
    }

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border-left: 3px solid var(--accent-emerald) !important;
        padding: 0.75rem 0.9rem !important;
        border-radius: 6px !important;
        border-top: 1px solid var(--border-color);
        border-right: 1px solid var(--border-color);
        border-bottom: 1px solid var(--border-color);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 4px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 6px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 229, 153, 0.15) !important;
        color: var(--accent-emerald) !important;
    }

    [data-testid="stImage"] img {
        border-radius: 6px;
        border: 1px solid var(--border-color);
    }
    hr {
        border-color: var(--border-color) !important;
    }

    @media (max-width: 700px) {
        .hero { align-items: flex-start; flex-direction: column; padding: 1.25rem; }
        .hero h1 { font-size: 1.8rem; }
        .hero-stamp { align-self: flex-start; }
        .main .block-container { padding-top: 1rem; }
    }
    </style>
    <div class="hero">
      <div>
        <div class="hero-kicker">Perception lab / ExDark</div>
        <h1>Low-light, brought into focus.</h1>
        <p>Compare Zero-DCE++ enhancement and YOLOv8 detection, side by side.</p>
      </div>
      <div class="hero-stamp">Zero-DCE++ &nbsp; / &nbsp; YOLOv8</div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_models():
    enhancer = ZeroDCEEnhancer()
    detector = YOLOObjectDetector()
    return enhancer, detector

try:
    enhancer, detector = load_models()
    st.sidebar.markdown("**SYSTEM STATUS**  ·  Models ready")
except Exception as e:
    st.error(f"Could not load the enhancement and detection models: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Analysis controls")
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.05, 0.95, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Image source")
input_option = st.sidebar.radio("Choose Input Source:", ["Upload Image", "Sample ExDark Images"])

img_to_process = None

if input_option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload low-light JPEG/PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file).convert('RGB')
else:
    sample_files = sorted(glob.glob("data/test/images/*.jpg")) + sorted(glob.glob("images/ExDark/*/*.jpg"))
    if sample_files:
        selected_sample = st.sidebar.selectbox("Select Sample Image:", sample_files[:30])
        if selected_sample and os.path.exists(selected_sample):
            img_to_process = Image.open(selected_sample).convert('RGB')

if img_to_process is not None:
    st.markdown('<div class="section-label">01 / Image enhancement</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Original input")
            st.image(img_to_process, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Enhanced · Zero-DCE++")
            start_enh = time.time()
            enhanced_pil = enhancer.enhance_pil(img_to_process)
            enh_latency = (time.time() - start_enh) * 1000.0
            st.image(enhanced_pil, use_container_width=True)

    st.markdown('<div class="section-label">02 / Detection comparison</div>', unsafe_allow_html=True)

    det_col1, det_col2 = st.columns(2)

    with det_col1:
        with st.container(border=True):
            st.subheader("Baseline · Original → YOLOv8")
            dets_base, lat_base, vis_base = detector.predict(img_to_process, conf_threshold=conf_threshold)
            st.image(vis_base, use_container_width=True)
            base_metrics = st.columns(3)
            base_metrics[0].metric("Objects", len(dets_base))
            base_metrics[1].metric("Detection", f"{lat_base:.1f} ms")
            avg_conf_base = np.mean([d["confidence"] for d in dets_base]) if dets_base else None
            base_metrics[2].metric("Avg. confidence", f"{avg_conf_base:.2f}" if avg_conf_base is not None else "—")

    with det_col2:
        with st.container(border=True):
            st.subheader("Enhanced · Zero-DCE++ → YOLOv8")
            dets_enh, lat_enh, vis_enh = detector.predict(enhanced_pil, conf_threshold=conf_threshold)
            st.image(vis_enh, use_container_width=True)
            enh_metrics = st.columns(3)
            enh_metrics[0].metric("Objects", len(dets_enh))
            enh_metrics[1].metric("Detection", f"{lat_enh:.1f} ms")
            avg_conf_enh = np.mean([d["confidence"] for d in dets_enh]) if dets_enh else None
            enh_metrics[2].metric("Avg. confidence", f"{avg_conf_enh:.2f}" if avg_conf_enh is not None else "—")
            st.caption(f"Enhancement overhead: {enh_latency:.1f} ms")

    # Detailed Detections Table
    st.markdown('<div class="section-label">03 / Detection records</div>', unsafe_allow_html=True)
    st.subheader("Detected objects")

    tab1, tab2 = st.tabs(["Original detections", "Enhanced detections"])
    with tab1:
        if dets_base:
            st.dataframe(dets_base)
        else:
            st.info("No objects detected in original low-light image at current threshold.")

    with tab2:
        if dets_enh:
            st.dataframe(dets_enh)
        else:
            st.info("No objects detected in enhanced image at current threshold.")

else:
    st.markdown(
        """
        <div style="padding:2.5rem 1.8rem; border:1px dashed rgba(0, 229, 153, 0.3); border-radius:10px; background:rgba(18, 24, 38, 0.6); backdrop-filter:blur(10px);">
          <div class="section-label">Ready for analysis</div>
          <div style="font:600 1.4rem 'Space Grotesk',sans-serif; color:#F8FAFC; margin-top:0.4rem;">Choose an image to begin.</div>
          <div style="margin-top:0.5rem; color:#94A3B8;">Upload a low-light image or select a sample from the sidebar controls.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
