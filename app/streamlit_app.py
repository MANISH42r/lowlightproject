import os
import time
import glob
import torch
import numpy as np
from PIL import Image
import streamlit as st

from enhancement.inference import ZeroDCEEnhancer
from detection.inference import YOLOObjectDetector

st.set_page_config(
    page_title="Low-Light Robust Perception",
    page_icon="🌙",
    layout="wide"
)

st.title("🌙 Low-Light Robust Perception")
st.markdown("### Deep Image Enhancement for Improved Object Detection under Low Light")

@st.cache_resource
def load_models():
    enhancer = ZeroDCEEnhancer()
    detector = YOLOObjectDetector()
    return enhancer, detector

try:
    enhancer, detector = load_models()
    st.success("Models loaded successfully (Zero-DCE++ & YOLOv8)")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Sidebar controls
st.sidebar.header("Configuration & Parameters")
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.05, 0.95, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Input Selection")
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
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Low-Light Image")
        st.image(img_to_process, use_container_width=True)

    with col2:
        st.subheader("✨ Zero-DCE++ Enhanced Image")
        start_enh = time.time()
        enhanced_pil = enhancer.enhance_pil(img_to_process)
        enh_latency = (time.time() - start_enh) * 1000.0
        st.image(enhanced_pil, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Object Detection Comparison")

    det_col1, det_col2 = st.columns(2)

    with det_col1:
        st.subheader("Original Detection (Baseline)")
        dets_base, lat_base, vis_base = detector.predict(img_to_process, conf_threshold=conf_threshold)
        st.image(vis_base, use_container_width=True)
        st.metric("Objects Detected", len(dets_base))
        st.metric("Detection Latency", f"{lat_base:.1f} ms")
        if dets_base:
            avg_conf_base = np.mean([d["confidence"] for d in dets_base])
            st.metric("Avg Confidence", f"{avg_conf_base:.2f}")

    with det_col2:
        st.subheader("Enhanced Detection (Zero-DCE++ + YOLO)")
        dets_enh, lat_enh, vis_enh = detector.predict(enhanced_pil, conf_threshold=conf_threshold)
        st.image(vis_enh, use_container_width=True)
        st.metric("Objects Detected", len(dets_enh))
        st.metric("Detection Latency", f"{lat_enh:.1f} ms (+{enh_latency:.1f} ms enhancement)")
        if dets_enh:
            avg_conf_enh = np.mean([d["confidence"] for d in dets_enh])
            st.metric("Avg Confidence", f"{avg_conf_enh:.2f}")

    # Detailed Detections Table
    st.markdown("---")
    st.subheader("📋 Detected Objects Breakdown")

    tab1, tab2 = st.tab1, st.tab2 = st.tabs(["Original Detections", "Enhanced Detections"])
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
    st.info("👈 Please upload an image or select a sample from the sidebar to test.")
