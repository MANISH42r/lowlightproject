# LOW-LIGHT ROBUST PERCEPTION: Deep Image Enhancement for Improved Object Detection

An end-to-end deep learning pipeline evaluating whether zero-reference low-light image enhancement (**Zero-DCE++**) improves object detection performance (**YOLOv8**) under extremely low-light conditions on the **ExDark 384 dataset**.

---

## 📌 Research Question & Objective

**Main Research Question:**
> *"Does deep-learning-based low-light image enhancement improve object detection performance under extremely low-light conditions?"*

This repository provides quantitative evidence comparing:
- **Baseline System (A):** Direct Object Detection on original low-light images (`YOLOv8`).
- **Enhanced System (B):** Zero-Reference Image Enhancement (`Zero-DCE++`) followed by Object Detection (`YOLOv8`).

---

## 🏗️ Pipeline Architecture

```
                 LOW-LIGHT IMAGE (ExDark 384)
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
   +-----------------+                   +-----------------+
   | Direct Baseline |                   |   Zero-DCE++    |
   | Object Detector |                   |  Enhancement    |
   |    (YOLOv8)     |                   +-----------------+
   +-----------------+                            |
            |                                     v
            |                              ENHANCED IMAGE
            |                                     |
            |                                     v
            |                            +-----------------+
            |                            | Object Detector |
            |                            |    (YOLOv8)     |
            |                            +-----------------+
            v                                     v
   Baseline Detections                   Enhanced Detections
            |                                     |
            +------------------+------------------+
                               |
                               v
                     Controlled Comparison
                 & Robustness Evaluation
```

---

## 📊 Dataset: ExDark 384

- **Dataset:** ExDark (Extremely Low-Light Environment Dataset) resized to **384×384**.
- **Annotations:** COCO Format (`images`, `annotations`, `categories`).
- **Object Categories (12):** Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table.
- **Split Breakdown:**
  - **Train:** 5,666 images (18,285 annotations)
  - **Validation:** 1,001 images (3,186 annotations)
  - **Test:** 695 images (2,232 annotations)

---

## ⚡ Zero-DCE++ Enhancement & Loss Formulation

Zero-DCE++ reformulates light enhancement as an iterative curve estimation problem. It estimates image-specific higher-order curves without requiring paired normal-light ground-truth images:

$$LE_k(x) = LE_{k-1}(x) + \alpha_k(x) \cdot LE_{k-1}(x) \cdot (1 - LE_{k-1}(x)) \quad \text{for } k=1, \dots, 8$$

### Loss Components:
$$\mathcal{L}_{\text{total}} = w_{\text{spa}} \mathcal{L}_{\text{spa}} + w_{\text{exp}} \mathcal{L}_{\text{exp}} + w_{\text{col}} \mathcal{L}_{\text{col}} + w_{\text{tv}} \mathcal{L}_{\text{tv}}$$

1. **Spatial Consistency Loss ($\mathcal{L}_{\text{spa}}$):** Preserves spatial coherence across local $4 \times 4$ patches.
2. **Exposure Control Loss ($\mathcal{L}_{\text{exp}}$):** Drives average local patch intensity towards target exposure ($E = 0.6$).
3. **Color Constancy Loss ($\mathcal{L}_{\text{col}}$):** Corrects inter-channel color deviations (Gray-World assumption).
4. **Illumination Smoothness Loss ($\mathcal{L}_{\text{tv}}$):** Total Variation loss enforcing monotonic parameter transitions.

---

## 🔬 Experimental Evaluation & Results

### 1. Image Quality Evaluation (No-Reference Metrics)

> *Note: ExDark is an un-paired dataset. Full-reference metrics (PSNR/SSIM) are scientifically invalid without paired normal-light reference images. Therefore, non-reference quality metrics are evaluated.*

| Image State | Mean Brightness (0-255) | Contrast (Std Dev) | Spatial Frequency (Sharpness) | Entropy |
| :--- | :---: | :---: | :---: | :---: |
| **Original Low-Light** | 21.85 | 32.05 | 4.77 | 4.6600 |
| **Zero-DCE++ Enhanced** | 27.02 (+23.7%) | 37.86 (+18.1%) | 5.27 (+10.5%) | 4.8500 |

---

### 2. Controlled Comparison (Object Detection)

| Metric | Baseline (YOLOv8) | Enhanced (Zero-DCE++ + YOLOv8) | Difference |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.4614 | 0.1685 | -0.2929 |
| **Recall** | 0.0076 | 0.0078 | +0.0002 |
| **mAP@50** | 0.0010 | 0.0004 | -0.0006 |
| **mAP@50:95** | 0.0003 | 0.0001 | -0.0002 |
| **Latency (ms)** | 21.05 | 21.47 | +0.42 |
| **FPS** | 47.50 | 46.57 | -0.93 |

---

### 3. Synthetic Degradation Robustness

| Synthetic Degradation | Baseline mAP@50 | Enhanced mAP@50 | Difference |
| :--- | :---: | :---: | :---: |
| **Brightness Reduction (0.5x)** | 0.0073 | 0.0233 | +0.0160 |
| **Gaussian Noise ($\sigma=0.05$)** | 0.0020 | 0.0052 | +0.0032 |
| **Gaussian Blur ($k=5$)** | 0.0068 | 0.0143 | +0.0075 |
| **Motion Blur ($k=9$)** | 0.0075 | 0.0203 | +0.0128 |
| **Contrast Reduction (0.5x)** | 0.0067 | 0.0227 | +0.0161 |
| **Partial Occlusion (20%)** | 0.0065 | 0.0212 | +0.0147 |

---

## 💻 Interactive Streamlit Web Application

An interactive web application is available under `app/streamlit_app.py`.

### Features:
- Upload low-light images or select sample test images.
- Side-by-side visualization of Original vs Zero-DCE++ Enhanced images.
- Side-by-side Object Detection predictions with bounding boxes, class labels, and confidence scores.
- Real-time detection metrics breakdown, average confidence, and latency measurements.

Run locally:
```bash
streamlit run app/streamlit_app.py
```

---

## ⚙️ Project Structure

```
lowlightproject/
├── config.yaml                     # Central configuration parameters
├── run_pipeline.py                 # Unified CLI pipeline automation script
├── conftest.py                     # Pytest environment setup
├── app/
│   └── streamlit_app.py            # Streamlit interactive application
├── scripts/
│   ├── inspect_dataset.py          # ExDark dataset inspection script
│   ├── prepare_dataset.py          # COCO to YOLO conversion & dataset splitter
│   └── generate_comparison.py      # Controlled comparison table & figure generator
├── enhancement/
│   ├── model.py                    # Zero-DCE++ PyTorch architecture
│   ├── losses.py                   # Zero-reference spatial/exposure/color/TV losses
│   ├── train.py                    # Zero-DCE++ training script
│   ├── inference.py                # Zero-DCE++ inference wrapper
│   └── evaluate.py                 # Zero-DCE++ image quality evaluation runner
├── evaluation/
│   └── image_quality.py            # No-reference & reference image quality metrics
├── detection/
│   ├── train.py                    # YOLO detector training script
│   ├── evaluate.py                 # YOLO detector evaluation script
│   └── inference.py                # YOLO object detection wrapper
├── robustness/
│   ├── brightness.py               # Brightness reduction degradation
│   ├── noise.py                    # Gaussian noise degradation
│   ├── blur.py                     # Gaussian blur degradation
│   ├── motion_blur.py              # Motion blur degradation
│   ├── contrast.py                 # Contrast reduction degradation
│   ├── occlusion.py                # Partial rectangular occlusion
│   └── evaluate.py                 # Robustness experiment runner
├── tests/                          # Unit test suite
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_detection.py
│   ├── test_enhancement.py
│   └── test_metrics.py
└── docs/
    ├── project_report.md           # Comprehensive technical research report
    └── resume_bullets.md           # Resume bullet points based on actual work
```

---

## 🚀 Installation & Quick Start

### 1. Requirements & Setup
```bash
pip install torch torchvision ultralytics scikit-image pandas seaborn pytest streamlit pyyaml opencv-python Pillow
```

### 2. Run Complete Pipeline (Quick Test Mode)
```bash
python run_pipeline.py --quick-test
```

### 3. Run Individual Pipeline Steps
```bash
# Dataset Inspection
python run_pipeline.py --inspect

# Prepare Dataset (COCO -> YOLO)
python run_pipeline.py --prepare-data

# Train Zero-DCE++ Enhancement Model
python run_pipeline.py --train-enhancement

# Train YOLO Object Detector
python run_pipeline.py --train-detector

# Evaluate Baseline vs Enhanced Detection
python run_pipeline.py --evaluate

# Run Robustness Experiments
python run_pipeline.py --robustness

# Run End-to-End Pipeline
python run_pipeline.py --all
```

---

## ⚠️ Limitations & Future Work

- **Un-paired Low-Light Dataset:** Ground-truth normal-light images are unavailable in ExDark, necessitating no-reference image metrics.
- **Hardware Constraint Optimization:** Lightweight models (Zero-DCE++, YOLOv8n) were selected to guarantee compatibility with entry-level GPUs (RTX 3050) and CPU fallback.
- **End-to-End Joint Fine-Tuning:** Currently, Zero-DCE++ and YOLO are trained sequentially. Future work will investigate joint end-to-end backpropagation from detection loss into curve estimation network.
