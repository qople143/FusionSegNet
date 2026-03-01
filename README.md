# FusionSegNet: Multi-Modal Brain Tumor Segmentation using Dual-Branch Fusion and Channel Attention

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

**FusionSegNet** is a deep learning framework for **automatic brain tumor segmentation** using **multi-modal MRI scans**.
The project focuses on improving segmentation accuracy by:

* Learning **modality-specific features** from T1 and FLAIR MRI
* Performing **feature-level fusion**
* Enhancing fusion using **channel attention mechanisms**

This work is evaluated on the **BraTS 2020 dataset** and demonstrates significant improvements over a baseline fusion model.

## Motivation

Brain tumor segmentation is a critical task in medical imaging, supporting:

* Clinical diagnosis
* Treatment planning
* Disease monitoring

Manual segmentation is **time-consuming, subjective, and error-prone**.
FusionSegNet aims to provide a **robust and automated solution** by leveraging deep learning and multi-modal MRI information.

## Architecture

FusionSegNet follows a **dual-branch encoder–decoder architecture**:

```
T1 MRI ──▶ Encoder ──┐
                       ├─▶ Feature Fusion ─▶ Channel Attention ─▶ Decoder ─▶ Tumor Mask
FLAIR MRI ─▶ Encoder ─┘
```

### Key Components:

* **Dual encoders** for modality-specific feature extraction
* **Feature fusion** at intermediate layers
* **Channel attention** to adaptively reweight informative features
* **Decoder** for pixel-wise segmentation

## 🚀 Models Implemented

### 🔹 Baseline FusionSegNet

* Dual-branch encoders for T1 and FLAIR
* Feature concatenation
* Encoder–decoder segmentation network

### 🔹 Enhanced FusionSegNet (Proposed)

* All baseline components
* **Channel attention (Squeeze-and-Excitation)**
* Improved feature refinement
* Better boundary delineation and reduced false positives

## 📊 Results

### Quantitative Performance (BraTS 2020)

| Model                     | Dice Score | Training Loss |
| ------------------------- | ---------- | ------------- |
| Baseline FusionSegNet     | 0.6980     | 0.3316        |
| **Enhanced FusionSegNet** | **0.7933** | **0.2286**    |

✔ **Dice improvement:** +13.7%
✔ **Loss reduction:** −31.1%

## Qualitative Results

The enhanced model produces:

* Smoother tumor boundaries
* Better overlap with ground truth
* Reduced fragmented predictions

Examples include:

* T1 & FLAIR inputs
* Ground truth masks
* Prediction vs GT overlays
* MRI-based visualization

(See figures in the `figures/` directory)

## Dataset

This project uses the **BraTS 2020 Brain Tumor Segmentation Dataset**.

**Dataset features:**

* 369 multi-institutional MRI volumes
* Modalities: T1, T1ce, T2, FLAIR
* Expert-annotated ground truth masks

 Download:
[https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

> In this work, **T1 and FLAIR modalities** are used for segmentation.

## Preprocessing Pipeline

* Intensity normalization (min–max)
* 3D MRI volumes converted to 2D axial slices
* Resized to **256 × 256**
* Binary tumor mask generation
* ~57,000 training samples generated

## Installation

```bash
git clone https://github.com/<your-username>/FusionSegNet.git
cd FusionSegNet
pip install -r requirements.txt
```

## Running the Project

### Training

```bash
python src/main.py
```

### Inference & Visualization

Use the provided notebook:

```bash
notebooks/FusionSegNet.ipynb
```

## Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib
* OpenCV
  
## Limitations

* 2D slice-based segmentation (no volumetric context)
* Limited to T1 and FLAIR modalities
* Evaluated on a single benchmark dataset

## Future Work

* Extend to **3D volumetric segmentation**
* Incorporate additional modalities (T2, T1ce)
* Cross-dataset generalization studies
* Clinical validation and deployment

## Author

**Sinchana L & Sadiya Fathima N**
Final Year B.E. — Computer Science & Engineering
Atria Institute of Technology

---

## 📜 License

This project is licensed under the **MIT License**.
