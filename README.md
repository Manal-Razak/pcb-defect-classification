# Explainable PCB Defect Classification

**SE402 — Machine Learning and Smart Systems**  
**Manal Abdul Razak | University of Europe for Applied Sciences**

---

## Project Overview

This project develops and compares three CNN-based models for automated classification of six PCB (Printed Circuit Board) manufacturing defect types, with Grad-CAM and SHAP explainability, deployed as an interactive web application.

**Key finding:** A critical resolution mismatch was identified — original PCB images (2529×2530px) rendered defect regions invisible after standard resizing to 224×224. This was resolved by cropping defect regions using XML bounding-box annotations, expanding the dataset from 693 to 2,953 usable images.

---

## Defect Classes

| Class | Description |
|---|---|
| Missing Hole | Drill hole absent from pad |
| Mouse Bite | Irregular notch along board edge |
| Open Circuit | Break in copper trace |
| Short | Unintended copper bridge between traces |
| Spur | Unwanted copper protrusion |
| Spurious Copper | Excess copper outside intended boundaries |

---

## Models & Results

| Model | Test Accuracy | Macro F1 | Total Params |
|---|---|---|---|
| Custom CNN (Baseline) | 88.12% | 0.88 | 26.1M |
| MobileNetV2 (Transfer Learning) | 96.86% | 0.97 | 2.23M |
| ResNet50 (Transfer Learning) | 98.43% | 0.98 | 23.52M |

---

## Live Demo

**Hugging Face Space:** https://huggingface.co/spaces/Breadwwh/pcb-defect-classifier

Features:
- Live classification with real-time Grad-CAM explanation
- Model selector (all three models)
- Precomputed SHAP analysis tab
- Model comparison table

---

## Dataset

- **Name:** PCB Defects Dataset
- **Source:** Open Lab on Human-Robot Interaction, Peking University
- **Kaggle:** https://www.kaggle.com/datasets/akhatova/pcb-defects
- **Original images:** 693 high-resolution images
- **After cropping:** 2,953 defect-region crops
- **Split:** 70% train / 15% validation / 15% test

---

## Repository Structure

```
pcb-defect-classification/
├── app.py                          # Gradio web app (3-model selector + Grad-CAM + SHAP)
├── pcb-detection.ipynb             # Full training notebook (run on Kaggle, Tesla T4 GPU)
├── requirements.txt                # Python dependencies
├── shap_results.png                # Precomputed SHAP attribution map
├── Missing_hole_sample.jpg         # Sample images for demo
├── Mouse_bite_sample.jpg
├── Open_circuit_sample.jpg
├── Short_sample.jpg
├── Spur_sample.jpg
└── Spurious_copper_sample.jpg
```

> **Note:** Model weight files (`.pth`) are not included in this repo due to size constraints. They are loaded directly in the Hugging Face Space.

---

## Tech Stack

| Component | Tool |
|---|---|
| Deep Learning | PyTorch |
| Transfer Learning | torchvision (ResNet50, MobileNetV2) |
| Explainability | pytorch-grad-cam, SHAP |
| Web Interface | Gradio |
| Deployment | Hugging Face Spaces |
| Training Environment | Kaggle Notebooks (Tesla T4 GPU) |

---

## How to Run Locally

```bash
git clone https://github.com/Breadwwh/pcb-defect-classification
cd pcb-defect-classification
pip install -r requirements.txt
# Add your .pth weight files to the same directory
python app.py
```

---

## Explainability

- **Grad-CAM:** Visualizes which image regions drove each prediction. Heatmaps consistently localize to genuine defect regions (hole boundary, trace gap, copper bridge) rather than background.
- **SHAP:** Provides class-wise positive/negative attribution. Reveals why Mouse Bite and Spur are sometimes confused — overlapping attribution near trace edges indicates genuine visual ambiguity, not model failure.

---

## Declarations

Generative AI tools were used to assist with code generation and language editing. All outputs were reviewed and verified by the author.
