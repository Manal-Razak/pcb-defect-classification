# PCB Defect Classification

Classifying 6 types of manufacturing defects on printed circuit boards using deep learning. Compares a custom CNN baseline against ResNet50 and MobileNetV2 transfer learning models, with Grad-CAM and SHAP explainability.

---

## Defect Classes

| Class | Description |
|-------|-------------|
| Missing Hole | Drilled hole absent from board |
| Mouse Bite | Small notches along board edge |
| Open Circuit | Broken conductive path |
| Short | Unintended connection between traces |
| Spur | Unwanted conductive protrusion |
| Spurious Copper | Excess copper in unintended areas |

---

## Results

| Model | Test Accuracy | Macro F1 |
|-------|--------------|----------|
| Custom CNN (baseline) | 88.34% | 0.85 |
| ResNet50 (transfer learning) | ~99% | 0.99 |
| MobileNetV2 (transfer learning) | ~99% | 0.99 |

Transfer learning dramatically outperforms training from scratch, even with a small dataset. MobileNetV2 achieves comparable accuracy to ResNet50 at lower computational cost.

---

## Pipeline

```
Raw PCB Images + XML Annotations
        |
Bounding Box Crop Extraction (padding=20px)
        |
Train / Val / Test Split (70 / 15 / 15)
        |
Data Augmentation (flips, rotations, ImageNet normalization)
        |
Model Training x 3
        |
Evaluation + Confusion Matrix
        |
Grad-CAM + SHAP Explainability
        |
Gradio Interactive Demo
```

---

## Models

**Custom CNN (Baseline)**
Built from scratch with 3 convolutional blocks (Conv, BatchNorm, ReLU, MaxPool), Dropout regularization, and a fully connected classifier. Trained for 15 epochs with Adam (lr=0.001).

**ResNet50 (Transfer Learning)**
Pretrained on ImageNet. All layers frozen except layer4 and the final classifier, replaced with a 6-class linear layer. Fine-tuned with Adam (lr=0.0001) for 10 epochs.

**MobileNetV2 (Transfer Learning)**
Pretrained on ImageNet. Last 3 InvertedResidual blocks unfrozen for fine-tuning. Chosen to evaluate the accuracy vs computational efficiency trade-off.

---

## Explainability

**Grad-CAM**
Highlights which image regions the model focuses on when classifying each defect, verifying the model attends to actual defect areas rather than background noise.

**SHAP**
Identifies which pixel regions push the model's prediction toward or away from each defect class, providing a complementary pixel-level explanation.

---

## Interactive Demo

A Gradio interface allows uploading a cropped PCB image and returns the predicted defect class with confidence scores across all 6 classes.

Live demo: https://huggingface.co/spaces/YOUR_USERNAME/pcb-defect-classifier

---

## Dataset

PCB Defects by akhatova, available on Kaggle: https://www.kaggle.com/datasets/akhatova/pcb-defects

- 12 raw PCB board images per class
- XML bounding box annotations per defect instance
- Crops extracted per annotation for more training samples

**Option 1 — Kaggle (recommended)**
Add the dataset via Add Data and search "PCB Defects (akhatova)" in Kaggle notebooks.

**Option 2 — Local**
```bash
kaggle datasets download -d akhatova/pcb-defects
unzip pcb-defects.zip -d pcb_data
```
Place your kaggle.json at ~/.kaggle/kaggle.json first.

---

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow grad-cam shap gradio
```

---

## Running the Notebook

```bash
git clone https://github.com/Manal-Razak/pcb-defect-classification.git
cd pcb-defect-classification
```

Set up the dataset, then open and run pcb_defect_classification.ipynb top to bottom.

GPU recommended. Trained on Kaggle free T4 GPU. CPU will work but training will be slow.

---

## Repository Structure

```
pcb-defect-classification/
    pcb_defect_classification.ipynb
    requirements.txt
    README.md
    .gitignore
```

Model weights and the dataset are not included due to file size. Re-run the notebook to reproduce them.

---

## References

- Dataset: https://www.kaggle.com/datasets/akhatova/pcb-defects
- ResNet: Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385
- MobileNetV2: Inverted Residuals and Linear Bottlenecks, https://arxiv.org/abs/1801.04381
- Grad-CAM: Visual Explanations from Deep Networks, https://arxiv.org/abs/1610.02391
- SHAP: A Unified Approach to Interpreting Model Predictions, https://arxiv.org/abs/1705.07874

---

## Author

Manal Abdul Razak Mohammed  
Software Engineering Student, University of Europe for Applied Sciences, Potsdam  
https://github.com/Manal-Razak
