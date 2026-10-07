# Design Specification: CNN-based Shrimp Freshness Classification System

- **Date:** 2026-10-07
- **Target Platform:** Google Colab (with GPU acceleration, e.g., T4/V100)
- **Framework:** PyTorch & torchvision
- **Task:** Binary Image Classification (`fresh` vs `spoiled`)
- **Status:** Approved for Implementation

---

## 1. Executive Summary & Objectives

The goal of this project is to develop an automated computer vision pipeline using Deep Learning Convolutional Neural Networks (CNN) to classify shrimp images as **Fresh (`fresh`)** or **Spoiled (`spoiled`)**.

To ensure scientific rigor and provide a thorough benchmark, four distinct CNN architectures will be implemented, trained under identical data conditions, and evaluated against standardized metrics:
1. **Custom CNN:** Lightweight baseline trained from scratch (to establish an empirical baseline).
2. **MobileNetV3-Large:** Lightweight model optimized for edge devices and mobile inference.
3. **ResNet-50:** Classic, deep residual network serving as the industry/academic benchmark.
4. **EfficientNet-B0:** State-of-the-art compound scaling model optimizing accuracy per parameter.

---

## 2. Environment & Storage Architecture

### 2.1 Google Colab Runtime Constraints
- **GPU:** NVIDIA T4 Tensor Core (standard Colab GPU).
- **Session Timeout Risk:** Free Colab sessions may disconnect after idling or prolonged runs.
- **Drive I/O Bottleneck:** Directly reading thousands of individual image files from `/content/drive/` during training causes severe I/O lag, resulting in low GPU utilization.

### 2.2 Storage & Caching Strategy
1. **Google Drive Mount:** Dataset is mounted at `/content/drive/MyDrive/...`.
2. **Local SSD Caching:** The raw dataset folder or zip archive is copied to Colab's fast local SSD (`/content/dataset/`) before dataset initialization.
3. **Artifact Persistence:** Best model weights (`best_<model_name>.pth`), training histories (`history_<model_name>.json`), and evaluation figures are directly saved back to Google Drive (`/content/drive/MyDrive/shrimp_project/checkpoints/`).

---

## 3. Data Pipeline & Preprocessing

### 3.1 Dataset Structure & Stratified Splitting
- **Raw Organization:** Two directories in Google Drive:
  - `fresh/` (Images of fresh shrimp)
  - `spoiled/` (Images of spoiled/non-fresh shrimp)
- **Split Ratio:**
  - **Training Set:** 70%
  - **Validation Set:** 15%
  - **Test Set:** 15%
- **Partitioning Method:** `sklearn.model_selection.train_test_split` with `stratify=labels` and a deterministic random seed (`seed=42`) to guarantee identical distribution across classes and ensure all 4 models evaluate on the exact same data splits.

### 3.2 Preprocessing & Data Augmentation
All images are standardized to **224 × 224 pixels** with 3 RGB channels.

#### Training Transforms (`train_transforms`)
- `transforms.RandomResizedCrop(224, scale=(0.8, 1.0))`
- `transforms.RandomHorizontalFlip(p=0.5)`
- `transforms.RandomVerticalFlip(p=0.5)` (orientation-invariant shrimp orientation)
- `transforms.RandomRotation(degrees=15)`
- `transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)` (lighting variations)
- `transforms.ToTensor()`
- `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

#### Validation & Test Transforms (`eval_transforms`)
- `transforms.Resize(256)`
- `transforms.CenterCrop(224)`
- `transforms.ToTensor()`
- `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

## 4. Model Architectures & Transfer Learning Strategy

### 4.1 Model Specifications

| Model Name | Backbone Architecture | Pretrained Source | Parameter Count | Primary Role |
| :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | 4 Conv blocks (Conv2d-BN-ReLU-MaxPool) + GAP + FC | None (Scratch) | ~2.1M | Baseline without transfer learning |
| **MobileNetV3-Large** | Inverted Residuals + SE blocks + Hard-Swish | ImageNet-1K (`DEFAULT`) | ~5.4M | Lightweight, edge/mobile deployment |
| **ResNet-50** | 50-layer Residual Bottleneck blocks | ImageNet-1K (`DEFAULT`) | ~25.6M | Industry & research standard benchmark |
| **EfficientNet-B0** | MBConv blocks with compound scaling | ImageNet-1K (`DEFAULT`) | ~5.3M | High-efficiency state-of-the-art CNN |

### 4.2 Fine-Tuning Strategy for Pretrained Models
1. **Classifier Head Adaptation:** Replace final layer (`fc` or `classifier`) with:
   - `Dropout(p=0.3)`
   - `Linear(in_features, 2)`
2. **Two-Stage Fine-Tuning:**
   - **Phase 1 (Warmup / Frozen Backbone):** Freeze backbone weights, train classifier head for 3 epochs with learning rate $1\times 10^{-3}$.
   - **Phase 2 (Fine-Tuning):** Unfreeze entire network, train with lower learning rate ($1\times 10^{-4}$ for classifier, $1\times 10^{-5}$ for backbone).

---

## 5. Training Engine & Checkpointing

### 5.1 Optimization Setup
- **Loss Function:** `nn.CrossEntropyLoss()`
- **Optimizer:** `AdamW` with weight decay $= 1\times 10^{-4}$.
- **Learning Rate Scheduler:** `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)`.
- **Batch Size:** 32.
- **Maximum Epochs:** 20 per model.
- **Early Stopping:** Monitored metric: `val_loss`, `patience=5`. Automatically restores best weights when triggered.

### 5.2 Training Loop Engine
A single reusable `train_model()` function handles:
1. Training phase: Forward pass, loss calculation, backward propagation, gradient clipping (`max_norm=1.0`), and optimizer step.
2. Validation phase: Forward pass under `torch.no_grad()`, calculation of `val_loss` and `val_acc`.
3. Checkpoint trigger: If `val_loss < best_val_loss`, save state dictionary to Google Drive.
4. History tracking: Collects per-epoch train/val loss and accuracy into a Python dictionary.

---

## 6. Evaluation, Metrics & Visualizations

### 6.1 Quantitative Test Metrics
Evaluation is executed strictly on the unobserved 15% Test Set:
1. **Accuracy:** Overall correctness across all test images.
2. **Precision, Recall, F1-Score (Macro & Per-class):** Focus on class 1 (`spoiled`) Recall to minimize False Positives.
3. **Inference Latency:** Average inference time per image (milliseconds) computed over 100 test iterations.
4. **Model Size:** Size of the exported `.pth` weights file on disk (MB).

### 6.2 Visualizations & Comparative Analysis
1. **Comparison Leaderboard Table:** Summary DataFrame comparing all 4 models side-by-side.
2. **Learning Curves:** 2x2 subplot showing Train vs Validation Loss and Accuracy curves for all models.
3. **Confusion Matrices:** 4-panel Seaborn heatmap displaying True Positives, False Positives, True Negatives, and False Negatives.
4. **ROC-AUC Curves:** Overlay of Receiver Operating Characteristic curves with Area Under Curve scores.
5. **Grad-CAM (Explainability):** Visual activation heatmaps highlighting regions of interest (e.g., shell discoloration, head/tail decay) for model predictions.

---

## 7. Notebook Structure (Colab Implementation Map)

- **Cell 1:** Environment setup, GPU verification (`nvidia-smi`), Google Drive mount.
- **Cell 2:** Local caching (copy/extract dataset from Drive to `/content/dataset`).
- **Cell 3:** Stratified train/val/test split and PyTorch `DataLoader` setup.
- **Cell 4:** Model definitions and factory function `build_model(model_name)`.
- **Cell 5:** Universal `train_model()` and `evaluate_model()` helper functions with Early Stopping.
- **Cell 6:** Sequential training loop over the 4 models with progress tracking.
- **Cell 7:** Test evaluation and Leaderboard generation.
- **Cell 8:** Visualization plots (Loss/Acc curves, Confusion Matrices, ROC-AUC).
- **Cell 9:** Grad-CAM inspection for sample test predictions.
- **Cell 10:** Model saving and export to Google Drive.
