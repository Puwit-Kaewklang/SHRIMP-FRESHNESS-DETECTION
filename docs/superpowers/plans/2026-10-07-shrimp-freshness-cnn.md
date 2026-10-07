# Shrimp Freshness CNN Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, reproducible Deep Learning benchmark pipeline in PyTorch to train and compare 4 CNN models for classifying shrimp freshness from Google Drive's `shrimp_raw_jpg.zip` dataset on Google Colab.

**Architecture:** The solution comprises a modular Python library (`src/`) verified with unit tests, alongside a standalone, self-contained Google Colab notebook (`shrimp_freshness_classification.ipynb`). The pipeline automates Drive mounting, fast local zip extraction, stratified 70/15/15 splitting, transfer learning, checkpointing, confusion matrix plotting, and Grad-CAM interpretability.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, scikit-learn, albumentations/torchvision.transforms, matplotlib, seaborn, pandas, pytest.

**Spec:** `docs/specs/2026-10-07-shrimp-freshness-cnn-design.md`

## Global Constraints
- Target hardware: Google Colab GPU (T4 / V100 / A100) or CPU fallback.
- Dataset source path: `/content/drive/MyDrive/shrimp_raw_jpg.zip` containing `fresh` and `not_fresh` folders.
- Extracted local path: `/content/dataset/` to avoid Drive I/O bottlenecks.
- Classes: Binary (`fresh`: 0, `not_fresh`: 1).
- Split ratio: Deterministic 70% Train, 15% Validation, 15% Test with `seed=42`.
- Models: Custom CNN, MobileNetV3-Large, ResNet-50, EfficientNet-B0.

## Review Focus
1. Corrupted or non-image files inside the zip archive -> Filter non-image extensions (.DS_Store, .txt) gracefully during dataset loading.
2. Unequal class distribution -> Enforce stratified sampling across Train/Val/Test sets so positive/negative ratio remains constant.
3. GPU out of memory (OOM) on large architectures -> Ensure batch size is 32 and garbage collection / `torch.cuda.empty_cache()` runs between models.
4. Drive path missing or incorrect zip name -> Provide clear error message and path existence check before proceeding with extraction.
5. Colab session termination -> Auto-save best weights (`best_<model>.pth`) and training histories (`history_<model>.json`) directly to Google Drive.

---

### Task 1: Scaffolding, Environment Setup & Data Pipeline Module

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces: `extract_and_prepare_dataset(zip_path: str, extract_to: str) -> str`
- Produces: `get_data_loaders(data_dir: str, batch_size: int = 32, seed: int = 42, split: tuple = (0.7, 0.15, 0.15))` returning `(train_loader, val_loader, test_loader, class_names)`

- [ ] **Step 1: Write the failing test for dataset extraction and splitting**

```python
# tests/test_dataset.py
import os
import shutil
import zipfile
from PIL import Image
import pytest
from src.dataset import extract_and_prepare_dataset, get_data_loaders

@pytest.fixture
def dummy_zip(tmp_path):
    zip_dir = tmp_path / "raw"
    zip_dir.mkdir()
    fresh_dir = zip_dir / "fresh"
    not_fresh_dir = zip_dir / "not_fresh"
    fresh_dir.mkdir()
    not_fresh_dir.mkdir()
    
    # Create 10 dummy images per class
    for i in range(10):
        img = Image.new("RGB", (32, 32), color=(i * 20, 100, 50))
        img.save(fresh_dir / f"fresh_{i}.jpg")
        img.save(not_fresh_dir / f"not_fresh_{i}.jpg")
        
    zip_file = tmp_path / "shrimp_raw_jpg.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        for root, _, files in os.walk(zip_dir):
            for file in files:
                abs_p = os.path.join(root, file)
                rel_p = os.path.relpath(abs_p, tmp_path)
                zf.write(abs_p, rel_p)
    return str(zip_file)

def test_extract_and_loaders(dummy_zip, tmp_path):
    extract_to = str(tmp_path / "extracted")
    data_dir = extract_and_prepare_dataset(dummy_zip, extract_to)
    assert os.path.exists(data_dir)
    train_loader, val_loader, test_loader, classes = get_data_loaders(data_dir, batch_size=4, seed=42)
    assert set(classes) == {"fresh", "not_fresh"}
    assert len(train_loader.dataset) == 14  # 70% of 20
    assert len(val_loader.dataset) == 3    # 15% of 20
    assert len(test_loader.dataset) == 3   # 15% of 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 3: Implement `requirements.txt` and `src/dataset.py`**

Implement zip extraction with nested directory normalization (locating `fresh` and `not_fresh` directories even if zipped with parent folder), stratified splitting with fixed seed, PyTorch ImageFolder transforms (RandomResizedCrop, flips, rotations, color jitter for train; resize, center crop, normalize for eval), and return loaders.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/ tests/
git commit -m "feat: implement dataset extraction and stratified dataloaders"
```

---

### Task 2: Model Architectures & Factory

**Files:**
- Create: `src/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: PyTorch `nn.Module`, `torchvision.models`
- Produces: `CustomCNN(nn.Module)`
- Produces: `build_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module` supporting `custom_cnn`, `mobilenet_v3`, `resnet50`, `efficientnet_b0`

- [ ] **Step 1: Write the failing test for model factory and forward passes**

```python
# tests/test_models.py
import torch
import pytest
from src.models import build_model, SUPPORTED_MODELS

@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_forward_pass(model_name):
    model = build_model(model_name, num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Implement `src/models.py`**

Define `CustomCNN` with 4 Conv-BN-ReLU-MaxPool blocks and adaptive average pooling.
Implement `build_model` matching classifier heads for MobileNetV3-Large (`classifier[3]`), ResNet-50 (`fc`), and EfficientNet-B0 (`classifier[1]`). Support both pretrained and scratch modes.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py`
Expected: PASS (all 4 models produce `(2, 2)` output tensor)

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: implement 4 CNN model architectures and factory"
```

---

### Task 3: Training Engine, Early Stopping & Checkpointing

**Files:**
- Create: `src/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `model: nn.Module`, `train_loader`, `val_loader`
- Produces: `EarlyStopping(patience: int = 5, min_delta: float = 1e-4)`
- Produces: `train_model(model, train_loader, val_loader, device, epochs, lr, save_path) -> dict` returning history `{train_loss, train_acc, val_loss, val_acc}`

- [ ] **Step 1: Write the failing test for training engine**

```python
# tests/test_trainer.py
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src.models import build_model
from src.trainer import train_model

def test_train_model_single_epoch(tmp_path):
    X = torch.randn(8, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)
    
    model = build_model("custom_cnn", num_classes=2, pretrained=False)
    save_path = str(tmp_path / "best_model.pth")
    history = train_model(model, loader, loader, device=torch.device("cpu"), epochs=1, save_path=save_path)
    
    assert "val_loss" in history
    assert "val_acc" in history
    assert len(history["val_loss"]) == 1
    assert os.path.exists(save_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.trainer'`

- [ ] **Step 3: Implement `src/trainer.py`**

Implement `EarlyStopping`, AdamW optimizer configuration, Cosine Annealing scheduler, training loop with metric computation, checkpoint saving to `save_path` on new lowest validation loss, and return structured history dictionary.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trainer.py tests/test_trainer.py
git commit -m "feat: implement training engine with early stopping and checkpointing"
```

---

### Task 4: Evaluation Metrics & Visualizations

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `model`, `test_loader`, `class_names`, `device`
- Produces: `evaluate_model(model, test_loader, device) -> dict` returning accuracy, precision, recall, f1, inference_latency_ms, confusion_matrix, y_true, y_pred, y_probs
- Produces: `plot_comparison_curves(all_histories: dict, save_path: str)`
- Produces: `plot_confusion_matrices(all_cms: dict, class_names: list, save_path: str)`

- [ ] **Step 1: Write the failing test for evaluation metrics**

```python
# tests/test_metrics.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src.models import build_model
from src.metrics import evaluate_model

def test_evaluate_model():
    X = torch.randn(8, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)
    
    model = build_model("custom_cnn", num_classes=2, pretrained=False)
    results = evaluate_model(model, loader, device=torch.device("cpu"))
    
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "latency_ms" in results
    assert results["confusion_matrix"].shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Implement `src/metrics.py`**

Implement metric calculations using scikit-learn (`accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`), latency measurement using time benchmarks, and matplotlib/seaborn visualization functions.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: implement evaluation metrics and visualization helpers"
```

---

### Task 5: Complete Self-Contained Google Colab Notebook

**Files:**
- Create: `shrimp_freshness_classification.ipynb`
- Create: `README.md` (Colab execution instructions & guidelines)

**Interfaces:**
- Input: Google Drive path `/content/drive/MyDrive/shrimp_raw_jpg.zip`
- Output: Trained models saved to `/content/drive/MyDrive/shrimp_models/`, interactive Leaderboard table, Confusion Matrices, Loss curves, and Grad-CAM visualizer.

- [ ] **Step 1: Write generator / notebook structure for `shrimp_freshness_classification.ipynb`**

Include all 10 essential phases:
1. GPU Check & Google Drive Mount
2. Dataset Unzipping (`/content/drive/MyDrive/shrimp_raw_jpg.zip` -> `/content/dataset/`)
3. Data Pipeline & Stratified Split (`train`, `val`, `test` loaders)
4. Model Definitions (Custom CNN, MobileNetV3-Large, ResNet-50, EfficientNet-B0)
5. Training Loop with Early Stopping & Drive Checkpoint Persistence
6. Train 4 Models sequentially with progress reporting
7. Test Set Evaluation & Comparison Leaderboard DataFrame
8. Visualizations: Loss/Accuracy Curves & Confusion Matrices
9. Grad-CAM visual explanation on sample shrimp predictions
10. Download & Summary Guide

- [ ] **Step 2: Validate notebook syntax and JSON validity**

Verify the `.ipynb` is valid JSON and can be parsed by `nbformat`.

- [ ] **Step 3: Create `README.md` explaining exact step-by-step usage on Colab**

Provide clear Thai/English step-by-step instructions on how to upload the `.ipynb` to Google Drive or Colab, configure GPU runtime, verify Drive path, and interpret the outputs.

- [ ] **Step 4: Commit**

```bash
git add shrimp_freshness_classification.ipynb README.md
git commit -m "feat: add complete Colab notebook and execution guide"
```
