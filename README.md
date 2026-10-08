# 🦐 Shrimp Freshness AI Classification System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-29%2F29%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning system for automated shrimp freshness classification (`fresh` vs `not_fresh`). This project benchmarks **5 state-of-the-art architectures (4 CNNs + 1 Vision Transformer)** in PyTorch, resolves severe class imbalance using `WeightedRandomSampler`, and provides an interactive **Streamlit Web Application** equipped with **Explainable AI (Grad-CAM)** and real-time inference.

ระบบจำแนกและตรวจสอบคุณภาพความสดของกุ้งแบบอัตโนมัติด้วย Deep Learning (CNNs & Vision Transformer) พร้อมเว็บแอปพลิเคชัน Streamlit และระบบวิเคราะห์จุดตัดสินใจด้วย Grad-CAM

---

## 🌟 Key Features

* **🧠 Model Zoo (5 Distinct Architectures):**
  * **Custom CNN (Baseline):** 4-Block Convolutional Network trained from scratch (~2.1M params).
  * **MobileNetV3-Large:** Lightweight & ultra-fast for Edge IoT & Mobile devices (~5.4M params).
  * **ResNet-50:** Industry-standard residual learning with 50 layers (~25.6M params).
  * **EfficientNet-B0:** Compound scaling balancing accuracy and computational budget (~5.3M params).
  * **Vision Transformer (ViT-B/16):** Pure self-attention Transformer architecture (~86M params).
* **🖥️ Interactive Streamlit Web App (`app.py`):**
  * Modern **Ocean Blue UI** designed for maximum clarity and aesthetic appeal.
  * Enhanced **Prompt typography** providing optimal Thai and English legibility.
  * Full **Dynamic Theme Synchronization**: automatically switches text and card colors seamlessly between Light and Dark modes.
  * Real-time image upload (Drag & drop JPEG, PNG, WEBP).
  * Quick-select sample images for instant verification.
  * Model architecture selector with live latency benchmarking (ms).
* **🔍 Explainable AI Engine (`src/inference.py`):**
  * Built-in Grad-CAM visualizer support for CNN feature maps.
* **⚖️ Balanced Training Pipeline:**
  * Uses `WeightedRandomSampler` to enforce equal 50:50 sampling during training without artificial oversampling.
* **🧪 Robust Engineering & Testing:**
  * Modular architecture with **29 Unit and Integration Tests** passing 100% via `pytest`.

---

## 📁 Repository Structure

```
Shrimp_Classification/
├── app.py                      # 🌟 Main Streamlit Web Application
├── models/                     # 🧠 Model weights directory (.pth files)
│   ├── 5 models/               # (Place best_*.pth weights here)
│   └── README.md
├── sample_images/              # 🦐 Curated sample images for instant testing
│   ├── fresh/                  # Fresh shrimp samples
│   ├── not_fresh/              # Not fresh shrimp samples
│   └── README.md
├── notebooks/                  # 📓 Google Colab training notebooks & generators
│   ├── shrimp_freshness_classification.ipynb
│   └── generate_notebook.py
├── src/                        # 🛠️ Core Deep Learning modules
│   ├── dataset.py              # Stratified split, augmentation & weighted loaders
│   ├── models.py               # Model architectures & factory (build_model)
│   ├── trainer.py              # Early stopping, optimizer, lr-scheduler & checkpointing
│   ├── metrics.py              # Accuracy, F1, Latency & Confusion Matrix plotting
│   └── inference.py            # Inference engine, caching & Grad-CAM visualizer
├── tests/                      # 🧪 Test suite (29 tests, 100% pass)
│   ├── test_app.py             # Streamlit AppTest verification
│   ├── test_dataset.py         # Data loading and extraction tests
│   ├── test_e2e_real_models.py # End-to-end inference with real .pth weights
│   ├── test_inference.py       # Inference & Grad-CAM unit tests
│   ├── test_metrics.py         # Evaluation metrics tests
│   ├── test_models.py          # Forward-pass tensor shape tests
│   └── test_trainer.py         # Training loop & EarlyStopping tests
├── docs/                       # 📚 Design specs & Implementation plans
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # 📖 Project documentation (This file)
├── README_COLAB.md             # 📓 Original Colab execution guide
└── PROJECT_STATUS.md           # 📋 Project roadmap & status logs
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Shrimp_Classification.git
cd Shrimp_Classification
```

### 2. Set Up Virtual Environment

It is recommended to use Python 3.10, 3.11, or 3.12:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Weights Setup

Place the trained `.pth` model weights in `models/` or `models/5 models/`:

```
models/5 models/
├── best_resnet50.pth          # (~94 MB)
├── best_mobilenet_v3.pth      # (~17 MB)
├── best_efficientnet_b0.pth   # (~16 MB)
├── best_vit_b_16.pth          # (~343 MB)
└── best_custom_cnn.pth        # (~1.7 MB)
```

> **Note:** The web application includes an **automatic fallback**: if specific `.pth` files are missing, it initializes the base architecture for UI testing and loads weights seamlessly once available.

---

## 🖥️ How to Run

### 1. Launch the Streamlit Web Application

Ensure your virtual environment is active, then execute:

```bash
streamlit run app.py
```

* The web app will open automatically in your browser at `http://localhost:8501`.
* **Hardware Acceleration:** Automatically utilizes **Apple Silicon GPU (`mps`)** on macOS, **NVIDIA GPU (`cuda`)** on Linux/Windows, or falls back to **CPU**.

### 2. Train on Google Colab

To train or reproduce the benchmark models on Google Colab:
1. Upload [notebooks/shrimp_freshness_classification.ipynb](notebooks/shrimp_freshness_classification.ipynb) to Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Place `shrimp_raw_jpg.zip` in your Google Drive (`MyDrive/shrimp_raw_jpg.zip`).
4. Run all cells sequentially. Detailed instructions can be found in [README_COLAB.md](README_COLAB.md).

---

## 🧪 Running Automated Tests

Run the full automated test suite (Unit tests, UI tests, and Real-weight inference tests):

```bash
pytest tests/ -v
```

**Test Coverage Summary:**
* `tests/test_dataset.py`: Zip extraction, directory normalization, and Stratified splitting.
* `tests/test_models.py`: Architecture instantiation and forward-pass tensor validation `(B, 2)`.
* `tests/test_trainer.py`: Early stopping logic, single-epoch training, and history serialization.
* `tests/test_metrics.py`: Classification metrics (Accuracy, F1, Precision, Latency) and Confusion Matrices.
* `tests/test_inference.py`: Image preprocessing transforms, weight resolution, and Grad-CAM hooks.
* `tests/test_app.py`: Streamlit `AppTest` initial state and model selectbox rendering.
* `tests/test_e2e_real_models.py`: End-to-end integration tests using all 5 trained `.pth` models on real sample images.

---

## 📊 Model Benchmark Overview

| Architecture | Backbone Type | Parameters | Strengths / Best Use Case |
| :--- | :---: | :---: | :--- |
| **MobileNetV3-Large** | Inverted Residuals + SE | ~5.4M | **Fastest inference (<15 ms)**; ideal for edge cameras & mobile |
| **ResNet-50** | Residual Bottleneck | ~25.6M | **High stability & accuracy**; standard benchmark |
| **EfficientNet-B0** | Compound MBConv | ~5.3M | Excellent parameter efficiency and feature representation |
| **ViT-B/16** | Self-Attention Transformer | ~86.0M | Modern vision backbone capturing long-range spatial correlations |
| **Custom CNN** | 4 Conv Blocks (Scratch) | ~2.1M | Lightweight baseline without transfer learning |

---

## 💡 Note on Domain Shift & In-The-Wild Images

* Deep learning models evaluate with maximum accuracy on **In-Distribution images** (e.g. raw shrimp captured under consistent lighting and plain backgrounds similar to the training dataset).
* When testing with arbitrary internet images (e.g., cooked orange shrimp, market scenes with ice/condiments, or varied angles), performance may be affected by **Domain Shift (Shortcut Learning)**.
* For optimal real-world evaluation, place raw shrimp on neutral, uniform backgrounds with top-down angles, or use sample images provided in [sample_images/](sample_images/).

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
