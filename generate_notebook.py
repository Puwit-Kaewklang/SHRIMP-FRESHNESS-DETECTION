import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "shrimp_freshness_classification.ipynb",
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_md(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# ----------------- CELL 1: HEADER -----------------
add_md("""# 🦐 Deep Learning Pipeline: จำแนกภาพกุ้งสด / ไม่สด ด้วย CNN บน Google Colab

**เปรียบเทียบประสิทธิภาพ 4 สถาปัตยกรรม CNN:**
1. **Custom CNN:** Baseline โมเดล 4 Convolutional blocks เทรนจากศูนย์ (Scratch)
2. **MobileNetV3-Large:** Lightweight & Fast (Transfer Learning) เหมาะสำหรับ Edge Device
3. **ResNet-50:** Residual Network มาตรฐานงานวิจัยคอมพิวเตอร์วิทัศน์
4. **EfficientNet-B0:** Compound Scaling ให้ความแม่นยำสูงต่อน้ำหนักโมเดล

---
### ⚙️ ขั้นตอนการเตรียมก่อนรัน (Pre-flight Checklist):
1. ไปที่เมนู **Runtime** -> **Change runtime type** -> เลือก **T4 GPU**
2. ตรวจสอบว่าไฟล์ `shrimp_raw_jpg.zip` อยู่ใน Google Drive ที่:
   `MyDrive/shrimp_raw_jpg.zip`
""")

# ----------------- CELL 2: GPU CHECK & DRIVE MOUNT -----------------
add_md("""## 1. ตรวจสอบ GPU และเชื่อมต่อ Google Drive (Mount Drive)""")
add_code("""# ตรวจสอบการเชื่อมต่อ GPU
!nvidia-smi

# เชื่อมต่อ Google Drive
from google.colab import drive
drive.mount('/content/drive')
""")

# ----------------- CELL 3: EXTRACT DATASET -----------------
add_md("""## 2. คัดลอกและแตกไฟล์ Dataset สู่ Local SSD
> **เทคนิคสำคัญ:** การอ่านไฟล์ภาพจาก Google Drive โดยตรงจะช้ามาก (เกิด I/O Bottleneck)
> การแตกไฟล์ zip ลงใน Local SSD ของ Colab (`/content/dataset/`) จะทำให้การเทรนเร็วขึ้น 5-10 เท่า""")
add_code("""import os
import zipfile
import shutil

DATA_PATH = '/content/drive/MyDrive/shrimp_raw_jpg.zip'
EXTRACT_DIR = '/content/dataset'

# ตรวจสอบว่ามีไฟล์หรือโฟลเดอร์อยู่ใน Drive หรือไม่
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์หรือโฟลเดอร์ที่ {DATA_PATH} กรุณาตรวจสอบว่าชื่อไฟล์และโฟลเดอร์ใน Google Drive ตรงกัน")

if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)

if os.path.isdir(DATA_PATH):
    print(f"📁 ตรวจพบว่า {DATA_PATH} เป็นโฟลเดอร์ (Folder)")
    print(f"🚀 กำลังคัดลอกไฟล์มายัง Local SSD: {EXTRACT_DIR}...")
    shutil.copytree(DATA_PATH, EXTRACT_DIR)
else:
    print(f"📦 ตรวจพบว่า {DATA_PATH} เป็นไฟล์บีบอัด (.zip)")
    print(f"🚀 กำลังแตกไฟล์มายัง Local SSD: {EXTRACT_DIR}...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(DATA_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# ค้นหาโฟลเดอร์ที่มี fresh และ not_fresh
dataset_root = None
for root, dirs, _ in os.walk(EXTRACT_DIR):
    dir_set = set(d.lower() for d in dirs)
    if 'fresh' in dir_set and 'not_fresh' in dir_set:
        dataset_root = root
        break

if not dataset_root:
    # ตรวจสอบว่าอยู่ใน EXTRACT_DIR โดยตรงหรือไม่
    if os.path.isdir(os.path.join(EXTRACT_DIR, 'fresh')) and os.path.isdir(os.path.join(EXTRACT_DIR, 'not_fresh')):
        dataset_root = EXTRACT_DIR
    else:
        raise ValueError(f"❌ ไม่พบโฟลเดอร์ 'fresh' และ 'not_fresh' ใน {EXTRACT_DIR}")

print(f"✅ เตรียม Dataset สำเร็จ! Dataset Root Path: {dataset_root}")
""")

# ----------------- CELL 4: DATA VERIFICATION & DISPLAY -----------------
add_md("""## 3. ตรวจสอบจำนวนภาพและแสดงตัวอย่างภาพกุ้ง""")
add_code("""from PIL import Image
import matplotlib.pyplot as plt
import glob

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def get_image_files(folder):
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if f.lower().endswith(VALID_EXTS) and not f.startswith('.'):
                files.append(os.path.join(root, f))
    return sorted(files)

fresh_files = get_image_files(os.path.join(dataset_root, 'fresh'))
not_fresh_files = get_image_files(os.path.join(dataset_root, 'not_fresh'))

print(f"📊 สรุปจำนวนภาพในชุดข้อมูล:")
print(f" - ภาพกุ้งสด (fresh): {len(fresh_files)} ภาพ")
print(f" - ภาพกุ้งไม่สด (not_fresh): {len(not_fresh_files)} ภาพ")
print(f" - รวมทั้งหมด: {len(fresh_files) + len(not_fresh_files)} ภาพ")

# แสดงตัวอย่างภาพ
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i in range(4):
    if i < len(fresh_files):
        img = Image.open(fresh_files[i])
        axes[0, i].imshow(img)
        axes[0, i].set_title("Fresh (กุ้งสด)", color="green", fontweight="bold")
        axes[0, i].axis("off")
        
    if i < len(not_fresh_files):
        img = Image.open(not_fresh_files[i])
        axes[1, i].imshow(img)
        axes[1, i].set_title("Not Fresh (ไม่สด)", color="red", fontweight="bold")
        axes[1, i].axis("off")

plt.suptitle("ตัวอย่างภาพใน Dataset", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
""")

# ----------------- CELL 5: DATA LOADER & SPLITTING -----------------
add_md("""## 4. แบ่งข้อมูลแบบ Stratified Split (Train 70% / Val 15% / Test 15%) & Data Augmentation
> ใช้ `random_state=42` เพื่อรับประกันว่าทุกโมเดลจะได้รับการเทรนและประเมินผลบนข้อมูลชุดเดียวกัน 100%""")
add_code("""import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class ShrimpDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# รวบรวมข้อมูลทั้งหมด (Label: 0 = fresh, 1 = not_fresh)
all_samples = [(f, 0) for f in fresh_files] + [(f, 1) for f in not_fresh_files]
all_labels = [s[1] for s in all_samples]
CLASS_NAMES = ['fresh', 'not_fresh']

# แบ่ง 70% Train, 15% Val, 15% Test
train_samples, temp_samples, _, temp_labels = train_test_split(
    all_samples, all_labels, test_size=0.30, random_state=42, stratify=all_labels
)
val_samples, test_samples = train_test_split(
    temp_samples, test_size=0.50, random_state=42, stratify=temp_labels
)

print(f"📈 จำนวนข้อมูลแต่ละส่วน:")
print(f" - Train: {len(train_samples)} ภาพ (70%)")
print(f" - Validation: {len(val_samples)} ภาพ (15%)")
print(f" - Test: {len(test_samples)} ภาพ (15%)")

# Data Augmentation & Preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 32
train_loader = DataLoader(ShrimpDataset(train_samples, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(ShrimpDataset(val_samples, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(ShrimpDataset(test_samples, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
""")

# ----------------- CELL 6: MODEL DEFINITIONS -----------------
add_md("""## 5. กำหนดสถาปัตยกรรมทั้ง 4 โมเดล (Custom CNN, MobileNetV3, ResNet-50, EfficientNet-B0)""")
add_code("""import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights
)

class CustomCNN(nn.Module):
    \"\"\"Baseline Custom CNN ออกแบบเอง 4 Conv Blocks\"\"\"
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))

def build_model(model_name, num_classes=2):
    name = model_name.lower()
    if name == 'custom_cnn':
        return CustomCNN(num_classes)
    elif name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        in_feat = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feat, num_classes)
        )
        return model
    elif name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feat, num_classes)
        )
        return model
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feat, num_classes)
        )
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print("✅ Model Factory พร้อมใช้งาน! โมเดลที่รองรับ:")
for m in ['custom_cnn', 'mobilenet_v3', 'resnet50', 'efficientnet_b0']:
    mod = build_model(m)
    print(f" - {m:15s}: {count_params(mod):,} parameters")
""")

# ----------------- CELL 7: TRAINING & CHECKPOINT ENGINE -----------------
add_md("""## 6. ฟังก์ชันเทรนโมเดล พร้อม Early Stopping และบันทึก Checkpoint ลง Google Drive""")
add_code("""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")

# โฟลเดอร์สำหรับบันทึกโมเดลลง Google Drive โดยตรง
CHECKPOINT_DIR = '/content/drive/MyDrive/shrimp_models'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

def train_one_model(model_name, epochs=20, lr=1e-4):
    print(f"\\n{'='*60}")
    print(f"🔥 เริ่มเทรนโมเดล: {model_name.upper()} (Epochs: {epochs}, LR: {lr})")
    print(f"{'='*60}")
    
    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=5)
    
    save_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.pth")
    best_val_loss = float('inf')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        # 1. Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += torch.sum(preds == targets).item()
            train_total += targets.size(0)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        scheduler.step()
        
        # 2. Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += torch.sum(preds == targets).item()
                val_total += targets.size(0)
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Checkpoint: บันทึกเฉพาะรอบที่ Val Loss ต่ำที่สุด
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f" ⭐ บันทึก Best Model -> {save_path}")
            
        if early_stopping(val_loss):
            print(f" 🛑 Early Stopping ทำงานที่ Epoch {epoch} เพื่อป้องกัน Overfitting")
            break
            
    # โหลด weights ที่ดีที่สุดกลับมา
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    # บันทึก Training History ลง Google Drive เพื่อป้องกัน Session หลุด
    import json
    history_save_path = os.path.join(CHECKPOINT_DIR, f"history_{model_name}.json")
    with open(history_save_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f" 📜 บันทึก Training History -> {history_save_path}")
    
    return model, history
""")

# ----------------- CELL 8: TRAIN ALL 4 MODELS -----------------
add_md("""## 7. รันการเทรนทั้ง 4 โมเดลตามลำดับ
*(ใช้เวลาประมาณ 15-30 นาที ขึ้นอยู่กับจำนวนภาพ)*""")
add_code("""MODELS_TO_TRAIN = [
    ('custom_cnn', 20, 1e-3),       # Custom CNN ใช้ LR 1e-3
    ('mobilenet_v3', 20, 1e-4),     # Transfer learning ใช้ LR 1e-4
    ('resnet50', 20, 1e-4),
    ('efficientnet_b0', 20, 1e-4)
]

trained_models = {}
histories = {}

for name, eps, lr_val in MODELS_TO_TRAIN:
    mod, hist = train_one_model(name, epochs=eps, lr=lr_val)
    trained_models[name] = mod
    histories[name] = hist
    # เคลียร์ Cache GPU Memory ป้องกัน memory สะสม
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\\n🎉 เทรนโมเดลครบทั้ง 4 ตัวเรียบร้อยแล้ว!")
""")

# ----------------- CELL 9: EVALUATION & LEADERBOARD -----------------
add_md("""## 8. การประเมินผลบน Test Set & ตารางเปรียบเทียบ (Leaderboard Table)""")
add_code("""import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_on_test(model):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    start_time = time.perf_counter()
    total_imgs = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            total_imgs += inputs.size(0)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(targets.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_time) * 1000.0  # ms
    latency = total_time / max(total_imgs, 1)
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'latency_ms': latency,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }

eval_results = {}
for name, mod in trained_models.items():
    eval_results[name] = evaluate_on_test(mod)

# สร้าง Leaderboard สรุปผล
summary_data = []
for name, res in eval_results.items():
    mod = trained_models[name]
    summary_data.append({
        'Model Name': name,
        'Parameters (M)': f"{count_params(mod)/1e6:.2f}M",
        'Test Accuracy': f"{res['accuracy']*100:.2f}%",
        'Precision (Macro)': f"{res['precision']*100:.2f}%",
        'Recall (Macro)': f"{res['recall']*100:.2f}%",
        'F1-Score': f"{res['f1']*100:.2f}%",
        'Latency (ms/image)': f"{res['latency_ms']:.2f} ms"
    })

leaderboard_df = pd.DataFrame(summary_data)
# บันทึกตารางสรุปผลเป็น CSV ลง Drive
leaderboard_df.to_csv(os.path.join(CHECKPOINT_DIR, "shrimp_models_leaderboard.csv"), index=False)

print("🏆 ตารางสรุปผลการเปรียบเทียบทั้ง 4 โมเดล (Test Set):")
display(leaderboard_df)
""")

# ----------------- CELL 10: VISUALIZATION (LOSS/ACC & CONFUSION MATRIX) -----------------
add_md("""## 9. พล็อตกราฟเปรียบเทียบ & Confusion Matrices Heatmaps""")
add_code("""import seaborn as sns

# 1. พล็อตกราฟ Loss และ Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
for name, h in histories.items():
    epochs_range = range(1, len(h['val_loss']) + 1)
    ax1.plot(epochs_range, h['val_loss'], label=f"{name}")
    ax2.plot(epochs_range, h['val_acc'], label=f"{name}")

ax1.set_title("Validation Loss Comparison Across Models", fontsize=13, fontweight='bold')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.set_title("Validation Accuracy Comparison Across Models", fontsize=13, fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "validation_curves.png"), dpi=300)
plt.show()

# 2. พล็อต Confusion Matrices ของทั้ง 4 โมเดล
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, res) in enumerate(eval_results.items()):
    ax = axes[idx]
    cm = res['confusion_matrix']
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Fresh', 'Not Fresh'],
        yticklabels=['Fresh', 'Not Fresh'],
        ax=ax, cbar=False
    )
    ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.suptitle("Confusion Matrices Comparison (Test Set)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "confusion_matrices.png"), dpi=300)
plt.show()
""")

# ----------------- CELL 11: GRAD-CAM EXPLAINABILITY -----------------
add_md("""## 10. AI Explainability: พล็อต Grad-CAM Heatmap
> แสดงให้เห็นว่าโมเดลมองดูที่จุดไหนของตัวกุ้งในการตัดสินว่า "สด" หรือ "ไม่สด" (เช่น บริเวณเปลือก หัว หรือเนื้อกุ้ง)""")
add_code("""import cv2

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx

# สุ่มทดสอบ Grad-CAM บน ResNet-50
target_layer = trained_models['resnet50'].layer4[-1]
cam_generator = SimpleGradCAM(trained_models['resnet50'], target_layer)

sample_img_tensor, sample_label = test_loader.dataset[0]
input_tensor = sample_img_tensor.unsqueeze(0).to(device)
cam_map, pred_class = cam_generator.generate(input_tensor)

# แปลงภาพกลับมาเป็น RGB ปกติเพื่อแสดงผล
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
orig_img = inv_normalize(sample_img_tensor).permute(1, 2, 0).cpu().numpy()
orig_img = np.clip(orig_img, 0, 1)

heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
overlay = 0.6 * orig_img + 0.4 * heatmap[:, :, ::-1]
overlay = np.clip(overlay, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(orig_img)
axes[0].set_title(f"Original Image (True: {CLASS_NAMES[sample_label]})", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cam_map, cmap='jet')
axes[1].set_title("Grad-CAM Heatmap", fontweight='bold')
axes[1].axis('off')

axes[2].imshow(overlay)
axes[2].set_title(f"Overlay (Predicted: {CLASS_NAMES[pred_class]})", fontweight='bold')
axes[2].axis('off')

plt.suptitle("ResNet-50 Grad-CAM Visual Attention", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
""")

# ----------------- CELL 12: PREDICT CUSTOM IMAGE -----------------
add_md("""## 11. ฟังก์ชันทดสอบทำนายภาพกุ้งภาพเดี่ยว (Single Image Inference)""")
add_code("""def predict_shrimp_freshness(image_path, model_name='resnet50'):
    \"\"\"ฟังก์ชันสำหรับนำภาพกุ้งภาพใหม่มาทดสอบทำนายความสด\"\"\"
    model = trained_models[model_name]
    model.eval()
    
    img = Image.open(image_path).convert('RGB')
    tensor_img = eval_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_img)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        
    print(f"🦐 ผลการทำนายด้วยโมเดล: {model_name}")
    print(f" -> ผลลัพธ์: {CLASS_NAMES[pred_idx].upper()}")
    print(f" -> ความมั่นใจ (กุ้งสด): {probs[0]*100:.2f}%")
    print(f" -> ความมั่นใจ (กุ้งไม่สด): {probs[1]*100:.2f}%")

# ตัวอย่างการเรียกใช้:
# predict_shrimp_freshness('/content/dataset/fresh/some_sample.jpg', model_name='resnet50')
""")

# Save notebook
out_path = "/Users/macswit/.gemini/antigravity/scratch/shrimp-freshness-cnn/shrimp_freshness_classification.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook created successfully at: {out_path}")
