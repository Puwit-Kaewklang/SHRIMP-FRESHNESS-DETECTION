# 🦐 Shrimp Freshness CNN Classification System
ระบบจำแนกภาพกุ้งสด / กุ้งไม่สด ด้วยสถาปัตยกรรม CNN 4 โมเดล (PyTorch & Google Colab)

---

## 📋 ภาพรวมโครงการ (Project Overview)
โปรเจกต์นี้พัฒนาขึ้นเพื่อเปรียบเทียบประสิทธิภาพของสถาปัตยกรรม Deep Learning Convolutional Neural Networks (CNN) จำนวน 4 โมเดล ในการจำแนกความสดของกุ้ง (`fresh` vs `not_fresh`):

1. **Custom CNN:** โมเดล 4 Convolutional Blocks สร้างขึ้นเอง เทรนจาก Scratch เพื่อใช้เป็น Baseline
2. **MobileNetV3-Large:** โมเดลขนาดเล็กและประมวลผลเร็ว (Transfer Learning) เหมาะสำหรับ Edge AI / Smart Camera
3. **ResNet-50:** โมเดล Residual Connection ที่เป็นมาตรฐานสากลในงานวิจัยคอมพิวเตอร์วิทัศน์
4. **EfficientNet-B0:** โมเดล Compound Scaling ที่ให้ความแม่นยำสูงต่อน้ำหนักโมเดล

---

## 📁 โครงสร้างไฟล์ในโปรเจกต์ (Repository Structure)

```
shrimp-freshness-cnn/
├── shrimp_freshness_classification.ipynb   # 🌟 Colab Notebook ตัวเต็มสำหรับรันบน Google Colab
├── requirements.txt                        # รายการ Dependencies สำหรับรันใน Local
├── README.md                               # คู่มือการใช้งานฉบับสมบูรณ์
├── src/                                    # โมดูล Python สำหรับพัฒนาและทดสอบ
│   ├── dataset.py                          # Data Pipeline, Unzip, Stratified Split & Transforms
│   ├── models.py                           # สถาปัตยกรรมโมเดลและ Model Factory
│   ├── trainer.py                          # Training Loop, Early Stopping & Checkpointing
│   └── metrics.py                          # Metrics (Acc, F1, Latency) & Visualization Helpers
├── tests/                                  # Unit Tests (TDD)
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_trainer.py
│   └── test_metrics.py
└── docs/
    ├── specs/                              # เอกสาร Design Specification
    └── superpowers/plans/                  # เอกสาร Implementation Plan
```

---

## 🚀 วิธีการนำไปรันบน Google Colab (Step-by-Step Guide)

### ขั้นตอนที่ 1: ตรวจสอบไฟล์ Dataset บน Google Drive
ตรวจสอบให้แน่ใจว่าคุณได้วางไฟล์ zip ไว้ที่:
```
ไดรฟ์ของฉัน/shrimp_raw_jpg.zip
```
(หรือ `/content/drive/MyDrive/shrimp_raw_jpg.zip` ใน Colab) โดยข้างในมีโฟลเดอร์ `fresh` และ `not_fresh`

### ขั้นตอนที่ 2: อัปโหลดและเปิด Notebook บน Google Colab
1. เปิดเว็บไซต์ [Google Colab](https://colab.research.google.com/)
2. คลิก **Upload** แล้วเลือกไฟล์ `shrimp_freshness_classification.ipynb` จากเครื่องของคุณ
3. เปลี่ยน Runtime เป็น GPU:
   * ไปที่เมนู **Runtime** $\rightarrow$ **Change runtime type**
   * เลือก **T4 GPU** แล้วกด **Save**

### ขั้นตอนที่ 3: รันโค้ดตามลำดับเซลล์
1. **รันเซลล์ที่ 1 & 2:** เพื่อตรวจสอบ GPU และเชื่อมต่อ Google Drive (กดปุ่ม Connect to Google Drive เมื่อมีหน้าต่างปรากฏ)
2. **รันเซลล์ที่ 3:** โค้ดจะคัดลอกและแตกไฟล์ zip ไปยัง Local SSD (`/content/dataset/`) โดยอัตโนมัติ ช่วยลดปัญหา I/O Bottleneck ทำให้เทรนเร็วขึ้น 5-10 เท่า
3. **รันเซลล์ที่ 4 & 5:** ตรวจสอบภาพ และแบ่งข้อมูลแบบ Stratified Split (Train 70% / Val 15% / Test 15%) ด้วย `seed=42`
4. **รันเซลล์ที่ 6:** โหลดสถาปัตยกรรมทั้ง 4 โมเดล
5. **รันเซลล์ที่ 7 & 8:** เริ่มเทรนทั้ง 4 โมเดลตามลำดับ ระบบจะหยุดอัตโนมัติหาก Val Loss ไม่ดีขึ้น (Early Stopping) และบันทึกเฉพาะโมเดลที่ดีที่สุดลง Google Drive
6. **รันเซลล์ที่ 9 & 10:** ดูตารางเปรียบเทียบ (Leaderboard), กราฟ Loss/Accuracy, และ Confusion Matrices Heatmaps
7. **รันเซลล์ที่ 11:** แสดงผลลัพธ์ **Grad-CAM** เพื่อดูว่า AI มองจุดไหนของตัวกุ้งในการตัดสินใจว่าสดหรือไม่สด

---

## 💾 ไฟล์ผลลัพธ์ที่ถูกบันทึกลง Google Drive อัตโนมัติ

ผลลัพธ์ทั้งหมดจะถูกบันทึกไว้ในโฟลเดอร์ `MyDrive/shrimp_models/` ใน Google Drive ของคุณ:
- `best_custom_cnn.pth`: Weights โมเดล Custom CNN ที่ดีที่สุด
- `best_mobilenet_v3.pth`: Weights โมเดล MobileNetV3 ที่ดีที่สุด
- `best_resnet50.pth`: Weights โมเดล ResNet-50 ที่ดีที่สุด
- `best_efficientnet_b0.pth`: Weights โมเดล EfficientNet-B0 ที่ดีที่สุด
- `shrimp_models_leaderboard.csv`: ตารางสรุปผลเปรียบเทียบโมเดลในรูปแบบ CSV (นำไปเปิดใน Excel ได้ทันที)
- `validation_curves.png`: ภาพกราฟ Loss และ Accuracy เปรียบเทียบ
- `confusion_matrices.png`: ภาพ Confusion Matrix 4 โมเดลสำหรับใส่รายงาน

---

## 🧪 การรัน Unit Tests ในเครื่อง Local
```bash
source .venv/bin/activate
pytest tests/ -v
```
ผลลัพธ์: Tests ทั้งหมดผ่าน 100% (Green Suite).
