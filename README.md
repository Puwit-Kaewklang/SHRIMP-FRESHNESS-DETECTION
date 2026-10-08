# 🦐 Shrimp Freshness CNN & Transformer Classification System
ระบบจำแนกภาพกุ้งสด / กุ้งไม่สด ด้วยสถาปัตยกรรม CNN และ Vision Transformer 5 โมเดล (PyTorch & Google Colab)

---

## 📋 ภาพรวมโครงการ (Project Overview)
โปรเจกต์นี้พัฒนาขึ้นเพื่อเปรียบเทียบประสิทธิภาพของสถาปัตยกรรม Deep Learning จำนวน 5 โมเดล (4 CNNs + 1 Vision Transformer) ในการจำแนกความสดของกุ้ง (`fresh` vs `not_fresh`):

1. **Custom CNN:** โมเดล 4 Convolutional Blocks สร้างขึ้นเอง เทรนจาก Scratch เพื่อใช้เป็น Baseline
2. **MobileNetV3-Large:** โมเดลขนาดเล็กและประมวลผลเร็ว (Transfer Learning) เหมาะสำหรับ Edge AI / Smart Camera
3. **ResNet-50:** โมเดล Residual Connection ที่เป็นมาตรฐานสากลในงานวิจัยคอมพิวเตอร์วิทัศน์
4. **EfficientNet-B0:** โมเดล Compound Scaling ที่ให้ความแม่นยำสูงต่อน้ำหนักโมเดล
5. **Vision Transformer (ViT-B/16):** โมเดล Transformer สถาปัตยกรรม Self-Attention ระดับ State-of-the-art

### ⚖️ การจัดการ Class Imbalance:
* จัดการความไม่สมดุลของข้อมูล (`fresh` 886 ภาพ vs `not_fresh` 509 ภาพ) โดยใช้ **`WeightedRandomSampler`** ในชุด Train สุ่มตัวอย่างให้มีสัดส่วน 50:50 เท่ากัน
* คงชุด Validation และ Test ไว้ตามสัดส่วนจริง เพื่อการวัดผลที่เที่ยงตรง ไม่เกิด Data Leakage หรือการประเมินที่หลอกตา

---

## 📁 โครงสร้างไฟล์ในโปรเจกต์ (Repository Structure)

```
Shrimp_Classification/
├── app.py                      # 🌟 หน้าเว็บแอปพลิเคชัน Streamlit (Phase 2)
├── models/                     # 🧠 โฟลเดอร์สำหรับวางไฟล์ .pth ทั้ง 5 โมเดล
│   └── README.md
├── sample_images/              # 🦐 ภาพตัวอย่างสำหรับทดสอบระบบบนเว็บ
│   ├── fresh/
│   ├── not_fresh/
│   └── README.md
├── notebooks/                  # 📓 Colab Notebook และสคริปต์ที่เกี่ยวข้อง
│   ├── shrimp_freshness_classification.ipynb
│   └── generate_notebook.py
├── src/                        # 🛠️ โมดูลระบบ Deep Learning
│   ├── dataset.py              # Data Pipeline, WeightedRandomSampler & Transforms
│   ├── models.py               # สถาปัตยกรรม 5 โมเดล (CNNs & ViT)
│   ├── trainer.py              # Training Engine, Early Stopping & Checkpointing
│   └── metrics.py              # Metrics (Acc, F1, Latency) & Visualization Helpers
├── tests/                      # 🧪 Unit Tests (TDD - 12/12 ผ่านทั้งหมด)
├── docs/                       # 📚 เอกสาร Specification และบันทึกแผนงาน
├── requirements.txt            # รายการ Dependencies
├── README.md                   # คู่มือการใช้งานฉบับสมบูรณ์
└── PROJECT_STATUS.md           # บันทึกสถานะโปรเจกต์และ Roadmap
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
4. **รันเซลล์ที่ 6:** โหลดสถาปัตยกรรมทั้ง 5 โมเดล (Custom CNN, MobileNetV3, ResNet-50, EfficientNet-B0, ViT-B/16)
5. **รันเซลล์ที่ 7 & 8:** เริ่มเทรนทั้ง 5 โมเดลตามลำดับ (มีระบบ Smart Resume โหลดโมเดลเดิมที่เคยเทรนเสร็จแล้วอัตโนมัติ)
6. **รันเซลล์ที่ 9 & 10:** ดูตารางเปรียบเทียบ (Leaderboard CSV), กราฟ Loss/Accuracy, และ Confusion Matrices Heatmaps
7. **รันเซลล์ที่ 11:** แสดงผลลัพธ์ **Grad-CAM** เพื่อดูว่า AI มองจุดไหนของตัวกุ้งในการตัดสินใจว่าสดหรือไม่สด

---

## 💾 ไฟล์ผลลัพธ์ที่ถูกบันทึกลง Google Drive อัตโนมัติ

ผลลัพธ์ทั้งหมดจะถูกบันทึกไว้ในโฟลเดอร์ `MyDrive/shrimp_models/` ใน Google Drive ของคุณ:
- `best_custom_cnn.pth` & `history_custom_cnn.json`
- `best_mobilenet_v3.pth` & `history_mobilenet_v3.json`
- `best_resnet50.pth` & `history_resnet50.json`
- `best_efficientnet_b0.pth` & `history_efficientnet_b0.json`
- `best_vit_b_16.pth` & `history_vit_b_16.json`
- `shrimp_models_leaderboard.csv`: ตารางสรุปผลเปรียบเทียบโมเดลในรูปแบบ CSV (นำไปเปิดใน Excel ได้ทันที)
- `validation_curves.png`: ภาพกราฟ Loss และ Accuracy เปรียบเทียบ
- `confusion_matrices.png`: ภาพ Confusion Matrix 5 โมเดลสำหรับใส่รายงาน

---

## 🖥️ การรันเว็บแอปพลิเคชัน (Streamlit Web App)
สามารถสั่งรันหน้าเว็บทดสอบการจำแนกความสดของกุ้งในเครื่อง Local ได้ด้วยคำสั่ง:

```bash
source .venv/bin/activate
streamlit run app.py
```
* **ฟีเจอร์เด่น:**
  * เลือกสลับทดสอบได้ทั้ง 5 โมเดล (ResNet-50, MobileNetV3, EfficientNet-B0, ViT, Custom CNN)
  * อัปโหลดภาพกุ้งหรือคลิกภาพตัวอย่างจากเมนูได้ทันที
  * แสดงผลระดับความสด (กุ้งสด/ไม่สด), % ความมั่นใจ, และ Latency
  * เปิด-ปิดแสดง **Heatmap (Grad-CAM)** เพื่อดูจุดที่โมเดลใช้ตัดสินใจ

---

## 🧪 การรัน Unit & Integration Tests ในเครื่อง Local
```bash
source .venv/bin/activate
pytest tests/ -v
```
ผลลัพธ์: Tests ทั้งหมดผ่าน 100% (**29/29 Green Suite** ครอบคลุมทั้ง Data Pipeline, Models, Trainer, Inference, Grad-CAM, Real Weights E2E, และ Streamlit UI).
