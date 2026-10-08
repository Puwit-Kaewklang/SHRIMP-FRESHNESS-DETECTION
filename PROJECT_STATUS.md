# 🦐 Shrimp Freshness AI Project Status & Roadmap

> **สถานะปัจจุบัน:** ✅ **Phase 1 เสร็จสมบูรณ์ 100% (Model Training, Pipeline & Evaluation)**  
> **เป้าหมายถัดไป:** 🚀 **Phase 2 (Streamlit Web Application Development)**  
> **อัปเดตล่าสุด:** 2026-10-08

---

## 📌 สรุปความคืบหน้าภาพรวม (Milestones Summary)

| ระยะงาน (Phase) | รายละเอียด | สถานะ | สิ่งที่ส่งมอบ (Deliverables) |
| :--- | :--- | :---: | :--- |
| **Phase 1: Model & Pipeline** | เทรนและเปรียบเทียบโมเดล 5 สถาปัตยกรรมบน Google Colab พร้อมแก้ปัญหา Imbalance และ Pipeline ทั้งหมด | ✅ **เสร็จสมบูรณ์** | `shrimp_freshness_classification.ipynb`, โมดูล `src/`, Unit Tests 12/12 ผ่าน |
| **Phase 2: Web Application** | สร้างเว็บแอป Streamlit สำหรับอัปโหลดภาพกุ้ง ทำนายความสด (สด/ไม่สด) แบบ Real-time พร้อม Grad-CAM | ⏳ **พร้อมเริ่มดำเนินการ** | `app.py`, โฟลเดอร์ `models/`, ทดสอบ Local & Cloud |
| **Phase 3: Deployment & Report** | นำเสนอผลสรุป (Leaderboard, Confusion Matrices) และ Deploy เว็บแอปขึ้น Cloud | 📋 **วางแผนแล้ว** | สรุปรายงานวิจัย/โปรเจกต์ + ลิงก์เว็บแอปใช้งานจริง |

---

## 🧠 สถาปัตยกรรมทั้ง 5 โมเดลที่พัฒนา (Model Zoo)

1. **Custom CNN (Baseline):** 4 Convolutional Blocks ออกแบบเอง เทรนจาก Scratch (~2.1M params) เพื่อดูค่าอ้างอิง
2. **MobileNetV3-Large:** Lightweight & Fast Transfer Learning (~5.4M params) เหมาะกับ Mobile/Edge IoT
3. **ResNet-50:** Residual Connection 50 Layers (~25.6M params) มาตรฐานอ้างอิงสากล
4. **EfficientNet-B0:** Compound Scaling (~5.3M params) ความแม่นยำสูงต่อน้ำหนักโมเดล
5. **Vision Transformer (ViT-B/16):** Pure Self-Attention Transformer (~86M params) สถาปัตยกรรมยุคใหม่

---

## 🛠️ บันทึกประวัติการแก้ปัญหาทางเทคนิค (Engineering Log)

1. **การแก้ปัญหา I/O Bottleneck บน Colab:**
   * ดึงภาพจาก Google Drive มาแคชไว้ใน Local SSD (`/content/dataset/`) ช่วยเร่งความเร็วการเทรน 5–10 เท่า
2. **การแก้ปัญหา `IsADirectoryError`:**
   * เขียนระบบตรวจสอบอัตโนมัติ: หาก `shrimp_raw_jpg.zip` ใน Drive เป็นโฟลเดอร์จะสั่ง `copytree` แต่ถ้าเป็นไฟล์ zip จะสั่ง `extractall`
3. **การจัดการ Class Imbalance (กุ้งสด 886 ภาพ vs กุ้งไม่สด 509 ภาพ):**
   * ใช้ **`WeightedRandomSampler`** ในชุด Train เพื่อสุ่มข้อมูลทั้งสองคลาสมาเทรนในอัตราส่วน **50% : 50% เท่ากัน** โดยไม่เปลืองพื้นที่สร้างไฟล์ใน Drive
   * คงชุด Validation และ Test ไว้ตามสัดส่วนจริง (30%) เพื่อความเที่ยงตรงในการวัดผล
4. **การแก้ปัญหา Colab RAM OOM (`DataLoader worker is killed by signal: Killed`):**
   * เปลี่ยน `num_workers=0` และเพิ่มระบบล้างแรม `gc.collect()` + `torch.cuda.empty_cache()` ป้องกัน Subprocess โดน SIGKILL จาก Linux Kernel เมื่อเทรนโมเดลใหญ่อย่าง ViT
5. **ระบบ Smart Checkpoint Resume:**
   * ตรวจสอบไฟล์ `.pth` และ `history_<model>.json` ใน Google Drive (`MyDrive/shrimp_models/`) หากโมเดลใดเทรนเสร็จแล้ว จะโหลดมาใช้ทันที ไม่ต้องเสียเวลารันซ้ำ

---

## 💾 ที่ตั้งของไฟล์โมเดลที่เทรนเสร็จแล้ว (Artifacts on Google Drive)

ไฟล์น้ำหนักโมเดลทั้งหมดถูกบันทึกไว้ใน Google Drive ที่โฟลเดอร์:  
📂 **`MyDrive/shrimp_models/`**
* `best_custom_cnn.pth` & `history_custom_cnn.json`
* `best_mobilenet_v3.pth` & `history_mobilenet_v3.json`
* `best_resnet50.pth` & `history_resnet50.json`
* `best_efficientnet_b0.pth` & `history_efficientnet_b0.json`
* `best_vit_b_16.pth` & `history_vit_b_16.json`
* `shrimp_models_leaderboard.csv` (ตารางเปรียบเทียบผล Test Set)
* `validation_curves.png` & `confusion_matrices.png`

---

## 🚀 แผนการทำงาน Phase 2: พัฒนา Streamlit Web App (Next Action Plan)

เมื่อย้ายโฟลเดอร์โปรเจกต์ไปยังตำแหน่งใหม่แล้ว จะเริ่มพัฒนา Web Application ทันทีตามขั้นตอนดังนี้:

### 1. โครงสร้างไฟล์ในโปรเจกต์:
```
Shrimp_Classification/
├── app.py                      # 🌟 ไฟล์หลักของ Streamlit Web Application
├── models/                     # 🧠 โฟลเดอร์สำหรับวางไฟล์ .pth ที่ดาวน์โหลดมาจาก Drive
│   ├── best_resnet50.pth       # (หรือโมเดลตัวอื่นๆ ที่ต้องการทดสอบ)
│   └── README.md
├── sample_images/              # 🦐 ภาพตัวอย่างกุ้งสด/ไม่สด สำหรับให้ผู้ใช้คลิกทดสอบทันที
│   ├── fresh/
│   ├── not_fresh/
│   └── README.md
├── notebooks/                  # 📓 Colab Notebook สำหรับการเทรนบนคลาวด์
│   ├── shrimp_freshness_classification.ipynb
│   └── generate_notebook.py
└── ...
```

### 2. ฟีเจอร์หลักของหน้าเว็บ (`app.py`):
* **🎛️ Model Selector (Sidebar):** Dropdown ให้เลือกสลับโมเดลได้ทั้ง 5 สถาปัตยกรรม
* **⚡ Caching ด้วย `@st.cache_resource`:** โหลดโมเดลเข้า RAM ครั้งเดียว ทำนายผลได้ทันทีในระดับมิลลิวินาที (Real-time Inference)
* **🖼️ Image Drag & Drop:** อัปโหลดภาพกุ้ง (`.jpg`, `.jpeg`, `.png`, `.webp`)
* **🟢/🔴 Prediction Banner:** แสดงผลลัพธ์เด่นชัด: **กุ้งสด (FRESH)** หรือ **กุ้งไม่สด (NOT FRESH)**
* **📊 Confidence Gauge:** แสดง % ความมั่นใจของแต่ละคลาส
* **⏱️ Latency Monitor:** แสดงเวลาที่โมเดลใช้ในการประมวลผล (ms)
* **🔍 Explainable AI (Grad-CAM):** เมนูกดดูภาพ Heatmap วิเคราะห์จุดที่ AI ใช้ตัดสินใจ

### 3. คำสั่งสำหรับการรัน Web App:
```bash
pip install streamlit
streamlit run app.py
```

---

## 🔄 คำแนะนำเมื่อเปิดโปรเจกต์หลังย้ายโฟลเดอร์

### 1. เปิดโฟลเดอร์ใน IDE:
เปิดโฟลเดอร์ `shrimp-freshness-cnn` ใน Antigravity หรือ VS Code

### 2. ⚠️ วิธีแก้ปัญหาและรีเซ็ต Virtual Environment (`.venv`):
> **สาเหตุ:** ใน Python บน macOS เมื่อมีการย้ายโฟลเดอร์โปรเจกต์ โฟลเดอร์ `.venv` ตัวเดิมจะยังคงจำ Path เก่าที่อยู่ใน `~/.gemini/...` ทำให้เกิดปัญหา Path เพี้ยนหรือ `bad interpreter`  
> **วิธีแก้:** ให้ลบ `.venv` เก่าทิ้ง แล้วสร้างใหม่ในโฟลเดอร์ใหม่ด้วยคำสั่งบรรทัดเดียวนี้ใน Terminal:

```bash
# รันคำสั่งนี้ครั้งเดียวในโฟลเดอร์ใหม่ (Terminal)
rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### 3. เริ่มงาน Phase 2:
เมื่อเสร็จแล้ว แจ้ง AI ได้ทันทีว่า:  
💬 *"ย้ายโฟลเดอร์เรียบร้อยแล้ว เริ่มทำ Phase 2 (Streamlit Web App) ได้เลย"* เพื่อพัฒนาหน้าเว็บแอปต่อทันที!
