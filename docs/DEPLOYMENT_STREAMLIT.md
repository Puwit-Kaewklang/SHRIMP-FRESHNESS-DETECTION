# 🚀 คู่มือการ Deploy บน Streamlit Community Cloud (ฟรี 100%)

คู่มือนี้จะแนะนำขั้นตอนการนำระบบ **Shrimp Freshness AI Classification System** ขึ้นสู่ **Streamlit Community Cloud** โดยไม่มีค่าใช้จ่าย และจัดการปัญหาข้อจำกัดทางเทคนิค (RAM 1GB และขนาดไฟล์ Git 100MB) อย่างถูกต้อง

---

## 📋 ภาพรวมสถาปัตยกรรม (Cloud Architecture)

* **Web UI & Inference:** รันบน Streamlit Community Cloud (CPU-only, Free Tier)
* **Model Storage:** ฝากไฟล์น้ำหนัก `.pth` ไว้บน **Hugging Face Model Hub** (ฟรี ไม่จำกัดขนาดไฟล์ และแบนด์วิดท์เร็วมาก)
* **On-Demand Loading:** เมื่อผู้ใช้เลือกโมเดลบนเว็บ ตัวแอปจะดาวน์โหลดน้ำหนักลงมาอัตโนมัติ และสลับเก็บในหน่วยความจำทีละโมเดล (`max_entries=1` + Garbage Collection) ป้องกัน RAM เกิน 1GB

---

## 🛠️ ขั้นตอนที่ 1: อัปโหลดไฟล์โมเดลไปที่ Hugging Face (ฟรี)

เนื่องจากโมเดล `best_vit_b_16.pth` มีขนาด ~327 MB (เกิน 100 MB ของ GitHub) เราจึงฝากไฟล์ไว้ที่ Hugging Face Hub:

### 1.1 สร้าง Model Repository บน Hugging Face
1. สมัคร/เข้าสู่ระบบที่ [huggingface.co](https://huggingface.co/)
2. ไปที่ [huggingface.co/new](https://huggingface.co/new) เลือกสร้าง **New Model**
3. ตั้งชื่อ เช่น `shrimp-freshness-models` และเลือกสถานะเป็น **Public** (หรือ Private ก็ได้)
4. จะได้ Repo ID ในรูปแบบ `<username>/shrimp-freshness-models`

### 1.2 สร้าง Access Token (สำหรับอัปโหลด)
1. ไปที่ [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. กด **Create new token** -> เลือก Type เป็น **Write**
3. คัดลอก Token เก็บไว้

### 1.3 รันสคริปต์อัปโหลดอัตโนมัติ
ในเครื่องของคุณ (มีไฟล์โมเดลทั้ง 5 ตัวอยู่ใน `models/5 models/` เรียบร้อยแล้ว) รันคำสั่งนี้ใน Terminal:

```bash
python scripts/upload_models_to_hf.py --repo <YOUR_USERNAME>/shrimp-freshness-models --token <YOUR_HF_TOKEN>
```

> **ทางเลือกสำรอง (ไม่ต้องใช้สคริปต์):** สามารถเปิดหน้าเว็บ Hugging Face Model ที่สร้างไว้ แล้วลากไฟล์ `.pth` ทั้ง 5 ตัวไปวางในแท็บ **Files and versions** ผ่านหน้าเว็บโดยตรงได้เช่นกัน

---

## 🐙 ขั้นตอนที่ 2: Push โค้ดขึ้น GitHub

โปรเจกต์นี้ได้รับการตั้งค่า `.gitignore` เพื่อไม่ให้ไฟล์ `.pth` ถูกดึงเข้า Git โดยตรง:

```bash
git add .
git commit -m "feat: ready for streamlit cloud deployment"
git push origin main
```

---

## ☁️ ขั้นตอนที่ 3: Deploy บน Streamlit Community Cloud

1. เข้าสู่ระบบที่ [share.streamlit.io](https://share.streamlit.io/) ด้วยบัญชี GitHub
2. กดปุ่ม **"Create app"** (หรือ **"New app"**)
3. กรอกรายละเอียด:
   * **Repository:** เลือก Repository ของคุณ เช่น `username/Shrimp_Classification`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **App URL (กำหนดเองได้):** เช่น `shrimp-guard-ai.streamlit.app`

---

## 🔑 ขั้นตอนที่ 4: ตั้งค่า Secrets ใน Streamlit Cloud

เพื่อให้แอปทราบว่าจะต้องไปดาวน์โหลดโมเดลจาก Hugging Face Repo ไหน:

1. ก่อนกด Deploy หรือหลังสร้างแอปแล้ว ให้คลิกที่ **Advanced settings...** (หรือเข้าไปที่ **App Settings → Secrets**)
2. เพิ่มค่าตัวแปรดังนี้:

```toml
# ใส่ Repo ID ของคุณที่สร้างไว้ในขั้นตอนที่ 1
HF_MODELS_REPO = "YOUR_USERNAME/shrimp-freshness-models"

# (กรณี Repo เป็น Private ให้ใส่ Token เพิ่มด้วย แต่ถ้าเป็น Public ไม่ต้องใส่)
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

3. กด **Save** แล้วกด **Deploy!** 🚀

---

## ✅ การตรวจสอบผลการทำงาน

1. เมื่อแอป Build เสร็จ หน้าเว็บจะเปิดขึ้นมา
2. สังเกตที่เมนูด้านซ้าย:
   * **Inference Engine:** จะแสดงเป็น `PyTorch (CPU)`
   * **Model Weights:** เมื่อเปิดใช้งานครั้งแรก ระบบจะดาวน์โหลด weights และแสดงสถานะ `ACTIVE (<ขนาด> MB)`
3. ลองทดสอบอัปโหลดภาพกุ้ง หรือกดเลือกรูปตัวอย่างเพื่อทดสอบ Inference และ Grad-CAM
4. ทดลองสลับโมเดล — ระบบจะเคลียร์โมเดลเดิมออกจาก RAM โดยอัตโนมัติ ทำให้ใช้งานได้อย่างเสถียรต่อเนื่อง

---

## 💡 คำแนะนำเพิ่มเติม (FAQ)

* **ถ้าไม่ตั้งค่า `HF_MODELS_REPO` จะเกิดอะไรขึ้น?**
  ระบบจะยังคงเปิดหน้าเว็บได้ปกติ แต่จะทำงานในโหมด `FALLBACK (Base Architecture)` ซึ่งใช้โมเดลเปล่าที่ยังไม่ได้ผ่านการเทรน
* **ค่าใช้จ่ายจริง ๆ คือเท่าไร?**
  **0 บาท ฟรี 100% ตลอดอายุการใช้งาน** ทั้งบน Streamlit Community Cloud, GitHub และ Hugging Face Model Hub
