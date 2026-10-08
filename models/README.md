# 🧠 Shrimp Freshness Model Weights (`models/`)

วางไฟล์น้ำหนักโมเดล (`.pth`) ที่คุณเทรนเสร็จแล้วจาก Google Drive (`MyDrive/shrimp_models/`) ไว้ในโฟลเดอร์นี้

### 📦 รายชื่อไฟล์โมเดลทั้ง 5 สถาปัตยกรรม:
1. `best_resnet50.pth` (แนะนำ - แม่นยำสูงและเสถียรที่สุด)
2. `best_mobilenet_v3.pth` (แนะนำ - น้ำหนักเบา ประมวลผลเร็ว เหมาะกับงาน Real-time/Edge)
3. `best_efficientnet_b0.pth` (แม่นยำสูงและขนาดกะทัดรัด)
4. `best_vit_b_16.pth` (Vision Transformer สถาปัตยกรรมใหม่ล่าสุด)
5. `best_custom_cnn.pth` (Baseline โมเดล 4 Convolutional Blocks)

> **คำแนะนำ:** คุณสามารถดาวน์โหลดเฉพาะตัวที่ดีที่สุด (เช่น `best_resnet50.pth` หรือ `best_mobilenet_v3.pth`) เพียง 1 ไฟล์มาวางก่อนได้ เว็บแอปพลิเคชันจะทำงานได้สมบูรณ์แบบทันทีโดยไม่จำเป็นต้องมีครบทั้ง 5 ไฟล์
