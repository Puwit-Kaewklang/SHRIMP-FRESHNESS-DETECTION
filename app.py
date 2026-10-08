from typing import Optional, Dict, Any, List
import os
from pathlib import Path
from PIL import Image
import torch
import streamlit as st

from src.models import SUPPORTED_MODELS
from src.inference import (
    load_model,
    predict_image,
    find_model_weights,
    CLASS_NAMES,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shrimp Freshness AI Classifier",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling: Modern Blue Theme & High Legibility Font
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    /* Global Typography: Prompt Font */
    html, body, [class*="css"], .stApp, h1, h2, h3, h4, h5, h6, p, div, span, button {
        font-family: 'Prompt', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Main Headers: Adapt dynamically to Streamlit theme using CSS variables */
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.2rem;
        letter-spacing: -0.01em;
        line-height: 1.3;
    }
    .header-accent {
        color: #0284C7; /* Ocean Sky Blue: High contrast in both Light & Dark modes */
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    /* Blue Theme Containers */
    .blue-card {
        background: rgba(2, 132, 199, 0.06);
        border: 1px solid rgba(2, 132, 199, 0.22);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Status Cards */
    .status-card-fresh {
        background: rgba(16, 185, 129, 0.12);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.12);
    }
    .status-card-not-fresh {
        background: rgba(239, 68, 68, 0.12);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.12);
    }
    .status-title-fresh {
        font-size: 2.1rem;
        font-weight: 800;
        color: #10B981;
        margin-bottom: 0.4rem;
    }
    .status-title-not-fresh {
        font-size: 2.1rem;
        font-weight: 800;
        color: #EF4444;
        margin-bottom: 0.4rem;
    }
    .status-desc {
        color: var(--text-color);
        font-size: 1.05rem;
        margin: 0;
        opacity: 0.88;
        line-height: 1.5;
    }

    /* Blue Accent Badges */
    .metric-badge-blue {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        background-color: rgba(2, 132, 199, 0.15);
        color: #0284C7;
        border: 1px solid rgba(2, 132, 199, 0.35);
        margin-right: 0.5rem;
    }

    /* Image Container */
    .image-preview-box {
        border: 1px solid rgba(2, 132, 199, 0.25);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(0, 0, 0, 0.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model Metadata Dictionary
# ---------------------------------------------------------
MODEL_DETAILS: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "display_name": "ResNet-50",
        "description": "Residual Network 50 Layers (Industry Standard)",
        "params": "~25.6M parameters",
        "type": "CNN",
    },
    "mobilenet_v3": {
        "display_name": "MobileNetV3-Large",
        "description": "Lightweight & Fast (Edge AI & IoT)",
        "params": "~5.4M parameters",
        "type": "CNN",
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "description": "Compound Scaling Architecture",
        "params": "~5.3M parameters",
        "type": "CNN",
    },
    "vit_b_16": {
        "display_name": "Vision Transformer (ViT-B/16)",
        "description": "Pure Self-Attention Transformer",
        "params": "~86.0M parameters",
        "type": "Transformer",
    },
    "custom_cnn": {
        "display_name": "Custom CNN (Baseline)",
        "description": "4-Block Convolutional Network from scratch",
        "params": "~2.1M parameters",
        "type": "CNN",
    },
}

# ---------------------------------------------------------
# Helper Functions with Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล AI...")
def get_cached_model(model_name: str) -> Any:
    device = get_device()
    weights_path = find_model_weights(model_name)
    model = load_model(model_name, weights_path=weights_path, device=device)
    return model, weights_path


def load_available_samples() -> Dict[str, List[str]]:
    samples: Dict[str, List[str]] = {"fresh": [], "not_fresh": []}
    for category in ["fresh", "not_fresh"]:
        dir_path = Path(f"sample_images/{category}")
        if dir_path.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                samples[category].extend([str(p) for p in dir_path.glob(ext)])
    return samples


# ---------------------------------------------------------
# Sidebar Configuration: Clean Ocean Blue
# ---------------------------------------------------------
device = get_device()
device_label = "Apple Silicon GPU (MPS)" if device.type == "mps" else ("NVIDIA CUDA GPU" if device.type == "cuda" else "CPU")

st.sidebar.markdown("### 🦐 ตั้งค่าโมเดล AI")

model_options = list(MODEL_DETAILS.keys())
selected_model_key = st.sidebar.selectbox(
    "เลือกสถาปัตยกรรมโมเดล:",
    options=model_options,
    format_func=lambda k: f"{MODEL_DETAILS[k]['display_name']} ({MODEL_DETAILS[k]['type']})",
    index=0,
)

model_info = MODEL_DETAILS[selected_model_key]
st.sidebar.caption(f"ℹ️ **รายละเอียด:** {model_info['description']}")
st.sidebar.caption(f"📊 **ขนาดโมเดล:** {model_info['params']}")

# Load Model
model, resolved_weights_path = get_cached_model(selected_model_key)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ ข้อมูลระบบและการประมวลผล")
st.sidebar.markdown(f"**⚡ ฮาร์ดแวร์:** `{device_label}`")

if resolved_weights_path:
    file_size_mb = os.path.getsize(resolved_weights_path) / (1024 * 1024)
    st.sidebar.success(f"✅ โหลด Weights: `{os.path.basename(resolved_weights_path)}` ({file_size_mb:.1f} MB)")
else:
    st.sidebar.warning("⚠️ ไม่พบไฟล์ .pth (รันในโหมดโครงสร้างเริ่มต้น)")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        🦐 ระบบจำแนกความสดของกุ้งด้วย AI
    </div>
    <div class="sub-header">
        ระบบตรวจสอบคุณภาพและจำแนกความสดของกุ้งแบบอัตโนมัติด้วย Deep Learning (CNNs & Vision Transformer)
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Image Input Section
# ---------------------------------------------------------
tab_upload, tab_sample = st.tabs(["📤 อัปโหลดภาพของคุณ", "🖼️ เลือกจากภาพตัวอย่างในระบบ"])

current_image: Optional[Image.Image] = None
image_source_label = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "ลากและวางไฟล์ภาพกุ้งที่นี่ หรือคลิกเพื่อเลือกไฟล์",
        type=["jpg", "jpeg", "png", "webp"],
        help="รองรับไฟล์ JPEG, PNG และ WEBP",
    )
    if uploaded_file is not None:
        try:
            current_image = Image.open(uploaded_file).convert("RGB")
            image_source_label = f"ไฟล์ที่อัปโหลด: {uploaded_file.name}"
        except Exception as e:
            st.error(f"ไม่สามารถเปิดไฟล์ภาพได้: {e}")

with tab_sample:
    samples = load_available_samples()
    col_fresh, col_not_fresh = st.columns(2)
    
    with col_fresh:
        st.markdown("**🟢 ภาพตัวอย่างกุ้งสด (Fresh):**")
        for s_path in samples["fresh"]:
            filename = os.path.basename(s_path)
            if st.button(f"เลือก: {filename}", key=f"btn_{s_path}", use_container_width=True):
                current_image = Image.open(s_path).convert("RGB")
                image_source_label = f"ภาพตัวอย่างกุ้งสด: {filename}"
                
    with col_not_fresh:
        st.markdown("**🔴 ภาพตัวอย่างกุ้งไม่สด (Not Fresh):**")
        for s_path in samples["not_fresh"]:
            filename = os.path.basename(s_path)
            if st.button(f"เลือก: {filename}", key=f"btn_{s_path}", use_container_width=True):
                current_image = Image.open(s_path).convert("RGB")
                image_source_label = f"ภาพตัวอย่างกุ้งไม่สด: {filename}"

# ---------------------------------------------------------
# Inference & Results Section (Clean & Focused, No Heatmap)
# ---------------------------------------------------------
if current_image is not None:
    st.divider()
    st.markdown(f"#### 🔎 ผลการวิเคราะห์ภาพ ({image_source_label})")

    # Run Prediction
    with st.spinner("โมเดลกำลังประมวลผล..."):
        result = predict_image(model, current_image, device=device)

    is_fresh = result["class_name"] == "fresh"
    confidence = result["confidence"]
    prob_fresh = result["probabilities"]["fresh"]
    prob_not_fresh = result["probabilities"]["not_fresh"]
    latency_ms = result["latency_ms"]

    col_view, col_metrics = st.columns([1, 1], gap="large")

    with col_view:
        st.markdown("**🖼️ ภาพที่ทำการวิเคราะห์:**")
        st.image(current_image, use_container_width=True)

    with col_metrics:
        # Status Result Card
        if is_fresh:
            st.markdown(
                """
                <div class="status-card-fresh">
                    <div class="status-title-fresh">🟢 กุ้งสด (FRESH)</div>
                    <p class="status-desc">
                        ระดับความสดอยู่ในเกณฑ์มาตรฐาน เหมาะสำหรับการบริโภคหรือแปรรูป
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-card-not-fresh">
                    <div class="status-title-not-fresh">🔴 กุ้งไม่สด (NOT FRESH)</div>
                    <p class="status-desc">
                        ตรวจพบสัญญาณการเสื่อมสภาพ ไม่แนะนำสำหรับการบริโภคสด
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("##### 📊 ความน่าจะเป็นของแต่ละคลาส (Class Probabilities)")
        
        # Progress bars with blue and status indicators
        st.write(f"**กุ้งสด (Fresh):** `{prob_fresh:.1f}%`")
        st.progress(min(prob_fresh / 100.0, 1.0))

        st.write(f"**กุ้งไม่สด (Not Fresh):** `{prob_not_fresh:.1f}%`")
        st.progress(min(prob_not_fresh / 100.0, 1.0))

        st.divider()
        st.markdown("##### ⏱️ ข้อมูลประสิทธิภาพการทำนาย (Benchmark Metrics)")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("ความมั่นใจ (Confidence)", f"{confidence:.1f}%")
        m_col2.metric("เวลาประมวลผล (Latency)", f"{latency_ms:.1f} ms")
        m_col3.metric("โมเดลที่ใช้งาน", model_info["display_name"])

else:
    st.info("👆 กรุณาอัปโหลดภาพกุ้ง หรือคลิกเลือกภาพตัวอย่างจากแถบด้านบนเพื่อเริ่มการวิเคราะห์")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "🦐 **Shrimp Freshness Classification System** | พัฒนาด้วย PyTorch & Streamlit | รองรับ 5 โมเดล (Custom CNN, MobileNetV3, ResNet-50, EfficientNet-B0, ViT-B/16)"
)
