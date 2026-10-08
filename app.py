from typing import Optional, Dict, Any, List
import os
import io
import base64
from pathlib import Path
from PIL import Image, ImageOps
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
    page_title="ShrimpGuard AI — Quality Analysis",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling: ShrimpGuard AI Technical Workspace Theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Prompt:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    /* Global Typography Base - Scaled up for comfort and legibility */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Prompt', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 16px;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        font-family: 'Prompt', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* CRITICAL: Preserve Streamlit native icon fonts and ligatures */
    [class*="material-symbols"],
    [class*="material-icons"],
    [data-testid="stIcon"],
    .material-symbols-rounded,
    .material-icons {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
    }

    /* ------------------------------------------------------------- */
    /* Remove Streamlit header toolbar to prevent overlap & keep layout clean */
    /* ------------------------------------------------------------- */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    .stAppHeader {
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        pointer-events: none !important;
        overflow: visible !important;
        border: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Push Main Content up cleanly */
    .main .block-container,
    .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"] {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    /* Sidebar: Keep collapse button in DOM (for JS click) while removing top gap */
    [data-testid="stSidebarHeader"] {
        position: absolute !important;
        top: 0 !important;
        right: 0 !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Hide Streamlit auto-generated header anchor link icons (🔗) */
    .main-title a,
    [data-testid="stHeaderActionElements"],
    a.header-anchor {
        display: none !important;
        visibility: hidden !important;
    }

    /* Eliminate phantom height from style and script containers */
    div[data-testid="stElementContainer"]:has(style),
    div.element-container:has(style),
    div[data-testid="stHtml"]:has(script:only-child) {
        position: absolute !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        pointer-events: none !important;
    }

    /* Topbar Breadcrumb & Badge */
    .topbar-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0;
        margin-bottom: 12px;
        padding-bottom: 4px;
    }
    .workspace-crumb {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 13.5px;
        font-family: 'Outfit', sans-serif;
        color: var(--text-color);
        opacity: 0.75;
        letter-spacing: 0.3px;
    }
    .sidebar-toggle-btn {
        background: transparent;
        border: 1px solid rgba(125, 140, 160, 0.25);
        border-radius: 6px;
        padding: 4px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        color: var(--text-color);
        opacity: 0.75;
        transition: all 0.18s ease;
        line-height: 1;
        margin-right: 2px;
    }
    .sidebar-toggle-btn:hover {
        opacity: 1;
        color: #0284c7;
        background: rgba(2, 132, 199, 0.1);
        border-color: rgba(2, 132, 199, 0.35);
        transform: scale(1.06);
    }
    .sidebar-toggle-btn:active {
        transform: scale(0.95);
    }
    .crumb-separator {
        opacity: 0.45;
        display: inline-flex;
        align-items: center;
    }
    .topbar-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Outfit', sans-serif;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: #0284c7;
        background: rgba(2, 132, 199, 0.08);
        border: 1px solid rgba(2, 132, 199, 0.22);
        padding: 5px 12px;
        border-radius: 999px;
    }

    /* Page Header */
    .page-header {
        margin-top: 0;
        margin-bottom: 20px;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'Outfit', sans-serif;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.6px;
        color: #0284c7;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .eyebrow-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #0284c7;
    }
    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #000000 !important;
        margin: 0 0 6px 0;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }
    .main-title .title-black-text {
        color: #000000 !important;
    }
    .main-title .ai-highlight {
        color: #00a7f5 !important;
        font-weight: 800;
    }
    .main-subtitle {
        font-size: 15px;
        color: var(--text-color);
        opacity: 0.8;
        margin: 0;
        line-height: 1.5;
    }

    /* Step Headings */
    .step-heading-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(125, 140, 160, 0.2);
        padding-bottom: 10px;
        margin-top: 18px;
        margin-bottom: 16px;
    }
    .step-heading-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .step-heading-left h3 {
        font-size: 20px;
        font-weight: 600;
        color: var(--text-color);
        margin: 0;
    }
    .step-label {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #0284c7;
        background: rgba(2, 132, 199, 0.1);
        border: 1px solid rgba(2, 132, 199, 0.25);
        padding: 5px 12px;
        border-radius: 6px;
    }
    .status-pill {
        font-family: 'Outfit', 'Prompt', sans-serif;
        font-size: 13px;
        font-weight: 600;
        padding: 4px 11px;
        border-radius: 999px;
        background: rgba(2, 132, 199, 0.14);
        color: #0284c7;
    }
    .status-pill-idle {
        font-family: 'Outfit', 'Prompt', sans-serif;
        font-size: 13px;
        font-weight: 600;
        padding: 4px 11px;
        border-radius: 999px;
        background: rgba(125, 140, 160, 0.15);
        color: var(--text-color);
        opacity: 0.75;
    }

    /* Upload Zone Interactive */
    .upload-zone-box {
        background: var(--secondary-background-color);
        border: 2px dashed rgba(2, 132, 199, 0.4);
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        margin-top: 6px;
        margin-bottom: 12px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        user-select: none;
    }
    .upload-zone-box:hover {
        border-color: #00a7f5;
        background: rgba(2, 132, 199, 0.05);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 167, 245, 0.09);
    }
    .upload-zone-box.drag-active {
        border-color: #00a7f5 !important;
        border-style: solid !important;
        background: rgba(2, 132, 199, 0.12) !important;
        transform: scale(1.01);
    }
    .upload-zone-has-file {
        border-color: rgba(16, 185, 129, 0.5) !important;
        background: rgba(16, 185, 129, 0.04) !important;
    }
    .upload-zone-has-file:hover {
        border-color: #10B981 !important;
        background: rgba(16, 185, 129, 0.08) !important;
    }
    .upload-icon-circle {
        width: 58px;
        height: 58px;
        background: rgba(2, 132, 199, 0.12);
        color: #0284c7;
        border-radius: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
        transition: transform 0.2s ease;
    }
    .upload-zone-box:hover .upload-icon-circle {
        transform: scale(1.08);
    }
    .upload-icon-circle.success-circle {
        background: rgba(16, 185, 129, 0.14);
        color: #10B981;
    }
    .upload-zone-box h4 {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-color);
        margin: 4px 0 6px 0;
    }
    .upload-zone-box p {
        font-size: 14.5px;
        color: var(--text-color);
        opacity: 0.75;
        margin: 0 0 10px 0;
    }
    .file-formats-note {
        font-family: 'Outfit', sans-serif;
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.6;
    }

    /* Completely hide Streamlit's lower native file uploader bar */
    [data-testid="stFileUploader"] {
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Sample Cards */
    .sample-tag-fresh {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        color: #10B981;
        background: rgba(16, 185, 129, 0.14);
        padding: 3px 9px;
        border-radius: 5px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .sample-tag-not-fresh {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        color: #EF4444;
        background: rgba(239, 68, 68, 0.14);
        padding: 3px 9px;
        border-radius: 5px;
        display: inline-block;
        margin-bottom: 6px;
    }

    /* Uniform Sample Image Cards in tab_sample */
    [data-testid="stTabs"] [data-testid="stImage"] {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stTabs"] [data-testid="stImage"] img {
        aspect-ratio: 1 / 1 !important;
        object-fit: cover !important;
        width: 100% !important;
        max-height: 140px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(125, 140, 160, 0.2) !important;
    }
    .sample-filename-text {
        font-family: 'Outfit', monospace;
        font-size: 13.5px;
        font-weight: 600;
        color: var(--text-color);
        margin: 4px 0 8px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Empty / Awaiting State Card */
    .awaiting-card {
        background: var(--secondary-background-color);
        border: 1.5px dashed rgba(2, 132, 199, 0.35);
        border-radius: 16px;
        padding: 42px 28px;
        text-align: center;
        margin-top: 14px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.02);
    }
    .awaiting-icon {
        font-size: 40px;
        margin-bottom: 14px;
        display: inline-block;
    }
    .awaiting-card h4 {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-color);
        margin: 0 0 10px 0;
    }
    .awaiting-card p {
        font-size: 15px;
        color: var(--text-color);
        opacity: 0.8;
        max-width: 580px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Preview Panel */
    .preview-panel {
        background: var(--secondary-background-color);
        border: 1px solid rgba(125, 140, 160, 0.22);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
    }
    .preview-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        font-size: 15px;
        font-weight: 600;
        color: var(--text-color);
    }
    .preview-filename {
        font-family: 'Outfit', monospace;
        font-size: 13px;
        background: rgba(125, 140, 160, 0.16);
        color: var(--text-color);
        padding: 4px 10px;
        border-radius: 6px;
        max-width: 240px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .preview-image-container {
        border-radius: 12px;
        overflow: hidden;
        position: relative;
        background: rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(125, 140, 160, 0.15);
    }
    .preview-badge {
        position: absolute;
        bottom: 14px;
        left: 14px;
        background: rgba(15, 23, 42, 0.88);
        backdrop-filter: blur(4px);
        color: #38bdf8;
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        padding: 5px 11px;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .preview-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 14px;
        font-family: 'Outfit', sans-serif;
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.7;
    }

    /* Classification Result Cards */
    .fresh-result-card {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.38);
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 22px 24px;
    }
    .not-fresh-result-card {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.38);
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 22px 24px;
    }
    .result-icon-fresh {
        background: #10B981;
        width: 56px;
        height: 56px;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 700;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
    }
    .result-icon-not-fresh {
        background: #EF4444;
        width: 56px;
        height: 56px;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 700;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.35);
    }
    .result-overline-fresh {
        font-family: 'Outfit', sans-serif;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #10B981;
        text-transform: uppercase;
    }
    .result-overline-not-fresh {
        font-family: 'Outfit', sans-serif;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #EF4444;
        text-transform: uppercase;
    }
    .result-title-fresh {
        font-size: 30px;
        font-weight: 700;
        color: #10B981;
        margin: 4px 0 6px 0;
        line-height: 1.2;
    }
    .result-title-not-fresh {
        font-size: 30px;
        font-weight: 700;
        color: #EF4444;
        margin: 4px 0 6px 0;
        line-height: 1.2;
    }
    .result-desc {
        color: var(--text-color);
        font-size: 15px;
        opacity: 0.9;
        margin: 0;
        line-height: 1.5;
    }

    /* Probability Panel */
    .probability-panel {
        background: var(--secondary-background-color);
        border: 1px solid rgba(125, 140, 160, 0.22);
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
    }
    .prob-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }
    .prob-title-row h4 {
        font-size: 17px;
        font-weight: 600;
        color: var(--text-color);
        margin: 0;
    }
    .prob-sub-badge {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        letter-spacing: 0.8px;
        color: #0284c7;
        background: rgba(2, 132, 199, 0.12);
        padding: 4px 9px;
        border-radius: 6px;
        font-weight: 700;
    }
    .prob-item {
        margin-bottom: 16px;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 15px;
    }
    .prob-label {
        display: flex;
        align-items: center;
        gap: 9px;
        font-weight: 500;
        color: var(--text-color);
    }
    .prob-dot-fresh {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10B981;
        display: inline-block;
    }
    .prob-dot-not-fresh {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #EF4444;
        display: inline-block;
    }
    .prob-val-fresh {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #10B981;
        font-size: 18px;
    }
    .prob-val-not-fresh {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #EF4444;
        font-size: 18px;
    }
    .custom-progress-bg {
        width: 100%;
        height: 9px;
        background: rgba(125, 140, 160, 0.18);
        border-radius: 999px;
        overflow: hidden;
    }
    .custom-progress-fill-fresh {
        height: 100%;
        background: linear-gradient(90deg, #10B981, #34D399);
        border-radius: 999px;
    }
    .custom-progress-fill-not-fresh {
        height: 100%;
        background: linear-gradient(90deg, #EF4444, #F87171);
        border-radius: 999px;
    }

    /* Result Metrics */
    .result-metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 16px;
        padding-top: 18px;
        border-top: 1px solid rgba(125, 140, 160, 0.18);
    }
    .metric-box {
        background: rgba(2, 132, 199, 0.05);
        border: 1px solid rgba(2, 132, 199, 0.16);
        border-radius: 12px;
        padding: 12px 14px;
    }
    .metric-box-label {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        color: var(--text-color);
        opacity: 0.65;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .metric-box-value {
        font-family: 'Outfit', 'Prompt', sans-serif;
        font-size: 16px;
        font-weight: 700;
        color: var(--text-color);
    }

    /* Notice Box */
    .analysis-notice-card {
        background: rgba(2, 132, 199, 0.06);
        border: 1px solid rgba(2, 132, 199, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 18px;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }
    .analysis-notice-card p {
        font-size: 14px;
        line-height: 1.6;
        color: var(--text-color);
        opacity: 0.9;
        margin: 0;
    }

    /* Footer */
    .main-footer {
        border-top: 1px solid rgba(125, 140, 160, 0.2);
        color: var(--text-color);
        opacity: 0.65;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 40px;
        padding: 22px 0;
        font-size: 12.5px;
    }

    /* Sidebar Custom Elements */
    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-top: 0 !important;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(125, 140, 160, 0.2);
        margin-top: 0 !important;
        margin-bottom: 16px;
    }
    .brand-mark {
        width: 44px;
        height: 44px;
        background: rgba(2, 132, 199, 0.12);
        border: 1px solid rgba(2, 132, 199, 0.25);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(0, 167, 245, 0.14);
    }
    .sidebar-detail h1 {
        font-family: 'Outfit', 'Prompt', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #000000 !important;
        margin: 0;
        line-height: 1.2;
    }
    .sidebar-detail h1 .title-black-text {
        color: #000000 !important;
    }
    .sidebar-detail h1 .ai-highlight {
        color: #00a7f5 !important;
        font-weight: 800;
    }
    .sidebar-detail p {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        letter-spacing: 1.6px;
        color: #64748b;
        margin: 3px 0 0 0;
        text-transform: uppercase;
        font-weight: 600;
    }
    .sidebar-section-title {
        font-family: 'Prompt', sans-serif !important;
        font-size: 1.22rem;
        font-weight: 700;
        color: var(--text-color);
        margin-top: 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 2px solid rgba(2, 132, 199, 0.3);
        padding-bottom: 6px;
        word-break: keep-all;
        line-height: 1.35;
    }
    .model-family-text {
        font-family: 'Outfit', 'Prompt', sans-serif;
        font-size: 14px;
        color: #0284c7;
        font-weight: 600;
        margin-top: 6px;
        margin-bottom: 10px;
    }
    .model-specs-grid {
        display: flex;
        flex-direction: column;
        gap: 8px;
        border-top: 1px solid rgba(125, 140, 160, 0.15);
        padding-top: 12px;
        margin-top: 8px;
    }
    .spec-item {
        display: flex;
        justify-content: space-between;
        font-family: 'Outfit', 'Prompt', sans-serif;
        font-size: 13px;
    }
    .spec-item span {
        color: var(--text-color);
        opacity: 0.7;
    }
    .spec-item strong {
        color: var(--text-color);
        font-weight: 600;
    }
    .sidebar-note {
        background: rgba(2, 132, 199, 0.05);
        border: 1px solid rgba(2, 132, 199, 0.16);
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 14px;
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.8;
        line-height: 1.5;
    }
    .system-status-box {
        background: rgba(2, 132, 199, 0.06);
        border: 1px solid rgba(2, 132, 199, 0.2);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 12px;
    }
    .system-specs {
        display: flex;
        flex-direction: column;
        gap: 10px;
        font-family: 'Outfit', 'Prompt', sans-serif;
        font-size: 13px;
    }
    .system-specs div {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .system-specs span {
        color: var(--text-color);
        opacity: 0.7;
    }
    .system-specs strong {
        color: var(--text-color);
        font-weight: 600;
    }
    .active-pill {
        color: #10B981;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        font-size: 12px;
    }
    .system-track {
        width: 100%;
        height: 5px;
        background: rgba(125, 140, 160, 0.2);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 4px;
    }
    .system-track-fill {
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, #0284c7, #38bdf8);
    }
    .status-line {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Outfit', sans-serif;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: #10B981;
        margin-top: 16px;
        padding: 7px 12px;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 6px #10B981;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model Metadata Dictionary (Aligned with ShrimpGuard AI)
# ---------------------------------------------------------
MODEL_DETAILS: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "id": "resnet50",
        "display_name": "ResNet-50",
        "name_full": "ResNet-50 (CNN)",
        "family": "Residual Neural Network",
        "params": "25.6M",
        "input": "224 × 224",
        "layers": "50 layers",
        "type": "CNN",
    },
    "vit_b_16": {
        "id": "vit_b_16",
        "display_name": "Vision Transformer",
        "name_full": "Vision Transformer (ViT)",
        "family": "Vision Transformer · ViT-B/16",
        "params": "86.6M",
        "input": "224 × 224",
        "layers": "12 encoder blocks",
        "type": "Transformer",
    },
    "efficientnet_b0": {
        "id": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "name_full": "EfficientNet-B0",
        "family": "EfficientNet · Convolutional Network",
        "params": "5.3M",
        "input": "224 × 224",
        "layers": "MBConv blocks",
        "type": "CNN",
    },
    "mobilenet_v3": {
        "id": "mobilenet_v3",
        "display_name": "MobileNetV3-Large",
        "name_full": "MobileNetV3-Large",
        "family": "MobileNet · Lightweight CNN",
        "params": "5.4M",
        "input": "224 × 224",
        "layers": "Inverted residual blocks",
        "type": "CNN",
    },
    "custom_cnn": {
        "id": "custom_cnn",
        "display_name": "Custom CNN",
        "name_full": "Custom CNN (Baseline)",
        "family": "Custom 4-Block ConvNet",
        "params": "2.1M",
        "input": "224 × 224",
        "layers": "4 Conv blocks",
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
                samples[category].extend(sorted([str(p) for p in dir_path.glob(ext)]))
    return samples


def image_to_base64(image: Image.Image, max_dim: int = 800) -> str:
    preview = image.copy()
    preview.thumbnail((max_dim, max_dim))
    buffered = io.BytesIO()
    preview.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_confidence_tier(conf: float) -> str:
    if conf >= 95.0:
        return "Very High"
    elif conf >= 80.0:
        return "High"
    elif conf >= 65.0:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------
# Sidebar Configuration: ShrimpGuard AI Technical Sidebar
# ---------------------------------------------------------
device = get_device()
device_label = "Apple Silicon GPU (MPS)" if device.type == "mps" else ("NVIDIA CUDA GPU" if device.type == "cuda" else "CPU")

# Brand Row
st.sidebar.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">
            <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#00a7f5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M11 12h.01" stroke-width="2.5" />
                <path d="M13 22c.5-.5 1.12-1 2.5-1-1.38 0-2-.5-2.5-1" />
                <path d="M14 2a3.28 3.28 0 0 1-3.227 1.798l-6.17-.561A2.387 2.387 0 1 0 4.387 8H15.5a1 1 0 0 1 0 13 1 1 0 0 0 0-5H12a7 7 0 0 1-7-7V8" />
                <path d="M14 8a8.5 8.5 0 0 1 0 8" />
                <path d="M16 16c2 0 4.5-4 4-6" />
            </svg>
        </div>
        <div class="sidebar-detail">
            <h1><span class="title-black-text" style="color: #000000 !important;">ShrimpGuard</span> <span class="ai-highlight" style="color: #00a7f5 !important;">AI</span></h1>
            <p>QUALITY ANALYSIS</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section 1: Model Settings
st.sidebar.markdown(
    """
    <div class="sidebar-section-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;">
            <path d="M11 12h.01" stroke-width="2.6" />
            <path d="M13 22c.5-.5 1.12-1 2.5-1-1.38 0-2-.5-2.5-1" />
            <path d="M14 2a3.28 3.28 0 0 1-3.227 1.798l-6.17-.561A2.387 2.387 0 1 0 4.387 8H15.5a1 1 0 0 1 0 13 1 1 0 0 0 0-5H12a7 7 0 0 1-7-7V8" />
            <path d="M14 8a8.5 8.5 0 0 1 0 8" />
            <path d="M16 16c2 0 4.5-4 4-6" />
        </svg>
        <span>ตั้งค่าโมเดล AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

model_options = list(MODEL_DETAILS.keys())
selected_model_key = st.sidebar.selectbox(
    "โมเดลที่ใช้วิเคราะห์:",
    options=model_options,
    format_func=lambda k: f"{MODEL_DETAILS[k]['name_full']}",
    index=0,
)

model_info = MODEL_DETAILS[selected_model_key]

# Model specs breakdown
st.sidebar.markdown(
    f"""
    <div class="model-family-text">{model_info['family']}</div>
    <div class="model-specs-grid">
        <div class="spec-item">
            <span>Parameters</span>
            <strong>{model_info['params']}</strong>
        </div>
        <div class="spec-item">
            <span>Input size</span>
            <strong>{model_info['input']} px</strong>
        </div>
        <div class="spec-item">
            <span>Architecture</span>
            <strong>{model_info['layers']}</strong>
        </div>
    </div>
    <div class="sidebar-note">
        ผลการวิเคราะห์ขึ้นอยู่กับสถาปัตยกรรมโมเดลและคุณภาพของภาพ
    </div>
    """,
    unsafe_allow_html=True,
)

# Load Model Weights
model, resolved_weights_path = get_cached_model(selected_model_key)

# Section 2: System Status
st.sidebar.markdown(
    """
    <div class="sidebar-section-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;">
            <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
            <circle cx="12" cy="12" r="3"/>
        </svg>
        <span>ข้อมูลระบบและการประมวลผล</span>
    </div>
    """,
    unsafe_allow_html=True,
)

weights_filename = os.path.basename(resolved_weights_path) if resolved_weights_path else "None"
weights_size_mb = (os.path.getsize(resolved_weights_path) / (1024 * 1024)) if resolved_weights_path else 0.0

st.sidebar.markdown(
    f"""
    <div class="system-status-box">
        <div class="system-specs">
            <div>
                <span>Inference Engine</span>
                <strong>PyTorch ({device_label})</strong>
            </div>
            <div>
                <span>Model Weights</span>
                <strong class="active-pill">ACTIVE ({weights_size_mb:.1f} MB)</strong>
            </div>
            <div class="system-track">
                <div class="system-track-fill"></div>
            </div>
        </div>
    </div>
    <div class="status-line">
        <span class="status-dot"></span>
        <span>ONLINE · PRODUCTION WORKSPACE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main Page Header & Eyebrow
# ---------------------------------------------------------
st.markdown(
    """
    <div class="topbar-row">
        <div class="workspace-crumb">
            <button id="sidebar-toggle-btn" class="sidebar-toggle-btn" type="button" title="ซ่อน/แสดงแถบด้านข้าง (Toggle Sidebar)">
                <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect width="18" height="18" x="3" y="3" rx="2" />
                    <path d="M9 3v18" />
                </svg>
            </button>
            <span>Workspace</span>
            <span class="crumb-separator">
                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.45; vertical-align: middle;">
                    <path d="m9 18 6-6-6-6" />
                </svg>
            </span>
            <strong>Freshness analysis</strong>
        </div>
        <span class="topbar-badge">
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
                <path d="m9 12 2 2 4-4" />
            </svg>
            AI QUALITY INSPECTION
        </span>
    </div>
    <header class="page-header">
        <div class="eyebrow">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; flex-shrink: 0;">
                <path d="M11 12h.01" stroke-width="2.5" />
                <path d="M13 22c.5-.5 1.12-1 2.5-1-1.38 0-2-.5-2.5-1" />
                <path d="M14 2a3.28 3.28 0 0 1-3.227 1.798l-6.17-.561A2.387 2.387 0 1 0 4.387 8H15.5a1 1 0 0 1 0 13 1 1 0 0 0 0-5H12a7 7 0 0 1-7-7V8" />
                <path d="M14 8a8.5 8.5 0 0 1 0 8" />
                <path d="M16 16c2 0 4.5-4 4-6" />
            </svg>
            SHRIMP FRESHNESS DETECTION
        </div>
        <!-- ระบบจำแนกความสดของกุ้งด้วย AI -->
        <h2 class="main-title"><span class="title-black-text" style="color: #000000 !important;">ระบบจำแนกความสดของกุ้งด้วย</span> <span class="ai-highlight" style="color: #00a7f5 !important;">AI</span></h2>
        <p class="main-subtitle">ตรวจสอบคุณภาพความสดของกุ้งด้วยเทคโนโลยี Deep Learning (CNNs & Vision Transformer)</p>
    </header>
    """,
    unsafe_allow_html=True,
)

# JavaScript helper for Interactive Sidebar Toggle
st.html(
    """
    <script>
    (function() {
        function getDoc() {
            try {
                return window.parent ? window.parent.document : document;
            } catch(e) {
                return document;
            }
        }

        function toggleSidebar() {
            const doc = getDoc();
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            let isExpanded = true;
            if (sidebar) {
                const ariaExp = sidebar.getAttribute('aria-expanded');
                if (ariaExp !== null) {
                    isExpanded = (ariaExp === 'true');
                } else {
                    const rect = sidebar.getBoundingClientRect();
                    isExpanded = rect.width > 50;
                }
            }

            if (isExpanded) {
                const collapseBtn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                                    doc.querySelector('[data-testid="stSidebarCollapseButton"]');
                if (collapseBtn) {
                    collapseBtn.click();
                    return;
                }
            } else {
                const expandBtn = doc.querySelector('[data-testid="stExpandSidebarButton"]') ||
                                  doc.querySelector('[data-testid="stExpandSidebarButton"] button');
                if (expandBtn) {
                    expandBtn.click();
                    return;
                }
            }

            // Fallback
            const anyCollapse = doc.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                                doc.querySelector('[data-testid="stSidebarCollapseButton"]');
            const anyExpand = doc.querySelector('[data-testid="stExpandSidebarButton"]') ||
                              doc.querySelector('[data-testid="stExpandSidebarButton"] button');
            if (isExpanded && anyCollapse) {
                anyCollapse.click();
            } else if (anyExpand) {
                anyExpand.click();
            } else if (anyCollapse) {
                anyCollapse.click();
            }
        }

        function initSidebarToggle() {
            const doc = getDoc();
            const btn = doc.getElementById('sidebar-toggle-btn');
            if (btn && btn.dataset.bound !== 'true') {
                btn.dataset.bound = 'true';
                btn.onclick = function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    toggleSidebar();
                };
            }
        }

        function initUploadZone() {
            const doc = getDoc();
            const zone = doc.getElementById('upload-zone-box');
            if (zone && zone.dataset.bound !== 'true') {
                zone.dataset.bound = 'true';

                // Click on zone opens the native file chooser
                zone.addEventListener('click', function(e) {
                    if (e.target.closest('button')) return;
                    const input = doc.querySelector('[data-testid="stFileUploaderDropzoneInput"]') ||
                                  doc.querySelector('section[data-testid="stFileUploaderDropzone"] input') ||
                                  doc.querySelector('[data-testid="stFileUploader"] input[type="file"]');
                    if (input) {
                        input.click();
                    }
                });

                // Drag over effect
                zone.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    zone.classList.add('drag-active');
                });

                zone.addEventListener('dragleave', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    zone.classList.remove('drag-active');
                });

                // Drop file into zone
                zone.addEventListener('drop', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    zone.classList.remove('drag-active');

                    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                        const input = doc.querySelector('[data-testid="stFileUploaderDropzoneInput"]') ||
                                      doc.querySelector('section[data-testid="stFileUploaderDropzone"] input') ||
                                      doc.querySelector('[data-testid="stFileUploader"] input[type="file"]');
                        if (input) {
                            try {
                                const dt = new DataTransfer();
                                for (let i = 0; i < e.dataTransfer.files.length; i++) {
                                    dt.items.add(e.dataTransfer.files[i]);
                                }
                                input.files = dt.files;
                                input.dispatchEvent(new Event('change', { bubbles: true }));
                            } catch(err) {
                                console.error('DataTransfer upload error:', err);
                            }
                        }

                        const dropzone = doc.querySelector('[data-testid="stFileUploaderDropzone"]');
                        if (dropzone) {
                            dropzone.dispatchEvent(new DragEvent('drop', {
                                dataTransfer: e.dataTransfer,
                                bubbles: true,
                                cancelable: true
                            }));
                        }
                    }
                });
            }
        }

        initSidebarToggle();
        initUploadZone();
        setInterval(function() {
            initSidebarToggle();
            initUploadZone();
        }, 250);

        const doc = getDoc();
        if (doc && !doc._sidebarKeybound) {
            doc._sidebarKeybound = true;
            doc.addEventListener('keydown', function(e) {
                if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
                    e.preventDefault();
                    toggleSidebar();
                }
            });
        }
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

# ---------------------------------------------------------
# Step 1: Image Input Section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="step-heading-row">
        <div class="step-heading-left">
            <h3>แหล่งภาพ</h3>
        </div>
        <span class="step-label">01 / IMAGE INPUT</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize active sample to None so NO PREDICTION runs on startup
if "active_sample_path" not in st.session_state:
    st.session_state["active_sample_path"] = None

available_samples = load_available_samples()

tab_upload, tab_sample = st.tabs(["📤 อัปโหลดภาพของคุณ", "🖼️ ภาพตัวอย่าง"])

current_image: Optional[Image.Image] = None
current_filename = ""
is_uploaded = False
file_size_mb = 0.0

with tab_upload:
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    uploaded_file = st.file_uploader(
        "เลือกไฟล์ภาพกุ้ง:",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state['uploader_key']}",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="upload-zone-box" id="upload-zone-box" title="คลิกเพื่อเลือกไฟล์ภาพจากอุปกรณ์ หรือลากภาพมาวางที่นี่">
                <div class="upload-icon-circle">
                    <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#00a7f5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                        <path d="M12 12v9" />
                        <path d="m16 16-4-4-4 4" />
                    </svg>
                </div>
                <h4>ลากภาพกุ้งมาวางที่นี่</h4>
                <p>หรือคลิกเลือกไฟล์ภาพจากอุปกรณ์ของคุณ</p>
                <div class="file-formats-note">JPG, PNG, WEBP • สูงสุด 20 MB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="upload-zone-box upload-zone-has-file" id="upload-zone-box" title="คลิกเพื่อเลือกภาพใหม่ หรือลากภาพมาวาง">
                <div class="upload-icon-circle success-circle">
                    <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                    </svg>
                </div>
                <h4 style="color: #10B981;">โหลดภาพเรียบร้อย: {uploaded_file.name}</h4>
                <p>ขนาดไฟล์: {uploaded_file.size / (1024 * 1024):.2f} MB · คลิกหรือลากภาพใหม่มาวาง เพื่อเปลี่ยนภาพ</p>
                <div class="file-formats-note">JPG, PNG, WEBP • สูงสุด 20 MB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_clear_space, c_clear_act = st.columns([5, 1])
        with c_clear_act:
            if st.button("🗑️ ล้างภาพ", key="btn_clear_uploaded_file", use_container_width=True):
                st.session_state["uploader_key"] += 1
                st.session_state["active_sample_path"] = None
                st.rerun()

    if uploaded_file is not None:
        try:
            current_image = Image.open(uploaded_file).convert("RGB")
            current_filename = uploaded_file.name
            file_size_mb = uploaded_file.size / (1024 * 1024)
            is_uploaded = True
            st.session_state["active_sample_path"] = None  # Clear sample if user uploads file
        except Exception as e:
            st.error(f"ไม่สามารถเปิดไฟล์ภาพได้: {e}")

with tab_sample:
    st.caption("เลือกภาพกุ้งจากชุดข้อมูลจริงเพื่อส่งเข้าโมเดลวิเคราะห์ความสดทันที:")
    col_h1, col_h2 = st.columns(2, gap="medium")
    with col_h1:
        st.markdown("**🟢 ตัวอย่างกุ้งสด (Fresh Samples):**")
    with col_h2:
        st.markdown("**🔴 ตัวอย่างกุ้งไม่สด (Not Fresh Samples):**")

    fresh_list = available_samples.get("fresh", [])
    not_fresh_list = available_samples.get("not_fresh", [])
    max_rows = max(len(fresh_list), len(not_fresh_list))

    for r_idx in range(max_rows):
        row_c1, row_c2 = st.columns(2, gap="medium")
        with row_c1:
            if r_idx < len(fresh_list):
                s_path = fresh_list[r_idx]
                fn = os.path.basename(s_path)
                item_c1, item_c2 = st.columns([1, 2], vertical_alignment="center")
                with item_c1:
                    try:
                        thumb = ImageOps.fit(Image.open(s_path).convert("RGB"), (260, 260), method=Image.Resampling.LANCZOS)
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        pass
                with item_c2:
                    st.markdown('<span class="sample-tag-fresh">FRESH SAMPLE</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="sample-filename-text"><code>{fn}</code></div>', unsafe_allow_html=True)
                    if st.button("เลือกภาพนี้เพื่อวิเคราะห์", key=f"btn_{s_path}", use_container_width=True):
                        st.session_state["active_sample_path"] = s_path
                        st.rerun()

        with row_c2:
            if r_idx < len(not_fresh_list):
                s_path = not_fresh_list[r_idx]
                fn = os.path.basename(s_path)
                item_c1, item_c2 = st.columns([1, 2], vertical_alignment="center")
                with item_c1:
                    try:
                        thumb = ImageOps.fit(Image.open(s_path).convert("RGB"), (260, 260), method=Image.Resampling.LANCZOS)
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        pass
                with item_c2:
                    st.markdown('<span class="sample-tag-not-fresh">NOT FRESH SAMPLE</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="sample-filename-text"><code>{fn}</code></div>', unsafe_allow_html=True)
                    if st.button("เลือกภาพนี้เพื่อวิเคราะห์", key=f"btn_{s_path}", use_container_width=True):
                        st.session_state["active_sample_path"] = s_path
                        st.rerun()

# If user clicked a sample in tab 2 and has not uploaded a separate file
if current_image is None and st.session_state.get("active_sample_path"):
    sample_p = st.session_state["active_sample_path"]
    if os.path.exists(sample_p):
        current_image = Image.open(sample_p).convert("RGB")
        current_filename = os.path.basename(sample_p)
        file_size_mb = os.path.getsize(sample_p) / (1024 * 1024)
        is_uploaded = False

# ---------------------------------------------------------
# Step 2: Analysis Result Section
# ---------------------------------------------------------
if current_image is not None:
    demo_badge_text = "วิเคราะห์เรียลไทม์" if is_uploaded else "ตัวอย่างชุดข้อมูล"

    # Header with Clear/Reset Option
    col_hdr_left, col_hdr_right = st.columns([4, 1])
    with col_hdr_left:
        st.markdown(
            f"""
            <div class="step-heading-row" style="margin-top: 24px; border-bottom: none; margin-bottom: 0;">
                <div class="step-heading-left">
                    <h3>⚡ ผลการวิเคราะห์</h3>
                    <span class="status-pill">{demo_badge_text}</span>
                </div>
                <span class="step-label">02 / ANALYSIS RESULT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_hdr_right:
        st.write("")
        if st.button("✕ ล้างผลการวิเคราะห์", key="btn_reset_image", use_container_width=True):
            st.session_state["active_sample_path"] = None
            st.rerun()

    # Run PyTorch Model Prediction
    with st.spinner("โมเดล AI กำลังประมวลผลภาพ..."):
        result = predict_image(model, current_image, device=device)

    is_fresh = result["class_name"] == "fresh"
    confidence = result["confidence"]
    prob_fresh = result["probabilities"]["fresh"]
    prob_not_fresh = result["probabilities"]["not_fresh"]
    latency_ms = result["latency_ms"]
    confidence_tier = get_confidence_tier(confidence)

    # Encode image thumbnail for preview card
    img_b64 = image_to_base64(current_image)
    image_badge_text = "UPLOADED IMAGE" if is_uploaded else "SAMPLE IMAGE"
    preview_tag = "LOCAL PREVIEW" if is_uploaded else "DATASET SAMPLE"

    col_preview, col_analysis = st.columns([1, 1], gap="large")

    with col_preview:
        st.markdown(
            f"""
            <div class="preview-panel">
                <div class="preview-header">
                    <span>📷 ภาพที่ทำการวิเคราะห์</span>
                    <span class="preview-filename">{current_filename}</span>
                </div>
                <div class="preview-image-container">
                    <img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; aspect-ratio: 4/3; object-fit: cover; display: block;" alt="Shrimp Image" />
                    <span class="preview-badge">{image_badge_text}</span>
                </div>
                <div class="preview-footer">
                    <span>{file_size_mb:.2f} MB / {preview_tag}</span>
                    <span>100% QUALITY</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_analysis:
        # 1. Classification Banner Card
        if is_fresh:
            st.markdown(
                """
                <div class="fresh-result-card">
                    <div class="result-icon-fresh">✓</div>
                    <div>
                        <div class="result-overline-fresh">AI CLASSIFICATION RESULT</div>
                        <h3 class="result-title-fresh">กุ้งสด (FRESH)</h3>
                        <p class="result-desc">ระดับความสดอยู่ในเกณฑ์มาตรฐาน เหมาะสำหรับการบริโภคหรือแปรรูป</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="not-fresh-result-card">
                    <div class="result-icon-not-fresh">✕</div>
                    <div>
                        <div class="result-overline-not-fresh">AI CLASSIFICATION RESULT</div>
                        <h3 class="result-title-not-fresh">กุ้งไม่สด (NOT FRESH)</h3>
                        <p class="result-desc">ตรวจพบสัญญาณการเสื่อมสภาพ ไม่แนะนำสำหรับการบริโภคสด</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Probability Panel with Styled Meters
        st.markdown(
            f"""
            <div class="probability-panel">
                <div class="prob-title-row">
                    <h4>ความน่าจะเป็นของแต่ละคลาส</h4>
                    <span class="prob-sub-badge">CLASS PROBABILITIES</span>
                </div>
                <div class="prob-item">
                    <div class="prob-header">
                        <span class="prob-label"><i class="prob-dot-fresh"></i> กุ้งสด <small style="opacity:0.65;">(Fresh)</small></span>
                        <strong class="prob-val-fresh">{prob_fresh:.1f}%</strong>
                    </div>
                    <div class="custom-progress-bg">
                        <div class="custom-progress-fill-fresh" style="width: {prob_fresh:.1f}%;"></div>
                    </div>
                </div>
                <div class="prob-item" style="margin-top: 14px;">
                    <div class="prob-header">
                        <span class="prob-label"><i class="prob-dot-not-fresh"></i> กุ้งไม่สด <small style="opacity:0.65;">(Not Fresh)</small></span>
                        <strong class="prob-val-not-fresh">{prob_not_fresh:.1f}%</strong>
                    </div>
                    <div class="custom-progress-bg">
                        <div class="custom-progress-fill-not-fresh" style="width: {prob_not_fresh:.1f}%;"></div>
                    </div>
                </div>
                <div class="result-metrics-grid">
                    <div class="metric-box">
                        <div class="metric-box-label">CONFIDENCE LEVEL</div>
                        <div class="metric-box-value">{confidence_tier} ({confidence:.1f}%)</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-box-label">SELECTED MODEL</div>
                        <div class="metric-box-value">{model_info['display_name']}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-box-label">LATENCY</div>
                        <div class="metric-box-value">{latency_ms:.1f} ms</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Quality Assurance Notice Card
    st.markdown(
        """
        <div class="analysis-notice-card">
            <span style="font-size: 18px; line-height: 1;">💡</span>
            <p>
                <strong>ข้อสังเกต:</strong> ผลการวิเคราะห์ประเมินด้วยแบบจำลอง Deep Learning สำหรับการคัดกรองเบื้องต้น ควรใช้ร่วมกับการประเมินทางกายภาพ (กลิ่น สี สัมผัสความแน่นของเนื้อ) และการควบคุมอุณหภูมิ เพื่อความปลอดภัยสูงสุดในการบริโภค
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # Awaiting state on startup (No prediction at start)
    st.markdown(
        """
        <div class="step-heading-row" style="margin-top: 24px;">
            <div class="step-heading-left">
                <h3>⚡ ผลการวิเคราะห์</h3>
                <span class="status-pill-idle">รอเลือกภาพหรืออัปโหลด</span>
            </div>
            <span class="step-label">02 / ANALYSIS RESULT</span>
        </div>
        <div class="awaiting-card">
            <div class="awaiting-icon">🔍</div>
            <h4>ระบบพร้อมสำหรับการวิเคราะห์ความสดของกุ้ง</h4>
            <p>
                กรุณาอัปโหลดภาพกุ้งของคุณในแท็บ <b>"📤 อัปโหลดภาพของคุณ"</b> หรือคลิกเลือกตัวอย่างในแท็บ <b>"🖼️ ภาพตัวอย่าง"</b> ด้านบนเพื่อเริ่มการวิเคราะห์ความสดด้วย AI
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <footer class="main-footer">
        <span>SHRIMPGUARD AI · DEEP LEARNING QUALITY INSPECTION</span>
        <span>POWERED BY PYTORCH & STREAMLIT</span>
    </footer>
    """,
    unsafe_allow_html=True,
)
