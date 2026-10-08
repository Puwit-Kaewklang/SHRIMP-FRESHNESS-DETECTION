# Streamlit Community Cloud Deployment & Memory Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable 100% free, zero-cost deployment of the Shrimp Freshness AI classification system on Streamlit Community Cloud with automatic on-demand weight downloads and memory optimization (preventing OOM crashes within the ~1 GB RAM limit).

**Architecture:** Hybrid weight-loading pipeline that prioritizes local `.pth` files, falling back to on-demand downloads from Hugging Face Hub / GitHub Releases when hosted on cloud instances where Git excludes heavy weights. Memory management uses Streamlit's `max_entries=1` cache and explicit garbage collection to prevent multi-model RAM exhaustion.

**Tech Stack:** Python 3.10+, PyTorch (CPU-only index), Streamlit 1.28+, Hugging Face Hub (`hf_hub_download`), pytest.

**Spec:** Feasibility review and requirements for Streamlit Community Cloud free tier deployment.

## Global Constraints

- Must work 100% free on Streamlit Community Cloud without requiring paid tiers.
- Must not break existing local development or existing test suites (`tests/`).
- Must adhere to GitHub's 100 MB per-file upload ceiling (preventing direct push of `best_vit_b_16.pth` 327 MB).
- Keep local inference instantaneous when `.pth` files already reside in `models/` or `models/5 models/`.

## Review Focus

- Missing weights in cloud environment gracefully trigger on-demand download rather than uninitialized random model.
- Switching between all 5 models sequentially on a 1 GB RAM machine does not accumulate memory.
- `requirements.txt` successfully instructs pip on Linux cloud to install PyTorch CPU wheels instead of 2.5 GB CUDA wheels.
- Network failure or missing Hugging Face repo falls back gracefully with user-friendly warnings in the UI.
- All existing 29 unit and integration tests continue to pass.

---

### Task 1: Add Cloud-Ready PyTorch and Hugging Face dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `.streamlit/config.toml`

**Interfaces:**
- Produces: CPU-optimized requirements file with `huggingface_hub` dependency, plus Streamlit production config.

- [ ] **Step 1: Update `requirements.txt` with PyTorch CPU index and `huggingface_hub`**
Add `--extra-index-url https://download.pytorch.org/whl/cpu` at the top and include `huggingface_hub>=0.20.0`.
- [ ] **Step 2: Create `.streamlit/config.toml` for production settings**
Configure `server.maxUploadSize = 25`, `browser.gatherUsageStats = false`, headless mode, and clean styling defaults.
- [ ] **Step 3: Verify `.streamlit/config.toml` formatting and requirements syntax**
- [ ] **Step 4: Commit**
```bash
git add requirements.txt .streamlit/config.toml
git commit -m "chore: configure cpu-only torch, huggingface_hub, and streamlit cloud settings"
```

---

### Task 2: Implement On-Demand Weight Resolution & Remote Downloading in `src/inference.py`

**Files:**
- Modify: `src/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: `find_model_weights(model_name)`
- Produces: `download_model_weights_if_missing(model_name, repo_id=None) -> Optional[str]` integrated seamlessly into `load_model`.

- [ ] **Step 1: Write unit test in `tests/test_inference.py` testing `download_model_weights_if_missing` behavior**
Test that local weights are returned without remote calls, and simulate missing weight resolution.
- [ ] **Step 2: Run test to observe failure/behavior**
Run: `.venv/bin/pytest tests/test_inference.py -v`
- [ ] **Step 3: Implement `download_model_weights_if_missing` in `src/inference.py`**
Implement function using `huggingface_hub.hf_hub_download` with configurable repo (defaulting to environment variable `HF_MODELS_REPO` or Streamlit secrets or fallback) with local file caching in `models/`.
- [ ] **Step 4: Update `load_model` to resolve weights via `find_model_weights` or `download_model_weights_if_missing`**
- [ ] **Step 5: Run tests to verify all tests pass**
Run: `.venv/bin/pytest tests/test_inference.py -v`
- [ ] **Step 6: Commit**
```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: add on-demand remote weight downloading from hugging face"
```

---

### Task 3: Optimize Memory Management & Model Caching in `app.py`

**Files:**
- Modify: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `get_cached_model(model_name)`
- Produces: Single-model LRU cache (`max_entries=1`) with garbage collection on eviction to fit inside 1 GB RAM.

- [ ] **Step 1: Update `get_cached_model` in `app.py` to use `max_entries=1`**
Update decorator to `@st.cache_resource(max_entries=1, show_spinner="กำลังโหลดโมเดล AI...")` and call `gc.collect()`.
- [ ] **Step 2: Add visual weight source indicator in app UI (Local vs Cloud Downloaded vs Fallback)**
Display clear tag in model technical info box to let user know if weights are active.
- [ ] **Step 3: Run `tests/test_app.py` to ensure AppTest continues to succeed**
Run: `.venv/bin/pytest tests/test_app.py -v`
- [ ] **Step 4: Commit**
```bash
git add app.py tests/test_app.py
git commit -m "perf: limit model cache to 1 entry and add garbage collection for cloud ram limits"
```

---

### Task 4: Provide Helper Script & Comprehensive Deployment Guide

**Files:**
- Create: `scripts/upload_models_to_hf.py`
- Create: `docs/DEPLOYMENT_STREAMLIT.md`

**Interfaces:**
- Produces: Interactive CLI script to upload `.pth` weights to free Hugging Face repo, plus comprehensive Thai/English deployment walkthrough for Streamlit Community Cloud.

- [ ] **Step 1: Create `scripts/upload_models_to_hf.py`**
Script checks for `models/5 models/*.pth` and allows 1-click upload to the user's Hugging Face model repository.
- [ ] **Step 2: Create `docs/DEPLOYMENT_STREAMLIT.md`**
Detailed step-by-step instructions:
1. Setting up free Hugging Face Model Hub (or GitHub Releases).
2. Uploading the 5 `.pth` files.
3. Deploying on share.streamlit.io (connecting GitHub, setting entry point `app.py`).
4. Configuring Secrets (e.g., `HF_MODELS_REPO`).
- [ ] **Step 3: Run complete pytest suite across the entire project**
Run: `.venv/bin/pytest`
- [ ] **Step 4: Commit**
```bash
git add scripts/upload_models_to_hf.py docs/DEPLOYMENT_STREAMLIT.md
git commit -m "docs: add free streamlit community cloud deployment guide and hf upload script"
```
