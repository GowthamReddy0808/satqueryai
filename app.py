"""
SatQuery AI - Remote-Sensing Vision-Language Assistant
Problem Statement: ST-09 / SIH26167 — ISRO Remote-Sensing Vision-Language Assistant
Professional Geospatial Intelligence & Earth Observation Platform
"""

import os
import io
import time
import base64
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from dotenv import load_dotenv

# Core & Tools Packages
from core import (
    AgenticQueryOrchestrator,
    TaskType,
    FeatureEvidenceState,
    BoundingBox,
    VisualOutput,
    TraceStep,
    SatQueryResult
)
from tools import (
    inspect_raster_metadata,
    calculate_ndvi,
    generate_spectral_masks,
    validate_bi_temporal_pair,
    compute_bi_temporal_change_map,
    process_optical_sar_pair,
    draw_bounding_boxes
)
from models import (
    TeammateModelAdapter,
    ModelRegistry,
    query_multimodal_vlm,
    extract_features
)
from evaluation import BENCHMARK_DATASETS

# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="SatQuery AI • Earth Observation Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External Clean CSS
CSS_FILE = Path(__file__).parent / "style.css"
if CSS_FILE.exists():
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Single-Image Analysis"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Single Image State
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "image_name" not in st.session_state:
    st.session_state.image_name = None
if "image_type" not in st.session_state:
    st.session_state.image_type = None
if "image_meta" not in st.session_state:
    st.session_state.image_meta = {}

# Bi-Temporal State
if "t1_bytes" not in st.session_state:
    st.session_state.t1_bytes = None
if "t1_name" not in st.session_state:
    st.session_state.t1_name = None
if "t2_bytes" not in st.session_state:
    st.session_state.t2_bytes = None
if "t2_name" not in st.session_state:
    st.session_state.t2_name = None

# Optical-SAR State
if "opt_bytes" not in st.session_state:
    st.session_state.opt_bytes = None
if "opt_name" not in st.session_state:
    st.session_state.opt_name = None
if "sar_bytes" not in st.session_state:
    st.session_state.sar_bytes = None
if "sar_name" not in st.session_state:
    st.session_state.sar_name = None

# Teammate API Configuration
if "teammate_api_url" not in st.session_state:
    st.session_state.teammate_api_url = os.getenv("TEAMMATE_MODEL_API_URL", "http://localhost:8000")
if "teammate_api_key" not in st.session_state:
    st.session_state.teammate_api_key = os.getenv("TEAMMATE_MODEL_API_KEY", "")
if "use_teammate_model" not in st.session_state:
    st.session_state.use_teammate_model = False

# Active Analysis Result
if "active_result" not in st.session_state:
    st.session_state.active_result = None

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None


# -----------------------------------------------------------------------------
# Helper Functions: Secrets & Data
# -----------------------------------------------------------------------------
def get_api_key() -> Optional[str]:
    """Retrieve OpenRouter API key supporting session state, env vars, and Streamlit secrets."""
    if "custom_api_key" in st.session_state and st.session_state["custom_api_key"]:
        return str(st.session_state["custom_api_key"]).strip()

    env_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
    if env_key:
        return env_key.strip()

    try:
        if hasattr(st, "secrets"):
            if "OPENROUTER_API_KEY" in st.secrets:
                return str(st.secrets["OPENROUTER_API_KEY"]).strip()
            if "OPEN_ROUTER_API_KEY" in st.secrets:
                return str(st.secrets["OPEN_ROUTER_API_KEY"]).strip()
    except Exception:
        pass

    return None


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Convert PIL image to PNG byte stream."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_sample_single_image():
    """Load the bundled 512x513 satellite scene."""
    sample_path = Path(__file__).parent / "satellite.png"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            data = f.read()
        st.session_state.image_bytes = data
        st.session_state.image_name = "satellite_sentinel2_cir.png"
        st.session_state.image_type = "image/png"
        st.session_state.image_meta = inspect_raster_metadata(data, "satellite_sentinel2_cir.png")
        st.session_state.messages = []
        st.session_state.active_result = None
        st.toast("Loaded sample satellite scene (Sentinel-2 CIR)!", icon="🛰️")
        st.rerun()
    else:
        st.error("Sample image 'satellite.png' was not found.")


def load_sample_bitemporal_pair():
    """Generate and load a simulated bi-temporal acquisition pair (T1 & T2) for instant demo."""
    sample_path = Path(__file__).parent / "satellite.png"
    if not sample_path.exists():
        st.error("Sample base image 'satellite.png' not found.")
        return

    with open(sample_path, "rb") as f:
        t1_data = f.read()

    # Create T2 by simulating a landscape transition (clearing / new built structure)
    pil_t1 = Image.open(io.BytesIO(t1_data)).convert("RGB")
    arr_t2 = np.array(pil_t1)
    h, w, _ = arr_t2.shape

    # Simulate development in southeast quadrant
    y1, y2 = int(h * 0.6), int(h * 0.85)
    x1, x2 = int(w * 0.55), int(w * 0.85)
    # Brighten / alter reflectance to simulate urban construction clearing
    arr_t2[y1:y2, x1:x2, 0] = np.clip(arr_t2[y1:y2, x1:x2, 0] * 1.6 + 50, 0, 255)
    arr_t2[y1:y2, x1:x2, 1] = np.clip(arr_t2[y1:y2, x1:x2, 1] * 1.5 + 40, 0, 255)
    arr_t2[y1:y2, x1:x2, 2] = np.clip(arr_t2[y1:y2, x1:x2, 2] * 1.4 + 30, 0, 255)

    pil_t2 = Image.fromarray(arr_t2)
    buf_t2 = io.BytesIO()
    pil_t2.save(buf_t2, format="PNG")
    t2_data = buf_t2.getvalue()

    st.session_state.t1_bytes = t1_data
    st.session_state.t1_name = "Scene_2023_T1_Earlier.png"
    st.session_state.t2_bytes = t2_data
    st.session_state.t2_name = "Scene_2024_T2_Later.png"
    st.session_state.active_result = None
    st.toast("Loaded bi-temporal sample pair (2023 vs 2024)!", icon="🔄")
    st.rerun()


def load_sample_optical_sar_pair():
    """Load sample Optical and SAR pair."""
    sample_path = Path(__file__).parent / "satellite.png"
    if not sample_path.exists():
        st.error("Sample base image 'satellite.png' not found.")
        return

    with open(sample_path, "rb") as f:
        opt_data = f.read()

    # Synthesize SAR backscatter from optical image
    pil_opt = Image.open(io.BytesIO(opt_data)).convert("L")
    # Radar roughness simulation: edges + speckle noise
    edges = pil_opt.filter(ImageFilter.FIND_EDGES)
    arr_edges = np.array(edges, dtype=float)
    arr_opt = np.array(pil_opt, dtype=float)
    sar_combined = arr_edges * 1.2 + arr_opt * 0.4
    # Add multiplicative speckle noise characteristic of SAR
    np.random.seed(42)
    speckle = np.random.gamma(4, 0.25, sar_combined.shape)
    sar_simulated = np.clip(sar_combined * speckle, 0, 255).astype(np.uint8)

    sar_img = Image.fromarray(sar_simulated)
    sar_buf = io.BytesIO()
    sar_img.save(sar_buf, format="PNG")
    sar_data = sar_buf.getvalue()

    st.session_state.opt_bytes = opt_data
    st.session_state.opt_name = "Sentinel2_Optical_RGB.png"
    st.session_state.sar_bytes = sar_data
    st.session_state.sar_name = "Sentinel1_SAR_Backscatter.png"
    st.session_state.active_result = None
    st.toast("Loaded sample Optical + SAR pair!", icon="📡")
    st.rerun()


def clear_all_inputs():
    """Reset all uploaded files, results, and increment uploader key."""
    st.session_state.image_bytes = None
    st.session_state.image_name = None
    st.session_state.image_type = None
    st.session_state.image_meta = {}
    st.session_state.t1_bytes = None
    st.session_state.t1_name = None
    st.session_state.t2_bytes = None
    st.session_state.t2_name = None
    st.session_state.opt_bytes = None
    st.session_state.opt_name = None
    st.session_state.sar_bytes = None
    st.session_state.sar_name = None
    st.session_state.active_result = None
    st.session_state.messages = []
    st.session_state.uploader_key += 1
    st.rerun()


def generate_comprehensive_report_markdown(result: SatQueryResult, image_name: str, query: str) -> str:
    """Generate formatted Markdown analysis report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    meta = result.metadata
    tags_str = ", ".join(result.feature_tags) if result.feature_tags else "General Earth Observation"

    trace_md = ""
    for t in result.execution_trace:
        trace_md += f"- **[{t.status}]** `{t.step_name}` via *{t.tool_or_model}* ({t.duration_ms}ms): {t.detail}\n"

    limitations_md = ""
    if result.limitations:
        limitations_md = "### Limitations & Scientific Notices:\n" + "\n".join([f"- {lim}" for lim in result.limitations]) + "\n"

    reasoning_md = ""
    if result.reasoning:
        reasoning_md = f"### Internal Model Reasoning:\n```text\n{result.reasoning}\n```\n"

    return f"""# SatQuery AI • Earth Observation Analysis Report
**Generated:** {timestamp}  
**Problem Statement:** ST-09 / SIH26167 — ISRO Remote-Sensing Vision-Language Assistant  
**Task Category:** `{result.task_type.value}`  
**Status:** `{result.status.upper()}`  

---

## 1. Input Scene Metadata
- **Image Filename:** `{image_name}`
- **Dimensions:** {meta.get('width', 'N/A')} × {meta.get('height', 'N/A')} px ({meta.get('megapixels', 'N/A')} MP)
- **Format / Modality:** {meta.get('format', 'N/A')} • {meta.get('modality', 'Optical')}
- **Coordinate Reference System (CRS):** `{meta.get('crs', 'Non-georeferenced pixel coordinates')}`
- **Spatial Resolution:** {meta.get('resolution', 'Standard Pixel Space')}

---

## 2. Analytical Inquiry
> "{query}"

---

## 3. Evidence-Grounded Assessment
**Identified Signatures:** {tags_str}  
**Total Processing Duration:** {result.duration_sec}s  

{result.answer}

{reasoning_md}
{limitations_md}
---

## 4. Auditable Execution Trace
{trace_md}

---
*Generated by SatQuery AI — Multi-modal Remote Sensing Intelligence Platform.*
"""


def generate_comprehensive_report_html(result: SatQueryResult, image_name: str, query: str) -> str:
    """Generate printable HTML report."""
    answer_html = result.answer.replace("\n", "<br>")
    tags_html = " ".join([f"<span class='badge badge-cyan'>{t}</span>" for t in result.feature_tags])
    meta = result.metadata
    trace_rows = "".join([
        f"<tr><td><code>{t.status}</code></td><td>{t.step_name}</td><td>{t.tool_or_model}</td><td>{t.duration_ms}ms</td><td>{t.detail}</td></tr>"
        for t in result.execution_trace
    ])

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SatQuery AI Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #0f172a; max-width: 850px; margin: 30px auto; padding: 25px; }}
  h1 {{ color: #0284c7; border-bottom: 2px solid #0284c7; padding-bottom: 8px; margin-top: 5px; }}
  h2 {{ color: #1e293b; margin-top: 24px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }}
  blockquote {{ background: #f8fafc; border-left: 4px solid #0284c7; margin: 12px 0; padding: 12px 18px; font-style: italic; }}
  .tag {{ display: inline-block; background: #e0f2fe; color: #0369a1; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; margin-right: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
  th {{ background: #f1f5f9; }}
  .footer {{ margin-top: 40px; font-size: 0.8rem; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
</style>
</head>
<body>
<div style="font-size: 0.8rem; text-transform: uppercase; color: #0284c7; font-weight: bold;">SatQuery AI • ISRO Remote-Sensing Vision-Language Assistant</div>
<h1>Geospatial Intelligence Assessment</h1>
<p><strong>Scene:</strong> {image_name} | <strong>Task:</strong> {result.task_type.value} | <strong>Status:</strong> {result.status.upper()}</p>
<h2>1. Inquiry</h2>
<blockquote>"{query}"</blockquote>
<h2>2. Evidence & Signatures</h2>
<p>{tags_html}</p>
<h2>3. Model Assessment</h2>
<div>{answer_html}</div>
<h2>4. Auditable Execution Trace</h2>
<table>
  <tr><th>Status</th><th>Step</th><th>Tool / Model</th><th>Latency</th><th>Detail</th></tr>
  {trace_rows}
</table>
<div class="footer">SatQuery AI Platform • Grounded Multimodal Earth Observation Pipeline</div>
</body>
</html>"""


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM TELEMETRY
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 1.8rem;">🛰️</span>
            <div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; line-height: 1.1;">SatQuery AI</div>
                <div style="font-size: 0.72rem; color: #38bdf8; letter-spacing: 0.05em; text-transform: uppercase;">ISRO Remote-Sensing Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("➕ New Analysis", use_container_width=True, type="primary"):
        clear_all_inputs()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 7 Sidebar Navigation Pages as required by Section 3
    nav_options = [
        "Single-Image Analysis",
        "Change Detection (Bi-Temporal)",
        "Optical + SAR Analysis",
        "Analysis History",
        "Dataset Information",
        "Model Status & Registry",
        "About the Project"
    ]

    selected_nav = st.radio(
        "Workflow Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.nav_page) if st.session_state.nav_page in nav_options else 0,
        label_visibility="collapsed"
    )
    st.session_state.nav_page = selected_nav

    st.markdown("---")

    # API Connection Status Indicator
    openrouter_key = get_api_key()
    api_connected = bool(openrouter_key)

    adapter = TeammateModelAdapter(
        api_base_url=st.session_state.teammate_api_url,
        api_key=st.session_state.teammate_api_key
    )
    tm_health = adapter.check_health()
    teammate_online = tm_health.get("available", False)

    if teammate_online:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #34d399; margin-bottom: 6px;">
                <span class="status-indicator"></span> <strong>Teammate RS Model: Online</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif api_connected:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #38bdf8; margin-bottom: 6px;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#38bdf8;"></span> <strong>OpenRouter VLM: Connected</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #f87171; margin-bottom: 6px;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#ef4444;"></span> <strong>API Key Missing</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        user_key = st.text_input(
            "Enter OpenRouter Key",
            type="password",
            placeholder="sk-or-v1-...",
            help="Configure OPENROUTER_API_KEY or enter your key here."
        )
        if user_key:
            st.session_state.custom_api_key = user_key.strip()
            st.rerun()

    # Model Configuration in Expander
    with st.expander("⚙️ Inference Parameters", expanded=False):
        model_name = st.selectbox(
            "OpenRouter Vision Model",
            options=[
                "nvidia/nemotron-3.5-super-vl:free",
                "google/gemma-4-26b-a4b-it:free",
                "qwen/qwen3-vl-8b-instruct"
            ],
            index=0
        )
        st.session_state.use_teammate_model = st.checkbox(
            "Prioritize Teammate RS Model",
            value=st.session_state.use_teammate_model,
            help="When checked, routes queries to your teammate's custom model endpoint if available."
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.slider("Max Tokens", 300, 2500, 1500, 100)

    # Session stats
    st.caption(f"Queries Recorded: {len(st.session_state.history)}")


# -----------------------------------------------------------------------------
# WORKFLOW CONTROLLER INSTANTIATION
# -----------------------------------------------------------------------------
orchestrator = AgenticQueryOrchestrator(teammate_adapter=adapter)


# -----------------------------------------------------------------------------
# PAGE 1: SINGLE-IMAGE ANALYSIS
# -----------------------------------------------------------------------------
if st.session_state.nav_page == "Single-Image Analysis":

    # VIEW A: WELCOME & UPLOAD SCREEN
    if st.session_state.image_bytes is None:
        st.markdown(
            """
            <div style="text-align: center; max-width: 820px; margin: 15px auto 30px auto;">
                <div class="brand-pill">🛰️ SatQuery AI • Problem Statement ST-09 / SIH26167</div>
                <div class="brand-title">Explore Earth with SatQuery AI</div>
                <div class="brand-subtitle">
                    Ask questions about satellite imagery, locate important regions, compare observations, and discover evidence-grounded insights.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4 Core Capability HUD Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🌿</div>
                    <div class="feature-title">Vegetation Analysis</div>
                    <div class="feature-desc">Calibrated CIR canopy reflectance and physical NDVI calculation with genuine NIR band verification.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">Region Grounding</div>
                    <div class="feature-desc">Text-guided spatial boundary localization with semi-transparent bounding box overlays.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🗺️</div>
                    <div class="feature-title">Land Cover Analysis</div>
                    <div class="feature-desc">Distinguish supported classes including vegetation, hydrology, bare soil, and built-up areas.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Auditable Trace</div>
                    <div class="feature-desc">Every step audited: model selection, raster metadata, latency metrics, and honest abstentions.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        up_col1, up_col2 = st.columns([2.4, 1], gap="medium")
        with up_col1:
            st.markdown(
                """
                <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">
                    Upload Satellite Imagery
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
                    Select an optical, multispectral, or SAR image in GeoTIFF (.tif, .tiff), PNG, or JPEG format.
                </div>
                """,
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Upload Single Satellite Scene",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key=f"single_up_{st.session_state.uploader_key}",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                st.session_state.image_bytes = bytes_data
                st.session_state.image_name = uploaded_file.name
                st.session_state.image_type = uploaded_file.type or "image/png"
                st.session_state.image_meta = inspect_raster_metadata(bytes_data, uploaded_file.name)
                st.session_state.messages = []
                st.session_state.active_result = None
                st.rerun()

        with up_col2:
            st.markdown(
                """
                <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 18px;">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-bottom: 6px;">
                        Quick Hackathon Demo
                    </div>
                    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.45; margin-bottom: 14px;">
                        Instant testing with our bundled 512×513 false-colour infrared Sentinel-2 satellite scene.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Load Sample Scene", use_container_width=True, type="primary"):
                load_sample_single_image()

    # VIEW B: ACTIVE WORKSPACE (IMAGE LOADED)
    else:
        col_left, col_right = st.columns([1.1, 1.4], gap="large")

        # LEFT COLUMN: SATELLITE VIEWER & RASTER STATS
        with col_left:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">
                        🛰️ Satellite Viewer
                    </div>
                    <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 500;">
                        {st.session_state.image_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Left Tabs: Scene, Spectral Masks, Metadata
            tab_scene, tab_masks, tab_meta = st.tabs([
                "🖼️ Scene View",
                "🔬 Spectral Overlays",
                "📊 Geospatial Metadata"
            ])

            with tab_scene:
                st.image(
                    st.session_state.image_bytes,
                    caption=f"{st.session_state.image_name} ({st.session_state.image_meta.get('width')}×{st.session_state.image_meta.get('height')} px)",
                    use_container_width=True
                )

            with tab_masks:
                masks = generate_spectral_masks(st.session_state.image_bytes)
                mask_choice = st.radio(
                    "Spectral Layer",
                    options=["🌿 Vegetation Cover", "💧 Water Body / Absorption", "🏜️ Bare Soil & Albedo", "🔥 Radiometric Heatmap"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                m_key = "vegetation"
                if "Water" in mask_choice:
                    m_key = "water"
                elif "Soil" in mask_choice:
                    m_key = "soil"
                elif "Heatmap" in mask_choice:
                    m_key = "heatmap"

                gen_img, cov_pct, desc_str = masks[m_key]
                st.image(gen_img, caption=f"{mask_choice} (Coverage: {cov_pct}%)", use_container_width=True)
                st.caption(desc_str)
                st.download_button(
                    label=f"📥 Download {m_key.capitalize()} Mask (.png)",
                    data=pil_to_png_bytes(gen_img),
                    file_name=f"SatQuery_{m_key}_mask.png",
                    mime="image/png",
                    use_container_width=True,
                    type="secondary"
                )

            with tab_meta:
                meta = st.session_state.image_meta
                meta_table = f"""
                | Parameter | Measured Specification |
                | :--- | :--- |
                | **Filename** | `{meta.get('filename')}` |
                | **Dimensions** | {meta.get('width')} × {meta.get('height')} px |
                | **Band Count** | {meta.get('band_count')} channels |
                | **Modality** | {meta.get('modality')} |
                | **Data Type** | `{meta.get('dtype')}` |
                | **CRS** | `{meta.get('crs')}` |
                | **Spatial Resolution** | {meta.get('resolution')} |
                | **File Size** | {meta.get('size_kb')} KB |
                """
                st.markdown(meta_table)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("❌ Remove Image", use_container_width=True, type="secondary"):
                clear_all_inputs()

        # RIGHT COLUMN: AGENTIC QUERY & RESULTS WORKSPACE
        with col_right:
            st.markdown(
                """
                <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">
                    💬 Earth Observation Intelligence Assistant
                </div>
                <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 12px;">
                    Ask questions regarding land-cover features, vegetation canopy, hydrology, or object localization.
                </div>
                """,
                unsafe_allow_html=True
            )

            triggered_query = None

            # Prominent Search Form
            with st.form(key="search_form", clear_on_submit=False):
                st.markdown(
                    """
                    <div style="font-size: 0.82rem; font-weight: 600; color: #38bdf8; margin-bottom: 6px;">
                        🔍 SATELLITE QUERY & SCENE SEARCH
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                search_val = st.text_input(
                    "Search Input",
                    placeholder="e.g. Describe the land-cover and major objects in this image...",
                    label_visibility="collapsed"
                )

                b_col1, b_col2 = st.columns([1, 1], gap="small")
                with b_col1:
                    run_submitted = st.form_submit_button("🔍 Run Analysis", type="primary", use_container_width=True)
                with b_col2:
                    clear_submitted = st.form_submit_button("🧹 Clear Chat", type="secondary", use_container_width=True)

            if clear_submitted:
                st.session_state.messages = []
                st.session_state.active_result = None
                st.rerun()

            if run_submitted and search_val.strip():
                triggered_query = search_val.strip()

            # Suggested Inquiry Chips
            st.markdown("<div style='font-size: 0.78rem; color: #cbd5e1; font-weight: 500; margin: 6px 0;'>💡 Suggested Inquiries:</div>", unsafe_allow_html=True)
            q_c1, q_c2 = st.columns(2, gap="small")
            with q_c1:
                if st.button("🌿 Where is vegetation visible?", use_container_width=True, type="secondary"):
                    triggered_query = "Where is vegetation visible in this image? Explain the spectral colours."
                if st.button("🗺️ Describe land-cover & major objects", use_container_width=True, type="secondary"):
                    triggered_query = "Describe the land-cover and major objects in this image."
            with q_c2:
                if st.button("💧 Highlight the water body", use_container_width=True, type="secondary"):
                    triggered_query = "Highlight the water body referred to in my question and identify its channel."
                if st.button("🔢 Count supported objects", use_container_width=True, type="secondary"):
                    triggered_query = "Count supported objects in the image and locate them."

            # Execute Query via Agentic Orchestrator
            if triggered_query:
                st.session_state.messages.append({"role": "user", "content": triggered_query})

                with st.spinner("Executing agentic workflow: classifying query, verifying feasibility, invoking specialist tools..."):
                    result: SatQueryResult = orchestrator.execute_workflow(
                        query=triggered_query,
                        image_bytes_list=[st.session_state.image_bytes],
                        image_names=[st.session_state.image_name],
                        image_types=[st.session_state.image_type or "image/png"],
                        active_model_name=model_name,
                        openrouter_api_key=openrouter_key,
                        use_teammate_model=st.session_state.use_teammate_model
                    )

                st.session_state.active_result = result
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "result": result
                })

                st.session_state.history.append({
                    "id": f"SQ-{len(st.session_state.history)+1}",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_name": st.session_state.image_name,
                    "task_type": result.task_type.value,
                    "question": triggered_query,
                    "answer": result.answer,
                    "tags": result.feature_tags,
                    "result": result
                })
                st.rerun()

            # Dedicated Results Workspace (Tabs mandated by Section 15)
            if st.session_state.active_result is not None:
                res: SatQueryResult = st.session_state.active_result

                st.markdown("---")
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">
                            🎯 Analysis Results & Grounded Evidence
                        </div>
                        <div class="badge badge-cyan">
                            Task: {res.task_type.value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Determine dynamic tabs based on active outputs
                tab_names = ["💬 Answer & Evidence"]
                has_detections = len(res.detections) > 0
                has_visuals = len(res.visual_outputs) > 0
                
                if has_detections:
                    tab_names.append("🎯 Annotated Regions")
                if has_visuals:
                    tab_names.append("🔬 Visual Products")
                tab_names.append("⚡ Execution Trace")
                tab_names.append("📥 Export Report")

                res_tabs = st.tabs(tab_names)

                # Tab 1: Answer & Evidence
                with res_tabs[0]:
                    if res.status == "abstention":
                        st.warning(f"⚠️ **Abstention Notice:** {res.abstention_reason}")

                    tags_html = "".join([f"<span class='badge badge-cyan'>{t}</span>" for t in res.feature_tags])
                    st.markdown(f"<div style='margin-bottom: 10px;'>{tags_html}</div>", unsafe_allow_html=True)
                    st.markdown(res.answer)

                    if res.reasoning:
                        with st.expander("🧠 Internal Model Reasoning Chain", expanded=False):
                            st.code(res.reasoning, language="text")

                    if res.limitations:
                        with st.expander("⚠️ Limitations & Scientific Notices", expanded=True):
                            for lim in res.limitations:
                                st.markdown(f"- {lim}")

                # Tab 2: Annotated Regions (if available)
                tab_idx = 1
                if has_detections:
                    with res_tabs[tab_idx]:
                        for vis in res.visual_outputs:
                            if vis.type == "annotated_image":
                                st.image(vis.image, caption=vis.title, use_container_width=True)
                                st.caption(vis.description)
                                st.download_button(
                                    "📥 Download Annotated PNG",
                                    data=pil_to_png_bytes(vis.image),
                                    file_name="SatQuery_Annotated_Regions.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                    tab_idx += 1

                # Tab 3: Visual Products (if available)
                if has_visuals:
                    with res_tabs[tab_idx]:
                        for vis in res.visual_outputs:
                            if vis.type != "annotated_image":
                                st.image(vis.image, caption=vis.title, use_container_width=True)
                                st.caption(vis.description)
                                st.download_button(
                                    f"📥 Download {vis.title} (.png)",
                                    data=pil_to_png_bytes(vis.image),
                                    file_name=f"SatQuery_{vis.type}.png",
                                    mime="image/png",
                                    key=f"dl_vis_{vis.type}",
                                    use_container_width=True
                                )
                    tab_idx += 1

                # Tab 4: Execution Trace (Section 13)
                with res_tabs[tab_idx]:
                    st.markdown(
                        f"""
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 10px;">
                            Total Pipeline Latency: <strong>{res.duration_sec}s</strong> • Audited Workflow Steps:
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    for step in res.execution_trace:
                        color = "#38bdf8" if step.status == "SUCCESS" else "#f59e0b" if step.status == "WARNING" else "#ef4444"
                        st.markdown(
                            f"""
                            <div class="trace-step-card" style="border-left-color: {color};">
                                <div class="trace-step-title">
                                    <span>{step.step_name}</span>
                                    <span style="color: {color}; font-size: 0.78rem;">[{step.status}] • {step.duration_ms}ms</span>
                                </div>
                                <div class="trace-step-detail">Tool: {step.tool_or_model} | {step.detail}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    tab_idx += 1

                # Tab 5: Export Report (Section 16)
                with res_tabs[tab_idx]:
                    last_q = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "Satellite Query"
                    rep_md = generate_comprehensive_report_markdown(res, st.session_state.image_name, last_q)
                    rep_html = generate_comprehensive_report_html(res, st.session_state.image_name, last_q)

                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            "📥 Download Report (.md)",
                            data=rep_md,
                            file_name=f"SatQuery_Report_{int(time.time())}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    with dl2:
                        st.download_button(
                            "📄 Download Printable (.html)",
                            data=rep_html,
                            file_name=f"SatQuery_Report_{int(time.time())}.html",
                            mime="text/html",
                            use_container_width=True
                        )


# -----------------------------------------------------------------------------
# PAGE 2: BI-TEMPORAL CHANGE DETECTION
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Change Detection (Bi-Temporal)":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="brand-pill">Section 9 • Temporal Satellite Analysis</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">🔄 Bi-Temporal Change Detection & Analysis</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">
                Compare earlier (T1) and later (T2) satellite observations of the same geographic area to observe landscape shifts, urban growth, or deforestation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    t_col1, t_col2 = st.columns(2, gap="medium")

    with t_col1:
        st.markdown("**1. Earlier Observation (Date T1)**")
        t1_file = st.file_uploader(
            "Upload Earlier Image (T1)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key=f"t1_up_{st.session_state.uploader_key}"
        )
        if t1_file is not None:
            st.session_state.t1_bytes = t1_file.getvalue()
            st.session_state.t1_name = t1_file.name

        if st.session_state.t1_bytes:
            st.image(st.session_state.t1_bytes, caption=f"T1: {st.session_state.t1_name}", use_container_width=True)

    with t_col2:
        st.markdown("**2. Later Observation (Date T2)**")
        t2_file = st.file_uploader(
            "Upload Later Image (T2)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key=f"t2_up_{st.session_state.uploader_key}"
        )
        if t2_file is not None:
            st.session_state.t2_bytes = t2_file.getvalue()
            st.session_state.t2_name = t2_file.name

        if st.session_state.t2_bytes:
            st.image(st.session_state.t2_bytes, caption=f"T2: {st.session_state.t2_name}", use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Load Sample Bi-Temporal Pair (2023 vs 2024 Demo)", type="secondary"):
        load_sample_bitemporal_pair()

    # Analytical Query Section for Change
    st.markdown("---")
    st.markdown("### 🔍 Temporal Inquiry & Analysis")

    q_change = st.text_input(
        "Ask a question about the temporal change",
        value="What changed between these two dates? Has the built-up area increased, decreased, or remained unchanged?",
        placeholder="e.g. Has the built-up area increased or decreased?"
    )

    c_chips = st.columns(3)
    with c_chips[0]:
        if st.button("What changed between these dates?", use_container_width=True, type="secondary"):
            q_change = "What changed between these two dates?"
    with c_chips[1]:
        if st.button("Where did the change occur?", use_container_width=True, type="secondary"):
            q_change = "Where did the change occur?"
    with c_chips[2]:
        if st.button("Has built-up area increased?", use_container_width=True, type="secondary"):
            q_change = "Has the built-up area increased, decreased, or remained unchanged?"

    if st.button("🔄 Compute Change Detection Map", type="primary", use_container_width=True):
        if not st.session_state.t1_bytes or not st.session_state.t2_bytes:
            st.warning("⚠️ **Abstention:** Bi-temporal change detection requires both an earlier (T1) and later (T2) satellite scene.")
        else:
            with st.spinner("Executing radiometric delta mapping, spatial alignment validation, and change categorization..."):
                change_res = orchestrator.execute_workflow(
                    query=q_change,
                    image_bytes_list=[st.session_state.t1_bytes, st.session_state.t2_bytes],
                    image_names=[st.session_state.t1_name or "T1_Scene", st.session_state.t2_name or "T2_Scene"],
                    image_types=["image/png", "image/png"]
                )

            st.session_state.active_result = change_res

            # Display Change Results Tabs
            st.markdown("---")
            c_tab1, c_tab2, c_tab3 = st.tabs(["🗺️ Change Maps & Overlays", "💬 Change Assessment", "⚡ Execution Trace"])

            with c_tab1:
                col_m1, col_m2 = st.columns(2)
                for vis in change_res.visual_outputs:
                    if vis.type == "change_overlay":
                        with col_m1:
                            st.image(vis.image, caption=vis.title, use_container_width=True)
                            st.caption(vis.description)
                            st.download_button(
                                "📥 Download Classified Change Map (.png)",
                                data=pil_to_png_bytes(vis.image),
                                file_name="SatQuery_Change_Overlay.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    elif vis.type == "change_heatmap":
                        with col_m2:
                            st.image(vis.image, caption=vis.title, use_container_width=True)
                            st.caption(vis.description)
                            st.download_button(
                                "📥 Download Radiometric Delta Heatmap (.png)",
                                data=pil_to_png_bytes(vis.image),
                                file_name="SatQuery_Change_Heatmap.png",
                                mime="image/png",
                                use_container_width=True
                            )

            with c_tab2:
                st.markdown(change_res.answer)

            with c_tab3:
                for step in change_res.execution_trace:
                    st.markdown(
                        f"""
                        <div class="trace-step-card">
                            <div class="trace-step-title">
                                <span>{step.step_name}</span>
                                <span style="color: #38bdf8;">[{step.status}] • {step.duration_ms}ms</span>
                            </div>
                            <div class="trace-step-detail">Tool: {step.tool_or_model} | {step.detail}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# -----------------------------------------------------------------------------
# PAGE 3: OPTICAL + SAR ANALYSIS
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Optical + SAR Analysis":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="brand-pill">Section 10 • Cross-Modal Remote Sensing</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">📡 Optical + SAR Joint Analysis</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">
                Combine optical surface reflectance with Synthetic Aperture Radar (SAR) structural roughness and dielectric properties.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    os_col1, os_col2 = st.columns(2, gap="medium")
    with os_col1:
        st.markdown("**1. Optical / Multispectral Observation**")
        opt_f = st.file_uploader("Upload Optical Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key=f"opt_up_{st.session_state.uploader_key}")
        if opt_f:
            st.session_state.opt_bytes = opt_f.getvalue()
            st.session_state.opt_name = opt_f.name
        if st.session_state.opt_bytes:
            st.image(st.session_state.opt_bytes, caption=f"Optical: {st.session_state.opt_name}", use_container_width=True)

    with os_col2:
        st.markdown("**2. SAR (Synthetic Aperture Radar) Observation**")
        sar_f = st.file_uploader("Upload SAR Backscatter Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key=f"sar_up_{st.session_state.uploader_key}")
        if sar_f:
            st.session_state.sar_bytes = sar_f.getvalue()
            st.session_state.sar_name = sar_f.name
        if st.session_state.sar_bytes:
            st.image(st.session_state.sar_bytes, caption=f"SAR: {st.session_state.sar_name}", use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Load Sample Optical + SAR Pair Demo", type="secondary"):
        load_sample_optical_sar_pair()

    st.markdown("---")
    st.markdown("### 🔍 Joint Sensor Inquiry")
    q_sar = st.text_input(
        "Joint Inquiry Query",
        value="What information do the optical and SAR images provide together? What structural information is visible in the SAR image?"
    )

    if st.button("📡 Execute Optical–SAR Joint Analysis", type="primary", use_container_width=True):
        if not st.session_state.opt_bytes or not st.session_state.sar_bytes:
            st.warning("⚠️ **Abstention:** Joint analysis requires both an optical reflectance image and a SAR backscatter scene.")
        else:
            with st.spinner("Processing cross-modal sensor synthesis..."):
                sar_res = orchestrator.execute_workflow(
                    query=q_sar,
                    image_bytes_list=[st.session_state.opt_bytes, st.session_state.sar_bytes],
                    image_names=[st.session_state.opt_name or "Optical", st.session_state.sar_name or "SAR"],
                    image_types=["image/png", "image/png"]
                )

            st.session_state.active_result = sar_res

            st.markdown("---")
            sar_tab1, sar_tab2, sar_tab3 = st.tabs(["🔬 Complementary Visuals", "💬 Modality Assessment", "⚡ Execution Trace"])

            with sar_tab1:
                c1, c2 = st.columns(2)
                for vis in sar_res.visual_outputs:
                    if vis.type == "optical_sar_composite":
                        with c1:
                            st.image(vis.image, caption=vis.title, use_container_width=True)
                            st.caption(vis.description)
                            st.download_button(
                                "📥 Download Structural-Spectral Composite (.png)",
                                data=pil_to_png_bytes(vis.image),
                                file_name="SatQuery_Optical_SAR_Composite.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    elif vis.type == "sar_backscatter":
                        with c2:
                            st.image(vis.image, caption=vis.title, use_container_width=True)
                            st.caption(vis.description)

            with sar_tab2:
                st.markdown(sar_res.answer)

            with sar_tab3:
                for step in sar_res.execution_trace:
                    st.markdown(
                        f"""
                        <div class="trace-step-card">
                            <div class="trace-step-title">
                                <span>{step.step_name}</span>
                                <span style="color: #38bdf8;">[{step.status}] • {step.duration_ms}ms</span>
                            </div>
                            <div class="trace-step-detail">Tool: {step.tool_or_model} | {step.detail}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# -----------------------------------------------------------------------------
# PAGE 4: ANALYSIS HISTORY
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Analysis History":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="brand-pill">Audit Trail & Session Records</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">📜 Analysis History</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">Review questions, detected signatures, and assessments from this session.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:
        st.info("No analyses recorded in this session yet. Upload a satellite scene or pair to begin.")
    else:
        for item in reversed(st.session_state.history):
            with st.container():
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.18); border-radius: 12px; padding: 18px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div style="font-size: 0.84rem; color: #38bdf8; font-weight: 600;">
                                {item['id']} • {item['image_name']}
                            </div>
                            <div style="font-size: 0.78rem; color: #94a3b8;">
                                🕒 {item['timestamp']} • Task: <code>{item.get('task_type', 'VQA')}</code>
                            </div>
                        </div>
                        <div style="font-size: 0.98rem; font-weight: 600; color: #f8fafc; margin-bottom: 8px;">
                            Q: "{item['question']}"
                        </div>
                        <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 12px;">
                            {item['answer'][:350]}...
                        </div>
                        <div>
                            {' '.join([f"<span class='badge badge-cyan'>{t}</span>" for t in item.get('tags', [])])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                h_c1, h_c2 = st.columns([1, 4])
                with h_c1:
                    res_obj = item.get("result")
                    if res_obj:
                        st.download_button(
                            f"📥 Export Report ({item['id']})",
                            data=generate_comprehensive_report_markdown(res_obj, item['image_name'], item['question']),
                            file_name=f"SatQuery_Report_{item['id']}.md",
                            mime="text/markdown",
                            key=f"hist_dl_{item['id']}",
                            use_container_width=True
                        )


# -----------------------------------------------------------------------------
# PAGE 5: DATASET INFORMATION (BENCHMARK DATASETS)
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Dataset Information":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="brand-pill">Section 11 • Remote Sensing Datasets</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">📚 Benchmark Datasets & Specifications</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">
                Official remote-sensing vision-language benchmark datasets identified for fine-tuning and evaluation under problem statement ST-09 / SIH26167.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    d_tabs = st.tabs(list(BENCHMARK_DATASETS.keys()))

    for idx, (name, dinfo) in enumerate(BENCHMARK_DATASETS.items()):
        with d_tabs[idx]:
            st.markdown(f"### {dinfo['title']}")
            st.markdown(f"**Primary Objective:** {dinfo['purpose']}")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(
                    f"""
                    - **Archive Scale:** {dinfo['size']}
                    - **Sensor Modalities:** {dinfo['modalities']}
                    - **License:** `{dinfo.get('license', 'Open Research Access')}`
                    """
                )
            with col_d2:
                st.markdown(
                    f"""
                    - **Supported Vision-Language Tasks:**
                    """
                    + "\n".join([f"  - `{t}`" for t in dinfo['supported_tasks']])
                )

            if "class_examples" in dinfo:
                with st.expander("Corine Land Cover Classes (BigEarthNet 19 Classes)", expanded=False):
                    st.write(", ".join(dinfo["class_examples"]))

            if "evaluation_metrics" in dinfo:
                st.markdown(f"**Standard Quantitative Metrics:** {', '.join(dinfo['evaluation_metrics'])}")

            st.caption(f"Citation: {dinfo['citation']}")


# -----------------------------------------------------------------------------
# PAGE 6: MODEL STATUS & REGISTRY
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Model Status & Registry":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="brand-pill">Section 6 • Model Adapter Layer</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">⚙️ Model Status & Teammate API Registry</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">
                Configure the integration adapter for your teammate's fine-tuned remote sensing model or inspect connected inference pipelines.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 1. Teammate Model Adapter Configuration")
    st.markdown(
        """
        Your teammate's specialized model connects via our standardized REST adapter (`models/teammate_adapter.py`).
        Configure the endpoint below or use the server environment variable `TEAMMATE_MODEL_API_URL`.
        """
    )

    c_cfg1, c_cfg2 = st.columns(2)
    with c_cfg1:
        new_url = st.text_input("Teammate API Base URL", value=st.session_state.teammate_api_url)
        if new_url != st.session_state.teammate_api_url:
            st.session_state.teammate_api_url = new_url
            st.rerun()

    with c_cfg2:
        new_key = st.text_input("Bearer Auth Token (Optional)", value=st.session_state.teammate_api_key, type="password")
        if new_key != st.session_state.teammate_api_key:
            st.session_state.teammate_api_key = new_key
            st.rerun()

    # Ping Test Button
    if st.button("📡 Test Connection to Teammate API", type="primary"):
        test_adapter = TeammateModelAdapter(api_base_url=st.session_state.teammate_api_url, api_key=st.session_state.teammate_api_key)
        h = test_adapter.check_health()
        if h["available"]:
            st.success(f"Connected successfully! Endpoint latency: {h['latency_ms']}ms. Model: {h.get('model_id')}")
        else:
            st.error(f"Inference server unreachable at {st.session_state.teammate_api_url}: {h.get('error')}")

    st.markdown("---")
    st.markdown("### 2. Registered Model Capabilities Matrix")

    registry = ModelRegistry()
    models = registry.get_registered_models()

    for m in models:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">
                        {m['name']}
                    </div>
                    <div class="badge {'badge-emerald' if m['is_specialized'] else 'badge-cyan'}">
                        {m['type']}
                    </div>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px;">
                    Endpoint: <code>{m['endpoint']}</code> | Provider: {m['provider']}
                </div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">
                    Supported Tasks: {', '.join([f'<code>{t}</code>' for t in m['supported_tasks']])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 3. Integration Contract Reference")
    st.info("The complete REST specification for the teammate's model is documented in `docs/TEAMMATE_API_CONTRACT.md`.")


# -----------------------------------------------------------------------------
# PAGE 7: ABOUT THE PROJECT
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "About the Project":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="brand-pill">ISRO Remote-Sensing Vision-Language Assistant</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">🛰️ About SatQuery AI</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">Problem Statement: ST-09 / SIH26167 • Multimodal Earth Observation Intelligence.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c_info1, c_info2 = st.columns([1.5, 1], gap="large")

    with c_info1:
        st.markdown(
            """
            ### 🌍 Overview
            **SatQuery AI** is an interactive Vision-Language Assistant designed for multimodal remote-sensing image analysis.
            It enables analysts, environmental researchers, and disaster monitoring teams to query satellite imagery using natural language
            and receive evidence-grounded answers accompanied by visual spatial outputs: bounding boxes, calibrated vegetation products,
            bi-temporal change maps, and joint optical–SAR structural representations.

            ### 🔬 Remote-Sensing Principles
            - **False-Colour Infrared (CIR):** Healthy vegetation mesophyll cells strongly reflect Near-Infrared (NIR) radiation (700-1100 nm). In CIR products, NIR is assigned to the Red display channel, rendering dense forests, crops, and grasslands bright crimson.
            - **True Physical NDVI:** Calibrated ratio `(NIR - Red) / (NIR + Red)`. Standard 3-band RGB imagery lacks a physical NIR band; SatQuery AI honestly distinguishes true NDVI from CIR canopy proxies to guarantee scientific fidelity.
            - **Bi-Temporal Radiometric Change:** Detects pixel intensity shifts across co-registered acquisition dates while accounting for seasonality, sun elevation angle, and registration displacement.
            - **Synthetic Aperture Radar (SAR):** Measures volumetric backscatter, surface roughness, and double-bounce corner reflection (urban buildings, ships), penetrating atmospheric cloud cover.

            ### ⚙️ Modular Engineering Pipeline
            - **Agentic Routing Controller:** Classifies analyst intent, verifies input feasibility, selects specialist tools, and returns auditable execution traces.
            - **Specialized Model Adapter:** Standardized interface for fine-tuned remote-sensing models (`models/teammate_adapter.py`) with OpenRouter fallback.
            - **Geospatial Raster Engine:** Rasterio GeoTIFF parsing (CRS, bounds, resolution) and NumPy radiometric calculations.
            - **Quantitative Evaluation:** Evaluator harness covering Grounding IoU@0.5, VQA accuracy, and change detection F1 scores.
            """
        )

    with c_info2:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 20px;">
                <div style="font-size: 1rem; font-weight: 600; color: #38bdf8; margin-bottom: 12px;">
                    Technical Architecture Specifications
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.7;">
                    • <strong>Problem Statement:</strong> ST-09 / SIH26167<br>
                    • <strong>Primary VLM:</strong> Nemotron 3.5 Super VL & Gemma 4<br>
                    • <strong>Teammate Model Adapter:</strong> REST / Bearer Auth Ready<br>
                    • <strong>Raster Engine:</strong> Rasterio 1.4.4 & NumPy 1.26<br>
                    • <strong>Change Detection:</strong> Radiometric Delta + Gain/Loss<br>
                    • <strong>Cross-Modal:</strong> Optical + SAR Complementary View<br>
                    • <strong>Benchmark Profiles:</strong> BigEarthNet, VRSBench, RSVQA, CDVQA<br>
                    • <strong>Security:</strong> Zero client-side API exposure
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
