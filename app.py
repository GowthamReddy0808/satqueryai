"""
SatQuery AI - Remote-Sensing Vision-Language Assistant
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
from PIL import Image
import numpy as np
from dotenv import load_dotenv

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

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "image_name" not in st.session_state:
    st.session_state.image_name = None

if "image_type" not in st.session_state:
    st.session_state.image_type = None

if "image_meta" not in st.session_state:
    st.session_state.image_meta = {}

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Satellite Analysis"

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# -----------------------------------------------------------------------------
# Helper Functions: Image & Telemetry Processing
# -----------------------------------------------------------------------------
def get_api_key() -> Optional[str]:
    """Retrieve the OpenRouter API key supporting session state, env vars, and Streamlit secrets."""
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


def load_image_from_bytes(image_bytes: bytes, filename: str, mime_type: str = "image/png"):
    """Parse image metadata and store in session state."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        width, height = pil_image.size
        size_kb = len(image_bytes) / 1024.0

        st.session_state.image_bytes = image_bytes
        st.session_state.image_name = filename
        st.session_state.image_type = mime_type
        st.session_state.image_meta = {
            "filename": filename,
            "width": width,
            "height": height,
            "format": pil_image.format or "PNG",
            "mode": pil_image.mode,
            "size_kb": size_kb,
            "megapixels": round((width * height) / 1_000_000, 2)
        }
    except Exception as e:
        st.error(f"Error loading image: {e}")


def load_sample_image():
    """Load the built-in sample satellite image from repository."""
    sample_path = Path(__file__).parent / "satellite.png"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            data = f.read()
        load_image_from_bytes(data, "satellite.png", "image/png")
        st.session_state.messages = []
        st.toast("Loaded sample satellite scene!", icon="🛰️")
        st.rerun()
    else:
        st.error("Sample image 'satellite.png' was not found.")


def clear_current_image():
    """Clear uploaded image, increment uploader key to wipe widget state, and reset messages."""
    st.session_state.image_bytes = None
    st.session_state.image_name = None
    st.session_state.image_type = None
    st.session_state.image_meta = {}
    st.session_state.messages = []
    st.session_state.uploader_key += 1
    st.rerun()


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Convert PIL image to PNG byte stream for download."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_spectral_overlay(image_bytes: bytes, mode: str) -> Tuple[Image.Image, float, str]:
    """
    Generate pixel-level Earth Observation spectral masks & annotated visualizations:
    Returns (Generated PIL Image, Coverage percentage, Description)
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(pil_img, dtype=float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    overlay = arr.copy()
    coverage_pct = 0.0

    if mode == "vegetation":
        # CIR Vegetation Index: NIR (Red) reflectance dominance over visible green
        veg_idx = (r - g) / (r + g + 1e-5)
        mask = (veg_idx > 0.25) & (r > 60)
        coverage_pct = round(float(np.mean(mask)) * 100, 1)

        # Highlight vegetation in vibrant emerald green
        overlay[mask, 0] = overlay[mask, 0] * 0.25 + 16 * 0.75
        overlay[mask, 1] = overlay[mask, 1] * 0.25 + 230 * 0.75
        overlay[mask, 2] = overlay[mask, 2] * 0.25 + 130 * 0.75
        desc = f"Identified healthy canopy reflectance across {coverage_pct}% of the scene."

    elif mode == "water":
        # Water bodies strongly absorb NIR (low Red) and have low overall reflectance
        brightness = (r + g + b) / 3.0
        mask = (r < 55) & (brightness < 60)
        coverage_pct = round(float(np.mean(mask)) * 100, 1)

        # Highlight water channels in electric cyan
        overlay[mask, 0] = overlay[mask, 0] * 0.15 + 6 * 0.85
        overlay[mask, 1] = overlay[mask, 1] * 0.15 + 182 * 0.85
        overlay[mask, 2] = overlay[mask, 2] * 0.15 + 248 * 0.85
        desc = f"Identified water absorption channels across {coverage_pct}% of the scene."

    elif mode == "soil":
        # Bare soil / high albedo terrain (neutral/balanced reflectance across bands)
        diff = np.abs(r - g) + np.abs(g - b)
        mask = (diff < 35) & (r > 70)
        coverage_pct = round(float(np.mean(mask)) * 100, 1)

        # Highlight soil in warm amber
        overlay[mask, 0] = overlay[mask, 0] * 0.25 + 245 * 0.75
        overlay[mask, 1] = overlay[mask, 1] * 0.25 + 158 * 0.75
        overlay[mask, 2] = overlay[mask, 2] * 0.25 + 11 * 0.75
        desc = f"Identified bare terrain/sediment across {coverage_pct}% of the scene."

    else:  # heatmap
        # Radiometric NIR Intensity Gradient Map
        norm_r = np.clip((r / (np.max(r) + 1e-5)) * 255, 0, 255)
        overlay[:, :, 0] = norm_r
        overlay[:, :, 1] = 255 - np.abs(norm_r - 128) * 2
        overlay[:, :, 2] = 255 - norm_r
        coverage_pct = 100.0
        desc = "Continuous false-colour radiometric thermal & near-infrared intensity heatmap."

    out_image = Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))
    return out_image, coverage_pct, desc


def extract_features(text: str) -> List[str]:
    """Identify land-cover and spectral features mentioned in model output."""
    tags = []
    text_lower = text.lower()

    mapping = {
        "🌿 Vegetation": ["vegetation", "canopy", "plant", "forest", "tree", "crop", "grass"],
        "💧 Water Body": ["water", "river", "lake", "wetland", "reservoir", "stream", "ocean", "marsh"],
        "🏙️ Built-up Area": ["built-up", "urban", "building", "settlement", "structure", "road", "infrastructure"],
        "🏜️ Bare Soil / Sand": ["bare", "soil", "sand", "arid", "ground", "sediment", "dry land"],
        "☁️ Cloud / Shadow": ["cloud", "cumulus", "shadow", "atmospheric", "haze"],
        "🔴 False-Colour NIR": ["false-color", "false-colour", "infrared", "near-infrared", "nir", "red hue"]
    }

    for tag, keywords in mapping.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)

    return tags


def compute_spectral_stats(image_bytes: bytes) -> Dict[str, Any]:
    """Compute radiometric distribution across RGB bands using NumPy."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(pil_img)
        mean_r = float(np.mean(arr[:, :, 0]))
        mean_g = float(np.mean(arr[:, :, 1]))
        mean_b = float(np.mean(arr[:, :, 2]))
        total = mean_r + mean_g + mean_b + 1e-6
        return {
            "mean_r": round(mean_r, 1),
            "mean_g": round(mean_g, 1),
            "mean_b": round(mean_b, 1),
            "pct_r": round((mean_r / total) * 100, 1),
            "pct_g": round((mean_g / total) * 100, 1),
            "pct_b": round((mean_b / total) * 100, 1),
        }
    except Exception:
        return {}


def generate_report_markdown(item: Dict[str, Any]) -> str:
    """Generate a clean, professional Markdown report for export."""
    meta = item.get("meta", {})
    tags_str = ', '.join(item.get('tags', [])) if item.get('tags') else 'General Earth Observation'
    timestamp_str = item.get('timestamp') or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    reasoning_block = ""
    if item.get("reasoning"):
        reasoning_block = f"### Internal Reasoning Chain:\n```text\n{item.get('reasoning')}\n```\n"

    return f"""# SatQuery AI • Earth Observation Analysis Report
**Generated:** {timestamp_str}  
**Report ID:** {item.get('id', 'N/A')}  
**Application:** SatQuery AI (Remote-Sensing Vision-Language Assistant)

---

## 1. Image Technical Metadata
- **Filename:** `{item.get('image_name', 'N/A')}`
- **Dimensions:** {meta.get('width', 'N/A')} × {meta.get('height', 'N/A')} pixels ({meta.get('megapixels', 'N/A')} MP)
- **Format:** {meta.get('format', 'N/A')}
- **Color Mode:** {meta.get('mode', 'N/A')}
- **File Size:** {meta.get('size_kb', 0):.2f} KB

---

## 2. Analytical Inquiry
**User Prompt:**
> "{item.get('question', '')}"

---

## 3. Geospatial Model Assessment
**Detected Classes & Signatures:**  
{tags_str}

**Model Response:**
{item.get('answer', '')}

{reasoning_block}
---

## 4. Verification & Limitations
*Notice: This report was generated using remote-sensing computer vision. False-colour spectral bands (e.g. Near-Infrared) represent physical properties such as chlorophyll reflectance rather than human natural visual colors. Quantitative ground measurements must be validated with calibrated ground truth.*
"""


def generate_report_html(item: Dict[str, Any]) -> str:
    """Generate an interactive, printable HTML report."""
    answer_html = str(item.get('answer', '')).replace("\n", "<br>")
    tags_html = ' '.join([f'<span class="tag">{t}</span>' for t in item.get('tags', [])])
    meta = item.get('meta', {})

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SatQuery AI Analysis Report - {item.get('image_name', '')}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #1e293b; max-width: 800px; margin: 40px auto; padding: 20px; }}
  h1 {{ color: #0284c7; border-bottom: 2px solid #0284c7; padding-bottom: 8px; }}
  h2 {{ color: #0f172a; margin-top: 24px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }}
  blockquote {{ background: #f8fafc; border-left: 4px solid #0284c7; margin: 12px 0; padding: 12px 18px; font-style: italic; }}
  pre {{ background: #f1f5f9; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; }}
  .tag {{ display: inline-block; background: #e0f2fe; color: #0369a1; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; margin-right: 6px; }}
  .footer {{ margin-top: 40px; font-size: 0.8rem; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
</style>
</head>
<body>
<div style="font-size: 0.8rem; text-transform: uppercase; color: #0284c7; font-weight: bold;">SatQuery AI • Remote-Sensing Vision-Language Report</div>
<h1>Geospatial Intelligence Assessment</h1>
<p><strong>Timestamp:</strong> {item.get('timestamp')} | <strong>Report ID:</strong> {item.get('id')}</p>
<h2>1. Image Metadata</h2>
<ul>
  <li><strong>Filename:</strong> {item.get('image_name')}</li>
  <li><strong>Dimensions:</strong> {meta.get('width', 0)} &times; {meta.get('height', 0)} px</li>
  <li><strong>Size:</strong> {meta.get('size_kb', 0):.2f} KB</li>
</ul>
<h2>2. Analytical Inquiry</h2>
<blockquote>{item.get('question')}</blockquote>
<h2>3. Identified Signatures</h2>
<p>{tags_html}</p>
<h2>4. Model Analysis</h2>
<p>{answer_html}</p>
<div class="footer">
  SatQuery AI Geospatial Intelligence System • OpenRouter Remote Sensing Vision Pipeline
</div>
</body>
</html>"""


# -----------------------------------------------------------------------------
# Core Model Analysis API Call
# -----------------------------------------------------------------------------
def query_satellite_model(
    image_bytes: bytes,
    mime_type: str,
    question: str,
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 1500
) -> Dict[str, Any]:
    """Execute analysis query against OpenRouter multimodal endpoint."""
    api_key = get_api_key()
    if not api_key:
        return {
            "success": False,
            "error": "OpenRouter API Key not found. Please provide an OPENROUTER_API_KEY in the sidebar, .env file, or Streamlit Cloud Secrets."
        }

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    system_prompt = (
        "You are SatQuery AI, an expert satellite image analysis assistant. "
        "Carefully inspect the provided remote-sensing satellite image and answer the user's question with scientific accuracy. "
        "Describe visible land features, possible water bodies, vegetation patterns, and built-up areas. "
        "The image may use false colours (such as Near-Infrared where healthy vegetation appears red). "
        "Explain what the colours represent when appropriate. "
        "Distinguish visible evidence from uncertainty. If a feature cannot be conclusively identified, state so clearly. "
        "Never return an empty response."
    )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://satquery.ai",
        "X-Title": "SatQuery AI Hackathon"
    }

    start_time = time.time()
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        elapsed_sec = round(time.time() - start_time, 2)

        if not response.ok:
            return {
                "success": False,
                "error": f"API Error ({response.status_code}): {response.text[:500]}",
                "duration": elapsed_sec
            }

        res_json = response.json()
        choice = res_json.get("choices", [{}])[0]
        msg = choice.get("message", {})

        answer = msg.get("content")
        reasoning = msg.get("reasoning")

        # Handle list responses if returned
        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", "")
                for item in answer
                if isinstance(item, dict) and item.get("type") == "text"
            )

        # Fallback to reasoning if content is empty or literal 'None'
        if not answer or str(answer).strip().lower() in ("none", ""):
            if reasoning and str(reasoning).strip():
                answer = f"**Analytical Assessment (Synthesized from Reasoning Chain):**\n\n{reasoning}"
            else:
                return {
                    "success": False,
                    "error": "The AI model returned an empty completion. Please try rephrasing your question.",
                    "duration": elapsed_sec
                }

        tags = extract_features(str(answer))

        return {
            "success": True,
            "answer": answer,
            "reasoning": reasoning,
            "tags": tags,
            "duration": elapsed_sec
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "The request timed out after 120 seconds. OpenRouter upstream was slow, please retry."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Connection error: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error during processing: {e}"
        }


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 1.8rem;">🛰️</span>
            <div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; line-height: 1.1;">SatQuery AI</div>
                <div style="font-size: 0.72rem; color: #38bdf8; letter-spacing: 0.05em; text-transform: uppercase;">Earth Observation Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Prominent New Analysis Action
    if st.button("➕ New Analysis", use_container_width=True, type="primary"):
        clear_current_image()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Navigation Selection
    nav_selection = st.radio(
        "Navigation Menu",
        options=["Satellite Analysis", "Analysis History", "About the Project"],
        index=["Satellite Analysis", "Analysis History", "About the Project"].index(st.session_state.nav_page),
        label_visibility="collapsed"
    )
    st.session_state.nav_page = nav_selection

    st.markdown("---")

    # Current Image Status Widget
    if st.session_state.image_bytes is not None:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 600; margin-bottom: 4px;">ACTIVE SATELLITE SCENE</div>
                <div style="font-size: 0.88rem; color: #f8fafc; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    📄 {st.session_state.image_name}
                </div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px;">
                    {st.session_state.image_meta.get('width', 0)} × {st.session_state.image_meta.get('height', 0)} px • {st.session_state.image_meta.get('size_kb', 0):.1f} KB
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("❌ Remove Image", use_container_width=True, type="secondary"):
            clear_current_image()
    else:
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px;">
                No satellite image currently loaded.
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🚀 Load Sample Image", use_container_width=True, type="secondary"):
            load_sample_image()

    st.markdown("---")

    # API Connection Status
    current_key = get_api_key()
    api_key_found = bool(current_key)
    if api_key_found:
        status_html = """
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #34d399; margin-bottom: 8px;">
            <span class="status-indicator"></span> <strong>API Connected</strong> (OpenRouter)
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
        if st.session_state.get("custom_api_key"):
            if st.button("🔑 Clear Custom Key", use_container_width=True, type="secondary"):
                del st.session_state["custom_api_key"]
                st.rerun()
    else:
        status_html = """
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #f87171; margin-bottom: 8px;">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#ef4444;"></span> <strong>API Key Missing</strong>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
        user_key_input = st.text_input(
            "Enter OpenRouter Key",
            type="password",
            placeholder="sk-or-v1-...",
            help="Enter your API key or configure OPENROUTER_API_KEY in Streamlit Secrets."
        )
        if user_key_input:
            st.session_state["custom_api_key"] = user_key_input.strip()
            st.rerun()

    # Configuration Section
    with st.expander("⚙️ Model Configuration", expanded=False):
        model_selection = st.selectbox(
            "Vision Model",
            options=[
                "nvidia/nemotron-3.5-super-vl:free",
                "google/gemma-4-26b-a4b-it:free",
                "qwen/qwen3-vl-8b-instruct"
            ],
            index=0
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        max_tokens = st.slider("Max Tokens", min_value=300, max_value=2500, value=1500, step=100)

    # History summary
    history_count = len(st.session_state.history)
    st.caption(f"Queries Recorded: {history_count}")
    if history_count > 0:
        if st.button("Clear History", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.session_state.messages = []
            st.rerun()


# -----------------------------------------------------------------------------
# PAGE 1: SATELLITE ANALYSIS (MAIN WORKSPACE)
# -----------------------------------------------------------------------------
if st.session_state.nav_page == "Satellite Analysis":

    # VIEW A: WELCOME SCREEN (No image uploaded)
    if st.session_state.image_bytes is None:
        st.markdown(
            """
            <div style="text-align: center; max-width: 820px; margin: 15px auto 30px auto;">
                <div class="brand-pill">🛰️ SatQuery AI • Earth Observation Intelligence</div>
                <div class="brand-title">Explore Earth with SatQuery AI</div>
                <div class="brand-subtitle" style="margin: 0 auto 15px auto;">
                    Upload a satellite image. Ask questions. Discover meaningful insights about our planet.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4 Feature Cards Grid
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🌿</div>
                    <div class="feature-title">Vegetation Analysis</div>
                    <div class="feature-desc">Identify and analyse vegetation cover, canopy health, and infrared false-colour reflectance.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🗺️</div>
                    <div class="feature-title">Land Cover Classification</div>
                    <div class="feature-desc">Distinguish supported classes such as vegetation, water, bare soil, and built-up infrastructure.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🛰️</div>
                    <div class="feature-title">Change Observation</div>
                    <div class="feature-desc">Observe radiometric differences, water line shifts, and seasonal landscape patterns.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <div class="feature-title">Natural Language Q&A</div>
                    <div class="feature-desc">Ask complex conversational queries about any visible feature in your satellite imagery.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        # Prominent Upload Container
        up_col1, up_col2 = st.columns([2.4, 1], gap="medium")

        with up_col1:
            st.markdown(
                """
                <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">
                    Upload Satellite Imagery
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
                    Select an Earth observation image (Sentinel, Landsat, Planet, or drone aerial capture) in PNG, JPG, or JPEG format.
                </div>
                """,
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Upload Satellite Image",
                type=["png", "jpg", "jpeg"],
                key=f"main_uploader_{st.session_state.uploader_key}",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                if st.session_state.image_bytes != file_bytes or st.session_state.image_name != uploaded_file.name:
                    load_image_from_bytes(
                        file_bytes,
                        uploaded_file.name,
                        uploaded_file.type or "image/png"
                    )
                    st.session_state.messages = []
                    st.rerun()

        with up_col2:
            st.markdown(
                """
                <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 18px;">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-bottom: 6px;">
                        Quick Hackathon Demo
                    </div>
                    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.45; margin-bottom: 14px;">
                        Want to test immediately? Load our bundled 512×513 false-colour infrared satellite scene.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Load Sample Scene", use_container_width=True, type="primary"):
                load_sample_image()

    # VIEW B: ACTIVE WORKSPACE (Image Loaded)
    else:
        col_left, col_right = st.columns([1.1, 1.4], gap="large")

        # ---------------------------------------------------------------------
        # LEFT COLUMN: SATELLITE VIEWER & VISUAL SPECTRAL OUTPUT
        # ---------------------------------------------------------------------
        with col_left:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">
                        🛰️ Satellite Viewer & Visual Output
                    </div>
                    <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 500;">
                        {st.session_state.image_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Interactive Tabs: Natural Image, Spectral Segmentation, and Telemetry Specs
            tab_natural, tab_masks, tab_telemetry = st.tabs([
                "🖼️ Scene View",
                "🔬 Visual Spectral Masks",
                "📊 Telemetry & Metadata"
            ])

            with tab_natural:
                st.image(
                    st.session_state.image_bytes,
                    caption=f"Original Scene: {st.session_state.image_name} ({st.session_state.image_meta.get('width')}×{st.session_state.image_meta.get('height')} px)",
                    use_container_width=True
                )

            with tab_masks:
                st.markdown(
                    """
                    <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 8px;">
                        Visual segmentation overlays computed directly from sensor band channels:
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                mask_mode = st.radio(
                    "Visual Analysis Layer",
                    options=["🌿 Vegetation Cover Highlight", "💧 Water Body / Absorption Mask", "🏜️ Bare Soil & High-Albedo Mask", "🔥 Radiometric NIR Heatmap"],
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed"
                )

                mode_key = "vegetation"
                if "Water" in mask_mode:
                    mode_key = "water"
                elif "Soil" in mask_mode:
                    mode_key = "soil"
                elif "Heatmap" in mask_mode:
                    mode_key = "heatmap"

                gen_img, cov_pct, desc_str = generate_spectral_overlay(st.session_state.image_bytes, mode_key)

                st.image(
                    gen_img,
                    caption=f"{mask_mode} (Coverage: {cov_pct}%)",
                    use_container_width=True
                )
                st.caption(desc_str)

                # Download Generated Mask
                st.download_button(
                    label=f"📥 Download {mask_mode.split()[1]} Image (.png)",
                    data=pil_to_png_bytes(gen_img),
                    file_name=f"SatQuery_{mode_key}_mask.png",
                    mime="image/png",
                    use_container_width=True,
                    type="secondary"
                )

            with tab_telemetry:
                stats = compute_spectral_stats(st.session_state.image_bytes)
                if stats:
                    st.markdown(
                        """
                        <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px;">
                            Mean radiometric channel intensity (0-255). In False-Colour NIR imagery, elevated Red reflects dense healthy chlorophyll.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    c_r, c_g, c_b = st.columns(3)
                    with c_r:
                        st.metric("Red (NIR)", f"{stats['mean_r']}", f"{stats['pct_r']}%")
                    with c_g:
                        st.metric("Green", f"{stats['mean_g']}", f"{stats['pct_g']}%")
                    with c_b:
                        st.metric("Blue", f"{stats['mean_b']}", f"{stats['pct_b']}%")

                    # Channel Balance Visualization
                    st.markdown(
                        f"""
                        <div style="margin-top: 10px;">
                            <div style="font-size: 0.76rem; color: #cbd5e1; margin-bottom: 4px;">Radiometric Spectral Distribution</div>
                            <div style="display: flex; height: 10px; border-radius: 6px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
                                <div style="width: {stats['pct_r']}%; background: #ef4444;" title="Red Channel: {stats['pct_r']}%"></div>
                                <div style="width: {stats['pct_g']}%; background: #22c55e;" title="Green Channel: {stats['pct_g']}%"></div>
                                <div style="width: {stats['pct_b']}%; background: #3b82f6;" title="Blue Channel: {stats['pct_b']}%"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                meta = st.session_state.image_meta
                meta_table = f"""
                | Parameter | Measured Value |
                | :--- | :--- |
                | **Filename** | `{meta.get('filename')}` |
                | **Spatial Dimensions** | {meta.get('width')} × {meta.get('height')} px |
                | **Resolution** | {meta.get('megapixels')} Megapixels |
                | **Color Channels** | {meta.get('mode')} |
                | **Encoding Format** | {meta.get('format')} |
                | **Uncompressed Size** | {meta.get('size_kb', 0):.2f} KB |
                """
                st.markdown(meta_table)

            # Replace / Clear Image Control
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            with st.expander("🔄 Replace Satellite Image", expanded=False):
                replace_file = st.file_uploader(
                    "Select replacement satellite image",
                    type=["png", "jpg", "jpeg"],
                    key=f"replace_uploader_{st.session_state.uploader_key}"
                )
                if replace_file is not None:
                    r_bytes = replace_file.getvalue()
                    if st.session_state.image_bytes != r_bytes or st.session_state.image_name != replace_file.name:
                        load_image_from_bytes(r_bytes, replace_file.name, replace_file.type or "image/png")
                        st.session_state.messages = []
                        st.rerun()

            if st.button("❌ Remove Image", use_container_width=True, type="secondary", key="remove_img_btn_main"):
                clear_current_image()

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: CONVERSATIONAL ASSISTANT & NON-OVERLAPPING SEARCH BAR
        # ---------------------------------------------------------------------
        with col_right:
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc;">
                        💬 Earth Observation Assistant
                    </div>
                    <div class="badge badge-cyan">
                        Active Vision Model
                    </div>
                </div>
                <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 12px;">
                    Ask questions regarding land-cover features, vegetation patterns, hydrology, or spectral signatures.
                </div>
                """,
                unsafe_allow_html=True
            )

            # PREPARE PENDING PROMPT FROM QUICK INQUIRIES IF CLICKED
            triggered_query = None

            # PROMINENT SEARCH / QUESTION BAR (CLEAN, NO HORIZONTAL OVERLAP)
            with st.form(key="search_question_form", clear_on_submit=False):
                st.markdown(
                    """
                    <div style="font-size: 0.82rem; font-weight: 600; color: #38bdf8; margin-bottom: 6px;">
                        🔍 SATELLITE QUERY & SCENE SEARCH
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Use pending prompt if one was clicked, else default
                default_text = ""
                if st.session_state.quick_prompt:
                    default_text = st.session_state.quick_prompt
                    st.session_state.quick_prompt = None

                search_box_val = st.text_input(
                    "Search or inquiry query",
                    value=default_text,
                    placeholder="e.g., Where is vegetation visible? Describe the river bed and land cover...",
                    label_visibility="collapsed"
                )

                # Dedicated button row underneath input to guarantee ZERO horizontal overlapping
                btn_row_col1, btn_row_col2 = st.columns([1, 1], gap="small")
                with btn_row_col1:
                    search_submitted = st.form_submit_button("🔍 Run Analysis", type="primary", use_container_width=True)
                with btn_row_col2:
                    clear_chat_submitted = st.form_submit_button("🧹 Clear Conversation", type="secondary", use_container_width=True)

            if clear_chat_submitted:
                st.session_state.messages = []
                st.rerun()

            if search_submitted and search_box_val.strip():
                triggered_query = search_box_val.strip()

            # Quick Prompt Chips (Clickable questions)
            st.markdown("<div style='font-size: 0.78rem; color: #cbd5e1; font-weight: 500; margin: 6px 0 6px 0;'>💡 Quick Suggested Inquiries:</div>", unsafe_allow_html=True)
            p_cols = st.columns(2, gap="small")
            with p_cols[0]:
                if st.button("🌿 Where is vegetation visible?", use_container_width=True, type="secondary"):
                    triggered_query = "Where is vegetation visible in this image? Explain the spectral colours."
                if st.button("🏗️ Describe major land-cover features", use_container_width=True, type="secondary"):
                    triggered_query = "Describe the major land-cover features and structures in this image."
            with p_cols[1]:
                if st.button("💧 Identify water bodies and bare land", use_container_width=True, type="secondary"):
                    triggered_query = "Identify water bodies, river courses, and bare land in this scene."
                if st.button("🎨 Explain false-colour NIR bands", use_container_width=True, type="secondary"):
                    triggered_query = "Explain what the false-colour spectral representation means in this satellite scene."

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Render Existing Conversation Thread
            if len(st.session_state.messages) == 0:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 25px 15px; background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(56, 189, 248, 0.2); border-radius: 12px; margin: 10px 0;">
                        <div style="font-size: 1.8rem; margin-bottom: 6px;">🛰️</div>
                        <div style="font-size: 0.92rem; font-weight: 600; color: #f8fafc; margin-bottom: 2px;">Ready for Analysis</div>
                        <div style="font-size: 0.82rem; color: #94a3b8;">
                            Type a question in the search bar above or click any suggested prompt.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            for idx, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="chat-bubble-user">
                            <div style="font-size: 0.74rem; color: #38bdf8; font-weight: 600; margin-bottom: 4px;">ANALYST QUESTION</div>
                            {msg['content']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    tags_html = "".join([f"<span class='badge badge-cyan'>{t}</span>" for t in msg.get("tags", [])])
                    duration_str = f"⚡ {msg['duration']}s" if msg.get("duration") else ""

                    st.markdown(
                        f"""
                        <div class="chat-bubble-assistant">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <span style="font-size: 1.1rem;">🛰️</span>
                                    <span style="font-size: 0.82rem; font-weight: 600; color: #38bdf8;">SATQUERY AI</span>
                                </div>
                                <div style="font-size: 0.74rem; color: #94a3b8;">{duration_str}</div>
                            </div>
                            <div style="margin-bottom: 10px;">{tags_html}</div>
                            <div style="font-size: 0.94rem; line-height: 1.6; color: #f1f5f9;">
                                {msg['content']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Collapsible Chain-of-Thought Reasoning
                    if msg.get("reasoning"):
                        with st.expander("🧠 View Geospatial Reasoning Chain", expanded=False):
                            st.markdown(f"```text\n{msg['reasoning']}\n```")

                    # Download Report Button for this analysis
                    report_data = {
                        "id": f"SQ-{idx+1}-{int(time.time())}",
                        "timestamp": msg.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "image_name": st.session_state.image_name,
                        "question": st.session_state.messages[idx-1]["content"] if idx > 0 else "Image Query",
                        "answer": msg["content"],
                        "reasoning": msg.get("reasoning"),
                        "tags": msg.get("tags", []),
                        "meta": st.session_state.image_meta
                    }

                    dl_col1, dl_col2 = st.columns([1, 1], gap="small")
                    with dl_col1:
                        st.download_button(
                            "📥 Download Report (.md)",
                            data=generate_report_markdown(report_data),
                            file_name=f"SatQuery_Report_{idx+1}.md",
                            mime="text/markdown",
                            key=f"dl_md_{idx}",
                            use_container_width=True
                        )
                    with dl_col2:
                        st.download_button(
                            "📄 Download Printable (.html)",
                            data=generate_report_html(report_data),
                            file_name=f"SatQuery_Report_{idx+1}.html",
                            mime="text/html",
                            key=f"dl_html_{idx}",
                            use_container_width=True
                        )

            # Execute Model Inference if question was triggered
            if triggered_query:
                st.session_state.messages.append({
                    "role": "user",
                    "content": triggered_query,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                with st.spinner("Analyzing satellite scene with multimodal vision pipeline..."):
                    result = query_satellite_model(
                        image_bytes=st.session_state.image_bytes,
                        mime_type=st.session_state.image_type or "image/png",
                        question=triggered_query,
                        model_name=model_selection,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                if result["success"]:
                    assistant_msg = {
                        "role": "assistant",
                        "content": result["answer"],
                        "reasoning": result.get("reasoning"),
                        "tags": result.get("tags", []),
                        "duration": result.get("duration", 0),
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.messages.append(assistant_msg)

                    st.session_state.history.append({
                        "id": f"SQ-{len(st.session_state.history)+1}",
                        "timestamp": assistant_msg["timestamp"],
                        "image_name": st.session_state.image_name,
                        "question": triggered_query,
                        "answer": result["answer"],
                        "reasoning": result.get("reasoning"),
                        "tags": result.get("tags", []),
                        "meta": st.session_state.image_meta
                    })
                    st.rerun()
                else:
                    st.error(result["error"])


# -----------------------------------------------------------------------------
# PAGE 2: ANALYSIS HISTORY
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Analysis History":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="brand-pill">Audit Trail & Records</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">📜 Analysis History</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">Review questions, identified features, and assessments from this session.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:
        st.info("No analyses recorded in this session yet. Upload a satellite image to start.")
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
                                🕒 {item['timestamp']}
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
                h_col1, h_col2 = st.columns([1, 4])
                with h_col1:
                    st.download_button(
                        f"📥 Export Report ({item['id']})",
                        data=generate_report_markdown(item),
                        file_name=f"Report_{item['id']}.md",
                        mime="text/markdown",
                        key=f"hist_dl_{item['id']}",
                        use_container_width=True
                    )


# -----------------------------------------------------------------------------
# PAGE 3: ABOUT THE PROJECT
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "About the Project":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="brand-pill">National Hackathon Showcase</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">🛰️ About SatQuery AI</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">Remote-Sensing Vision-Language Architecture & Operational Pipeline.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c_info1, c_info2 = st.columns([1.5, 1], gap="large")

    with c_info1:
        st.markdown(
            """
            ### 🌍 Overview
            **SatQuery AI** is an Earth observation intelligence application designed to assist geospatial analysts, environmental scientists, and disaster response teams in understanding remote-sensing imagery through natural language dialogues and calibrated spectral segmentation.

            ### 🔬 Remote-Sensing Principles
            - **False-Colour Infrared (CIR):** Satellite sensors record Near-Infrared (NIR) wavelengths (700-1100 nm). Because healthy vegetation strongly reflects NIR radiation through its leaf mesophyll structure, CIR imagery assigns NIR to the Red channel. Consequently, dense forests, crops, and marshes appear distinctly red.
            - **Hydrology Signatures:** Clear water bodies absorb radiation across infrared and red bands, appearing dark navy or black.
            - **Built-up Surfaces:** Concrete, asphalt, and urban settlements appear cyan, grey, or white depending on radiometric reflectance.

            ### ⚙️ Technical Architecture
            - **Frontend:** Streamlit with customized ChatGPT-grade dark UI, glassmorphic HUD cards, and responsive columns.
            - **Backend Inference:** OpenRouter Multimodal Vision-Language API (`nvidia/nemotron-3.5-super-vl:free`).
            - **Visual Output Pipeline:** Real NumPy radiometric channel index computing pixel-level spectral masks for vegetation canopy, hydrology water absorption, and soil albedo.
            - **Export Engine:** Formatted Markdown and printable HTML report generator.
            """
        )

    with c_info2:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 20px;">
                <div style="font-size: 1rem; font-weight: 600; color: #38bdf8; margin-bottom: 12px;">
                    Technical Specifications
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.7;">
                    • <strong>Primary Engine:</strong> Nemotron 3.5 Super VL<br>
                    • <strong>API Endpoint:</strong> OpenRouter Chat Completions<br>
                    • <strong>Modality:</strong> Image + Text (Multimodal)<br>
                    • <strong>Visual Output:</strong> Pixel Spectral Segmentation Masks<br>
                    • <strong>Coverage Telemetry:</strong> NumPy CIR/NDVI & Absorption %<br>
                    • <strong>Export:</strong> Markdown & Printable HTML
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
