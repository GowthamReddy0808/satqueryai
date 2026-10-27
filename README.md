# 🛰️ SatQuery AI

**Earth Observation Intelligence & Remote-Sensing Vision-Language Assistant**

SatQuery AI is an AI-powered satellite imagery analysis platform designed for environmental researchers, urban planners, and geospatial intelligence teams. It pairs multimodal vision-language reasoning with pixel-level radiometric spectral analysis to provide natural language insights into Earth observation data.

---

## 🌟 Key Features

- **💬 Conversational Earth Observation:** Ask natural language questions about visible features, terrain types, infrastructure, and ecological health.
- **🔬 Visual Spectral Segmentation:** Real pixel-level spectral masks generated directly from sensor channels:
  - 🌿 **Vegetation Canopy Highlight:** CIR Vegetation Index with quantified scene coverage percentage.
  - 💧 **Hydrology & Water Course Mask:** Near-infrared absorption channel detection.
  - 🏜️ **Soil & Albedo Terrain Mask:** High-reflectance bare surface classification.
  - 🔥 **NIR Radiometric Heatmap:** False-colour infrared intensity gradient mapping.
- **📊 Radiometric Telemetry:** Automated channel distribution analysis (mean Red/NIR, Green, Blue balance).
- **🧠 Chain-of-Thought Geospatial Tracing:** Inspect the AI model's internal reasoning chain for mission-critical verification.
- **📥 Audit & Export Engine:** Download analysis dossiers in structured Markdown (`.md`) or printable HTML formats.
- **🌌 ChatGPT-Grade Dark UI:** High-contrast geospatial dark theme built with customized Streamlit and custom CSS.

---

## 🛠️ Tech Stack

- **Frontend & App Engine:** Streamlit with custom CSS design system
- **Multimodal Vision Backend:** OpenRouter Multimodal Vision-Language API (`nvidia/nemotron-3.5-super-vl:free` / `qwen/qwen3-vl-8b-instruct`)
- **Spectral Telemetry:** NumPy, Pillow (PIL)
- **Environment Management:** python-dotenv

---

## 🚀 Quickstart Guide

### 1. Clone Repository
```bash
git clone https://github.com/GowthamReddy0808/satqueryai.git
cd satqueryai
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install streamlit requests pillow numpy python-dotenv
```

### 4. Configure API Key
Create a `.env` file in the root directory (based on `.env.example`):
```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
```
> **Security Notice:** Never commit your actual `.env` file to version control. The repository includes a `.gitignore` that automatically excludes `.env` and credential files.

### 5. Launch Application
```bash
streamlit run app.py --server.port 8502
```
Open [http://localhost:8502](http://localhost:8502) in your browser.

---

## 📄 License
This project was developed for national-level hackathon demonstration. Open source under the MIT License.
"# satqueryai" 
"# satqueryai" 
