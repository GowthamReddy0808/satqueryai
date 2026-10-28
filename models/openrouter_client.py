"""
SatQuery AI - OpenRouter Multimodal Vision-Language Client
Interfaces with high-capability open-weight remote sensing vision models:
- Nemotron 3.5 Super VL
- Qwen 2.5 / 3 VL
- Gemma 4 26B
"""

import os
import re
import time
import base64
from typing import Dict, Any, List, Optional
import requests
from core.schema import BoundingBox


def extract_coordinate_boxes(text: str) -> List[BoundingBox]:
    """
    Parse grounded bounding box coordinates if returned by vision-language models
    in formats such as [ymin, xmin, ymax, xmax] or {"box_2d": [...]}.
    """
    boxes = []
    # Match patterns like [0.12, 0.34, 0.56, 0.78] or [120, 340, 560, 780]
    coord_pattern = re.compile(r'\[\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*\]')
    matches = coord_pattern.findall(text)
    for idx, match in enumerate(matches):
        try:
            coords = [float(v) for v in match]
            # Check if normalized or pixel
            is_norm = all(0.0 <= c <= 1.0 for c in coords)
            boxes.append(BoundingBox(
                box_2d=coords,
                label=f"Grounded Region {idx+1}",
                confidence=None,
                is_normalized=is_norm
            ))
        except Exception:
            continue
    return boxes


def extract_features(text: str) -> List[str]:
    """Identify remote sensing features and spectral signatures in text."""
    tags = []
    text_lower = text.lower()
    mapping = {
        "🌿 Vegetation Canopy": ["vegetation", "canopy", "plant", "forest", "tree", "crop", "grass", "chlorophyll"],
        "💧 Water Body": ["water", "river", "lake", "wetland", "reservoir", "stream", "ocean", "marsh", "drainage"],
        "🏙️ Built-up Area": ["built-up", "urban", "building", "settlement", "structure", "road", "infrastructure"],
        "🏜️ Bare Soil / Sand": ["bare", "soil", "sand", "arid", "ground", "sediment", "dry land", "alluvial"],
        "☁️ Cloud / Shadow": ["cloud", "cumulus", "shadow", "atmospheric", "haze"],
        "🔴 False-Colour NIR": ["false-color", "false-colour", "infrared", "near-infrared", "nir", "red hue"],
        "📡 SAR Backscatter": ["sar", "radar", "backscatter", "dielectric", "roughness"]
    }
    for tag, keywords in mapping.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags


def query_multimodal_vlm(
    image_bytes: bytes,
    mime_type: str,
    question: str,
    model_name: str = "nvidia/nemotron-3.5-super-vl:free",
    api_key: Optional[str] = None,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500
) -> Dict[str, Any]:
    """
    Execute multimodal query against OpenRouter API.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
    if not key:
        return {
            "success": False,
            "error": "OpenRouter API Key not found. Please provide an OPENROUTER_API_KEY in the sidebar, .env file, or Streamlit Cloud Secrets."
        }

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    system_prompt = system_instruction or (
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
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {key}",
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
        detections = extract_coordinate_boxes(str(answer))

        return {
            "success": True,
            "answer": answer,
            "reasoning": reasoning,
            "tags": tags,
            "detections": detections,
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
