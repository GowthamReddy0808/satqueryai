"""
SatQuery AI - Geospatial Annotation & Bounding Box Renderer
Draws grounded regions, bounding boxes, segmentation masks, and class tags
with high visual fidelity on remote sensing imagery.
"""

import io
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from core.schema import BoundingBox


def draw_bounding_boxes(
    image: Image.Image,
    detections: List[BoundingBox],
    box_width: int = 3,
    show_labels: bool = True
) -> Image.Image:
    """
    Render bounding boxes with translucent background tags and confidence scores.
    Handles both normalized coordinates [0, 1] and absolute pixel coordinates [ymin, xmin, ymax, xmax].
    """
    if not detections:
        return image.copy()

    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(annotated)

    w, h = annotated.size

    # Try loading a clean font or fallback to default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    color_palette = [
        (56, 189, 248, 255),   # Cyan
        (34, 197, 94, 255),    # Emerald
        (249, 115, 22, 255),   # Amber
        (236, 72, 153, 255),   # Pink
        (168, 85, 247, 255),   # Purple
        (239, 68, 68, 255)     # Red
    ]

    for idx, det in enumerate(detections):
        box = det.box_2d
        if len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = box

        # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
        if det.is_normalized or (ymax <= 1.0 and xmax <= 1.0 and ymax > 0):
            left = int(xmin * w)
            top = int(ymin * h)
            right = int(xmax * w)
            bottom = int(ymax * h)
        else:
            left = int(xmin)
            top = int(ymin)
            right = int(xmax)
            bottom = int(ymax)

        # Clip to image boundaries
        left = max(0, min(w - 1, left))
        top = max(0, min(h - 1, top))
        right = max(left + 2, min(w, right))
        bottom = max(top + 2, min(h, bottom))

        color = color_palette[idx % len(color_palette)]
        fill_color = (color[0], color[1], color[2], 40)  # Translucent fill

        # Draw semi-transparent bounding box interior
        draw_overlay.rectangle([left, top, right, bottom], fill=fill_color, outline=color, width=box_width)

        # Label tag text
        conf_str = f" ({round(det.confidence * 100, 1)}%)" if det.confidence is not None else ""
        label_text = f"{det.label}{conf_str}" if show_labels else ""

        if label_text:
            text_bbox = draw_base.textbbox((left, max(0, top - 20)), label_text, font=font)
            # Draw label badge background
            draw_overlay.rectangle(
                [text_bbox[0] - 4, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
                fill=(15, 23, 42, 220),
                outline=color,
                width=1
            )
            # Draw text
            draw_overlay.text(
                (text_bbox[0], text_bbox[1]),
                label_text,
                fill=(255, 255, 255, 255),
                font=font
            )

    combined = Image.alpha_composite(annotated, overlay)
    return combined.convert("RGB")


def create_segmentation_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color_rgb: Tuple[int, int, int] = (56, 189, 248),
    alpha: float = 0.6
) -> Image.Image:
    """
    Overlay a boolean/float segmentation mask on an image with adjustable transparency.
    """
    base_img = image.copy().convert("RGB")
    arr = np.array(base_img, dtype=float)

    if mask.shape[:2] != arr.shape[:2]:
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (base_img.width, base_img.height), Image.Resampling.NEAREST
        )
        mask = np.array(pil_mask) > 128

    overlay = arr.copy()
    overlay[mask, 0] = overlay[mask, 0] * (1.0 - alpha) + color_rgb[0] * alpha
    overlay[mask, 1] = overlay[mask, 1] * (1.0 - alpha) + color_rgb[1] * alpha
    overlay[mask, 2] = overlay[mask, 2] * (1.0 - alpha) + color_rgb[2] * alpha

    return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))
