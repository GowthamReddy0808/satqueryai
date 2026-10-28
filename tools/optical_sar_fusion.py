"""
SatQuery AI - Optical + SAR Joint Analysis Engine
Integrates Optical (surface reflectance / spectral signature) with
Synthetic Aperture Radar (SAR roughness, dielectric constant, volumetric scatter).
"""

import io
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image


def process_optical_sar_pair(
    optical_bytes: bytes,
    sar_bytes: bytes,
    optical_name: str = "Optical_Scene",
    sar_name: str = "SAR_Scene"
) -> Tuple[Image.Image, Image.Image, Image.Image, Dict[str, Any]]:
    """
    Process joint Optical + SAR observations:
    Returns:
    1. Optical scene (RGB)
    2. SAR backscatter amplitude visualizer (greyscale intensity)
    3. Joint Structural-Spectral Complementary Composite (False-color fusion)
    4. Modality contribution breakdown and physical evidence report.
    """
    opt_img = Image.open(io.BytesIO(optical_bytes)).convert("RGB")
    sar_img = Image.open(io.BytesIO(sar_bytes)).convert("L")  # Single-channel backscatter intensity

    # Match dimensions for complementary visual alignment
    if opt_img.size != sar_img.size:
        sar_img = sar_img.resize(opt_img.size, Image.Resampling.BILINEAR)

    opt_arr = np.array(opt_img, dtype=float)
    sar_arr = np.array(sar_img, dtype=float)

    # Convert SAR to 3-channel greyscale
    sar_rgb = np.stack([sar_arr, sar_arr, sar_arr], axis=2).astype(np.uint8)
    sar_vis = Image.fromarray(sar_rgb)

    # Joint Structural-Spectral Composite:
    # Channel 0 (Red): Optical Red reflectance (biochemical signature / dry land)
    # Channel 1 (Green): Optical Green reflectance (vegetation)
    # Channel 2 (Blue): SAR Backscatter amplitude (surface roughness, building dihedral bounce, metal reflection)
    fused_arr = np.zeros_like(opt_arr)
    fused_arr[:, :, 0] = opt_arr[:, :, 0] * 0.8
    fused_arr[:, :, 1] = opt_arr[:, :, 1] * 0.8
    fused_arr[:, :, 2] = sar_arr * 0.95

    composite_img = Image.fromarray(np.uint8(np.clip(fused_arr, 0, 255)))

    # Compute physical statistics
    opt_mean = float(np.mean(opt_arr))
    sar_mean = float(np.mean(sar_arr))
    sar_std = float(np.std(sar_arr))

    # Detect high radar bounce (urban structures, ships, metallic infrastructure)
    high_backscatter_mask = sar_arr > (sar_mean + 1.5 * sar_std)
    high_bounce_pct = round(float(np.mean(high_backscatter_mask)) * 100, 2)

    evidence_report = {
        "optical_name": optical_name,
        "sar_name": sar_name,
        "dimensions": opt_img.size,
        "optical_contribution": (
            "Spectral chlorophyll absorption, water boundary delineation, and surface albedo. "
            "Susceptible to cloud obscuration and illumination variations."
        ),
        "sar_contribution": (
            "Dielectric properties, double-bounce corner reflection (urban/built-up structures), "
            "and all-weather cloud-penetrating structural sensing."
        ),
        "high_backscatter_features_pct": high_bounce_pct,
        "modality_alignment_fidelity": "Dimension-matched spatial composite. Note: Physical phase fusion requires complex-valued SLC (Single Look Complex) radar products.",
        "scientific_notice": (
            "Scientific Integrity Notice: The joint visualization represents a calibrated structural-spectral display composite "
            "(Optical R/G + SAR Backscatter Blue). It provides complementary geometric and spectral evidence without fabricating unmeasured radar phase properties."
        )
    }

    return opt_img, sar_vis, composite_img, evidence_report
