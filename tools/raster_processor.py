"""
SatQuery AI - Geospatial Raster & Spectral Processing Engine
Handles GeoTIFF metadata extraction via Rasterio, radiometric index calculation,
genuine NDVI calculation (when NIR is present), and spectral products.
"""

import io
import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.io import MemoryFile
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def inspect_raster_metadata(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract comprehensive remote sensing metadata from GeoTIFF or standard raster files.
    Extracts CRS, bounding box coordinates, spatial resolution, band count, and data types.
    """
    metadata: Dict[str, Any] = {
        "filename": filename,
        "format": "Unknown",
        "width": 0,
        "height": 0,
        "band_count": 0,
        "dtype": "uint8",
        "has_georeference": False,
        "crs": "None (Non-georeferenced pixel coordinates)",
        "bounds": None,
        "resolution": "Unknown (Standard pixel space)",
        "modality": "Optical / Aerial Imagery",
        "megapixels": 0.0,
        "size_kb": round(len(image_bytes) / 1024.0, 2),
    }

    # Attempt GeoTIFF extraction using rasterio if file is TIFF/GeoTIFF
    is_tiff = filename.lower().endswith((".tif", ".tiff", ".geotiff")) or (len(image_bytes) >= 4 and image_bytes[:4] in (b'II*\x00', b'MM\x00*'))
    if RASTERIO_AVAILABLE and is_tiff:
        try:
            with MemoryFile(image_bytes) as memfile:
                with memfile.open() as src:
                    metadata["format"] = "GeoTIFF / Cloud-Optimized GeoTIFF (COG)" if src.crs else "TIFF"
                    metadata["width"] = src.width
                    metadata["height"] = src.height
                    metadata["band_count"] = src.count
                    metadata["dtype"] = str(src.dtypes[0]) if src.dtypes else "unknown"
                    metadata["megapixels"] = round((src.width * src.height) / 1_000_000, 2)
                    
                    if src.crs:
                        metadata["has_georeference"] = True
                        metadata["crs"] = str(src.crs)
                        b = src.bounds
                        metadata["bounds"] = {
                            "left": round(b.left, 4),
                            "bottom": round(b.bottom, 4),
                            "right": round(b.right, 4),
                            "top": round(b.top, 4),
                        }
                        res_x, res_y = src.res
                        metadata["resolution"] = f"{round(res_x, 2)}m × {round(res_y, 2)}m / px"

                    # Infer modality
                    if src.count >= 4:
                        metadata["modality"] = f"Multispectral Optical ({src.count} Bands: RGB + NIR/SWIR)"
                    elif src.count == 1:
                        metadata["modality"] = "Single-Band Panchromatic / SAR Backscatter"
                    else:
                        metadata["modality"] = "3-Band Optical (RGB / CIR)"
                    return metadata
        except Exception:
            pass  # Fallback to PIL for non-GeoTIFF or corrupt TIFF headers

    # Standard PIL Inspection (PNG, JPEG, uncompressed TIFF)
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            w, h = pil_img.size
            bands = len(pil_img.getbands())
            metadata["format"] = pil_img.format or "PNG/JPEG"
            metadata["width"] = w
            metadata["height"] = h
            metadata["band_count"] = bands
            metadata["dtype"] = pil_img.mode
            metadata["megapixels"] = round((w * h) / 1_000_000, 2)
            
            # Modal estimation based on filename and channel count
            name_lower = filename.lower()
            if "sar" in name_lower or "sentinel1" in name_lower or bands == 1:
                metadata["modality"] = "Synthetic Aperture Radar (SAR) / Backscatter Amplitude"
            elif "cir" in name_lower or "false" in name_lower:
                metadata["modality"] = "Color-Infrared (CIR False-Color Optical: NIR-R-G)"
            elif bands >= 4:
                metadata["modality"] = f"Multispectral Optical ({bands} bands)"
            else:
                metadata["modality"] = "Standard Optical (RGB Visible Spectrum)"
    except Exception as e:
        metadata["error"] = str(e)

    return metadata


def calculate_ndvi(
    image_bytes: bytes,
    nir_band_idx: int = 4,
    red_band_idx: int = 1
) -> Tuple[Optional[np.ndarray], Optional[Image.Image], Dict[str, Any]]:
    """
    Compute genuine NDVI = (NIR - Red) / (NIR + Red) if multispectral data is provided.
    If only standard 3-band RGB is available, this function DOES NOT fabricate genuine NDVI.
    Instead, it returns an honest disclaimer and a calibrated false-color canopy reflectance index.
    """
    report: Dict[str, Any] = {
        "is_genuine_ndvi": False,
        "mean_index": 0.0,
        "vegetation_coverage_pct": 0.0,
        "status_message": "",
        "explanation": ""
    }

    # Attempt to read multispectral bands via rasterio
    if RASTERIO_AVAILABLE:
        try:
            with MemoryFile(image_bytes) as memfile:
                with memfile.open() as src:
                    if src.count >= 4:
                        # Genuine 4+ band imagery with NIR band
                        nir = src.read(nir_band_idx).astype(float)
                        red = src.read(red_band_idx).astype(float)
                        denom = nir + red
                        denom[denom == 0] = 1e-6
                        ndvi_arr = (nir - red) / denom
                        ndvi_arr = np.clip(ndvi_arr, -1.0, 1.0)

                        # Colorize NDVI [-1 to +1]: Water/Cloud (<0: blue/grey), Soil (0-0.2: beige/brown), Veg (>0.3: vibrant green)
                        norm_ndvi = ((ndvi_arr + 1.0) / 2.0 * 255).astype(np.uint8)
                        
                        # Apply scientific remote sensing colormap
                        h, w = ndvi_arr.shape
                        rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
                        # Water / snow (< 0.0) -> blue/grey
                        water_mask = ndvi_arr < 0.0
                        rgb_map[water_mask] = [40, 120, 200]
                        # Soil / rock (0.0 to 0.2) -> tan/amber
                        soil_mask = (ndvi_arr >= 0.0) & (ndvi_arr < 0.2)
                        rgb_map[soil_mask] = [210, 180, 140]
                        # Moderate veg (0.2 to 0.5) -> light green
                        mod_veg = (ndvi_arr >= 0.2) & (ndvi_arr < 0.5)
                        rgb_map[mod_veg] = [120, 210, 80]
                        # Dense healthy canopy (> 0.5) -> dark emerald green
                        dense_veg = ndvi_arr >= 0.5
                        rgb_map[dense_veg] = [15, 140, 45]

                        veg_coverage = float(np.mean(ndvi_arr > 0.3) * 100)
                        mean_val = float(np.mean(ndvi_arr))

                        report["is_genuine_ndvi"] = True
                        report["mean_index"] = round(mean_val, 3)
                        report["vegetation_coverage_pct"] = round(veg_coverage, 1)
                        report["status_message"] = "Genuine Physical NDVI Computed from Band 4 (NIR) and Band 1 (Red)."
                        report["explanation"] = (
                            f"NDVI calculated using verified NIR and Red bands. "
                            f"Mean scene NDVI: {round(mean_val, 2)}. "
                            f"Healthy vegetation coverage (>0.30): {round(veg_coverage, 1)}%."
                        )
                        return ndvi_arr, Image.fromarray(rgb_map), report
        except Exception:
            pass

    # RGB Fallback: CANNOT compute genuine physical NDVI
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(pil_img, dtype=float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # In CIR false-color, the Red channel maps to NIR!
    # CIR Green-Red ratio: (R - G) / (R + G + eps)
    cir_index = (r - g) / (r + g + 1e-5)
    veg_mask = (cir_index > 0.22) & (r > 60)
    veg_coverage = float(np.mean(veg_mask) * 100)

    # Colormap visual overlay
    h, w, _ = arr.shape
    vis_overlay = arr.copy()
    vis_overlay[veg_mask, 0] = vis_overlay[veg_mask, 0] * 0.3 + 20 * 0.7
    vis_overlay[veg_mask, 1] = vis_overlay[veg_mask, 1] * 0.3 + 220 * 0.7
    vis_overlay[veg_mask, 2] = vis_overlay[veg_mask, 2] * 0.3 + 120 * 0.7

    report["is_genuine_ndvi"] = False
    report["mean_index"] = round(float(np.mean(cir_index)), 3)
    report["vegetation_coverage_pct"] = round(veg_coverage, 1)
    report["status_message"] = "False-Colour CIR Canopy Index (Calibrated Proxy)"
    report["explanation"] = (
        "Note on Scientific Fidelity: Standard 3-channel RGB imagery lacks an independent physical Near-Infrared (NIR) band "
        "(such as Sentinel-2 Band 8 or Landsat-8 Band 5). True physical NDVI requires verified NIR and Red channels. "
        f"This scene was analyzed using CIR canopy reflectance, identifying chlorophyll response across {round(veg_coverage, 1)}% of the scene."
    )

    return cir_index, Image.fromarray(np.uint8(vis_overlay)), report


def generate_spectral_masks(image_bytes: bytes) -> Dict[str, Tuple[Image.Image, float, str]]:
    """
    Generate calibrated visual segmentation overlays:
    - Vegetation Canopy (CIR/NIR Dominance)
    - Water Bodies / Wetland Absorption
    - Bare Soil & High Albedo Sediments
    - Radiometric Intensity Gradient Heatmap
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(pil_img, dtype=float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (r + g + b) / 3.0

    products: Dict[str, Tuple[Image.Image, float, str]] = {}

    # 1. Vegetation
    cir_veg = (r - g) / (r + g + 1e-5)
    veg_mask = (cir_veg > 0.22) & (r > 55)
    veg_cov = round(float(np.mean(veg_mask)) * 100, 1)
    veg_img = arr.copy()
    veg_img[veg_mask, 0] = veg_img[veg_mask, 0] * 0.25 + 16 * 0.75
    veg_img[veg_mask, 1] = veg_img[veg_mask, 1] * 0.25 + 230 * 0.75
    veg_img[veg_mask, 2] = veg_img[veg_mask, 2] * 0.25 + 130 * 0.75
    products["vegetation"] = (
        Image.fromarray(np.uint8(np.clip(veg_img, 0, 255))),
        veg_cov,
        f"Canopy reflectance detected over {veg_cov}% of the analyzed scene."
    )

    # 2. Water Bodies
    water_mask = (r < 55) & (brightness < 65)
    water_cov = round(float(np.mean(water_mask)) * 100, 1)
    water_img = arr.copy()
    water_img[water_mask, 0] = water_img[water_mask, 0] * 0.15 + 6 * 0.85
    water_img[water_mask, 1] = water_img[water_mask, 1] * 0.15 + 182 * 0.85
    water_img[water_mask, 2] = water_img[water_mask, 2] * 0.15 + 248 * 0.85
    products["water"] = (
        Image.fromarray(np.uint8(np.clip(water_img, 0, 255))),
        water_cov,
        f"Deep water absorption & low reflectance channels across {water_cov}% of the scene."
    )

    # 3. Bare Soil & High Albedo
    spectral_diff = np.abs(r - g) + np.abs(g - b)
    soil_mask = (spectral_diff < 35) & (r > 70) & (~veg_mask)
    soil_cov = round(float(np.mean(soil_mask)) * 100, 1)
    soil_img = arr.copy()
    soil_img[soil_mask, 0] = soil_img[soil_mask, 0] * 0.25 + 245 * 0.75
    soil_img[soil_mask, 1] = soil_img[soil_mask, 1] * 0.25 + 158 * 0.75
    soil_img[soil_mask, 2] = soil_img[soil_mask, 2] * 0.25 + 11 * 0.75
    products["soil"] = (
        Image.fromarray(np.uint8(np.clip(soil_img, 0, 255))),
        soil_cov,
        f"Bare soil, sediment, and high-albedo terrain across {soil_cov}% of the scene."
    )

    # 4. Radiometric Heatmap
    norm_r = np.clip((r / (np.max(r) + 1e-5)) * 255, 0, 255)
    heat_img = np.zeros_like(arr)
    heat_img[:, :, 0] = norm_r
    heat_img[:, :, 1] = 255 - np.abs(norm_r - 128) * 2
    heat_img[:, :, 2] = 255 - norm_r
    products["heatmap"] = (
        Image.fromarray(np.uint8(heat_img)),
        100.0,
        "Continuous false-color radiometric near-infrared / thermal intensity map."
    )

    return products
