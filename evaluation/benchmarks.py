"""
SatQuery AI - Remote Sensing Benchmark Dataset Profiles & Specifications
Profiles for BigEarthNet, VRSBench, RSVQA, and CDVQA as mandated by problem statement ST-09.
"""

from typing import Dict, Any, List

BENCHMARK_DATASETS: Dict[str, Dict[str, Any]] = {
    "BigEarthNet": {
        "title": "BigEarthNet (Sentinel-2 & Sentinel-1)",
        "purpose": "Multi-modal Earth observation representation adaptation & multi-label land-cover classification.",
        "size": "590,326 paired Sentinel-2 (multispectral) and Sentinel-1 (dual-pol SAR) image patches.",
        "modalities": "Optical Multispectral (12 spectral bands, 10m-60m GSD) + SAR (VV, VH polarizations).",
        "supported_tasks": ["Multi-label Land Cover Classification", "Multimodal Pretraining", "Optical-SAR Representation Learning"],
        "license": "Community Data License Agreement (CDLA-Permissive-1.0)",
        "classes": 19,
        "class_examples": [
            "Urban fabric", "Industrial/commercial units", "Arable land", "Permanent crops",
            "Pastures", "Complex cultivation patterns", "Broad-leaved forest", "Coniferous forest",
            "Mixed forest", "Natural grassland", "Moors and heathlands", "Sparsely vegetated areas",
            "Inland wetlands", "Maritime wetlands", "Inland waters", "Marine waters"
        ],
        "citation": "Sumbul et al., 'BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding', IEEE IGARSS 2019."
    },
    "VRSBench": {
        "title": "VRSBench (Visual Remote Sensing Benchmark)",
        "purpose": "Comprehensive benchmark for remote sensing vision-language tasks including captioning, visual question answering, and text-guided region grounding.",
        "size": "102,400+ high-resolution aerial and satellite scene-text pairs.",
        "modalities": "High-resolution Optical Aerial / Satellite Imagery (0.1m - 0.5m GSD).",
        "supported_tasks": [
            "Remote Sensing Scene Captioning",
            "Visual Question Answering (VQA)",
            "Text-Guided Visual Grounding (Bounding Boxes)",
            "Object Counting"
        ],
        "evaluation_metrics": ["CIDEr", "BLEU-4", "METEOR", "Grounding IoU@0.5", "VQA Accuracy"],
        "citation": "Ling et al., 'VRSBench: A Comprehensive Benchmark for Visual Remote Sensing', 2024."
    },
    "RSVQA": {
        "title": "RSVQA (Remote Sensing Visual Question Answering)",
        "purpose": "Standardized benchmark for remote-sensing visual question answering across both high and low resolution sensors.",
        "size": "Two subsets: RSVQA-LR (Sentinel-2, 772 images, 72,000+ QA pairs) and RSVQA-HR (Aerial, 10,659 images, 1,000,000+ QA pairs).",
        "modalities": "Sentinel-2 (10m) and High-Resolution Orthophotos (0.15m).",
        "supported_tasks": ["Single-Image VQA", "Presence/Absence QA", "Object Counting QA", "Area Comparison QA"],
        "evaluation_metrics": ["Overall Accuracy (OA)", "Average Accuracy (AA)", "Per-Category Accuracy"],
        "citation": "Lobry et al., 'RSVQA: Visual Question Answering for Remote Sensing Data', IEEE TGRS 2020."
    },
    "CDVQA": {
        "title": "CDVQA (Change Detection Visual Question Answering)",
        "purpose": "Bi-temporal change-based visual question answering, evaluating model capacity to reason over temporal pairs.",
        "size": "10,000+ bi-temporal satellite pairs with temporal comparison queries.",
        "modalities": "Bi-temporal optical pairs (T1 earlier, T2 later).",
        "supported_tasks": [
            "Change-Based VQA",
            "Land Cover Transition Verification",
            "Building Demolition/Construction Tracking",
            "Vegetation / Forest Loss Verification"
        ],
        "evaluation_metrics": ["VQA Accuracy", "Temporal Change IoU", "F1 Score"],
        "citation": "Yuan et al., 'CDVQA: Change Detection Visual Question Answering Benchmark', 2023."
    }
}
