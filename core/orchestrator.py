"""
SatQuery AI - Agentic Query Orchestrator & Multi-Tool Controller
Classifies natural-language remote-sensing queries, checks input feasibility,
selects specialist tools, audits execution traces, and applies honest abstention rules.
"""

import time
import io
import re
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from core.schema import (
    TaskType,
    FeatureEvidenceState,
    BoundingBox,
    VisualOutput,
    TraceStep,
    SatQueryResult
)
from tools.raster_processor import (
    inspect_raster_metadata,
    calculate_ndvi,
    generate_spectral_masks
)
from tools.change_detector import (
    validate_bi_temporal_pair,
    compute_bi_temporal_change_map
)
from tools.optical_sar_fusion import process_optical_sar_pair
from tools.annotator import draw_bounding_boxes
from models.teammate_adapter import TeammateModelAdapter
from models.openrouter_client import query_multimodal_vlm, extract_features


class AgenticQueryOrchestrator:
    """
    Modular controller executing evidence-grounded remote-sensing workflows.
    """

    def __init__(self, teammate_adapter: Optional[TeammateModelAdapter] = None):
        self.teammate_adapter = teammate_adapter or TeammateModelAdapter()

    def classify_task(self, query: str, num_images: int = 1) -> TaskType:
        """
        Classify analyst intent using semantic pattern matching.
        """
        q = query.lower().strip()

        # Change detection patterns
        if any(w in q for w in ["change", "changed", "between these two", "earlier and later", "over time", "has the built-up area increased", "decreased"]):
            return TaskType.CHANGE_DETECTION if num_images > 1 else TaskType.CHANGE_VQA

        # Optical + SAR patterns
        if any(w in q for w in ["optical and sar", "optical + sar", "radar and optical", "sar image", "backscatter"]):
            return TaskType.OPTICAL_SAR

        # Region Grounding & Localization
        if any(w in q for w in ["locate", "highlight the", "bounding box", "where is the", "point out", "localize"]):
            return TaskType.GROUNDING

        # Object Counting
        if any(w in q for w in ["count", "how many", "number of buildings", "number of ships"]):
            return TaskType.COUNTING

        # Spectral / NDVI Analysis
        if any(w in q for w in ["ndvi", "vegetation index", "spectral index", "nir band", "chlorophyll index"]):
            return TaskType.SPECTRAL

        # Scene Captioning / Summary
        if any(w in q for w in ["caption", "describe the scene", "summarize this image", "overview of the scene"]):
            return TaskType.CAPTIONING

        # Land Cover Analysis
        if any(w in q for w in ["land cover", "land-cover", "classification", "water body", "urban area", "forest cover"]):
            return TaskType.LANDCOVER

        return TaskType.VQA

    def execute_workflow(
        self,
        query: str,
        image_bytes_list: List[bytes],
        image_names: List[str],
        image_types: List[str],
        active_model_name: str = "nvidia/nemotron-3.5-super-vl:free",
        openrouter_api_key: Optional[str] = None,
        use_teammate_model: bool = False
    ) -> SatQueryResult:
        """
        Execute full 10-step agentic remote sensing pipeline.
        """
        overall_start = time.time()
        trace: List[TraceStep] = []
        visual_outputs: List[VisualOutput] = []
        limitations: List[str] = []
        detections: List[BoundingBox] = []

        num_images = len(image_bytes_list)

        # Step 1: Input Validation
        t0 = time.time()
        if num_images == 0:
            return SatQueryResult(
                status="abstention",
                task_type=TaskType.GENERAL,
                answer="No satellite imagery provided. Please upload an Earth observation scene to begin analysis.",
                abstention_reason="Empty input image container",
                evidence_state=FeatureEvidenceState.INPUT_UNSUITABLE,
                limitations=["No image uploaded"]
            )

        meta_primary = inspect_raster_metadata(image_bytes_list[0], image_names[0])
        trace.append(TraceStep(
            step_name="1. Input Inspection & Metadata Extraction",
            tool_or_model="rasterio / PIL inspector",
            status="SUCCESS",
            detail=f"Parsed {meta_primary['format']} ({meta_primary['width']}×{meta_primary['height']} px, {meta_primary['band_count']} bands, CRS: {meta_primary['crs']})",
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        # Step 2: Task Classification
        t1 = time.time()
        task_type = self.classify_task(query, num_images)
        trace.append(TraceStep(
            step_name="2. Query Intent Classification",
            tool_or_model="Agentic Routing Controller",
            status="SUCCESS",
            detail=f"Classified inquiry as '{task_type.value}' based on natural language structure.",
            duration_ms=round((time.time() - t1) * 1000, 1)
        ))

        # Step 3: Feasibility & Abstention Checks
        t2 = time.time()
        if task_type in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA) and num_images < 2:
            trace.append(TraceStep(
                step_name="3. Feasibility Verification",
                tool_or_model="Integrity Controller",
                status="WARNING",
                detail="Bi-temporal change detection requested with only 1 image provided. Triggering honest abstention.",
                duration_ms=round((time.time() - t2) * 1000, 1)
            ))
            return SatQueryResult(
                status="abstention",
                task_type=task_type,
                answer=(
                    "**Abstention — Insufficient Temporal Observations:**\n\n"
                    "Bi-temporal change detection requires two co-registered satellite observations (an earlier date T1 and a later date T2) "
                    "of the same geographic area to observe landscape evolution, urban expansion, or deforestation.\n\n"
                    "Please navigate to the **Change Detection (Bi-Temporal)** workflow in the sidebar and upload both acquisition dates."
                ),
                metadata=meta_primary,
                execution_trace=trace,
                limitations=["Only one acquisition date provided for bi-temporal query."],
                abstention_reason="Missing second temporal image pair",
                evidence_state=FeatureEvidenceState.INPUT_UNSUITABLE
            )

        if task_type == TaskType.OPTICAL_SAR and num_images < 2:
            trace.append(TraceStep(
                step_name="3. Feasibility Verification",
                tool_or_model="Integrity Controller",
                status="WARNING",
                detail="Joint Optical-SAR analysis requested with only 1 image provided. Triggering honest abstention.",
                duration_ms=round((time.time() - t2) * 1000, 1)
            ))
            return SatQueryResult(
                status="abstention",
                task_type=task_type,
                answer=(
                    "**Abstention — Missing Cross-Modal Input:**\n\n"
                    "Optical–SAR joint analysis requires both an optical or multispectral image (surface reflectance) and a "
                    "Synthetic Aperture Radar (SAR) backscatter scene of the same geographic area.\n\n"
                    "Please navigate to the **Optical + SAR Analysis** workspace in the sidebar and upload both sensor modalities."
                ),
                metadata=meta_primary,
                execution_trace=trace,
                limitations=["Single modality provided for cross-sensor query."],
                abstention_reason="Missing SAR or Optical companion scene",
                evidence_state=FeatureEvidenceState.INPUT_UNSUITABLE
            )

        trace.append(TraceStep(
            step_name="3. Feasibility Verification",
            tool_or_model="Integrity Controller",
            status="SUCCESS",
            detail="Input parameters satisfied for selected task workflow.",
            duration_ms=round((time.time() - t2) * 1000, 1)
        ))

        # ---------------------------------------------------------------------
        # WORKFLOW A: BI-TEMPORAL CHANGE DETECTION (2 IMAGES)
        # ---------------------------------------------------------------------
        if task_type in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA) and num_images >= 2:
            t_change = time.time()
            valid, msg, pair_meta = validate_bi_temporal_pair(
                image_bytes_list[0], image_bytes_list[1], image_names[0], image_names[1]
            )

            if not valid:
                limitations.append(msg)
                return SatQueryResult(
                    status="abstention",
                    task_type=task_type,
                    answer=f"**Analysis Aborted:** {msg}",
                    metadata=pair_meta,
                    limitations=limitations,
                    execution_trace=trace,
                    evidence_state=FeatureEvidenceState.INPUT_UNSUITABLE
                )

            heat_map, classified_overlay, metrics = compute_bi_temporal_change_map(
                image_bytes_list[0], image_bytes_list[1]
            )

            visual_outputs.append(VisualOutput(
                type="change_overlay",
                title="Classified Change Map",
                image=classified_overlay,
                description=f"Gain/Expansion (Green): {metrics['gain_expansion_pct']}%, Loss/Reduction (Red): {metrics['loss_reduction_pct']}%.",
                coverage_pct=metrics["total_changed_pct"]
            ))

            visual_outputs.append(VisualOutput(
                type="change_heatmap",
                title="Radiometric Difference Heatmap",
                image=heat_map,
                description="Absolute pixel intensity delta between T1 and T2.",
                coverage_pct=metrics["total_changed_pct"]
            ))

            trace.append(TraceStep(
                step_name="4. Radiometric Change Mapping",
                tool_or_model="tools/change_detector.py",
                status="SUCCESS",
                detail=f"Computed pixel delta. Changed Area: {metrics['total_changed_pct']}% (Gain: {metrics['gain_expansion_pct']}%, Loss: {metrics['loss_reduction_pct']}%)",
                duration_ms=round((time.time() - t_change) * 1000, 1)
            ))

            answer_text = (
                f"### Bi-Temporal Change Analysis Results\n\n"
                f"- **Earlier Observation (T1):** `{image_names[0]}`\n"
                f"- **Later Observation (T2):** `{image_names[1]}`\n"
                f"- **Total Identified Landscape Shift:** **{metrics['total_changed_pct']}%** of the scene.\n"
                f"- **Expansion / Gain Signatures:** **{metrics['gain_expansion_pct']}%** (Highlighted in Green on T2).\n"
                f"- **Reduction / Loss Signatures:** **{metrics['loss_reduction_pct']}%** (Highlighted in Red on T2).\n\n"
                f"**Confounding Factors Evaluated:**\n"
                + "\n".join([f"- {cf}" for cf in metrics["confounding_factors"]])
            )

            return SatQueryResult(
                status="success",
                task_type=task_type,
                answer=answer_text,
                visual_outputs=visual_outputs,
                metadata={**meta_primary, **metrics},
                execution_trace=trace,
                feature_tags=["🔄 Bi-Temporal Change", "🛰️ Change Map"],
                duration_sec=round(time.time() - overall_start, 2)
            )

        # ---------------------------------------------------------------------
        # WORKFLOW B: OPTICAL + SAR JOINT ANALYSIS (2 MODALITIES)
        # ---------------------------------------------------------------------
        if task_type == TaskType.OPTICAL_SAR and num_images >= 2:
            t_sar = time.time()
            opt_img, sar_vis, composite_img, report = process_optical_sar_pair(
                image_bytes_list[0], image_bytes_list[1], image_names[0], image_names[1]
            )

            visual_outputs.append(VisualOutput(
                type="sar_backscatter",
                title="SAR Backscatter Intensity",
                image=sar_vis,
                description="Greyscale amplitude of radar backscatter showing roughness and volumetric geometry."
            ))

            visual_outputs.append(VisualOutput(
                type="optical_sar_composite",
                title="Structural-Spectral Complementary Product",
                image=composite_img,
                description="Optical Red/Green surface reflectance blended with SAR Blue radar backscatter."
            ))

            trace.append(TraceStep(
                step_name="4. Cross-Modal Sensor Synthesis",
                tool_or_model="tools/optical_sar_fusion.py",
                status="SUCCESS",
                detail=f"Synthesized complementary composite. High radar backscatter structures: {report['high_backscatter_features_pct']}%",
                duration_ms=round((time.time() - t_sar) * 1000, 1)
            ))

            answer_text = (
                f"### Optical + SAR Joint Intelligence Analysis\n\n"
                f"**Optical Contribution (`{image_names[0]}`):**\n"
                f"{report['optical_contribution']}\n\n"
                f"**SAR Radar Contribution (`{image_names[1]}`):**\n"
                f"{report['sar_contribution']}\n\n"
                f"- **High-Backscatter Signatures (Urban / Metallic / Corner Reflector):** **{report['high_backscatter_features_pct']}%** of the scene.\n\n"
                f"*{report['scientific_notice']}*"
            )

            return SatQueryResult(
                status="success",
                task_type=task_type,
                answer=answer_text,
                visual_outputs=visual_outputs,
                metadata={**meta_primary, **report},
                execution_trace=trace,
                feature_tags=["📡 SAR Backscatter", "🛰️ Optical Reflectance", "🔬 Sensor Fusion"],
                duration_sec=round(time.time() - overall_start, 2)
            )

        # ---------------------------------------------------------------------
        # WORKFLOW C: SINGLE IMAGE SPECTRAL ANALYSIS (NDVI / CIR / MASKS)
        # ---------------------------------------------------------------------
        if task_type == TaskType.SPECTRAL:
            t_spec = time.time()
            ndvi_arr, ndvi_vis, ndvi_report = calculate_ndvi(image_bytes_list[0])
            masks = generate_spectral_masks(image_bytes_list[0])

            if ndvi_vis is not None:
                visual_outputs.append(VisualOutput(
                    type="ndvi_map",
                    title="Vegetation Index Map (NDVI / CIR)",
                    image=ndvi_vis,
                    description=ndvi_report["status_message"],
                    coverage_pct=ndvi_report["vegetation_coverage_pct"]
                ))

            for k, (m_img, cov, desc) in masks.items():
                visual_outputs.append(VisualOutput(
                    type=f"mask_{k}",
                    title=f"Spectral Product: {k.capitalize()}",
                    image=m_img,
                    description=desc,
                    coverage_pct=cov
                ))

            trace.append(TraceStep(
                step_name="4. Spectral Product Computation",
                tool_or_model="tools/raster_processor.py",
                status="SUCCESS",
                detail=f"Computed vegetation index: {ndvi_report['status_message']}. Canopy: {ndvi_report['vegetation_coverage_pct']}%",
                duration_ms=round((time.time() - t_spec) * 1000, 1)
            ))

            answer_text = (
                f"### Spectral Vegetation Assessment\n\n"
                f"- **Analysis Mode:** {ndvi_report['status_message']}\n"
                f"- **Canopy Reflectance Coverage:** **{ndvi_report['vegetation_coverage_pct']}%**\n"
                f"- **Mean Spectral Index Value:** **{ndvi_report['mean_index']}**\n\n"
                f"**Methodological Explanation:**\n"
                f"{ndvi_report['explanation']}"
            )

            if not ndvi_report["is_genuine_ndvi"]:
                limitations.append("Image is 3-band RGB. Genuine physical NDVI requires genuine calibrated NIR band.")

            return SatQueryResult(
                status="success",
                task_type=task_type,
                answer=answer_text,
                visual_outputs=visual_outputs,
                metadata={**meta_primary, **ndvi_report},
                limitations=limitations,
                execution_trace=trace,
                feature_tags=["🌿 Vegetation Canopy", "🔬 Spectral Index"],
                duration_sec=round(time.time() - overall_start, 2)
            )

        # ---------------------------------------------------------------------
        # WORKFLOW D: SINGLE IMAGE VQA, CAPTIONING, OR REGION GROUNDING
        # ---------------------------------------------------------------------
        t_model = time.time()
        model_answer = ""
        model_reasoning = None
        tags = []

        # Check if teammate model is active & available
        teammate_health = self.teammate_adapter.check_health()
        if use_teammate_model and teammate_health["available"]:
            trace.append(TraceStep(
                step_name="4. Model Selection & Routing",
                tool_or_model=f"Teammate RS Adapter ({self.teammate_adapter.model_id})",
                status="SUCCESS",
                detail=f"Connected to custom inference server at {self.teammate_adapter.api_base_url}",
                duration_ms=teammate_health["latency_ms"]
            ))

            res = self.teammate_adapter.infer(
                task_type=task_type,
                query=query,
                image_bytes_list=[image_bytes_list[0]],
                image_formats=[image_types[0] if image_types else "image/png"],
                metadata=meta_primary
            )
            if res["success"]:
                model_answer = res["answer"]
                model_reasoning = res.get("reasoning")
                detections = res.get("detections", [])
            else:
                limitations.append(f"Teammate model returned error: {res.get('error')}. Falling back to OpenRouter.")
                use_teammate_model = False

        if not use_teammate_model or not model_answer:
            # Fallback to OpenRouter Multimodal Vision-Language Model
            trace.append(TraceStep(
                step_name="4. Model Selection & Routing",
                tool_or_model=f"OpenRouter VLM ({active_model_name})",
                status="SUCCESS",
                detail="Invoking multimodal vision-language model with Earth Observation prompt.",
                duration_ms=round((time.time() - t_model) * 1000, 1)
            ))

            # System prompt customized by task type
            if task_type == TaskType.CAPTIONING:
                sys_prompt = (
                    "You are SatQuery AI, an Earth observation specialist. Provide a detailed, scientific scene caption "
                    "for this remote-sensing satellite image. Identify major land cover classes (vegetation, hydrology, built-up, bare soil), "
                    "spatial layout, sensor modality (RGB vs False-Colour NIR), and notable geographic features."
                )
            elif task_type == TaskType.GROUNDING:
                sys_prompt = (
                    "You are SatQuery AI. Locate and identify the spatial boundaries of the requested geographic feature. "
                    "State the location clearly in cardinal directions (north, south, center) and if possible provide approximate normalized "
                    "bounding box coordinates in the format [ymin, xmin, ymax, xmax] with values between 0.0 and 1.0."
                )
            else:
                sys_prompt = (
                    "You are SatQuery AI, an expert satellite image analysis assistant. "
                    "Carefully inspect the provided remote-sensing satellite image and answer the user's question with scientific accuracy. "
                    "Describe visible land features, possible water bodies, vegetation patterns, and built-up areas. "
                    "The image may use false colours (such as Near-Infrared where healthy vegetation appears red). "
                    "Explain what the colours represent when appropriate. "
                    "Distinguish visible evidence from uncertainty. If a feature cannot be conclusively identified, state so clearly."
                )

            t_vlm = time.time()
            res = query_multimodal_vlm(
                image_bytes=image_bytes_list[0],
                mime_type=image_types[0] if image_types else "image/png",
                question=query,
                model_name=active_model_name,
                api_key=openrouter_api_key,
                system_instruction=sys_prompt
            )

            if not res["success"]:
                trace.append(TraceStep(
                    step_name="5. Model Inference Execution",
                    tool_or_model=active_model_name,
                    status="FAILED",
                    detail=res.get("error", "Unknown inference error"),
                    duration_ms=round((time.time() - t_vlm) * 1000, 1)
                ))
                return SatQueryResult(
                    status="error",
                    task_type=task_type,
                    answer=f"**Inference Error:** {res.get('error')}",
                    metadata=meta_primary,
                    limitations=[res.get("error", "API Error")],
                    execution_trace=trace
                )

            model_answer = res["answer"]
            model_reasoning = res.get("reasoning")
            tags = res.get("tags", [])
            detections = res.get("detections", [])

            trace.append(TraceStep(
                step_name="5. Model Inference Execution",
                tool_or_model=active_model_name,
                status="SUCCESS",
                detail=f"Inference completed in {res.get('duration', 0)}s. Identified {len(tags)} feature tags.",
                duration_ms=round((time.time() - t_vlm) * 1000, 1)
            ))

        # Render Bounding Box Annotations if detections found or Grounding task
        if detections:
            t_box = time.time()
            base_pil = Image.open(io.BytesIO(image_bytes_list[0]))
            annotated_img = draw_bounding_boxes(base_pil, detections)
            visual_outputs.append(VisualOutput(
                type="annotated_image",
                title=f"Annotated Scene ({len(detections)} Grounded Regions)",
                image=annotated_img,
                description=f"Model localized {len(detections)} spatial bounding regions."
            ))
            trace.append(TraceStep(
                step_name="6. Geospatial Grounding & Annotation",
                tool_or_model="tools/annotator.py",
                status="SUCCESS",
                detail=f"Rendered {len(detections)} bounding boxes on display scene.",
                duration_ms=round((time.time() - t_box) * 1000, 1)
            ))
        elif task_type == TaskType.GROUNDING:
            limitations.append("The connected model did not return exact bounding box coordinates for this prompt. Qualitative spatial explanation provided.")

        # Always generate spectral overlay products as companion evidence
        t_masks = time.time()
        spectral_masks = generate_spectral_masks(image_bytes_list[0])
        # Add primary mask relevant to features
        if "🌿 Vegetation Canopy" in tags:
            v_img, v_cov, v_desc = spectral_masks["vegetation"]
            visual_outputs.append(VisualOutput(
                type="mask_vegetation",
                title="Canopy Reflectance Overlay",
                image=v_img,
                description=v_desc,
                coverage_pct=v_cov
            ))
        elif "💧 Water Body" in tags:
            w_img, w_cov, w_desc = spectral_masks["water"]
            visual_outputs.append(VisualOutput(
                type="mask_water",
                title="Hydrology Absorption Overlay",
                image=w_img,
                description=w_desc,
                coverage_pct=w_cov
            ))

        trace.append(TraceStep(
            step_name="7. Evidence Synthesis & Formatting",
            tool_or_model="SatQuery Core Orchestrator",
            status="SUCCESS",
            detail=f"Synthesized {len(visual_outputs)} visual outputs and grounded textual evidence.",
            duration_ms=round((time.time() - t_masks) * 1000, 1)
        ))

        return SatQueryResult(
            status="success",
            task_type=task_type,
            answer=model_answer,
            reasoning=model_reasoning,
            detections=detections,
            visual_outputs=visual_outputs,
            metadata=meta_primary,
            limitations=limitations,
            execution_trace=trace,
            duration_sec=round(time.time() - overall_start, 2),
            feature_tags=tags if tags else extract_features(model_answer),
            evidence_state=FeatureEvidenceState.DETECTED if tags else FeatureEvidenceState.UNABLE_TO_DETERMINE
        )
