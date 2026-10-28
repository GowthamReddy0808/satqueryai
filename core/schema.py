"""
SatQuery AI - Core Data Schema & Abstention Definitions
Defines standardized data contracts for multi-modal remote sensing analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from PIL import Image


class TaskType(str, Enum):
    VQA = "visual_question_answering"
    CAPTIONING = "scene_captioning"
    GROUNDING = "region_grounding"
    COUNTING = "object_counting"
    LANDCOVER = "land_cover_analysis"
    SPECTRAL = "spectral_analysis"
    CHANGE_DETECTION = "bi_temporal_change_detection"
    CHANGE_VQA = "change_based_vqa"
    OPTICAL_SAR = "optical_sar_joint_analysis"
    GENERAL = "general_query"


class FeatureEvidenceState(str, Enum):
    DETECTED = "Feature detected with high confidence"
    NOT_DETECTED = "Feature not detected in visual evidence"
    CONFIRMED_ABSENT = "Feature confirmed absent within observable bands"
    UNABLE_TO_DETERMINE = "Unable to determine due to spatial/spectral resolution or cloud cover"
    INPUT_UNSUITABLE = "Input imagery unsuitable for the requested analytical task"


@dataclass
class BoundingBox:
    box_2d: List[float]  # [ymin, xmin, ymax, xmax] in normalized 0-1 or pixel coords
    label: str
    confidence: Optional[float] = None
    color: str = "#38bdf8"
    is_normalized: bool = True


@dataclass
class VisualOutput:
    type: str  # "annotated_image", "segmentation_mask", "change_map", "spectral_ndvi", "sar_intensity", etc.
    title: str
    image: Image.Image
    description: str
    coverage_pct: Optional[float] = None
    format: str = "PNG"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    step_name: str
    tool_or_model: str
    status: str  # "SUCCESS", "WARNING", "SKIPPED", "FAILED"
    detail: str
    duration_ms: float = 0.0


@dataclass
class SatQueryResult:
    status: str  # "success", "abstention", "error", "unsupported_task"
    task_type: TaskType
    answer: str
    reasoning: Optional[str] = None
    detections: List[BoundingBox] = field(default_factory=list)
    visual_outputs: List[VisualOutput] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    execution_trace: List[TraceStep] = field(default_factory=list)
    duration_sec: float = 0.0
    feature_tags: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    abstention_reason: Optional[str] = None
    evidence_state: Optional[FeatureEvidenceState] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "task_type": self.task_type.value,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "detections": [
                {
                    "box_2d": d.box_2d,
                    "label": d.label,
                    "confidence": d.confidence,
                    "color": d.color
                } for d in self.detections
            ],
            "visual_outputs": [
                {
                    "type": v.type,
                    "title": v.title,
                    "description": v.description,
                    "coverage_pct": v.coverage_pct,
                    "format": v.format
                } for v in self.visual_outputs
            ],
            "metadata": self.metadata,
            "limitations": self.limitations,
            "execution_trace": [
                {
                    "step": t.step_name,
                    "tool": t.tool_or_model,
                    "status": t.status,
                    "detail": t.detail,
                    "duration_ms": t.duration_ms
                } for t in self.execution_trace
            ],
            "duration_sec": self.duration_sec,
            "feature_tags": self.feature_tags,
            "confidence_score": self.confidence_score,
            "abstention_reason": self.abstention_reason,
            "evidence_state": self.evidence_state.value if self.evidence_state else None
        }
