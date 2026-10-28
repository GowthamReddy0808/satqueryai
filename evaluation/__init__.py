"""SatQuery AI Evaluation Package"""
from .benchmarks import BENCHMARK_DATASETS
from .evaluator import compute_bbox_iou, compute_change_detection_metrics, compute_vqa_accuracy

__all__ = [
    "BENCHMARK_DATASETS",
    "compute_bbox_iou",
    "compute_change_detection_metrics",
    "compute_vqa_accuracy"
]
