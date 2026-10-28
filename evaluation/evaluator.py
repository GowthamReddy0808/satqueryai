"""
SatQuery AI - Remote Sensing Quantitative Evaluation Harness
Implements standard evaluation metrics for RS vision-language tasks:
- Grounding IoU@0.5
- VQA Accuracy & Token Overlap
- Change Detection IoU, Precision, Recall, and F1
- Abstention Fidelity & Unsupported Claim Rate
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def compute_bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes:
    Format: [ymin, xmin, ymax, xmax].
    """
    y_min_a, x_min_a, y_max_a, x_max_a = box_a
    y_min_b, x_min_b, y_max_b, x_max_b = box_b

    # Intersection coordinates
    inter_y_min = max(y_min_a, y_min_b)
    inter_x_min = max(x_min_a, x_min_b)
    inter_y_max = min(y_max_a, y_max_b)
    inter_x_max = min(x_max_a, x_max_b)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (x_max_a - x_min_a) * (y_max_a - y_min_a))
    area_b = max(0.0, (x_max_b - x_min_b) * (y_max_b - y_min_b))

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def compute_change_detection_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard change detection metrics:
    Precision, Recall, F1-Score, and Intersection over Union (IoU).
    """
    pred = predicted_mask.astype(bool)
    gt = ground_truth_mask.astype(bool)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    oa = float((tp + tn) / (tp + fp + fn + tn)) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou": round(iou, 4),
        "overall_accuracy": round(oa, 4)
    }


def compute_vqa_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute Exact Match and Relaxed Token Overlap Accuracy for VQA pairs.
    """
    if not predictions or len(predictions) != len(references):
        return {"exact_match_accuracy": 0.0, "token_f1": 0.0, "sample_count": 0}

    exact_matches = 0
    token_f1s = []

    for pred, ref in zip(predictions, references):
        p_clean = pred.lower().strip()
        r_clean = ref.lower().strip()

        if p_clean == r_clean or r_clean in p_clean:
            exact_matches += 1

        p_tokens = set(p_clean.split())
        r_tokens = set(r_clean.split())
        inter = len(p_tokens & r_tokens)
        if len(p_tokens) + len(r_tokens) > 0:
            f1 = (2.0 * inter) / (len(p_tokens) + len(r_tokens))
            token_f1s.append(f1)
        else:
            token_f1s.append(1.0 if not p_tokens and not r_tokens else 0.0)

    return {
        "exact_match_accuracy": round(exact_matches / len(predictions), 4),
        "mean_token_f1": round(float(np.mean(token_f1s)), 4),
        "sample_count": len(predictions)
    }
