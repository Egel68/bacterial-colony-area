from typing import Dict

import numpy as np


def compute_segmentation_metrics(
    pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-6
) -> Dict[str, float]:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    tp = np.sum((pred == 1) & (gt == 1)).item()
    fp = np.sum((pred == 1) & (gt == 0)).item()
    fn = np.sum((pred == 0) & (gt == 1)).item()
    tn = np.sum((pred == 0) & (gt == 0)).item()

    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    accuracy = (tp + tn) / (tp + fp + fn + tn + smooth)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
