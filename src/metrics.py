import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(preds, targets):
    preds = preds.astype(np.uint8).flatten()
    targets = targets.astype(np.uint8).flatten()

    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)

    intersection = np.logical_and(preds == 1, targets == 1).sum()
    union = np.logical_or(preds == 1, targets == 1).sum()
    iou = intersection / (union + 1e-8)

    cm = confusion_matrix(targets, preds, labels=[0, 1])

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }