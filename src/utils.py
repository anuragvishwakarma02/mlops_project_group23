"""Shared helpers used by training and evaluation modules."""

"""Shared helpers for the NER pipeline."""

import random

import numpy as np
import torch


def set_seed_all(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name_map: dict) -> str:
    """Resolve runtime compute device with CUDA > MPS > CPU preference."""
    if torch.cuda.is_available():
        return device_name_map["cuda"]
    if torch.backends.mps.is_available():
        return device_name_map["mps"]
    return device_name_map["cpu"]


def compute_token_metrics(eval_pred, label_list, seqeval_metric):
    """Compute seqeval metrics from token-classification predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []
    valid_token_count = 0

    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags = []
        gold_tags = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                pred_tags.append(label_list[pred_id])
                gold_tags.append(label_list[label_id])
                valid_token_count += 1
        if gold_tags:
            true_predictions.append(pred_tags)
            true_labels.append(gold_tags)

    # seqeval raises ZeroDivisionError when there are no valid labels.
    if valid_token_count == 0 or not true_labels:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
