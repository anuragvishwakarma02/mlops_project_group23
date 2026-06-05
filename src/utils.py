"""Shared helpers used by training and evaluation modules."""

import torch
from sklearn.metrics import accuracy_score, f1_score


class MyDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping HuggingFace tokenizer encodings and integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """Return accuracy and weighted F1 for use with HuggingFace Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1':       f1_score(labels, preds, average='weighted'),
    }


def build_label_maps(labels):
    """
    Build deterministic label<->id mappings from a list of string labels.

    Returns:
        label2id (dict): str -> int
        id2label (dict): int -> str
    """
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label
