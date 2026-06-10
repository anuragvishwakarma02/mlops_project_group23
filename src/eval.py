"""Evaluate token-classification model and produce reports."""

import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

from config import (
    DEVICE_NAME,
    ENABLE_WANDB,
    EVAL_OUTPUT_DIR,
    LOCAL_MODEL_DIR,
    PROCESSED_DATA_FILE,
    TOP_K_CONFUSION,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
)
from utils import resolve_device
from wandb_utils import init_wandb_run


def log(message):
    print(f"[eval] {message}")


def flatten_valid_labels(pred_ids, true_ids):
    flat_true = []
    flat_pred = []
    for pred_seq, true_seq in zip(pred_ids, true_ids):
        for pred, true in zip(pred_seq, true_seq):
            if true != -100:
                flat_true.append(int(true))
                flat_pred.append(int(pred))
    return flat_true, flat_pred


def save_reports(flat_true, flat_pred, label_list, output_dir):
    target_names = [str(lbl) for lbl in label_list]
    report_text = classification_report(
        flat_true,
        flat_pred,
        labels=list(range(len(label_list))),
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
    report_dict = classification_report(
        flat_true,
        flat_pred,
        labels=list(range(len(label_list))),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    txt_path = os.path.join(output_dir, "classification_report.txt")
    json_path = os.path.join(output_dir, "classification_report.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    log("Saved classification report")
    log("\n" + "=" * 70)
    log("TOKEN-LEVEL CLASSIFICATION REPORT")
    log("=" * 70)
    log(report_text)
    return report_dict


def save_confusion(flat_true, flat_pred, label_list, label2id, output_dir):
    all_label_ids = list(range(len(label_list)))
    all_cm = confusion_matrix(flat_true, flat_pred, labels=all_label_ids)

    label_support = all_cm.sum(axis=1)
    o_id = label2id.get("O", None)

    entity_ids = [idx for idx in all_label_ids if idx != o_id and label_support[idx] > 0]
    entity_ids = sorted(entity_ids, key=lambda idx: label_support[idx], reverse=True)
    selected_ids = entity_ids[:TOP_K_CONFUSION]

    if o_id is not None and label_support[o_id] > 0:
        selected_ids = [o_id] + selected_ids

    if not selected_ids:
        selected_ids = [idx for idx in all_label_ids if label_support[idx] > 0][:TOP_K_CONFUSION]

    cm = confusion_matrix(flat_true, flat_pred, labels=selected_ids)
    selected_names = [label_list[idx] for idx in selected_ids]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    fig_w = max(12, min(24, int(0.5 * len(selected_names)) + 8))
    fig_h = max(8, min(20, int(0.5 * len(selected_names)) + 6))

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

    sns.heatmap(
        cm,
        ax=axes[0],
        cmap="Blues",
        xticklabels=selected_names,
        yticklabels=selected_names,
        linewidths=0.2,
        linecolor="gray",
    )
    axes[0].set_title(f"Token-level Confusion Matrix (top {len(selected_names)} labels)")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].tick_params(axis="x", rotation=90)

    sns.heatmap(
        cm_norm,
        ax=axes[1],
        cmap="magma",
        vmin=0,
        vmax=1,
        xticklabels=selected_names,
        yticklabels=selected_names,
        linewidths=0.2,
        linecolor="gray",
    )
    axes[1].set_title("Row-normalized Confusion Matrix")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].tick_params(axis="x", rotation=90)

    out_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(out_path)
    plt.close(fig)
    log(f"Saved confusion matrix figure: {out_path}")


def main():
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f"Processed data file not found: {PROCESSED_DATA_FILE}. Run data.py first."
        )

    if not os.path.exists(LOCAL_MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: {LOCAL_MODEL_DIR}. Run train.py first."
        )

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    with open(PROCESSED_DATA_FILE, "rb") as f:
        data = pickle.load(f)

    tokenized_dataset = data["tokenized_dataset"]
    label_list = data["label_list"]
    label2id = data["label2id"]
    eval_split = data["eval_split"]

    device = resolve_device(DEVICE_NAME)
    log(f"Using device: {device}")

    model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=EVAL_OUTPUT_DIR, report_to=[]),
        data_collator=data_collator,
    )

    eval_results = trainer.evaluate(tokenized_dataset[eval_split])

    # Print Trainer eval metrics summary
    log("\n" + "=" * 70)
    log("TRAINER EVALUATION METRICS")
    log("=" * 70)
    for key, val in sorted(eval_results.items()):
        if isinstance(val, float):
            log(f"  {key:<40} {val:.4f}")
        else:
            log(f"  {key:<40} {val}")
    log("=" * 70)

    pred_output = trainer.predict(tokenized_dataset[eval_split])
    pred_ids = np.argmax(pred_output.predictions, axis=2)
    true_ids = pred_output.label_ids

    flat_true, flat_pred = flatten_valid_labels(pred_ids, true_ids)
    report_dict = save_reports(flat_true, flat_pred, label_list, EVAL_OUTPUT_DIR)
    save_confusion(flat_true, flat_pred, label_list, label2id, EVAL_OUTPUT_DIR)

    # Print summary metrics to console
    for avg_key in ("macro avg", "weighted avg"):
        if avg_key in report_dict:
            m = report_dict[avg_key]
            log(f"{avg_key} — precision: {m['precision']:.4f} | recall: {m['recall']:.4f} | f1: {m['f1-score']:.4f} | support: {int(m['support'])}")

    if ENABLE_WANDB:
        wandb_enabled = init_wandb_run(
            project=WANDB_PROJECT,
            name=f"{WANDB_RUN_NAME}-eval",
            resume="allow",
        )
        if wandb_enabled:
            # 1. Trainer metrics (eval_loss, runtime, etc.)
            wandb.log({f"final/{k}": v for k, v in eval_results.items() if isinstance(v, (int, float))})

            # 2. Summary metrics (macro avg / weighted avg / accuracy)
            for avg_key in ("macro avg", "weighted avg", "accuracy"):
                if avg_key in report_dict:
                    m = report_dict[avg_key]
                    if isinstance(m, dict):
                        wandb.log({
                            f"final/{avg_key}/precision": m.get("precision", 0),
                            f"final/{avg_key}/recall": m.get("recall", 0),
                            f"final/{avg_key}/f1": m.get("f1-score", 0),
                        })
                    else:
                        wandb.log({f"final/{avg_key}": m})

            # 3. Artifacts
            artifact = wandb.Artifact("eval-report", type="evaluation")
            artifact.add_file(os.path.join(EVAL_OUTPUT_DIR, "classification_report.json"))
            artifact.add_file(os.path.join(EVAL_OUTPUT_DIR, "classification_report.txt"))
            artifact.add_file(os.path.join(EVAL_OUTPUT_DIR, "confusion_matrices.png"))
            wandb.log_artifact(artifact)
            wandb.finish()

    log(f"All evaluation artifacts saved to {EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
