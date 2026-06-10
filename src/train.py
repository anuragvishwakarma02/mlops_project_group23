"""
train.py — Model loading, Trainer setup, and fine-tuning loop.

Run:
    python train.py
"""


import os
import pickle

import evaluate
import torch
import wandb
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from config import (
    DEVICE_NAME,
    ENABLE_WANDB,
    LOCAL_MODEL_DIR,
    MODEL_NAME,
    PROCESSED_DATA_FILE,
    TRAINING_ARGS,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
)
from utils import compute_token_metrics, resolve_device
from wandb_utils import init_wandb_run


def log(message):
    print(f"[train] {message}")


def main():
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f"Processed data file not found: {PROCESSED_DATA_FILE}. Run data.py first."
        )

    with open(PROCESSED_DATA_FILE, "rb") as f:
        data = pickle.load(f)

    tokenized_dataset = data["tokenized_dataset"]
    label_list = data["label_list"]
    id2label = data["id2label"]
    label2id = data["label2id"]
    eval_split = data["eval_split"]

    device = resolve_device(DEVICE_NAME)
    log(f"Using device: {device}")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    if ENABLE_WANDB:
        wandb_enabled = init_wandb_run(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model_name": MODEL_NAME,
                "num_labels": len(label_list),
                "epochs": TRAINING_ARGS["num_train_epochs"],
                "train_batch_size": TRAINING_ARGS["per_device_train_batch_size"],
                "eval_batch_size": TRAINING_ARGS["per_device_eval_batch_size"],
                "task": "token-classification",
            },
        )
        if not wandb_enabled:
            log("WANDB_API_KEY not set; run logging is disabled")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval_metric = evaluate.load("seqeval")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            **TRAINING_ARGS,
            report_to=["wandb"] if ENABLE_WANDB else [],
            run_name=WANDB_RUN_NAME,
        ),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[eval_split],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_token_metrics(pred, label_list, seqeval_metric),
    )

    log("Starting NER fine-tuning")
    trainer.train()

    trainer.save_model(LOCAL_MODEL_DIR)
    log(f"Training complete. Model saved to {LOCAL_MODEL_DIR}")

    if ENABLE_WANDB and wandb.run is not None:
        wandb.finish()
        log("W&B run closed")


if __name__ == "__main__":
    main()
