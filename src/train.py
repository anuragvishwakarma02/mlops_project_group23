"""
train.py — Model loading, Trainer setup, and fine-tuning loop.

Run:
    python train.py

Expects:
    processed_genre_reviews_dict.pickle   — produced by data.py

Outputs:
    distilbert-reviews-genres/   — saved fine-tuned model + config
    results/                     — training checkpoints
    logs/                        — training logs
"""

import pickle

import torch
import wandb
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from config import (
    MODEL_NAME, DEVICE_NAME, PROCESSED_DATA_FILE, CACHED_MODEL_DIR,
    TRAINING_ARGS, WANDB_PROJECT, WANDB_RUN_NAME, MAX_LENGTH, SAMPLE_SIZE,
)
from utils import MyDataset, compute_metrics
from wandb_utils import init_wandb_run

def log(message):
    """Print a friendly status message for the training stage."""
    print(f"[train] {message}")

def main():
    log("Starting experiment tracking on Weights & Biases")
    wandb_enabled = init_wandb_run(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            'model':         MODEL_NAME,
            'epochs':        TRAINING_ARGS['num_train_epochs'],
            'batch_size':    TRAINING_ARGS['per_device_train_batch_size'],
            'learning_rate': TRAINING_ARGS['learning_rate'],
            'max_length':    MAX_LENGTH,
            'sample_size':   SAMPLE_SIZE,
            'dataset':       'UCSD Goodreads',
        },
    )
    if wandb_enabled:
        log("Using WANDB_API_KEY from environment for tracking")
    else:
        log("WANDB_API_KEY not set; W&B tracking disabled")

    log(f"Loading processed dataset from {PROCESSED_DATA_FILE}")
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    train_dataset = MyDataset(data['train_encodings'], data['train_labels_encoded'])
    test_dataset  = MyDataset(data['test_encodings'],  data['test_labels_encoded'])
    id2label      = data['id2label']
    label2id      = data['label2id']

    log("Checking GPU availability:")
    DEVICE= DEVICE_NAME['cuda'] if torch.cuda.is_available() else DEVICE_NAME['mps'] if torch.backends.mps.is_available() else DEVICE_NAME['cpu']
    log(f" Using device : {DEVICE}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**TRAINING_ARGS),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    log("Fine-tuning DistilBERT. This may take a while...")
    trainer.train()

    trainer.save_model(CACHED_MODEL_DIR)
    log(f"Training complete. Model saved to {CACHED_MODEL_DIR}/")

    wandb.finish()
    if wandb_enabled:
        log("W&B run closed successfully")


if __name__ == '__main__':
    main()
