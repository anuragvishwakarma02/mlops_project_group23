
"""
eval.py — Evaluation, metrics, and result artefacts.

Run:
    python eval.py

Expects:
    processed_genre_reviews_dict.pickle          — produced by data.py
    distilbert-reviews-genres/     — produced by train.py

Outputs:
    eval_results/classification_report.txt
    eval_results/confusion_heatmap.png
    eval_results/misclassification_heatmap.png
"""

import json
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch

from config import (
    PROCESSED_DATA_FILE, CACHED_MODEL_DIR, DEVICE_NAME, EVAL_OUTPUT_DIR,
    WANDB_PROJECT, WANDB_RUN_NAME,
)
from utils import MyDataset, compute_metrics
from wandb_utils import init_wandb_run

def log(message):
    """Print a friendly status message for the evaluation stage."""
    print(f"[eval] {message}")


def run_baseline(train_texts, train_labels, test_texts, test_labels):
    """TF-IDF + Logistic Regression baseline for comparison."""
    log("Running baseline: TF-IDF + Logistic Regression")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    lr_model = LogisticRegression(max_iter=1000).fit(X_train, train_labels)
    preds = lr_model.predict(X_test)
    print(classification_report(test_labels, preds))
    return preds


def predict_with_bert(model, test_dataset, id2label):
    """
    Run HuggingFace Trainer predict and map integer ids back to label strings.
    Also returns the raw eval metrics dict.
    """
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    log("Running Hugging Face evaluation loop")
    eval_results = trainer.evaluate(test_dataset)
    log(f"Evaluation metrics: {eval_results}")

    predicted = trainer.predict(test_dataset)
    label_ids = predicted.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in label_ids]
    return predicted_labels, eval_results


def save_report(test_labels, predicted_labels, output_dir):
    """Print, save, and return the sklearn classification report as a dict."""
    report_str  = classification_report(test_labels, predicted_labels)
    report_dict = classification_report(test_labels, predicted_labels, output_dict=True)
    log("BERT classification report:")
    print(report_str)

    txt_path  = os.path.join(output_dir, 'classification_report.txt')
    json_path = os.path.join(output_dir, 'classification_report.json')
    with open(txt_path, 'w') as f:
        f.write(report_str)
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    log(f"Saved classification report to {txt_path} and {json_path}")
    return report_dict


def save_heatmap(test_labels, predicted_labels, output_dir, exclude_diagonal=False):
    """Save a seaborn heatmap of genre classifications (or misclassifications)."""
    counts = defaultdict(int)
    for true, pred in zip(test_labels, predicted_labels):
        if not (exclude_diagonal and true == pred):
            counts[(true, pred)] += 1

    rows = [
        {'True Genre': t, 'Predicted Genre': p, 'Number of Classifications': c}
        for (t, p), c in counts.items()
    ]
    df_wide = pd.DataFrame(rows).pivot_table(
        index='True Genre',
        columns='Predicted Genre',
        values='Number of Classifications',
    )

    plt.figure(figsize=(9, 7))
    sns.set(style='ticks', font_scale=1.2)
    sns.heatmap(df_wide, linewidths=1, cmap='Purples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fname = 'misclassification_heatmap.png' if exclude_diagonal else 'confusion_heatmap.png'
    path = os.path.join(output_dir, fname)
    plt.savefig(path)
    plt.close()
    log(f"Saved heatmap to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    log("Starting evaluation and metrics logging")

    wandb_enabled = init_wandb_run(
        project=WANDB_PROJECT,
        name=f'{WANDB_RUN_NAME}-eval',
        resume='allow',
    )
    if wandb_enabled:
        log("Using WANDB_API_KEY from environment for tracking")
    else:
        log("WANDB_API_KEY not set; W&B tracking disabled")

    log(f"Loading processed dataset from {PROCESSED_DATA_FILE}")
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    test_dataset = MyDataset(data['test_encodings'], data['test_labels_encoded'])
    test_labels  = data['test_labels']
    id2label     = data['id2label']

    # Baseline
    run_baseline(
        data['train_texts'], data['train_labels'],
        data['test_texts'],  test_labels,
    )

    log("Checking GPU availability:")
    DEVICE= DEVICE_NAME['cuda'] if torch.cuda.is_available() else DEVICE_NAME['mps'] if torch.backends.mps.is_available() else DEVICE_NAME['cpu']
    log(f" Using device : {DEVICE}")

    # Load fine-tuned model and predict
    log(f"Loading fine-tuned model from {CACHED_MODEL_DIR}")
    model = DistilBertForSequenceClassification.from_pretrained(CACHED_MODEL_DIR).to(DEVICE)

    predicted_labels, eval_results = predict_with_bert(model, test_dataset, id2label)

    # Save results
    save_report(test_labels, predicted_labels, EVAL_OUTPUT_DIR)
    save_heatmap(test_labels, predicted_labels, EVAL_OUTPUT_DIR, exclude_diagonal=False)
    save_heatmap(test_labels, predicted_labels, EVAL_OUTPUT_DIR, exclude_diagonal=True)

    # Log final metrics to W&B
    wandb.log({
        'final/loss':     eval_results.get('eval_loss', None),
        'final/accuracy': accuracy_score(test_labels, predicted_labels),
        'final/f1':       f1_score(test_labels, predicted_labels, average='weighted'),
    })

    # Upload classification report as a versioned W&B Artifact
    artifact = wandb.Artifact('eval-report', type='evaluation')
    artifact.add_file(os.path.join(EVAL_OUTPUT_DIR, 'classification_report.json'))
    artifact.add_file(os.path.join(EVAL_OUTPUT_DIR, 'classification_report.txt'))
    artifact.add_file(os.path.join(EVAL_OUTPUT_DIR, 'confusion_heatmap.png'))
    artifact.add_file(os.path.join(EVAL_OUTPUT_DIR, 'misclassification_heatmap.png'))
    wandb.log_artifact(artifact)

    wandb.finish()
    log(f"All evaluation artifacts saved to {EVAL_OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
