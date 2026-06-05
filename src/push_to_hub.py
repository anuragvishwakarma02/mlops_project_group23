
"""
push_to_hub.py — Push the fine-tuned model and tokenizer to Hugging Face Hub.

Run standalone:
    python push_to_hub.py --repo <your-hf-username/repo-name> [--private]

The HF token must be supplied via the HF_TOKEN environment variable.
"""

import argparse
import os

import wandb
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from config import CACHED_MODEL_DIR, MODEL_NAME, WANDB_PROJECT, WANDB_RUN_NAME
from wandb_utils import init_wandb_run


def log(message):
    """Print a friendly status message for the hub publishing stage."""
    print(f"[hub] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description='Push fine-tuned model to Hugging Face Hub')
    parser.add_argument('--repo', required=True,
                        help='Hub repo id, e.g. your-username/distilbert-reviews-genres')
    parser.add_argument('--private', action='store_true',
                        help='Create the repository as private')
    return parser.parse_args()


def push(repo: str, private: bool = False):
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError(
            'HF_TOKEN environment variable is not set. '
            'Pass it with -e HF_TOKEN=hf_xxx when running Docker.'
        )

    if not os.path.exists(CACHED_MODEL_DIR):
        raise FileNotFoundError(
            f'Model directory not found: {CACHED_MODEL_DIR}. '
            'Run training first.'
        )

    log(f"Loading fine-tuned model from {CACHED_MODEL_DIR}")
    model = DistilBertForSequenceClassification.from_pretrained(CACHED_MODEL_DIR)

    log(f"Loading tokenizer from base model {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    log(f"Pushing model to Hugging Face Hub: {repo} (private={private})")
    model.push_to_hub(repo, token=token, private=private)

    log(f"Pushing tokenizer to Hugging Face Hub: {repo}")
    tokenizer.push_to_hub(repo, token=token, private=private)

    hf_url = f'https://huggingface.co/{repo}'
    log(f"Publish complete: {hf_url}")

    # Log the HF model URL into the W&B run summary
    try:
        wandb_enabled = init_wandb_run(
            project=WANDB_PROJECT,
            name=f'{WANDB_RUN_NAME}-eval',
            resume='allow',
        )
        if not wandb_enabled:
            log('WANDB_API_KEY not set; skipping W&B summary update')
            return
        wandb.run.summary['huggingface_model'] = hf_url
        wandb.finish()
        log('Logged Hugging Face model URL to W&B run summary')
    except Exception as exc:
        log(f'W&B summary update skipped: {exc}')


def main():
    args = parse_args()
    push(args.repo, args.private)


if __name__ == '__main__':
    main()
