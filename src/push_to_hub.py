
"""
push_to_hub.py — Push the fine-tuned model and tokenizer to Hugging Face Hub.

Run standalone:
    python push_to_hub.py --repo <your-hf-username/repo-name> [--private]

The HF token must be supplied via the HF_TOKEN environment variable.
"""

"""Push fine-tuned NER model and tokenizer to Hugging Face Hub."""

import argparse
import os

import wandb
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import HF_TOKEN, LOCAL_MODEL_DIR, WANDB_PROJECT, WANDB_RUN_NAME
from wandb_utils import init_wandb_run


def log(message):
    print(f"[hub] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description="Push fine-tuned NER model to Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="Hub repo id, e.g. your-username/mlops-group23-ner")
    parser.add_argument("--private", action="store_true", help="Create repository as private")
    return parser.parse_args()


def push(repo: str, private: bool = False):
    token = HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required for hub push.")

    if not os.path.exists(LOCAL_MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {LOCAL_MODEL_DIR}. Run train.py first.")

    log(f"Loading model and tokenizer from {LOCAL_MODEL_DIR}")
    model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

    log(f"Pushing model to Hub repo: {repo} (private={private})")
    model.push_to_hub(repo, token=token, private=private)
    tokenizer.push_to_hub(repo, token=token, private=private)

    hf_url = f"https://huggingface.co/{repo}"
    log(f"Publish complete: {hf_url}")

    try:
        wandb_enabled = init_wandb_run(
            project=WANDB_PROJECT,
            name=f"{WANDB_RUN_NAME}-hub",
            resume="allow",
        )
        if wandb_enabled and wandb.run is not None:
            wandb.run.summary["huggingface_model"] = hf_url
            wandb.finish()
            log("Logged Hugging Face model URL to W&B run summary")
    except Exception as exc:
        log(f"W&B summary update skipped: {exc}")


def main():
    args = parse_args()
    push(args.repo, args.private)


if __name__ == "__main__":
    main()
