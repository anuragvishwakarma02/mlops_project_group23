"""
main.py — End-to-end pipeline: data preparation → training → evaluation → hub push.

Run:
    python main.py [--skip-data] [--skip-train] [--skip-eval] [--run-inference]
                   [--push-to-hub --repo <user/repo> [--private]]

Flags:
    --skip-data    Skip data download/processing (requires existing pickle cache)
    --skip-train   Skip model fine-tuning (requires existing model directory)
    --skip-eval    Skip evaluation artefact generation
    --push-to-hub  Push fine-tuned model + tokenizer to Hugging Face Hub
    --repo         Hub repo id, e.g. your-username/distilbert-reviews-genres
    --private      Create the Hub repository as private
    --run-inference      Run inference after evaluation

The HF_TOKEN environment variable must be set when using --push-to-hub.
"""
"""Run the full NER pipeline: data -> train -> eval -> optional hub push."""

import argparse
import os

import data as data_module
import eval as eval_module
import inference as inference_module
import push_to_hub as push_module
import train as train_module
from config import LOCAL_MODEL_DIR, PROCESSED_DATA_FILE


def log(message):
    print(f"[pipeline] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description="NER MLOps pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip model evaluation")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model/tokenizer to Hugging Face Hub")
    parser.add_argument("--repo", default=None, help="Hub repo id, e.g. user/mlops-group23-ner")
    parser.add_argument("--private", action="store_true", help="Create Hub repository as private")
    parser.add_argument("--run-inference", action="store_true", help="Run inference after evaluation")
    return parser.parse_args()


def run_data():
    print("\n" + "=" * 60)
    print("STEP 1 - Data preparation")
    print("=" * 60)
    data_module.main()


def run_train():
    print("\n" + "=" * 60)
    print("STEP 2 - Model fine-tuning")
    print("=" * 60)
    # Ensure data step has been run before training
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f"Processed data file not found: {PROCESSED_DATA_FILE}. Run data step first."
        )
    train_module.main()


def run_eval():
    print("\n" + "=" * 60)
    print("STEP 3 - Evaluation")
    print("=" * 60)
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f"Processed data file not found: {PROCESSED_DATA_FILE}. Run data step first."
        )
    if not os.path.exists(LOCAL_MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: {LOCAL_MODEL_DIR}. Run training step first."
        )
    eval_module.main()


def run_inference():
    print("\n" + "=" * 60)
    print("STEP 4 - Inference")
    print("=" * 60)
    if not os.path.exists(LOCAL_MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: {LOCAL_MODEL_DIR}. Run training step first."
        )
    inference_module.main()


def run_push(args):
    print("\n" + "=" * 60)
    print("STEP 5 - Push to Hugging Face Hub")
    print("=" * 60)
    # --repo must be explicitly provided; no default to avoid accidental pushes
    if not args.repo:
        raise ValueError("--repo is required when --push-to-hub is used")
    push_module.push(args.repo, args.private)


def main():
    args = parse_args()

    if not args.skip_data:
        run_data()
    else:
        log("Skipping data step")

    if not args.skip_train:
        run_train()
    else:
        log("Skipping training step")

    if not args.skip_eval:
        run_eval()
    else:
        log("Skipping evaluation step")

    if args.run_inference:
        run_inference()
    else:
        log("Skipping inference step")

    if args.push_to_hub:
        run_push(args)
    else:
        log("Skipping hub push")

    print("\n" + "=" * 60)
    print("Pipeline complete")
    print("=" * 60)


if __name__ == "__main__":
    main()