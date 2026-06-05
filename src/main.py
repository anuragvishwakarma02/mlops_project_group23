
"""
main.py — End-to-end pipeline: data preparation → training → evaluation → hub push.

Run:
    python main.py [--skip-data] [--skip-train] [--skip-eval]
                   [--push-to-hub --repo <user/repo> [--private]]

Flags:
    --skip-data    Skip data download/processing (requires existing pickle cache)
    --skip-train   Skip model fine-tuning (requires existing model directory)
    --skip-eval    Skip evaluation artefact generation
    --push-to-hub  Push fine-tuned model + tokenizer to Hugging Face Hub
    --repo         Hub repo id, e.g. your-username/distilbert-reviews-genres
    --private      Create the Hub repository as private

The HF_TOKEN environment variable must be set when using --push-to-hub.
"""

import argparse
import os

import data as data_module
import eval as eval_module
import push_to_hub as push_module
import train as train_module


def log(message):
    """Print a friendly status message for the pipeline orchestrator."""
    print(f"[pipeline] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description='Full MLOps pipeline')
    parser.add_argument('--skip-data',   action='store_true', help='Skip data preparation')
    parser.add_argument('--skip-train',  action='store_true', help='Skip model training')
    parser.add_argument('--skip-eval',   action='store_true', help='Skip evaluation')
    parser.add_argument('--push-to-hub', action='store_true', help='Push model to Hugging Face Hub')
    parser.add_argument('--repo',    default=None, help='Hub repo id, e.g. user/distilbert-reviews-genres')
    parser.add_argument('--private', action='store_true', help='Create Hub repo as private')
    return parser.parse_args()


def run_data():
    print('\n' + '=' * 60)
    print('STEP 1 — Data preparation')
    print('=' * 60)
    log("Preparing dataset artifacts")
    data_module.main()


def run_train():
    print('\n' + '=' * 60)
    print('STEP 2 — Model fine-tuning')
    print('=' * 60)
    log("Starting model fine-tuning")
    if not os.path.exists(train_module.PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f'Processed data file not found: {train_module.PROCESSED_DATA_FILE}. '
            'Run data preparation first (omit --skip-data).'
        )
    train_module.main()


def run_eval():
    print('\n' + '=' * 60)
    print('STEP 3 — Evaluation')
    print('=' * 60)
    log("Running evaluation and artifact generation")
    if not os.path.exists(eval_module.PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f'Processed data file not found: {eval_module.PROCESSED_DATA_FILE}. '
            'Run data preparation first (omit --skip-data).'
        )
    if not os.path.exists(eval_module.CACHED_MODEL_DIR):
        raise FileNotFoundError(
            f'Model directory not found: {eval_module.CACHED_MODEL_DIR}. '
            'Run training first (omit --skip-train).'
        )
    eval_module.main()


def run_push(args):
    print('\n' + '=' * 60)
    print('STEP 4 — Push to Hugging Face Hub')
    print('=' * 60)
    log("Publishing model and tokenizer to Hugging Face Hub")
    if not args.repo:
        raise ValueError(
            '--repo is required when using --push-to-hub, '
            'e.g. --repo your-username/distilbert-reviews-genres'
        )
    push_module.push(args.repo, args.private)


def main():
    args = parse_args()

    if not args.skip_data:
        run_data()
    else:
        log('Skipping data preparation.')

    if not args.skip_train:
        run_train()
    else:
        log('Skipping model training.')

    if not args.skip_eval:
        run_eval()
    else:
        log('Skipping evaluation.')

    if args.push_to_hub:
        run_push(args)
    else:
        log('Skipping Hugging Face Hub push (use --push-to-hub to enable).')

    print('\n' + '=' * 60)
    print('Pipeline complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
