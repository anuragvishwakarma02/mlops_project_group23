"""Central configuration for the NER pipeline."""

import os
from datetime import datetime

import dotenv
from utils import resolve_device

# Load environment variables from .env file
dotenv.load_dotenv()


def _as_bool(value: str, default: bool) -> bool:
    # Accepts '1', 'true', 'yes', 'y', 'on' as True (case-insensitive)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}



SEED = int(os.getenv("SEED", "42"))

# Available pretrained NER models — change '0' to '1' or '2' to switch models
models = {
    '0': 'dslim/bert-base-NER',
    '1': 'elastic/distilbert-base-uncased-finetuned-conll03-english',
    '2': 'microsoft/deberta-v3-base'
}
MODEL_NAME = os.getenv("MODEL_NAME", models.get('0'))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

DATASET_NAME = os.getenv("DATASET_NAME", "yongsun-yoon/open-ner-english")

DEVICE_NAME = {
    "mps": "mps",   # Apple Silicon
    "cuda": "cuda", # NVIDIA GPU
    "cpu": "cpu",
}
DEVICE = resolve_device(DEVICE_NAME)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./ner_results")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", f"./models/{MODEL_NAME.replace('/', '-')}")
EVAL_OUTPUT_DIR = os.path.join(ARTIFACTS_DIR, "eval_results")

DATASET_PICKLE_PATH = os.getenv("DATASET_PICKLE_PATH", os.path.join(ARTIFACTS_DIR, "open_ner_english_dataset.pkl"))
CLEANED_DATASET_PICKLE_PATH = os.getenv("CLEANED_DATASET_PICKLE_PATH", os.path.join(ARTIFACTS_DIR, "open_ner_english_cleaned.pkl"))
PROCESSED_DATA_FILE = os.getenv("PROCESSED_DATA_FILE", os.path.join(ARTIFACTS_DIR, "processed_ner_data.pkl"))

# Maps label IDs to label names (e.g., 0 -> "O", 1 -> "B-PER")
ID2LABEL_FILE = os.getenv("ID2LABEL_FILE", os.path.join(ARTIFACTS_DIR, "id2label.json"))

USE_SUBSET = _as_bool(os.getenv("USE_SUBSET", "true"), True)

# Smaller sample counts on CPU to keep training feasible
_default_train_samples = "2000" if DEVICE == "cpu" else "5000"
_default_validation_samples = "400" if DEVICE == "cpu" else "1000"
TRAIN_MAX_SAMPLES = int(os.getenv("TRAIN_MAX_SAMPLES", _default_train_samples))
VALIDATION_MAX_SAMPLES = int(os.getenv("VALIDATION_MAX_SAMPLES", _default_validation_samples))

MAX_ENTITY_TYPES = int(os.getenv("MAX_ENTITY_TYPES", "500"))
RARE_ENTITY_POLICY = os.getenv("RARE_ENTITY_POLICY", "O").upper()
TOP_K_CONFUSION = int(os.getenv("TOP_K_CONFUSION", "15"))

ENABLE_WANDB = _as_bool(os.getenv("ENABLE_WANDB", "true"), True)
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mlops-group-23-project")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", f"ner-fine-tune-run-{MODEL_NAME.replace('/', '-')}")
WANDB_RUN_NAME = f"{WANDB_RUN_NAME}-{MODEL_NAME.replace('/', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# HF_TOKEN should be set in .env — never hardcode it
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO", "anuragvishwakarma02/mlops-group23-ner")


_default_learning_rate = "5e-5" if DEVICE == "cpu" else "3e-5"
_default_num_train_epochs = "5" if DEVICE == "cpu" else "3"

TRAINING_ARGS = dict(
    output_dir=OUTPUT_DIR,
    learning_rate=float(os.getenv("LEARNING_RATE", _default_learning_rate)),
    per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "16")),
    per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "16")),
    num_train_epochs=float(os.getenv("NUM_TRAIN_EPOCHS", _default_num_train_epochs)),
    weight_decay=float(os.getenv("WEIGHT_DECAY", "0.01")),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=int(os.getenv("LOGGING_STEPS", "50")),
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Best checkpoint selected based on F1 score
    push_to_hub=False,
    seed=SEED,
)