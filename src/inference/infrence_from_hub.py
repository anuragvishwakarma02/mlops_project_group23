"""Run NER inference from a Hugging Face Hub model and print token-level tags as JSON."""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE_NAME, HF_REPO, HF_TOKEN, MAX_LENGTH
from src.utils import resolve_device


def log(message):
    print(f"[inference-hub] {message}")


def load_model_and_tokenizer():
    model_id = os.getenv("HF_MODEL_ID") or os.getenv("HF_MODEL_NAME") or HF_REPO
    token = HF_TOKEN or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForTokenClassification.from_pretrained(model_id, token=token)
    return model_id, tokenizer, model


def predict_tokens(text, tokenizer, model, device):
    words = text.split()
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        pred_ids = torch.argmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()

    word_ids = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).word_ids()

    id2label = model.config.id2label or {}

    token_labels = []
    prev_word = None
    for token_pos, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word:
            prev_word = word_id
            continue

        label_id = pred_ids[token_pos]
        label = id2label.get(label_id, str(label_id))
        token_labels.append({"token": words[word_id], "label": label})
        prev_word = word_id

    entities = [row for row in token_labels if row["label"] != "O"]
    return token_labels, entities


def main():
    env_text = os.getenv("INPUT_TEXT", "").strip()
    if env_text:
        textset = [env_text]
    else:
        textset = [
            "Google launched a new model with 12B parameters in California",
            "I am from Pune",
            "India is a country in South Asia",
        ]

    device = resolve_device(DEVICE_NAME)
    model_id, tokenizer, model = load_model_and_tokenizer()
    model = model.to(device)
    model.eval()

    log(f"Loaded model from '{model_id}'. Running prediction...")
    for text in textset:
        token_labels, entities = predict_tokens(text, tokenizer, model, device)
        result = {
            "model_name": model.config._name_or_path,
            "input_text": text,
            "token_labels": token_labels,
            "entities": entities,
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
