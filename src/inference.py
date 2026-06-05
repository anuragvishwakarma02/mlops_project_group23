
"""Run single-text inference and print a friendly JSON response."""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_ID2LABEL_PATH = Path(__file__).resolve().parents[1] / "id2label.json"


def log(message):
    """Print a friendly status message for inference."""
    print(f"[inference] {message}")


def load_model():
    model_name = os.getenv("HF_MODEL_NAME", DEFAULT_MODEL)
    hf_token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
    return model_name, tokenizer, model


def predict(text, tokenizer, model):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

    id2label = model.config.id2label or {}
    if not id2label and DEFAULT_ID2LABEL_PATH.exists():
        with open(DEFAULT_ID2LABEL_PATH, "r", encoding="utf-8") as f:
            local_map = json.load(f)
        id2label = {int(k): v for k, v in local_map.items()}
    predicted_label = id2label.get(pred_idx, str(pred_idx))

    return {
        "predicted_label": predicted_label,
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx].item()),
        "model_name": model.config._name_or_path,
    }


def main():
    input_text = os.getenv("INPUT_TEXT")
    if not input_text:
        raise ValueError("INPUT_TEXT environment variable is required for inference.")

    model_name, tokenizer, model = load_model()
    log(f"Loaded model '{model_name}'. Running prediction...")
    result = predict(input_text, tokenizer, model)
    log("Prediction complete. Returning JSON output.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
