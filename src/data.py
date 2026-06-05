
"""
Prepare Goodreads review data for training and evaluation.

This script:
1) downloads and caches raw genre reviews
2) builds train/test splits
3) tokenizes text and encodes labels
4) writes reusable artifacts to disk
"""

import os
import certifi
import gzip
import json
import pickle
import random

import requests
from transformers import DistilBertTokenizerFast

from config import (
    MODEL_NAME, MAX_LENGTH, RAW_DATA_FILE, PROCESSED_DATA_FILE,
    HEAD, SAMPLE_SIZE, REVIEWS_PER_GENRE, TRAIN_RATIO, GENRE_URL_DICT, ID2LABEL_FILE,
)
from utils import build_label_maps

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()


def log(message):
    """Print a friendly status message for the data stage."""
    print(f"[data] {message}")

def load_reviews(url, head=HEAD, sample_size=SAMPLE_SIZE):
    """Stream reviews from a gzip JSONL source and return a random sample."""
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    log(f"Source responded with HTTP {response.status_code}")

    with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
        for line in f:
            reviews.append(json.loads(line)['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def download_and_sample(raw_data_file=RAW_DATA_FILE):
    """Download Goodreads genre reviews; return from disk cache if available."""

    if os.path.exists(raw_data_file):
        log(f"Using cached raw data from {raw_data_file}")
        with open(raw_data_file, 'rb') as f:
            return pickle.load(f)

    os.makedirs(os.path.dirname(raw_data_file), exist_ok=True)

    genre_reviews_dict = {}
    for genre, url in GENRE_URL_DICT.items():
        log(f"Downloading and sampling reviews for '{genre}'")
        genre_reviews_dict[genre] = load_reviews(url)

    with open(raw_data_file, 'wb') as f:
        pickle.dump(genre_reviews_dict, f)
    log(f"Saved raw cache to {raw_data_file}")
    
    return genre_reviews_dict


def split_data(genre_reviews_dict, reviews_per_genre=REVIEWS_PER_GENRE, train_ratio=TRAIN_RATIO):
    """Split per-genre reviews into train/test lists of texts and labels."""

    train_texts, train_labels = [], []
    test_texts, test_labels   = [], []

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        split_idx = int(len(sampled) * train_ratio)
        for r in sampled[:split_idx]:
            train_texts.append(r)
            train_labels.append(genre)
        for r in sampled[split_idx:]:
            test_texts.append(r)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


def encode_data(train_texts, train_labels, test_texts, test_labels,
                model_name=MODEL_NAME, max_length=MAX_LENGTH):
    """
    Tokenise texts with DistilBERT and integer-encode string labels.

    Returns tokeniser encodings as plain dicts (pickle-safe) plus label maps.
    """

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    label2id, id2label = build_label_maps(train_labels)

    log("Tokenizing training texts...")
    train_encodings = dict(
        tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    )
    log("Tokenizing test texts...")
    test_encodings = dict(
        tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)
    )

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded  = [label2id[y] for y in test_labels]

    return (train_encodings, test_encodings, train_labels_encoded, test_labels_encoded, label2id, id2label)


def main():
    genre_reviews_dict = download_and_sample()

    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)
    log(f"Prepared split sizes -> train: {len(train_texts)} | test: {len(test_texts)}")

    (train_encodings, test_encodings,
     train_labels_encoded, test_labels_encoded,
     label2id, id2label) = encode_data(train_texts, train_labels, test_texts, test_labels)

    # Keep artifacts together so train/eval scripts can consume one source of truth.
    processed = {
        'train_encodings':      train_encodings,
        'test_encodings':       test_encodings,
        'train_labels_encoded': train_labels_encoded,
        'test_labels_encoded':  test_labels_encoded,
        'train_texts':          train_texts,
        'train_labels':         train_labels,
        'test_texts':           test_texts,
        'test_labels':          test_labels,
        'label2id':             label2id,
        'id2label':             id2label,
    }

    with open(PROCESSED_DATA_FILE, 'wb') as f:
        pickle.dump(processed, f)
    log(f"Saved processed dataset to {PROCESSED_DATA_FILE}")

    with open(ID2LABEL_FILE, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2, ensure_ascii=False)
    log(f"Saved label mapping to {ID2LABEL_FILE}")


if __name__ == '__main__':
    main()
