"""Prepare NER data artifacts for training and evaluation."""

import json
import os
import pickle
import re
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

from config import (
    MODEL_NAME,
    CLEANED_DATASET_PICKLE_PATH,
    DATASET_NAME,
    DATASET_PICKLE_PATH,
    ID2LABEL_FILE,
    MAX_ENTITY_TYPES,
    MAX_LENGTH,
    PROCESSED_DATA_FILE,
    RARE_ENTITY_POLICY,
    SEED,
    TRAIN_MAX_SAMPLES,
    USE_SUBSET,
    VALIDATION_MAX_SAMPLES,
)


FRONT_MATTER_RE = re.compile(r"^---[\s\S]*?---\s*")
CITATION_RE = re.compile(r"\[@[^\]]+\]")
SECTION_TAG_RE = re.compile(r"\{#section[^}]*\}")
LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")
NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9\s]+")
MULTISPACE_RE = re.compile(r"\s+")


def log(message):
    print(f"[data] {message}")


def load_dataset_with_cache():
    os.makedirs(os.path.dirname(DATASET_PICKLE_PATH) or ".", exist_ok=True)
    try:
        if os.path.exists(DATASET_PICKLE_PATH):
            with open(DATASET_PICKLE_PATH, "rb") as f:
                dataset = pickle.load(f)
            log(f"Loaded dataset from pickle: {DATASET_PICKLE_PATH}")
        else:
            dataset = load_dataset(DATASET_NAME)
            with open(DATASET_PICKLE_PATH, "wb") as f:
                pickle.dump(dataset, f)
            log(f"Downloaded dataset from Hugging Face Hub: {DATASET_NAME}")
    except Exception:
        dataset = load_dataset(DATASET_NAME)
        with open(DATASET_PICKLE_PATH, "wb") as f:
            pickle.dump(dataset, f)
        log("Dataset cache load failed; downloaded a fresh copy.")
    return dataset


def apply_subset(dataset):
    if not USE_SUBSET:
        return dataset

    split_limits = {
        "train": TRAIN_MAX_SAMPLES,
        "validation": VALIDATION_MAX_SAMPLES,
    }

    for split_name, max_samples in split_limits.items():
        if split_name in dataset and max_samples > 0:
            n = min(max_samples, len(dataset[split_name]))
            dataset[split_name] = dataset[split_name].shuffle(seed=SEED).select(range(n))
            log(f"Subset applied -> {split_name}: {n} rows")
    return dataset


def strip_special_chars(text):
    if text is None:
        return ""

    text = str(text).replace("\\r\\n", " ").replace("\\n", " ").replace("\\r", " ").replace("\\xa0", " ")
    text = FRONT_MATTER_RE.sub(" ", text)
    text = CITATION_RE.sub(" ", text)
    text = SECTION_TAG_RE.sub(" ", text)
    text = LATEX_CMD_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text.lower()


def clean_entities_for_ner(entities):
    cleaned_entities = []
    if not isinstance(entities, list):
        return cleaned_entities

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        entity_type = entity.get("entity_type", entity.get("type", entity.get("label")))
        mentions = entity.get("entity_mentions", [])

        if isinstance(mentions, list):
            cleaned_mentions = []
            for mention in mentions:
                cleaned = strip_special_chars(mention)
                if cleaned:
                    cleaned_mentions.append(cleaned)
            if cleaned_mentions and entity_type is not None:
                cleaned_entities.append(
                    {
                        "entity_mentions": cleaned_mentions,
                        "entity_type": entity_type,
                    }
                )
        else:
            cleaned_entities.append(entity)

    return cleaned_entities


def clean_example(example):
    original = example.get("text", "")
    example["raw_text"] = original
    example["text"] = strip_special_chars(original)
    example["entities"] = clean_entities_for_ner(example.get("entities", []))
    return example


def build_tokens_with_spans(text):
    tokens = []
    spans = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        spans.append((match.start(), match.end()))
    return tokens, spans


def find_all_occurrences(text, phrase):
    if not phrase:
        return []
    matches = []
    for match in re.finditer(re.escape(phrase), text):
        matches.append((match.start(), match.end()))
    return matches


def normalize_label(label):
    if label is None:
        return None
    normalized = str(label).strip().upper().replace(" ", "_")
    normalized = re.sub(r"[^A-Z0-9_\-]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or None


def split_bio_tag(tag):
    if not isinstance(tag, str) or tag == "O" or "-" not in tag:
        return None, None
    prefix, entity_type = tag.split("-", 1)
    if prefix not in {"B", "I"} or not entity_type:
        return None, None
    return prefix, entity_type


def get_entity_spans(entity, text):
    spans = []
    if not isinstance(entity, dict):
        return spans

    start = entity.get("start", entity.get("start_offset", entity.get("begin_offset")))
    end = entity.get("end", entity.get("end_offset", entity.get("stop_offset")))
    label = entity.get("label", entity.get("type", entity.get("entity", entity.get("tag"))))

    if start is not None and end is not None and label is not None:
        try:
            s = int(start)
            e = int(end)
            norm_label = normalize_label(label)
            if s < e and norm_label is not None:
                spans.append((s, e, norm_label))
        except Exception:
            pass

    mentions = entity.get("entity_mentions", [])
    entity_type = normalize_label(entity.get("entity_type", label))
    if isinstance(mentions, list) and entity_type is not None:
        for mention in mentions:
            if isinstance(mention, str):
                for s, e in find_all_occurrences(text, mention):
                    spans.append((s, e, entity_type))

    return spans


def convert_text_entities_to_tokens_tags(example):
    text = example.get("text", "")
    entities = example.get("entities", [])

    tokens, token_spans = build_tokens_with_spans(text)
    tags = ["O"] * len(tokens)

    all_spans = []
    for entity in entities:
        all_spans.extend(get_entity_spans(entity, text))

    for start, end, label in all_spans:
        matched_token_indices = []
        for idx, (token_start, token_end) in enumerate(token_spans):
            if token_end > start and token_start < end:
                matched_token_indices.append(idx)

        if not matched_token_indices:
            continue

        first = matched_token_indices[0]
        if tags[first] == "O":
            tags[first] = f"B-{label}"

        for idx in matched_token_indices[1:]:
            if tags[idx] == "O":
                tags[idx] = f"I-{label}"

    example["tokens"] = tokens
    example["ner_str_tags"] = tags
    return example


def collapse_rare_entity_types(example, keep_types, rare_entity_policy="O"):
    collapsed = []
    for tag in example["ner_str_tags"]:
        if tag == "O":
            collapsed.append("O")
            continue

        prefix, entity_type = split_bio_tag(tag)
        if prefix is None:
            collapsed.append("O")
            continue

        if entity_type in keep_types:
            collapsed.append(tag)
        elif rare_entity_policy == "MISC":
            collapsed.append(f"{prefix}-MISC")
        else:
            collapsed.append("O")

    example["ner_str_tags"] = collapsed
    return example


def detect_columns(ds_split):
    token_col = "tokens" if "tokens" in ds_split.column_names else "words"
    if "ner_tags" in ds_split.column_names:
        label_col = "ner_tags"
    elif "tags" in ds_split.column_names:
        label_col = "tags"
    else:
        raise ValueError(f"Could not find a token-label column in: {ds_split.column_names}")
    return token_col, label_col


def tokenize_and_align_labels(examples, tokenizer, token_col, label_col):
    tokenized = tokenizer(
        examples[token_col],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
    )

    aligned_labels = []
    for batch_idx in range(len(examples[token_col])):
        word_ids = tokenized.word_ids(batch_index=batch_idx)
        labels = examples[label_col][batch_idx]

        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def main():
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE) or ".", exist_ok=True)

    dataset = load_dataset_with_cache()
    dataset = apply_subset(dataset)
    dataset = dataset.map(clean_example, load_from_cache_file=False)

    # --- VALIDATION 1: post-cleaning ---
    log("\n" + "=" * 60)
    log("STEP 1.1 - Data preparation: validation checks")
    log("=" * 60)
    for split_name in dataset.keys():
        n = len(dataset[split_name])
        empty_text = sum(1 for r in dataset[split_name] if not r.get("text", "").strip())
        total_entities = sum(len(r.get("entities", [])) for r in dataset[split_name])
        log(
            f"[validate-clean] {split_name}: {n} rows | "
            f"{empty_text} empty texts | "
            f"{total_entities} entity annotations"
        )
    if empty_text == len(dataset["train"]):
        raise ValueError("All training texts are empty after cleaning — check regex patterns in strip_special_chars.")

    # --- CLEAN STEP 1: drop rows with empty text ---
    log("\n[clean-step-1] Dropping rows with empty text...")
    for split_name in dataset.keys():
        before = len(dataset[split_name])
        dataset[split_name] = dataset[split_name].filter(
            lambda r: bool(r.get("text", "").strip()),
            load_from_cache_file=False,
        )
        after = len(dataset[split_name])
        log(f"  {split_name}: {before} -> {after} rows (dropped {before - after} empty-text rows)")

    with open(CLEANED_DATASET_PICKLE_PATH, "wb") as f:
        pickle.dump(dataset, f)
    log(f"Saved cleaned dataset pickle: {CLEANED_DATASET_PICKLE_PATH}")

    if "tokens" not in dataset["train"].column_names or (
        "ner_tags" not in dataset["train"].column_names and "tags" not in dataset["train"].column_names
    ):
        dataset = dataset.map(convert_text_entities_to_tokens_tags, load_from_cache_file=False)

        type_counter = Counter()
        for split_name in dataset.keys():
            for row in dataset[split_name]["ner_str_tags"]:
                for tag in row:
                    _, entity_type = split_bio_tag(tag)
                    if entity_type is not None:
                        type_counter[entity_type] += 1

        if len(type_counter) > MAX_ENTITY_TYPES:
            keep_types = {entity_type for entity_type, _ in type_counter.most_common(MAX_ENTITY_TYPES)}
            dataset = dataset.map(
                lambda ex: collapse_rare_entity_types(ex, keep_types, RARE_ENTITY_POLICY),
                load_from_cache_file=False,
            )
            log(
                f"Collapsed rare entity types: kept top {MAX_ENTITY_TYPES} of {len(type_counter)} "
                f"(policy={RARE_ENTITY_POLICY})."
            )

        unique_tag_set = {"O"}
        for split_name in dataset.keys():
            for row in dataset[split_name]["ner_str_tags"]:
                unique_tag_set.update(row)

        ordered_tags = ["O"] + sorted([tag for tag in unique_tag_set if tag != "O"])
        str2id = {tag: idx for idx, tag in enumerate(ordered_tags)}

        def str_tags_to_ids(example):
            example["ner_tags"] = [str2id[tag] for tag in example["ner_str_tags"]]
            return example

        dataset = dataset.map(str_tags_to_ids, load_from_cache_file=False)
        log("Converted dataset from text/entities to tokens/ner_tags format.")

        # --- VALIDATION 2: post token/tag conversion ---
        log("\n" + "=" * 60)
        log("STEP 1.2 - Data preparation: post-conversion validation")
        log("=" * 60)
        for split_name in dataset.keys():
            rows = dataset[split_name]
            empty_tokens = sum(1 for r in rows if len(r.get("tokens", [])) == 0)
            all_o_tags = sum(
                1 for r in rows if r.get("ner_str_tags") and all(t == "O" for t in r["ner_str_tags"])
            )
            has_entities = sum(
                1 for r in rows if any(t != "O" for t in r.get("ner_str_tags", []))
            )
            log(
                f"[validate-convert] {split_name}: "
                f"{empty_tokens} rows with empty tokens | "
                f"{all_o_tags} rows all-O tags | "
                f"{has_entities} rows with entity tags"
            )
        if empty_tokens == len(dataset["train"]):
            raise ValueError("All training rows have empty token lists after conversion — tokenizer input is broken.")

        # --- CLEAN STEP 2: drop rows with empty token lists ---
        log("\n[clean-step-2] Dropping rows with empty token lists...")
        for split_name in dataset.keys():
            before = len(dataset[split_name])
            dataset[split_name] = dataset[split_name].filter(
                lambda r: len(r.get("tokens", [])) > 0,
                load_from_cache_file=False,
            )
            after = len(dataset[split_name])
            log(f"  {split_name}: {before} -> {after} rows (dropped {before - after} empty-token rows)")

        # --- CLEAN STEP 3: drop rows where ALL tags are O (no entity signal) in train only ---
        log("\n[clean-step-3] Dropping training rows with zero entity tags (all-O)...")
        before = len(dataset["train"])
        dataset["train"] = dataset["train"].filter(
            lambda r: any(t != "O" for t in r.get("ner_str_tags", [])),
            load_from_cache_file=False,
        )
        after = len(dataset["train"])
        log(f"  train: {before} -> {after} rows (dropped {before - after} all-O rows)")
    else:
        label_feature = (
            dataset["train"].features["ner_tags"].feature
            if "ner_tags" in dataset["train"].features
            else dataset["train"].features["tags"].feature
        )
        if hasattr(label_feature, "names") and label_feature.names is not None:
            ordered_tags = list(label_feature.names)
        else:
            unique_ids = sorted(
                {
                    label
                    for split_name in dataset.keys()
                    for row in dataset[split_name]["ner_tags"]
                    for label in row
                }
            )
            ordered_tags = [str(idx) for idx in unique_ids]

    token_col, label_col = detect_columns(dataset["train"])

    label_list = list(ordered_tags)
    id2label = {idx: label for idx, label in enumerate(label_list)}
    label2id = {label: idx for idx, label in id2label.items()}
    eval_split = "validation" if "validation" in dataset else "test"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, token_col, label_col),
        batched=True,
        load_from_cache_file=False,
    )

    expected_num_labels = len(label_list)
    for split_name in tokenized_dataset.keys():
        max_seen = -1
        total_valid = 0
        rows_with_no_labels = 0
        for row in tokenized_dataset[split_name]["labels"]:
            valid = [label for label in row if label != -100]
            total_valid += len(valid)
            if valid:
                max_seen = max(max_seen, max(valid))
            else:
                rows_with_no_labels += 1

        # --- VALIDATION 3: post tokenization ---
        log(
            f"[validate-tokenize] {split_name}: "
            f"{total_valid} valid label tokens | "
            f"{rows_with_no_labels} rows with zero valid labels | "
            f"max label id = {max_seen}, expected < {expected_num_labels}"
        )
        if total_valid == 0:
            raise ValueError(
                f"No valid label tokens in {split_name} after tokenization. "
                "All labels are -100 — check tokenize_and_align_labels."
            )
        if max_seen >= expected_num_labels:
            raise ValueError(
                f"Out-of-range label id found in {split_name}: {max_seen} >= {expected_num_labels}."
            )

    # --- CLEAN STEP 4: drop tokenized rows with zero valid labels ---
    log("\n[clean-step-4] Dropping tokenized rows with zero valid labels...")
    for split_name in tokenized_dataset.keys():
        before = len(tokenized_dataset[split_name])
        tokenized_dataset[split_name] = tokenized_dataset[split_name].filter(
            lambda r: any(label != -100 for label in r["labels"]),
            load_from_cache_file=False,
        )
        after = len(tokenized_dataset[split_name])
        log(f"  {split_name}: {before} -> {after} rows (dropped {before - after} zero-label rows)")

    processed = {
        "dataset": dataset,
        "tokenized_dataset": tokenized_dataset,
        "label_list": label_list,
        "id2label": id2label,
        "label2id": label2id,
        "token_col": token_col,
        "label_col": label_col,
        "eval_split": eval_split,
    }

    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(processed, f)
    log(f"Saved processed dataset to {PROCESSED_DATA_FILE}")

    with open(ID2LABEL_FILE, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
    log(f"Saved label mapping to {ID2LABEL_FILE}")

    # --- VALIDATION 4: final summary ---
    log(f"[validate-final] label_list size: {len(label_list)}")
    log(f"[validate-final] label sample (first 10): {label_list[:10]}")
    for split_name in tokenized_dataset.keys():
        n = len(tokenized_dataset[split_name])
        log(f"[validate-final] {split_name}: {n} tokenized rows ready for training")
    log("[validate-final] Data pipeline complete — all checks passed.")


if __name__ == "__main__":
    main()
