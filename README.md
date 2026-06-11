# NER Fine-Tuning Pipeline — MLOps Assignment 3

MLOps Assignment 3 — IIT Jodhpur PGD AI Program

## Authors

| Roll Number | Name
|---|---|
| G25AIT2017 | Anurag Vishwakarma
| G25AIT2125 | Vinod Krishnan Panicke
| G25AIT2008 | Ananya Chaudhary

This project fine-tunes transformer-based NER models on `yongsun-yoon/open-ner-english` for token-level named entity recognition.

## Notebook

Primary notebook:

- `mlops-group-project.ipynb`

## Python Program (Script Pipeline)

This repository includes a Python script-based pipeline for non-notebook runs:

- `main.py`: orchestrates pipeline stages (`data -> train -> eval -> inference -> hub push`)
- `data.py`: loads `yongsun-yoon/open-ner-english`, cleans data, encodes labels, tokenizes, saves processed dataset
- `train.py`: fine-tunes a token classification model via Hugging Face `Trainer`, saves model/tokenizer/metrics
- `eval.py`: evaluates saved model and writes classification report + confusion matrix artifacts
- `inference.py`: runs sample sentence inference against the saved model
- `push_to_hub.py`: pushes tokenizer and model to Hugging Face Hub
- `config.py`: central configuration loaded from environment variables / `.env`
- `utils.py`: shared dataset utilities, cleaning helpers, metrics, and seed utilities
- `wandb_utils.py`: W&B run initialization helpers

### Run the Python Pipeline

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the repo root for secrets:

```env
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=hf_your_token_here

# Optional overrides
MODEL_NAME=dslim/bert-base-NER   [(`elastic/distilbert-base-uncased-finetuned-conll03-english`(quick run on gh), `microsoft/deberta-v3-base`]
DATASET_NAME=yongsun-yoon/open-ner-english
USE_SUBSET=true
TRAIN_MAX_SAMPLES=5000
VALIDATION_MAX_SAMPLES=1000
HF_REPO=anuragvishwakarma02/mlops-group23-ner
```

Notes:
- If running on CPU, `src/config.py` automatically reduces sample sizes to `2000 / 400`. `Added to hande the github pipeline run`
- If `WANDB_API_KEY` is missing, W&B logging is disabled automatically.

3. Run the full pipeline:

```bash
python src/main.py
```

### Useful Flags

```bash
# Skip data preparation (uses existing pickle cache)
python src/main.py --skip-data

# Skip training and evaluation, run inference only
python src/main.py --skip-train --skip-eval --run-inference

# Run inference after evaluation
python src/main.py --run-inference

# Push fine-tuned model to Hugging Face Hub
python src/main.py --push-to-hub --repo your-username/your-model

# Push to a private Hub repository
python src/main.py --push-to-hub --repo your-username/your-model --private
```

All supported flags:

| Flag | Description |
|---|---|
| `--skip-data` | Skip data download/processing |
| `--skip-train` | Skip model fine-tuning |
| `--skip-eval` | Skip evaluation artifact generation |
| `--run-inference` | Run sample inference after evaluation |
| `--push-to-hub` | Push model + tokenizer to Hugging Face Hub |
| `--repo` | Hub repo id, e.g. `your-username/mlops-group23-ner` |
| `--private` | Create the Hub repository as private |

## Run with Docker

Build image:

```bash
docker build -t mlops-group23-ner .
```

Run full pipeline:

```bash
docker run --rm \
  -e WANDB_API_KEY="<your-key>" \
  -e HF_TOKEN="<your-token>" \
  mlops-group23-ner
```

Run with inference:

```bash
docker run --rm \
  -e WANDB_API_KEY="<your-key>" \
  -e HF_TOKEN="<your-token>" \
  mlops-group23-ner --run-inference
```

Run and persist outputs locally:

```bash
mkdir -p artifacts models
docker run --rm \
  -e WANDB_API_KEY="<your-key>" \
  -e HF_TOKEN="<your-token>" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/models:/app/models" \
  mlops-group23-ner
```

Run and push model to Hugging Face Hub:

```bash
docker run --rm \
  -e WANDB_API_KEY="<your-key>" \
  -e HF_TOKEN="<your-token>" \
  mlops-group23-ner --push-to-hub --repo your-username/your-model
```

## Current Pipeline (As Implemented)

1. Load `yongsun-yoon/open-ner-english` from Hugging Face Datasets.
2. Clean and validate token–label alignment; apply rare-entity policy.
3. Encode entity labels into integer ids; persist `id2label.json`.
4. Optionally subset data based on `TRAIN_MAX_SAMPLES` / `VALIDATION_MAX_SAMPLES`.
5. Tokenize with the model's `AutoTokenizer`; align labels to subword tokens.
6. Fine-tune a token classification model via Hugging Face `Trainer`.
7. Evaluate with seqeval precision, recall, F1, accuracy; save classification report and confusion matrix.
8. Optionally log metrics and artifacts to W&B.
9. Optionally push tokenizer and model to Hugging Face Hub.

## Key Configuration

Model options (set via `MODEL_NAME` env var):

| Key | Model |
|---|---|
| `0` (default) | `dslim/bert-base-NER` |
| `1` | `elastic/distilbert-base-uncased-finetuned-conll03-english` |
| `2` | `microsoft/deberta-v3-base` |

Training arguments are device-dependent:

| Parameter | CPU default | GPU default |
|---|---|---|
| `TRAIN_MAX_SAMPLES` | 2000 | 5000 |
| `VALIDATION_MAX_SAMPLES` | 400 | 1000 |
| `MAX_LENGTH` | 256 | 256 |
| `USE_SUBSET` | true | true |

### Training Hyperparameter Configurations

Two hyperparameter profiles are available in the notebook configuration cell. Switch between them by commenting/uncommenting the relevant block.

**V1 — Default (faster convergence)**

| Parameter | Value |
|---|---|
| `LEARNING_RATE` | `3e-5` |
| `TRAIN_BATCH_SIZE` | `16` (GPU) / `8` (CPU) |
| `EVAL_BATCH_SIZE` | `16` (GPU) / `8` (CPU) |
| `NUM_TRAIN_EPOCHS` | `3` |
| `WEIGHT_DECAY` | `0.01` |
| `WARMUP_RATIO` | `0.1` |
| `LOGGING_STEPS` | `50` |

**V2 — Slower, more stable convergence**

| Parameter | Value |
|---|---|
| `LEARNING_RATE` | `5e-5` |
| `TRAIN_BATCH_SIZE` | `16` (GPU) / `8` (CPU) |
| `EVAL_BATCH_SIZE` | `16` (GPU) / `8` (CPU) |
| `NUM_TRAIN_EPOCHS` | `5` |
| `WEIGHT_DECAY` | `0.01` |
| `WARMUP_RATIO` | `0.1` |
| `LOGGING_STEPS` | `50` |

> V2 uses a higher learning rate over more epochs for slower, more stable convergence — recommended for full-dataset runs.

## Outputs

Generated by notebook/script execution:

- `artifacts/processed_ner_data.pkl`
- `artifacts/id2label.json`
- `artifacts/eval_results/classification_report.txt`
- `artifacts/eval_results/classification_report.json`
- `artifacts/eval_results/confusion_matrices.png`
- `models/<model-name>/` — saved model and tokenizer

## Weights & Biases

When `ENABLE_WANDB=true` and `WANDB_API_KEY` is set, training metrics are streamed to W&B project `mlops-group-23-project`.

Evaluation uploads as a W&B artifact:
- `classification_report.json`
- `classification_report.txt`
- confusion matrix image

## Links

- **GitHub:** [GitHub Repository](https://github.com/anuragvishwakarma02/mlops_project_group23)
- **Kaggle Notebook:** [Kaggle](https://www.kaggle.com/code/anuragg25ait2017/mlops-group-project/edit/run/326129505)
- **Hugging Face Model:** [anuragvishwakarma02/mlops-group23-ner](https://huggingface.co/anuragvishwakarma02/mlops-group23-ner)
- **Docker Image (Public):** [Docker Hub](https://hub.docker.com/r/g25ait2017/mlops/tags)
- **W&B Project Dashboard:** [W&B](https://api.wandb.ai/links/g25ait2017-prom-iit-rajasthan/mek3r89q)
- **Dataset:** [yongsun-yoon/open-ner-english](https://huggingface.co/datasets/yongsun-yoon/open-ner-english)

## Troubleshooting

- **Missing file errors:** Run the full pipeline at least once to generate cached artifacts and model files:

```bash
python src/main.py
```

- **Hub push fails:** Verify `HF_TOKEN` is set, `--repo` is provided, and the token has write access to the target repo.
- **W&B disabled unexpectedly:** Check that `WANDB_API_KEY` is exported in your environment or present in `.env`.
