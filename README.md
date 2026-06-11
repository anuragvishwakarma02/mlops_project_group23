# MLOps Group 23 - NER Pipeline

End-to-end Named Entity Recognition pipeline using Hugging Face Transformers.

The project runs:
- data preparation
- model fine-tuning
- evaluation (reports + confusion matrix)
- optional inference
- optional push to Hugging Face Hub

## Project Structure

- `src/data.py` - dataset loading, cleaning, label preparation, tokenization cache
- `src/train.py` - model training with `Trainer`
- `src/eval.py` - evaluation metrics, reports, confusion matrix, W&B logging
- `src/inference.py` - sample sentence inference
- `src/main.py` - pipeline orchestrator
- `src/config.py` - central configuration from environment variables
- `artifacts/` - processed data and evaluation outputs
- `models/` - local trained model directories

## Requirements

- Python 3.10+
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in repo root:

```env
# Optional but recommended
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=hf_your_token_here

# Optional overrides in environment
MODEL_NAME=dslim/bert-base-NER
DATASET_NAME=yongsun-yoon/open-ner-english
USE_SUBSET=true
TRAIN_MAX_SAMPLES=5000
VALIDATION_MAX_SAMPLES=1000
HF_REPO=anuragvishwakarma02/mlops-group23-ner
```

Notes:
- If running on CPU, defaults in `src/config.py` automatically reduce sample sizes to `2000/400`.
- If `WANDB_API_KEY` is missing, W&B logging is disabled automatically.

## Run the Pipeline

From repo root:

```bash
python src/main.py
```

### Useful Flags

```bash
python src/main.py --skip-data
python src/main.py --skip-train --skip-eval --run-inference
python src/main.py --run-inference
python src/main.py --push-to-hub --repo your-username/your-model
python src/main.py --push-to-hub --repo your-username/your-model --private
```

Supported flags in `src/main.py`:
- `--skip-data`
- `--skip-train`
- `--skip-eval`
- `--run-inference`
- `--push-to-hub`
- `--repo`
- `--private`

## Docker

Build image:

```bash
docker build -t mlops .
```

Run full pipeline:

```bash
docker run --rm \
	-e WANDB_API_KEY="<your-key>" \
	-e HF_TOKEN="<your-token>" \
	mlops
```

Run with inference:

```bash
docker run --rm \
	-e WANDB_API_KEY="<your-key>" \
	-e HF_TOKEN="<your-token>" \
	mlops --run-inference
```

Run and push to Hub:

```bash
docker run --rm \
	-e WANDB_API_KEY="<your-key>" \
	-e HF_TOKEN="<your-token>" \
	mlops --push-to-hub --repo your-username/your-model
```

## Outputs

Main outputs include:
- `artifacts/processed_ner_data.pkl`
- `artifacts/id2label.json`
- `artifacts/eval_results/classification_report.txt`
- `artifacts/eval_results/classification_report.json`
- `artifacts/eval_results/confusion_matrices.png`
- `models/<model-name>/` saved model and tokenizer

## Weights & Biases

When enabled, training and evaluation metrics are logged to W&B.

Evaluation uploads:
- `classification_report.json`
- `classification_report.txt`
- confusion matrix image

as a W&B evaluation artifact.

## Troubleshooting

-  IMP :If you get missing file errors, run full pipeline once it means the artifcats and models are missing in you local :

```bash
python src/main.py
```

- If push to Hub fails, verify:
	- `HF_TOKEN` is set
	- `--repo` is provided
	- token has write access to the target repo
