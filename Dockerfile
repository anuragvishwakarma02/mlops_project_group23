FROM python:3.11-slim

ARG HF_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_MODEL_NAME=${HF_MODEL_NAME}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
# COPY id2label.json ./id2label.json

ENTRYPOINT ["python", "src/main.py"]

# ── How to run locally ──────────────────────────────────────────────────────

# 1. Build:
#    docker build -t mlops .
#    docker build --build-arg HF_MODEL_NAME=your-username/your-model -t mlops .

# 2. Run pipeline (train + eval):
#    docker run --rm \
#      -e WANDB_API_KEY="<your-key>" \
#      -e HF_TOKEN="<your-token>" \
#      -v "$(pwd)/eval_results:/app/eval_results" \
#      -v "$(pwd)/results:/app/results" \
#      mlops

# 3. Run pipeline + push to Hugging Face Hub:
#    docker run --rm \
#      -e WANDB_API_KEY="<your-key>" \
#      -e HF_TOKEN="<your-token>" \
#      -v "$(pwd)/eval_results:/app/eval_results" \
#      -v "$(pwd)/results:/app/results" \
#      mlops --push-to-hub --repo your-username/your-model
# ────────────────────────────────────────────────────────────────────────────
