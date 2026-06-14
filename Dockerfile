FROM python:3.11-slim

ARG HF_MODEL_NAME=anuragvishwakarma02/mlops-group23-ner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_MODEL_NAME=${HF_MODEL_NAME}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENTRYPOINT ["python", "-m"]
CMD ["src.inference.infrence_from_hub"]
