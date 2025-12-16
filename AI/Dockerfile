FROM python:3.13-slim

# системные зависимости (если потребуется для sklearn/tensorflow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создадим папки под модели и данные
RUN mkdir -p /app/models /app/data
COPY service.py .

EXPOSE 7000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "1"]
