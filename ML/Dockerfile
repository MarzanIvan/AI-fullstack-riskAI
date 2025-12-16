FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Увеличиваем timeout и количество попыток
ENV PIP_DEFAULT_TIMEOUT=200 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=off

# Устанавливаем зависимости
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


FROM python:3.11-slim

# Устанавливаем только необходимые системные пакеты для работы приложения
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем все собранные зависимости из builder
COPY --from=builder /install /usr/local

# Создаём папки под модели и датасеты
RUN mkdir -p /app/models /app/data

# Копируем исходники приложения
COPY service.py .
COPY s3_client.py .

EXPOSE 9000

# Запуск uvicorn
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
