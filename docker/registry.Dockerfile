FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -e . numpy

EXPOSE 8000

CMD ["uvicorn", "secure_aggregation.storage.registry_service:app", "--host", "0.0.0.0", "--port", "8000"]
