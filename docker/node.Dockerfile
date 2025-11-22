FROM python:3.12-slim
WORKDIR /app

# Install system dependencies for SSL
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies including mnist and grpc
RUN pip install --no-cache-dir -e .[mnist] && \
    pip install --no-cache-dir grpcio grpcio-tools

CMD ["python", "-m", "secure_aggregation"]
