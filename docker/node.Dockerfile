FROM python:3.12-slim
WORKDIR /app

# Install system dependencies for SSL
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy entrypoint script first to leverage Docker layer caching
COPY scripts/node_entrypoint.sh /usr/local/bin/node-entrypoint.sh
RUN chmod +x /usr/local/bin/node-entrypoint.sh

# Copy project files
COPY . /app

# Install Python dependencies including mnist and grpc
RUN pip install --no-cache-dir -e .[mnist] && \
    pip install --no-cache-dir grpcio grpcio-tools

ENTRYPOINT ["node-entrypoint.sh"]
CMD ["python", "-m", "secure_aggregation"]
