# Use the official arm64 optimized image
FROM --platform=linux/arm64 python:3.9-slim

WORKDIR /app

# Install system dependencies (for building numpy/pandas if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command (can be overridden)
CMD ["python", "src/main.py"]