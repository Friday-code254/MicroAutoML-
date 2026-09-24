# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4

# Set work directory
WORKDIR /app

# Install system dependencies (e.g., for building some C extensions like LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better caching
COPY pyproject.toml .
RUN pip install setuptools wheel && pip install .

# Copy the rest of the application
COPY . .

# Run the command
ENTRYPOINT ["microautoml"]
CMD ["--help"]
