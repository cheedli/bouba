# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    libgomp1 \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements file
#COPY requirements.txt .

# Install Python dependencies
#RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Chainlit
EXPOSE 8000

# Set environment variable for Chainlit
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8000

# Command to run the application
CMD ["chainlit", "run", "app_chainlit.py", "--host", "0.0.0.0", "--port", "8000"]