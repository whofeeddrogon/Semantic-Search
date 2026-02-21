FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Install CPU-only version of PyTorch to prevent downloading 3GB of NVIDIA CUDA packages
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .
