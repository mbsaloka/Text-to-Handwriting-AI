# Base image with Python & CUDA (for Hugging Face GPU Spaces)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (non-torch ones)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Install torch and torchvision explicitly (CUDA-compatible)
RUN pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Copy application files
COPY . .

# Expose port (FastAPI default)
EXPOSE 7860

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
