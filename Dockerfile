# Stage 1: Build base image
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS ml-base

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Common system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
RUN pip install --force-reinstall --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install xformers with CUDA 11.8 support
RUN pip install xformers --extra-index-url https://download.pytorch.org/whl/cu118

# Set common environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create common directories
RUN mkdir -p /opt/ml/code /opt/ml/cache

WORKDIR /opt/ml/code

# Stage 2: Build training image
FROM ml-base

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install all Python dependencies from requirements.txt
RUN pip install -r /tmp/requirements.txt

# Clone Kohya's training scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git /kohya
WORKDIR /kohya

# Install SageMaker training toolkit
RUN pip install sagemaker-training

# Install library for Kohya (required)
RUN pip install -e .

# Copy SageMaker wrapper scripts
COPY train_wrapper.py /opt/ml/code/train_wrapper.py
COPY kohya_config.py /opt/ml/code/kohya_config.py

# Copy captioning and endpoint scripts
COPY auto_caption_s3_dataset.py /opt/ml/code/auto_caption_s3_dataset.py
COPY automated_schema_labeler.py /opt/ml/code/automated_schema_labeler.py
COPY simple_auto_caption.py /opt/ml/code/simple_auto_caption.py
COPY serve /opt/ml/code/serve

# Make serve script executable
RUN chmod +x /opt/ml/code/serve

# Set environment variables
ENV PYTHONPATH=/kohya:/opt/ml/code
ENV SAGEMAKER_PROGRAM=train_wrapper.py
ENV NETWORKX_BACKENDS=numpy

# Remove direct entrypoint to let SageMaker handle it properly
ENTRYPOINT []