FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Force reinstall PyTorch with proper CUDA support and system libraries
# Install after system libraries to ensure proper linking
RUN pip install --force-reinstall --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install all Python dependencies from requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install xformers separately with CUDA 11.8 support
RUN pip install xformers --extra-index-url https://download.pytorch.org/whl/cu118

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

WORKDIR /opt/ml/code

# Set environment variables
ENV PYTHONPATH=/kohya:/opt/ml/code
ENV SAGEMAKER_PROGRAM=train_wrapper.py
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV NETWORKX_BACKENDS=numpy

# Remove direct entrypoint to let SageMaker handle it properly
ENTRYPOINT []