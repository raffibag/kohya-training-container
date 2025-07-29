FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.1.2 and torchvision are already installed in the base image
# But let's ensure we have the exact compatible versions
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Clone Kohya's training scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git /kohya
WORKDIR /kohya

# Install Kohya requirements with compatible versions for PyTorch 2.1.2
RUN pip install accelerate
RUN pip install transformers
RUN pip install diffusers
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip install datasets
RUN pip install safetensors
RUN pip install opencv-python-headless
RUN pip install gradio
RUN pip install altair
RUN pip install easygui
RUN pip install einops
RUN pip install pytorch-lightning
RUN pip install bitsandbytes
RUN pip install tensorboard
RUN pip install wandb

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

# Remove direct entrypoint to let SageMaker handle it properly
ENTRYPOINT []