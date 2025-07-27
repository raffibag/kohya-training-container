FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch and torchvision first (specific versions)
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Clone Kohya's training scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git /kohya
WORKDIR /kohya

# Install Kohya requirements one by one to avoid conflicts
RUN pip install accelerate==0.21.0
RUN pip install transformers==4.30.0
RUN pip install diffusers==0.18.2
RUN pip install xformers==0.0.21 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install datasets
RUN pip install safetensors
RUN pip install gradio
RUN pip install altair
RUN pip install easygui
RUN pip install einops
RUN pip install pytorch-lightning==1.9.0
RUN pip install bitsandbytes==0.41.0
RUN pip install tensorboard
RUN pip install wandb

# Install SageMaker training toolkit
RUN pip install sagemaker-training

# Install library for Kohya (required)
RUN pip install -e .

# Copy SageMaker wrapper scripts
COPY train_wrapper.py /opt/ml/code/train.py
COPY kohya_config.py /opt/ml/code/kohya_config.py

WORKDIR /opt/ml/code

# Set environment variables
ENV PYTHONPATH=/kohya:/opt/ml/code
ENV SAGEMAKER_PROGRAM=train.py
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENTRYPOINT ["python", "-m", "sagemaker_training.trainer"]