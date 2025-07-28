FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.4.0 and torchvision are already installed in the base image
# But let's ensure we have the latest compatible versions
RUN pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# Clone Kohya's training scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git /kohya
WORKDIR /kohya

# Install Kohya requirements with latest compatible versions for PyTorch 2.4.0
RUN pip install accelerate
RUN pip install transformers
RUN pip install diffusers
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu124
RUN pip install datasets
RUN pip install safetensors
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
COPY debug_train.py /opt/ml/code/debug_train.py
COPY minimal_test.py /opt/ml/code/minimal_test.py
COPY train_wrapper.py /opt/ml/code/train.py

WORKDIR /opt/ml/code

# Set environment variables
ENV PYTHONPATH=/kohya:/opt/ml/code
ENV SAGEMAKER_PROGRAM=train.py
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Direct entrypoint to bypass sagemaker-training issues
ENTRYPOINT ["python", "/opt/ml/code/train.py"]