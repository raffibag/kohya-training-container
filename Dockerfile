# Ultra-lean CUDA + PyTorch base with Python preinstalled
FROM pytorch-base:ultra-lean

# Set timezone and minimize interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install git for cloning repos
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml/code

# Copy and install pinned requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# CRITICAL: Force NumPy 1.x before CLIP or Kohya
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Install OpenAI CLIP (no deps to avoid torch upgrade)
RUN pip install --no-cache-dir --no-deps git+https://github.com/openai/CLIP.git

# Clone Kohya and install as editable
RUN git clone https://github.com/kohya-ss/sd-scripts.git /kohya
WORKDIR /kohya
RUN pip install --no-cache-dir sagemaker-training && \
    pip install --no-cache-dir -e .

# Copy all code and wrappers
WORKDIR /opt/ml/code
COPY train_wrapper.py kohya_config.py ./
COPY auto_caption_s3_dataset.py automated_schema_labeler.py simple_auto_caption.py ./
COPY serve ./serve

# Make serve script executable and symlink it
RUN chmod +x serve && ln -s /opt/ml/code/serve /usr/local/bin/serve

# Set HuggingFace + CLIP cache directories
ENV HF_HOME=/opt/ml/cache/huggingface
ENV TRANSFORMERS_CACHE=/opt/ml/cache/huggingface
ENV HOME=/opt/ml/cache

# Create cache directories (models will be downloaded at runtime)
RUN mkdir -p /opt/ml/cache/huggingface /opt/ml/cache/clip

# Verify NumPy version
RUN python -c "import numpy; assert numpy.__version__.startswith('1.'), f'NumPy {numpy.__version__} is not 1.x'"

# Clean up pip cache and Python bytecode
RUN rm -rf /root/.cache /tmp/* /var/tmp/* && \
    find /usr/local -depth -type d -name __pycache__ -exec rm -rf {} + && \
    find /usr/local -name '*.pyc' -delete && \
    find /kohya -depth -type d -name __pycache__ -exec rm -rf {} +

# Set environment for SageMaker training
ENV PYTHONPATH=/kohya:/opt/ml/code
ENV SAGEMAKER_PROGRAM=train_wrapper.py
ENV NETWORKX_BACKENDS=numpy

# Let SageMaker manage entrypoint
ENTRYPOINT []
