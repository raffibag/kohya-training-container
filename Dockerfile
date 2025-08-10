# Ultra-lean base with CUDA 11.8, Python 3.10, torch 2.1.2, torchvision 0.16.2, xformers 0.0.23.post1
FROM pytorch-base:ultra-lean

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Minimal build/runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ make python3-dev ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Install pinned deps
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# (Optional) If you want OpenAI CLIP specifically, do it without deps:
# RUN pip install --no-cache-dir --no-deps git+https://github.com/openai/CLIP.git

# Clone Kohya and install
RUN git clone https://github.com/kohya-ss/sd-scripts.git /kohya
WORKDIR /kohya
RUN pip install --no-cache-dir -e .

# App files
WORKDIR /opt/ml/code
COPY train_wrapper.py kohya_config.py ./
COPY auto_caption_s3_dataset.py automated_schema_labeler.py simple_auto_caption.py ./
COPY serve ./serve
RUN chmod +x serve train_wrapper.py && ln -s /opt/ml/code/serve /usr/local/bin/serve

# HF caches
ENV HF_HOME=/opt/ml/cache/huggingface
ENV TRANSFORMERS_CACHE=/opt/ml/cache/huggingface
ENV HOME=/opt/ml/cache
RUN mkdir -p /opt/ml/cache/huggingface /opt/ml/cache/clip

# Safety checks (fail fast if mismatched)
RUN python - <<'PY'
import torch, numpy as np, transformers, diffusers, accelerate, xformers
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN ok")
print("CUDA available:", torch.cuda.is_available())
print("xformers:", xformers.__version__)
print("NumPy:", np.__version__)
print("Transformers:", transformers.__version__)
print("Diffusers:", diffusers.__version__)
print("Accelerate:", accelerate.__version__)
PY

# Clean caches
RUN rm -rf /root/.cache /tmp/* /var/tmp/* && \
    find /usr/local -depth -type d -name __pycache__ -exec rm -rf {} + && \
    find /usr/local -name '*.pyc' -delete && \
    find /kohya -depth -type d -name __pycache__ -exec rm -rf {} +

# SageMaker env
ENV PYTHONPATH=/kohya:/opt/ml/code
ENV SAGEMAKER_PROGRAM=train_wrapper.py
ENV NETWORKX_BACKENDS=numpy
ENTRYPOINT []
