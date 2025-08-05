# Use the pytorch-base container as base
FROM pytorch-base

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install all Python dependencies from requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install CLIP manually with torch constraint to avoid upgrades
RUN pip install --no-deps git+https://github.com/openai/CLIP.git

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

# CRITICAL: Force downgrade NumPy to 1.x as the very last step
# This ensures all dependencies are installed but we still get NumPy 1.x
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Verify NumPy version
RUN python -c "import numpy; assert numpy.__version__.startswith('1.'), f'NumPy {numpy.__version__} is not 1.x'"

# Set environment variables
ENV PYTHONPATH=/kohya:/opt/ml/code
ENV SAGEMAKER_PROGRAM=train_wrapper.py
ENV NETWORKX_BACKENDS=numpy

# Remove direct entrypoint to let SageMaker handle it properly
ENTRYPOINT []