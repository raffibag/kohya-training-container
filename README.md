# Kohya Training Container

Production-ready Docker container for Stable Diffusion training using Kohya's sd-scripts with DreamBooth + LoRA support.

## Features

- **Kohya's sd-scripts**: Industry-standard training framework
- **DreamBooth + LoRA**: Identity preservation with efficient fine-tuning
- **SageMaker Integration**: Ready for AWS SageMaker training jobs
- **Optimized Dependencies**: Compatible PyTorch 2.0.1 + CUDA 11.7 stack

## Container Contents

- PyTorch 2.0.1 with CUDA 11.7 support
- Kohya's sd-scripts (latest)
- xformers 0.0.21 (optimized for PyTorch 2.0.1)
- bitsandbytes 0.41.0 (8-bit optimization)
- SageMaker training toolkit
- Custom training wrapper for automatic model type detection

## Build with CodeBuild

This container is built using AWS CodeBuild for reliable, high-speed builds and ECR deployment.

**Target ECR:** `796245059390.dkr.ecr.us-west-2.amazonaws.com/kohya-training:latest`

## Training Types Supported

### Character Training (DreamBooth + LoRA)
- Identity preservation for people/characters
- Higher LoRA rank (64) for detail retention
- Prior preservation with regularization images

### Style Training (LoRA only)
- Artistic style transfer
- Lower LoRA rank (32) for style learning
- Faster training without regularization

## Usage in SageMaker

The container automatically detects training type based on instance prompts and adjusts hyperparameters accordingly.