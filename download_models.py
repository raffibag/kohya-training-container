#!/usr/bin/env python3
"""
Download BLIP and CLIP models for pre-caching in Docker container
"""

import os
from pathlib import Path

# Set cache directory
cache_dir = Path("model_cache/huggingface")
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)

print("📦 Downloading BLIP model for captioning...")
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')
    print("✅ BLIP model downloaded successfully")
    
    # Clear from memory
    del processor, model
    
except Exception as e:
    print(f"❌ Failed to download BLIP: {e}")

print("📦 Downloading CLIP model for labeling...")
try:
    import clip
    import torch
    
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    print("✅ CLIP model downloaded successfully")
    
    # Clear from memory
    del model, preprocess
    
except Exception as e:
    print(f"❌ Failed to download CLIP: {e}")

print("🎉 Model cache created successfully!")
print(f"Cache location: {cache_dir.absolute()}")

# List downloaded files
cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
print(f"Total cache size: {cache_size / 1024 / 1024:.1f} MB")