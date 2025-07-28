#!/usr/bin/env python
"""
Minimal test to verify Kohya can run in SageMaker environment
"""
import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging to both console AND file
def setup_logging():
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    
    log_file = os.path.join(model_dir, "training_debug.log")
    
    # Create logger that writes to both console and file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def main():
    logger = setup_logging()
    logger.info("=== MINIMAL KOHYA TEST ===")
    
    # Environment check
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    training_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Training dir: {training_dir}")
    logger.info(f"Training dir exists: {os.path.exists(training_dir)}")
    
    if os.path.exists(training_dir):
        files = os.listdir(training_dir)
        logger.info(f"Training files: {files[:10]}")  # First 10 files
        
        # Count images
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"Found {len(image_files)} image files")
    
    # Test 1: Can we import Kohya modules?
    try:
        sys.path.insert(0, '/kohya')
        logger.info("=== TESTING KOHYA IMPORTS ===")
        import accelerate
        logger.info("✓ accelerate imported")
        import transformers  
        logger.info("✓ transformers imported")
        import diffusers
        logger.info("✓ diffusers imported")
        logger.info("Kohya environment setup: OK")
    except Exception as e:
        logger.error(f"✗ Kohya import failed: {e}")
        return
    
    # Test 2: Can we run Kohya help command?
    try:
        logger.info("=== TESTING KOHYA SCRIPT ACCESS ===")
        result = subprocess.run(['python', '/kohya/sdxl_train_network.py', '--help'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("✓ Kohya script accessible")
            logger.info(f"Help output length: {len(result.stdout)} chars")
        else:
            logger.error(f"✗ Kohya script error: {result.stderr}")
    except Exception as e:
        logger.error(f"✗ Kohya script test failed: {e}")
    
    # Test 3: Create a minimal working config and test it
    try:
        logger.info("=== TESTING MINIMAL CONFIG ===")
        
        # Create minimal test data if training data exists
        if os.path.exists(training_dir) and image_files:
            # Create minimal TOML config
            test_config = f"""
[model_arguments]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

[additional_network_arguments]
network_module = "networks.lora"
network_dim = 4
network_alpha = 4

[dataset_arguments]
train_data_dir = "{training_dir}"
resolution = "512,512"
batch_size = 1
enable_bucket = false

[training_arguments]
output_dir = "{model_dir}"
output_name = "test_lora"
max_train_steps = 1
save_every_n_steps = 1
mixed_precision = "fp16"
"""
            
            config_path = "/tmp/minimal_test.toml"
            with open(config_path, 'w') as f:
                f.write(test_config)
            
            print(f"Created config at: {config_path}")
            
            # Try to run ONE training step
            cmd = ['python', '/kohya/sdxl_train_network.py', '--config_file', config_path]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            print(f"Exit code: {result.returncode}")
            print(f"STDOUT length: {len(result.stdout)}")  
            print(f"STDERR length: {len(result.stderr)}")
            
            if result.stdout:
                print("=== STDOUT SAMPLE ===")
                print(result.stdout[:1000])  # First 1000 chars
            
            if result.stderr:
                print("=== STDERR SAMPLE ===") 
                print(result.stderr[:1000])  # First 1000 chars
                
            # Check if any files were created
            if os.path.exists(model_dir):
                output_files = list(Path(model_dir).glob("*"))
                print(f"Output files created: {[str(f) for f in output_files]}")
        else:
            print("No training data found - skipping training test")
    
    except Exception as e:
        print(f"✗ Minimal config test failed: {e}")
    
    # Create summary file
    summary = {
        "test_completed": True,
        "training_dir_exists": os.path.exists(training_dir),
        "model_dir": model_dir,
        "python_path": sys.path,
        "env_vars": {k: v for k, v in os.environ.items() if k.startswith('SM_')}
    }
    
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "minimal_test_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()