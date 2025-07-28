#!/usr/bin/env python
"""
Ultra-simple test that just writes to SM_MODEL_DIR
"""
import os
import sys

def main():
    print("=== SIMPLE TEST STARTED ===")
    
    # Get SageMaker directories
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    training_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    
    print(f"Model dir: {model_dir}")
    print(f"Training dir: {training_dir}")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Write a simple test file
    test_file = os.path.join(model_dir, "simple_test_results.txt")
    with open(test_file, "w") as f:
        f.write("Simple test completed successfully!\n")
        f.write(f"Model dir: {model_dir}\n")
        f.write(f"Training dir: {training_dir}\n")
        f.write(f"Training dir exists: {os.path.exists(training_dir)}\n")
        
        if os.path.exists(training_dir):
            files = os.listdir(training_dir)
            f.write(f"Training files count: {len(files)}\n")
            f.write(f"First 5 files: {files[:5]}\n")
        
        f.write("Python version: " + sys.version + "\n")
        f.write("Environment variables:\n")
        for key, value in os.environ.items():
            if key.startswith('SM_'):
                f.write(f"  {key}={value}\n")
    
    print(f"Test file written to: {test_file}")
    print("=== SIMPLE TEST COMPLETED ===")

if __name__ == "__main__":
    main()