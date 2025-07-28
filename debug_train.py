#!/usr/bin/env python
"""
Debug script to test SageMaker environment and file locations
"""
import os
import sys
import json
import boto3
from pathlib import Path
from datetime import datetime

def main():
    # Create debug output
    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "environment": dict(os.environ),
        "sys_path": sys.path,
        "working_dir": os.getcwd(),
        "arguments": sys.argv
    }
    
    # Check standard SageMaker paths
    paths_to_check = {
        "SM_MODEL_DIR": os.environ.get("SM_MODEL_DIR", "NOT_SET"),
        "SM_CHANNEL_TRAINING": os.environ.get("SM_CHANNEL_TRAINING", "NOT_SET"),
        "SM_OUTPUT_DATA_DIR": os.environ.get("SM_OUTPUT_DATA_DIR", "NOT_SET"),
        "SM_OUTPUT_DIR": os.environ.get("SM_OUTPUT_DIR", "NOT_SET"),
        "/opt/ml/input": os.path.exists("/opt/ml/input"),
        "/opt/ml/model": os.path.exists("/opt/ml/model"),
        "/opt/ml/output": os.path.exists("/opt/ml/output"),
        "/opt/ml/input/data": os.path.exists("/opt/ml/input/data"),
        "/opt/ml/input/data/training": os.path.exists("/opt/ml/input/data/training")
    }
    debug_info["paths"] = paths_to_check
    
    # List directory contents
    directories_to_list = [
        "/opt/ml",
        "/opt/ml/input",
        "/opt/ml/input/data",
        os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    ]
    
    dir_contents = {}
    for dir_path in directories_to_list:
        if os.path.exists(dir_path):
            try:
                contents = os.listdir(dir_path)
                dir_contents[dir_path] = contents[:20]  # First 20 files
                
                # If it's the training dir, look for images
                if "training" in dir_path:
                    images = [f for f in contents if f.endswith(('.jpg', '.jpeg', '.png'))]
                    dir_contents[f"{dir_path}_images"] = images
            except Exception as e:
                dir_contents[dir_path] = f"Error: {str(e)}"
        else:
            dir_contents[dir_path] = "DOES_NOT_EXIST"
    
    debug_info["directory_contents"] = dir_contents
    
    # Write debug info to model directory
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    
    debug_file = os.path.join(model_dir, "debug_info.json")
    with open(debug_file, "w") as f:
        json.dump(debug_info, f, indent=2)
    
    print("=== DEBUG INFO WRITTEN ===")
    print(json.dumps(debug_info, indent=2))
    
    # Try multiple output locations
    output_attempts = []
    
    # Test 1: Write to SM_MODEL_DIR (standard location)
    try:
        test_file = os.path.join(model_dir, "test_model_dir.txt")
        with open(test_file, "w") as f:
            f.write(f"Test file in SM_MODEL_DIR: {model_dir}\n")
        output_attempts.append({"location": "SM_MODEL_DIR", "path": model_dir, "success": True})
    except Exception as e:
        output_attempts.append({"location": "SM_MODEL_DIR", "path": model_dir, "error": str(e)})
    
    # Test 2: Write to /opt/ml/model (hardcoded)
    try:
        os.makedirs("/opt/ml/model", exist_ok=True)
        test_file = "/opt/ml/model/test_hardcoded.txt"
        with open(test_file, "w") as f:
            f.write("Test file in /opt/ml/model\n")
        output_attempts.append({"location": "/opt/ml/model", "success": True})
    except Exception as e:
        output_attempts.append({"location": "/opt/ml/model", "error": str(e)})
    
    # Test 3: Write to SM_OUTPUT_DATA_DIR
    output_data_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    try:
        os.makedirs(output_data_dir, exist_ok=True)
        test_file = os.path.join(output_data_dir, "test_output_data.txt")
        with open(test_file, "w") as f:
            f.write(f"Test file in SM_OUTPUT_DATA_DIR: {output_data_dir}\n")
        output_attempts.append({"location": "SM_OUTPUT_DATA_DIR", "path": output_data_dir, "success": True})
    except Exception as e:
        output_attempts.append({"location": "SM_OUTPUT_DATA_DIR", "path": output_data_dir, "error": str(e)})
    
    # Test 4: Create a tar.gz file (SageMaker expects this format)
    try:
        import tarfile
        import io
        
        # Create dummy LoRA model file
        dummy_lora = os.path.join(model_dir, "franka_lora.safetensors")
        with open(dummy_lora, "wb") as f:
            f.write(b"DUMMY_LORA_MODEL_DATA")
        
        # Create model.tar.gz
        tar_path = os.path.join(model_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(dummy_lora, arcname="franka_lora.safetensors")
            tar.add(debug_file, arcname="debug_info.json")
        
        output_attempts.append({"location": "model.tar.gz", "path": tar_path, "success": True})
    except Exception as e:
        output_attempts.append({"location": "model.tar.gz", "error": str(e)})
    
    # Test 5: List all writable directories
    writable_dirs = []
    for test_dir in ["/opt/ml", "/tmp", "/opt/ml/model", "/opt/ml/output"]:
        try:
            test_file = os.path.join(test_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            writable_dirs.append(test_dir)
        except:
            pass
    
    debug_info["output_attempts"] = output_attempts
    debug_info["writable_directories"] = writable_dirs
    
    # Update debug file with results
    with open(debug_file, "w") as f:
        json.dump(debug_info, f, indent=2)
    
    print("=== OUTPUT TEST RESULTS ===")
    print(json.dumps(output_attempts, indent=2))
    print(f"Writable directories: {writable_dirs}")
    
if __name__ == "__main__":
    main()