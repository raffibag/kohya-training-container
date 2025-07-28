#!/usr/bin/env python
"""
SageMaker wrapper for Kohya's SD training scripts
Supports DreamBooth + LoRA training with SDXL
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path

# Add Kohya to path
sys.path.insert(0, '/kohya')

from kohya_config import create_kohya_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--training-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    
    # Training configuration
    parser.add_argument("--pretrained-model-name", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--instance-prompt", type=str, required=True)
    parser.add_argument("--class-prompt", type=str, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # LoRA specific
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    
    # DreamBooth specific (auto-enabled for character training)
    parser.add_argument("--prior-loss-weight", type=float, default=1.0)
    parser.add_argument("--with-prior-preservation", action="store_true")
    parser.add_argument("--class-data-dir", type=str, default=None)
    parser.add_argument("--num-class-images", type=int, default=200)
    
    # Optimization
    parser.add_argument("--mixed-precision", type=str, default="fp16")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--enable-xformers", action="store_true", default=True)
    parser.add_argument("--optimizer", type=str, default="AdamW8bit")
    parser.add_argument("--clip-skip", type=int, default=2)
    
    # Advanced
    parser.add_argument("--caption-extension", type=str, default=".txt")
    parser.add_argument("--shuffle-caption", action="store_true")
    parser.add_argument("--cache-latents", action="store_true", default=True)
    parser.add_argument("--cache-latents-to-disk", action="store_true")
    parser.add_argument("--color-aug", action="store_true")
    parser.add_argument("--flip-aug", action="store_true")
    
    return parser.parse_args()

def prepare_dataset(training_dir, instance_prompt):
    """Prepare dataset directory structure for Kohya"""
    logger.info(f"Training directory contents: {os.listdir(training_dir)}")
    
    # SageMaker mounts S3 data directly to training_dir
    # Check if images are in root or in a subdirectory
    dataset_dir = Path(training_dir)
    
    # Try to find images in various locations
    image_files = []
    search_dirs = [
        dataset_dir,
        dataset_dir / "dataset",
        dataset_dir / "training",
        dataset_dir / "data"
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            logger.info(f"Searching for images in: {search_dir}")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                found = list(search_dir.glob(ext))
                if found:
                    image_files.extend(found)
                    dataset_dir = search_dir
                    logger.info(f"Found {len(found)} {ext} files in {search_dir}")
                    break
            if image_files:
                break
    
    if not image_files:
        logger.error(f"No images found in any expected location!")
        logger.error(f"Searched in: {search_dirs}")
        # List all files recursively to debug
        all_files = list(Path(training_dir).rglob("*"))
        logger.error(f"All files in training_dir: {[str(f) for f in all_files[:20]]}")
        raise ValueError("No training images found")
    
    logger.info(f"Found {len(image_files)} training images in {dataset_dir}")
    
    # Create captions for each image
    for image_file in image_files:
        caption_file = image_file.with_suffix('.txt')
        if not caption_file.exists():
            with open(caption_file, 'w') as f:
                f.write(instance_prompt)
            logger.info(f"Created caption for {image_file.name}")
    
    return str(dataset_dir)

def run_kohya_training(args, dataset_path):
    """Run Kohya's training script with generated config"""
    
    # Generate Kohya config
    config_path = "/tmp/kohya_config.toml"
    create_kohya_config(args, dataset_path, config_path)
    
    # Determine which script to use
    if args.use_lora:
        script_name = "sdxl_train_network.py"
    else:
        script_name = "sdxl_train.py"
    
    # Build command
    cmd = [
        "python", f"/kohya/{script_name}",
        "--config_file", config_path,
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run training
    try:
        logger.info("=== STARTING KOHYA TRAINING ===")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("=== KOHYA TRAINING OUTPUT ===")
        logger.info(result.stdout)
        if result.stderr:
            logger.error("=== KOHYA TRAINING ERRORS ===")
            logger.error(result.stderr)
        logger.info("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

def detect_training_type(instance_prompt, training_dir):
    """Auto-detect if this is character or style training"""
    
    # Check prompt for character indicators
    character_keywords = ['person', 'man', 'woman', 'character', 'actor', 'model']
    style_keywords = ['style', 'photography', 'art', 'painting', 'aesthetic']
    
    prompt_lower = instance_prompt.lower()
    
    # Count indicators
    char_score = sum(1 for kw in character_keywords if kw in prompt_lower)
    style_score = sum(1 for kw in style_keywords if kw in prompt_lower)
    
    # Check directory structure for hints
    path_lower = str(training_dir).lower()
    if 'character' in path_lower or 'model-' in path_lower:
        char_score += 2
    elif 'style' in path_lower or 'photographer' in path_lower:
        style_score += 2
    
    is_character = char_score > style_score
    training_type = "character" if is_character else "style"
    
    logger.info(f"Auto-detected training type: {training_type}")
    logger.info(f"Character score: {char_score}, Style score: {style_score}")
    
    return is_character, training_type

def setup_dreambooth_regularization(args, training_type, dataset_path):
    """Setup regularization images for DreamBooth character training"""
    
    if training_type != "character":
        return None
        
    # Enable DreamBooth with prior preservation for characters
    args.with_prior_preservation = True
    
    reg_dir = Path("/tmp/regularization_images")
    reg_dir.mkdir(exist_ok=True)
    
    # Extract class name from instance prompt
    # "a photo of john person" -> "person"
    words = args.instance_prompt.lower().split()
    class_word = "person"  # default
    
    if "person" in words:
        class_word = "person"
    elif "man" in words:
        class_word = "man"
    elif "woman" in words:
        class_word = "woman"
    
    args.class_prompt = f"a photo of a {class_word}"
    args.class_data_dir = str(reg_dir)
    
    logger.info(f"DreamBooth enabled with class prompt: {args.class_prompt}")
    logger.info(f"Regularization directory: {args.class_data_dir}")
    
    return str(reg_dir)

def main():
    logger.info("=== KOHYA TRAINING WRAPPER STARTED ===")
    
    args = parse_args()
    
    # Debug environment
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if key.startswith('SM_'):
            logger.info(f"  {key}={value}")
    
    logger.info("Starting Kohya-based SDXL training")
    logger.info(f"Training directory: {args.training_dir}")
    logger.info(f"Model output directory: {args.model_dir}")
    logger.info(f"Instance prompt: {args.instance_prompt}")
    
    # Check if directories exist
    if not os.path.exists(args.training_dir):
        logger.error(f"Training directory does not exist: {args.training_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.model_dir):
        logger.info(f"Creating model directory: {args.model_dir}")
        os.makedirs(args.model_dir, exist_ok=True)
    
    # Auto-detect training type
    is_character, training_type = detect_training_type(args.instance_prompt, args.training_dir)
    
    # Prepare dataset
    dataset_path = prepare_dataset(args.training_dir, args.instance_prompt)
    
    # Setup DreamBooth for character training
    reg_dir = setup_dreambooth_regularization(args, training_type, dataset_path)
    
    # Adjust hyperparameters based on type
    if is_character:
        # Character training: Lower LR, more epochs, DreamBooth
        args.learning_rate = max(args.learning_rate * 0.5, 5e-5)  # Lower LR for faces
        args.num_train_epochs = max(args.num_train_epochs, 6)     # More epochs
        args.lora_rank = max(args.lora_rank, 64)                  # Higher rank for details
        logger.info("Optimized for character training")
    else:
        # Style training: Standard settings, no DreamBooth
        args.with_prior_preservation = False
        logger.info("Optimized for style training")
    
    # Run training
    success = run_kohya_training(args, dataset_path)
    
    if success:
        logger.info("Training completed successfully!")
        
        # Copy models to SageMaker output directory
        output_dir = Path(args.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and copy LoRA files from Kohya output
        lora_output_dir = Path("/tmp/lora_models")
        
        # First, let's see what files exist in /tmp
        logger.info("=== CHECKING /tmp FOR ANY OUTPUT FILES ===")
        tmp_files = list(Path("/tmp").rglob("*.safetensors"))
        tmp_files.extend(list(Path("/tmp").rglob("*.ckpt")))
        tmp_files.extend(list(Path("/tmp").rglob("*.pt")))
        logger.info(f"Found {len(tmp_files)} model files in /tmp:")
        for f in tmp_files:
            logger.info(f"  - {f}")
        
        # Also check the current working directory
        cwd_files = list(Path(".").rglob("*.safetensors"))
        cwd_files.extend(list(Path(".").rglob("*.ckpt")))
        if cwd_files:
            logger.info(f"Found {len(cwd_files)} model files in current dir:")
            for f in cwd_files:
                logger.info(f"  - {f}")
        
        # Check if output directory exists
        if not lora_output_dir.exists():
            logger.error(f"LoRA output directory not found: {lora_output_dir}")
            logger.info(f"Creating directory: {lora_output_dir}")
            lora_output_dir.mkdir(parents=True, exist_ok=True)
            
        # List all files in the output directory
        all_files = list(lora_output_dir.rglob("*"))
        logger.info(f"Files in {lora_output_dir}: {[str(f) for f in all_files]}")
        
        # Copy all model files
        import shutil
        model_files_copied = 0
        for model_file in lora_output_dir.rglob("*.safetensors"):
            dest_file = output_dir / model_file.name
            shutil.copy2(model_file, dest_file)
            logger.info(f"Copied {model_file.name} to {dest_file}")
            model_files_copied += 1
            
        # Also copy any .ckpt files
        for model_file in lora_output_dir.rglob("*.ckpt"):
            dest_file = output_dir / model_file.name
            shutil.copy2(model_file, dest_file)
            logger.info(f"Copied {model_file.name} to {dest_file}")
            model_files_copied += 1
            
        if model_files_copied == 0:
            logger.error("No model files found in expected location!")
            logger.info("Attempting fallback: searching entire /tmp for model files...")
            
            # Fallback: copy ANY model files found in /tmp
            for tmp_file in tmp_files:
                dest_file = output_dir / tmp_file.name
                shutil.copy2(tmp_file, dest_file)
                logger.info(f"Fallback: Copied {tmp_file} to {dest_file}")
                model_files_copied += 1
            
            if model_files_copied == 0:
                logger.error("No model files found anywhere!")
                logger.error(f"Expected files in: {lora_output_dir}")
                # Don't exit, create a debug file instead
                debug_file = output_dir / "training_debug.txt"
                with open(debug_file, "w") as f:
                    f.write("No model files were generated by training\n")
                    f.write(f"Searched in: {lora_output_dir}\n")
                    f.write(f"Also searched: /tmp\n")
                    f.write(f"Working directory: {os.getcwd()}\n")
                logger.info(f"Created debug file: {debug_file}")
        else:
            logger.info(f"Successfully copied {model_files_copied} model file(s)")
        
        # Create metadata
        metadata = {
            "model_type": "SDXL LoRA",
            "training_method": "Kohya DreamBooth + LoRA",
            "instance_prompt": args.instance_prompt,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "training_steps": args.max_train_steps or (args.num_train_epochs * 100),  # estimate
            "base_model": args.pretrained_model_name
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        logger.error("Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()