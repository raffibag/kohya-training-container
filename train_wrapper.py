#!/usr/bin/env python
"""
SageMaker wrapper for Kohya's SD training scripts
Supports DreamBooth + LoRA training with SDXL
"""

import os
import sys
import json
import subprocess
import logging
import time
from datetime import datetime
from pathlib import Path

# Add Kohya to path
sys.path.insert(0, '/kohya')

from kohya_config import create_kohya_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_hp(key, default=None, required=False):
    """Get SageMaker hyperparameter from hyperparameters.json file"""
    try:
        with open("/opt/ml/input/config/hyperparameters.json", "r") as f:
            hyperparams = json.load(f)
    except FileNotFoundError:
        logger.warning("Hyperparameters file not found, using defaults")
        hyperparams = {}
    
    # SageMaker uses kebab-case keys in hyperparameters.json
    val = hyperparams.get(key, default)
    if required and val is None:
        raise ValueError(f"Missing required hyperparameter: {key}")
    return val

def get_config():
    """Get configuration from environment variables using the cleaner helper"""
    
    class Config:
        def __init__(self):
            # SageMaker specific paths
            self.model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
            self.training_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
            self.output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")
            
            # Training configuration from hyperparameters
            self.pretrained_model_name = get_hp("pretrained-model-name", "stabilityai/stable-diffusion-xl-base-1.0")
            self.instance_prompt = get_hp("instance-prompt", "a photo of a person", required=True)
            self.class_prompt = get_hp("class-prompt")
            self.num_train_epochs = int(get_hp("num-train-epochs", "4"))
            self.train_batch_size = int(get_hp("train-batch-size", "1"))
            self.learning_rate = float(get_hp("learning-rate", "1e-4"))
            self.lr_scheduler = get_hp("lr-scheduler", "cosine_with_restarts")
            self.lr_warmup_steps = int(get_hp("lr-warmup-steps", "0"))
            max_steps = int(get_hp("max-train-steps", "0"))
            self.max_train_steps = max_steps if max_steps > 0 else None
            self.seed = int(get_hp("seed", "42"))
            
            # LoRA specific
            self.use_lora = get_hp("use-lora", "true").lower() in ["true", "1", "yes"]
            self.lora_rank = int(get_hp("lora-rank", "32"))
            self.lora_alpha = int(get_hp("lora-alpha", "32"))
            self.lora_dropout = float(get_hp("lora-dropout", "0.0"))
            
            # Kohya output directory
            self.kohya_output_dir = get_hp("kohya-output-dir", "/tmp/lora_models")
            
            # DreamBooth specific
            self.prior_loss_weight = float(get_hp("prior-loss-weight", "1.0"))
            self.with_prior_preservation = get_hp("with-prior-preservation", "false").lower() in ["true", "1", "yes"]
            self.class_data_dir = get_hp("class-data-dir")
            self.num_class_images = int(get_hp("num-class-images", "200"))
            
            # Optimization
            self.mixed_precision = get_hp("mixed-precision", "fp16")
            self.gradient_checkpointing = get_hp("gradient-checkpointing", "true").lower() in ["true", "1", "yes"]
            self.enable_xformers = get_hp("enable-xformers", "true").lower() in ["true", "1", "yes"]
            self.optimizer = get_hp("optimizer", "AdamW8bit")
            self.clip_skip = int(get_hp("clip-skip", "2"))
            
            # Advanced
            self.caption_extension = get_hp("caption-extension", ".txt")
            self.shuffle_caption = get_hp("shuffle-caption", "false").lower() in ["true", "1", "yes"]
            self.cache_latents = get_hp("cache-latents", "true").lower() in ["true", "1", "yes"]
            self.cache_latents_to_disk = get_hp("cache-latents-to-disk", "false").lower() in ["true", "1", "yes"]
            self.color_aug = get_hp("color-aug", "false").lower() in ["true", "1", "yes"]
            self.flip_aug = get_hp("flip-aug", "false").lower() in ["true", "1", "yes"]
    
    return Config()

def prepare_dataset(training_dir, instance_prompt):
    """Prepare dataset directory structure for Kohya"""
    logger.info(f"Training directory contents: {os.listdir(training_dir)}")
    
    # SageMaker mounts S3 data directly to training_dir
    # Check if images are in root or in a subdirectory
    source_dir = Path(training_dir)
    
    # Try to find images in various locations
    image_files = []
    search_dirs = [
        source_dir,
        source_dir / "dataset",
        source_dir / "training",
        source_dir / "data"
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            logger.info(f"Searching for images in: {search_dir}")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                found = list(search_dir.glob(ext))
                if found:
                    image_files.extend(found)
                    source_dir = search_dir
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
    
    logger.info(f"Found {len(image_files)} training images in {source_dir}")
    
    # Auto-detect training type for optimal repeat calculation
    _, training_type = detect_training_type(instance_prompt, str(source_dir))
    
    # Calculate optimal repeats
    optimal_repeats = calculate_optimal_repeats(len(image_files), training_type)
    
    # Create Kohya-compatible directory structure
    # Kohya expects: parent_dir/repeats_subfolder/images
    kohya_parent_dir = Path("/tmp/kohya_dataset")  
    kohya_images_dir = kohya_parent_dir / f"{optimal_repeats}_training"
    kohya_images_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating Kohya dataset structure at: {kohya_parent_dir}")
    logger.info(f"Images will be in: {kohya_images_dir}")
    logger.info(f"Using {optimal_repeats} repeats for {training_type} training")
    
    # Copy images and create captions in the Kohya structure
    import shutil
    for image_file in image_files:
        # Copy image
        dest_image = kohya_images_dir / image_file.name
        shutil.copy2(image_file, dest_image)
        
        # Create or copy caption
        source_caption = image_file.with_suffix('.txt')
        dest_caption = dest_image.with_suffix('.txt')
        
        if source_caption.exists():
            shutil.copy2(source_caption, dest_caption)
            logger.info(f"Copied caption for {image_file.name}")
        else:
            with open(dest_caption, 'w') as f:
                f.write(instance_prompt)
            logger.info(f"Created caption for {image_file.name}")
    
    logger.info(f"Prepared {len(image_files)} images in Kohya format")
    
    # Return the parent directory (what Kohya expects as train_data_dir)
    return str(kohya_parent_dir)

def run_kohya_training(config, dataset_path):
    """Run Kohya's training script with generated config"""
    
    # Generate Kohya config
    config_path = "/tmp/kohya_config.toml"
    create_kohya_config(config, dataset_path, config_path, config.kohya_output_dir)
    
    # Determine which script to use
    if config.use_lora:
        script_name = "sdxl_train_network.py"
    else:
        script_name = "sdxl_train.py"
    
    # Build command
    cmd = [
        "python", f"/kohya/{script_name}",
        "--config_file", config_path,
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run training with full output capture
    try:
        logger.info("=== STARTING KOHYA TRAINING ===")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run with output capture to see the actual error
        result = subprocess.run(
            cmd, 
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit, we'll handle it
        )
        
        # Log stdout
        if result.stdout:
            logger.info("=== KOHYA STDOUT ===")
            for line in result.stdout.splitlines():
                logger.info(f"KOHYA: {line}")
        
        # Log stderr
        if result.stderr:
            logger.error("=== KOHYA STDERR ===")
            for line in result.stderr.splitlines():
                logger.error(f"KOHYA ERR: {line}")
        
        # Always save training logs for debugging
        logs_path = Path(config.model_dir) / "kohya_training_logs.txt"
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(logs_path, "w") as f:
            f.write(f"Kohya training completed with return code: {result.returncode}\n\n")
            f.write("=== COMMAND ===\n")
            f.write(f"{' '.join(cmd)}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout or "(empty)")
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr or "(empty)")
            f.write("\n\n=== CONFIG FILE ===\n")
            if Path(config_path).exists():
                with open(config_path, "r") as cf:
                    f.write(cf.read())
        logger.info(f"Saved training logs to: {logs_path}")
        
        # Check if it succeeded
        if result.returncode != 0:
            logger.error(f"Kohya training failed with return code: {result.returncode}")
            return False
        
        logger.info("Training subprocess completed successfully!")
        
        # Validate that model files were created
        kohya_output_path = Path(config.kohya_output_dir)
        model_files = list(kohya_output_path.glob("*.safetensors")) + list(kohya_output_path.glob("*.ckpt"))
        
        if model_files:
            logger.info(f"✓ Found {len(model_files)} model files in {kohya_output_path}")
            for f in model_files:
                logger.info(f"  - {f.name} ({f.stat().st_size} bytes)")
        else:
            logger.warning(f"⚠ No model files found in {kohya_output_path}")
            # List what files DO exist
            all_files = list(kohya_output_path.glob("*")) if kohya_output_path.exists() else []
            logger.info(f"Files in output dir: {[f.name for f in all_files]}")
        
        return True
    except Exception as e:
        logger.error(f"Exception running Kohya training: {type(e).__name__}: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Save exception details
        error_details_path = Path(config.model_dir) / "kohya_exception_details.txt"
        error_details_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_details_path, "w") as f:
            f.write(f"Exception running Kohya training\n")
            f.write(f"Exception type: {type(e).__name__}\n")
            f.write(f"Exception message: {e}\n\n")
            f.write("=== FULL TRACEBACK ===\n")
            f.write(traceback.format_exc())
            f.write("\n\n=== COMMAND ===\n")
            f.write(f"{' '.join(cmd)}\n")
        logger.info(f"Saved exception details to: {error_details_path}")
        return False

def detect_training_type(instance_prompt, training_dir):
    """Auto-detect training type: character, style, or concept"""
    
    # Character indicators
    character_keywords = ['person', 'man', 'woman', 'character', 'actor', 'model', 'individual']
    # Style indicators  
    style_keywords = ['style', 'photography', 'art', 'painting', 'aesthetic', 'photographer']
    # Concept indicators
    concept_keywords = ['concept', 'object', 'building', 'vehicle', 'clothing', 'accessory']
    
    prompt_lower = instance_prompt.lower()
    
    # Count indicators
    char_score = sum(1 for kw in character_keywords if kw in prompt_lower)
    style_score = sum(1 for kw in style_keywords if kw in prompt_lower)
    concept_score = sum(1 for kw in concept_keywords if kw in prompt_lower)
    
    # Check directory structure for hints
    path_lower = str(training_dir).lower()
    if 'character' in path_lower or 'model-' in path_lower:
        char_score += 2
    elif 'style' in path_lower or 'photographer' in path_lower:
        style_score += 2
    elif 'concept' in path_lower or 'object' in path_lower:
        concept_score += 2
    
    # Determine training type
    scores = {'character': char_score, 'style': style_score, 'concept': concept_score}
    training_type = max(scores, key=scores.get)
    
    # Fallback to character if tied
    if char_score == style_score == concept_score:
        training_type = "character"
    
    is_character = training_type == "character"
    
    logger.info(f"Auto-detected training type: {training_type}")
    logger.info(f"Scores - Character: {char_score}, Style: {style_score}, Concept: {concept_score}")
    
    return is_character, training_type

def calculate_optimal_repeats(num_images, training_type, epochs=6, target_steps_range=(800, 1200)):
    """Calculate optimal repeat count based on training parameters"""
    
    # Target steps by training type
    type_targets = {
        'character': 1000,  # Need good identity learning
        'style': 800,       # Less repetition to avoid overfitting style quirks  
        'concept': 1200     # Abstract concepts need more exposure
    }
    
    target_steps = type_targets.get(training_type, 1000)
    
    # Calculate base repeats
    base_repeats = target_steps // (num_images * epochs)
    
    # Apply training type specific adjustments
    if training_type == "character":
        # Characters need consistent identity learning
        min_repeats, max_repeats = 8, 25
    elif training_type == "style":
        # Styles should avoid overfitting to specific compositions
        min_repeats, max_repeats = 4, 15
    elif training_type == "concept":
        # Concepts need more repetition to learn abstract features
        min_repeats, max_repeats = 10, 30
    else:
        # Default fallback
        min_repeats, max_repeats = 8, 20
    
    # Adjust based on dataset size
    if num_images <= 5:
        base_repeats = int(base_repeats * 1.5)  # Boost for small datasets
    elif num_images >= 20:
        base_repeats = int(base_repeats * 0.8)  # Reduce for large datasets
    
    # Clamp to reasonable range
    optimal_repeats = max(min_repeats, min(base_repeats, max_repeats))
    
    total_steps = num_images * optimal_repeats * epochs
    
    logger.info(f"Repeat calculation for {training_type} training:")
    logger.info(f"  Dataset: {num_images} images, {epochs} epochs")
    logger.info(f"  Optimal repeats: {optimal_repeats}")
    logger.info(f"  Total training steps: {total_steps}")
    
    return optimal_repeats

def setup_dreambooth_regularization(config, training_type, dataset_path):
    """Setup regularization images for DreamBooth character training"""
    
    if training_type != "character":
        return None
        
    # Enable DreamBooth with prior preservation for characters
    config.with_prior_preservation = True
    
    reg_dir = Path("/tmp/regularization_images")
    reg_dir.mkdir(exist_ok=True)
    
    # Extract class name from instance prompt
    # "a photo of john person" -> "person"
    words = config.instance_prompt.lower().split()
    class_word = "person"  # default
    
    if "person" in words:
        class_word = "person"
    elif "man" in words:
        class_word = "man"
    elif "woman" in words:
        class_word = "woman"
    
    config.class_prompt = f"a photo of a {class_word}"
    config.class_data_dir = str(reg_dir)
    
    logger.info(f"DreamBooth enabled with class prompt: {config.class_prompt}")
    logger.info(f"Regularization directory: {config.class_data_dir}")
    
    return str(reg_dir)

def main():
    # Immediate startup confirmation with flush
    print("=== KOHYA TRAINING WRAPPER STARTED ===")
    print(f"Python version: {sys.version}")
    print(f"Script location: {__file__}")
    print(f"Command line args: {sys.argv}")
    print(f"Number of command line args: {len(sys.argv)}")
    print(f"=== HYPERPARAMETERS DEBUG ===")
    try:
        with open("/opt/ml/input/config/hyperparameters.json", "r") as f:
            hyperparams = json.load(f)
        print(f"Hyperparameters file found with {len(hyperparams)} parameters:")
        for key, val in hyperparams.items():
            print(f"  {key}={val}")
    except FileNotFoundError:
        print("  NO hyperparameters.json file found!")
    except Exception as e:
        print(f"  Error reading hyperparameters.json: {e}")
        
    print(f"Environment SM_HP vars (legacy check):")
    hp_found = False
    for key, val in os.environ.items():
        if key.startswith('SM_HP_'):
            print(f"  {key}={val}")
            hp_found = True
    if not hp_found:
        print("  NO SM_HP_* variables found!")
    sys.stdout.flush()
    
    logger.info("=== KOHYA TRAINING WRAPPER STARTED ===")
    
    config = get_config()  # This will now raise ValueError if instance-prompt is missing
    
    # Debug environment
    print("=== ENVIRONMENT VARIABLES ===")
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if key.startswith('SM_'):
            print(f"{key}={value}")
            logger.info(f"  {key}={value}")
    sys.stdout.flush()
    
    # Give CloudWatch time to catch up
    print("=== STARTUP DELAY FOR CLOUDWATCH ===")
    for i in range(5):
        print(f"Startup check {i+1}/5...")
        sys.stdout.flush()
        time.sleep(1)
    
    # Ensure model_dir is set
    if not config.model_dir:
        logger.error("Missing model_dir (SM_MODEL_DIR is not set). Cannot write model output.")
        sys.exit(1)
    
    logger.info("Starting Kohya-based SDXL training")
    logger.info(f"Training directory: {config.training_dir}")
    logger.info(f"Model output directory: {config.model_dir}")
    logger.info(f"Instance prompt: {config.instance_prompt}")
    
    # Check if directories exist
    if not os.path.exists(config.training_dir):
        logger.error(f"Training directory does not exist: {config.training_dir}")
        sys.exit(1)
    
    if not os.path.exists(config.model_dir):
        logger.info(f"Creating model directory: {config.model_dir}")
        os.makedirs(config.model_dir, exist_ok=True)
    
    # Auto-detect training type
    is_character, training_type = detect_training_type(config.instance_prompt, config.training_dir)
    
    # Prepare dataset
    dataset_path = prepare_dataset(config.training_dir, config.instance_prompt)
    
    # Setup DreamBooth for character training
    reg_dir = setup_dreambooth_regularization(config, training_type, dataset_path)
    
    # Adjust hyperparameters based on type
    if is_character:
        # Character training: Lower LR, more epochs, DreamBooth
        config.learning_rate = max(config.learning_rate * 0.5, 5e-5)  # Lower LR for faces
        config.num_train_epochs = max(config.num_train_epochs, 6)     # More epochs
        config.lora_rank = max(config.lora_rank, 64)                  # Higher rank for details
        logger.info("Optimized for character training")
    else:
        # Style training: Standard settings, no DreamBooth
        config.with_prior_preservation = False
        logger.info("Optimized for style training")
    
    # Run training
    success = run_kohya_training(config, dataset_path)
    
    # CRITICAL: ALWAYS create debug info in /opt/ml/model regardless of training outcome
    output_dir = Path(config.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, let's see what files exist in /tmp ALWAYS
    logger.info("=== MANDATORY DEBUG: CHECKING ALL LOCATIONS FOR FILES ===")
    tmp_files = list(Path("/tmp").rglob("*.safetensors"))
    tmp_files.extend(list(Path("/tmp").rglob("*.ckpt")))
    tmp_files.extend(list(Path("/tmp").rglob("*.pt")))
    
    # Create comprehensive debug file ALWAYS (this ensures S3 upload happens)
    debug_file = output_dir / "training_debug.txt"
    with open(debug_file, "w") as f:
        f.write("=== KOHYA TRAINING DEBUG REPORT ===\n")
        f.write(f"Training script ran successfully: {success}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
        f.write(f"Model directory: {config.model_dir}\n")
        f.write(f"Training directory: {config.training_dir}\n")
        f.write(f"Kohya output directory: {config.kohya_output_dir}\n")
        f.write(f"Instance prompt: {config.instance_prompt}\n")
        f.write(f"\nFiles found in /tmp: {len(tmp_files)}\n")
        for f_tmp in tmp_files:
            f.write(f"  - {f_tmp} ({f_tmp.stat().st_size} bytes)\n")
        
        # Check kohya output directory
        lora_output_dir = Path(config.kohya_output_dir)
        if lora_output_dir.exists():
            kohya_files = list(lora_output_dir.rglob("*"))
            f.write(f"\nFiles in Kohya output dir ({lora_output_dir}): {len(kohya_files)}\n")
            for kf in kohya_files:
                if kf.is_file():
                    f.write(f"  - {kf} ({kf.stat().st_size} bytes)\n")
        else:
            f.write(f"\nKohya output directory {lora_output_dir} does not exist\n")
        
        # Log environment variables
        f.write(f"\nEnvironment variables:\n")
        for key, value in os.environ.items():
            if key.startswith('SM_'):
                f.write(f"  {key}={value}\n")
        
        f.write(f"\nArguments: {vars(config)}\n")
    
    # Ensure proper permissions on debug file
    import stat
    os.chmod(debug_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    logger.info(f"✅ ALWAYS created debug file: {debug_file}")
    
    if success:
        logger.info("Training completed successfully!")
        
        # Copy models to SageMaker output directory
        output_dir = Path(config.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and copy LoRA files from Kohya output
        lora_output_dir = Path(config.kohya_output_dir)
        
        # First, let's see what files exist in /tmp
        logger.info("=== CHECKING /tmp FOR ANY OUTPUT FILES ===")
        tmp_files = list(Path("/tmp").rglob("*.safetensors"))
        tmp_files.extend(list(Path("/tmp").rglob("*.ckpt")))
        tmp_files.extend(list(Path("/tmp").rglob("*.pt")))
        logger.info(f"Found {len(tmp_files)} model files in /tmp:")
        for f in tmp_files:
            logger.info(f"  - {f} ({f.stat().st_size} bytes)")
        
        # Also check the current working directory
        cwd_files = list(Path(".").rglob("*.safetensors"))
        cwd_files.extend(list(Path(".").rglob("*.ckpt")))
        if cwd_files:
            logger.info(f"Found {len(cwd_files)} model files in current dir:")
            for f in cwd_files:
                logger.info(f"  - {f} ({f.stat().st_size} bytes)")
        
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
        import stat
        model_files_copied = 0
        copied_files = []
        
        for model_file in lora_output_dir.rglob("*.safetensors"):
            dest_file = output_dir / model_file.name
            shutil.copy2(model_file, dest_file)
            # Ensure proper permissions
            os.chmod(dest_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            logger.info(f"Copied {model_file.name} to {dest_file} ({dest_file.stat().st_size} bytes)")
            model_files_copied += 1
            copied_files.append(dest_file)
            
        # Also copy any .ckpt files
        for model_file in lora_output_dir.rglob("*.ckpt"):
            dest_file = output_dir / model_file.name
            shutil.copy2(model_file, dest_file)
            # Ensure proper permissions
            os.chmod(dest_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            logger.info(f"Copied {model_file.name} to {dest_file} ({dest_file.stat().st_size} bytes)")
            model_files_copied += 1
            copied_files.append(dest_file)
            
        if model_files_copied == 0:
            logger.error("No model files found in expected location!")
            logger.info("Attempting fallback: searching entire /tmp for model files...")
            
            # Fallback: copy ANY model files found in /tmp
            for tmp_file in tmp_files:
                if tmp_file.stat().st_size > 1000:  # Only copy files > 1KB
                    dest_file = output_dir / tmp_file.name
                    shutil.copy2(tmp_file, dest_file)
                    # Ensure proper permissions
                    os.chmod(dest_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                    logger.info(f"Fallback: Copied {tmp_file} to {dest_file} ({dest_file.stat().st_size} bytes)")
                    model_files_copied += 1
                    copied_files.append(dest_file)
            
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
                    f.write(f"Training completed: {success}\n")
                    f.write(f"Args: {vars(config)}\n")
                # Ensure proper permissions
                os.chmod(debug_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                logger.info(f"Created debug file: {debug_file}")
                copied_files.append(debug_file)
        else:
            logger.info(f"Successfully copied {model_files_copied} model file(s)")
        
        # Create metadata
        metadata = {
            "model_type": "SDXL LoRA",
            "training_method": "Kohya DreamBooth + LoRA",
            "instance_prompt": config.instance_prompt,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "training_steps": config.max_train_steps or (config.num_train_epochs * 100),  # estimate
            "base_model": config.pretrained_model_name,
            "files_copied": [str(f.name) for f in copied_files],
            "total_files": len(copied_files)
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        # Ensure proper permissions
        os.chmod(metadata_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        
        # CRITICAL: Validate /opt/ml/model contents before finishing
        logger.info("=== FINAL VALIDATION OF /opt/ml/model ===")
        final_files = list(output_dir.glob("*"))
        logger.info(f"Final file count in {output_dir}: {len(final_files)}")
        
        total_size = 0
        for f in final_files:
            if f.is_file():
                size = f.stat().st_size
                total_size += size
                logger.info(f"  ✓ {f.name}: {size} bytes, permissions: {oct(f.stat().st_mode)}")
        
        logger.info(f"Total size of all files in /opt/ml/model: {total_size} bytes")
        
        if total_size == 0:
            logger.error("❌ CRITICAL: /opt/ml/model is empty! SageMaker will not upload anything to S3!")
            return False
        else:
            logger.info(f"✅ SUCCESS: /opt/ml/model contains {len(final_files)} files ({total_size} bytes total)")
        
        # Force filesystem sync
        logger.info("Forcing filesystem sync...")
        os.sync()
        
        return True
    else:
        logger.error("Training failed!")
        # Even on failure, create debug info in /opt/ml/model
        output_dir = Path(config.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        failure_file = output_dir / "training_failure.txt"
        with open(failure_file, "w") as f:
            f.write("Training failed\n")
            f.write(f"Args: {vars(config)}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write("Check CloudWatch logs for details\n")
        
        # Force filesystem sync before exit
        logger.info("Forcing filesystem sync after failure...")
        os.sync()
        time.sleep(2)
        
        return False

def cleanup_and_exit(success, config):
    """Proper cleanup and exit with SageMaker synchronization"""
    import time
    
    logger.info("=== FINAL CLEANUP AND SYNCHRONIZATION ===")
    
    # CRITICAL: Final comprehensive model directory listing
    if hasattr(config, 'model_dir') and config.model_dir:
        model_dir = Path(config.model_dir)
        if model_dir.exists():
            files = list(model_dir.glob("*"))
            logger.info(f"Final model directory contains {len(files)} files:")
            
            # Print to both logger AND stdout for absolute visibility
            print("=== Final model directory listing ===")
            total_size = 0
            for f in files:
                if f.is_file():
                    size = f.stat().st_size
                    total_size += size
                    logger.info(f"  - {f.name}: {size} bytes, permissions: {oct(f.stat().st_mode)}")
                    print(f"  - {f.name}: {size} bytes")
                else:
                    logger.info(f"  - {f.name}: <directory>")
                    print(f"  - {f.name}: <directory>")
            
            logger.info(f"✅ TOTAL SIZE IN /opt/ml/model: {total_size} bytes")
            print(f"✅ TOTAL SIZE IN /opt/ml/model: {total_size} bytes")
            
            if total_size == 0:
                logger.error("❌ CRITICAL: /opt/ml/model is empty! No S3 upload will occur!")
                print("❌ CRITICAL: /opt/ml/model is empty! No S3 upload will occur!")
        else:
            logger.error(f"❌ Model directory {model_dir} does not exist!")
            print(f"❌ Model directory {model_dir} does not exist!")
    
    # Also check for any other common model locations as fallback
    other_locations = ["/tmp", "/opt/ml/code", "/kohya"]
    for location in other_locations:
        location_path = Path(location)
        if location_path.exists():
            model_files = []
            model_files.extend(list(location_path.rglob("*.safetensors")))
            model_files.extend(list(location_path.rglob("*.ckpt")))
            model_files.extend(list(location_path.rglob("*.pt")))
            if model_files:
                logger.info(f"Model files found in {location}: {len(model_files)}")
                print(f"Model files found in {location}: {len(model_files)}")
                for mf in model_files[:5]:  # Show first 5
                    logger.info(f"  - {mf}: {mf.stat().st_size} bytes")
                    print(f"  - {mf}: {mf.stat().st_size} bytes")
    
    # Force all pending writes to disk
    logger.info("Performing final filesystem sync...")
    os.sync()
    
    # Flush all log handlers
    logger.info("Flushing all log handlers...")
    for handler in logging.getLogger().handlers:
        handler.flush()
    
    # Give SageMaker time to process logs and sync files
    logger.info("Final wait to allow SageMaker to flush logs and sync model directory...")
    print("Final wait to allow SageMaker to flush logs and sync model directory...")
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Critical: Wait for SageMaker to sync /opt/ml/model to S3
    time.sleep(15)  # Increased from 10 to 15 seconds
    
    if success:
        logger.info("✅ Training completed successfully - exiting with code 0")
        print("✅ Training completed successfully - exiting with code 0")
        sys.exit(0)
    else:
        logger.error("❌ Training failed - exiting with code 1")
        print("❌ Training failed - exiting with code 1")
        sys.exit(1)

if __name__ == "__main__":
    try:
        result = main()
        cleanup_and_exit(result, get_config())
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        import traceback
        
        # Print to both logger and stdout for visibility
        print("\n=== UNHANDLED EXCEPTION ===")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()
        
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Try to create error file in model dir
        try:
            config = get_config()
            if hasattr(config, 'model_dir') and config.model_dir:
                output_dir = Path(config.model_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                error_file = output_dir / "training_error.txt"
                with open(error_file, "w") as f:
                    f.write(f"Unhandled exception: {type(e).__name__}: {e}\n\n")
                    f.write("=== FULL TRACEBACK ===\n")
                    f.write(traceback.format_exc())
                    f.write("\n\n=== ENVIRONMENT ===\n")
                    for key, val in os.environ.items():
                        if key.startswith('SM_') or key.startswith('PYTHON'):
                            f.write(f"{key}={val}\n")
                
                logger.info(f"Created error file: {error_file}")
        except Exception as inner_e:
            logger.error(f"Failed to create error file: {inner_e}")
        
        cleanup_and_exit(False, config if 'config' in locals() else None)