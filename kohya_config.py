"""
Generate Kohya configuration files for SDXL training
"""

import os
import re
from pathlib import Path

def extract_model_name(instance_prompt, dataset_path):
    """Extract model name from instance prompt or dataset path"""
    
    # First try to extract from instance prompt
    # "a photo of franka person" -> "franka"
    # "a photo of sinisha style" -> "sinisha"
    prompt_match = re.search(r'a\s+photo\s+of\s+(\w+)', instance_prompt.lower())
    if prompt_match:
        return prompt_match.group(1)
    
    # Fallback to dataset path
    # "training/character/model-franka/dataset" -> "franka"
    # "training/style/photographer-sinisha/dataset" -> "sinisha"
    path_str = str(dataset_path).lower()
    
    # Look for model-X or photographer-X patterns
    model_match = re.search(r'(?:model|photographer)-(\w+)', path_str)
    if model_match:
        return model_match.group(1)
    
    # Look for character names in path segments
    path_parts = Path(dataset_path).parts
    for part in reversed(path_parts):
        if part.lower() not in ['dataset', 'training', 'character', 'style', 'conditioning']:
            # Clean up the part (remove prefixes)
            clean_name = re.sub(r'^(?:model|photographer)-', '', part.lower())
            if clean_name and len(clean_name) > 2:
                return clean_name
    
    # Final fallback
    return "sdxl"

def create_kohya_config(args, dataset_path, config_path, output_dir=None):
    """Create a Kohya TOML config file"""
    
    # Extract model name from instance prompt or dataset path
    model_name = extract_model_name(args.instance_prompt, dataset_path)
    
    # Use provided output_dir or fallback to kohya_output_dir
    kohya_output_dir = output_dir or getattr(args, 'kohya_output_dir', '/tmp/lora_models')
    
    # Calculate training steps
    if args.max_train_steps:
        max_train_steps = args.max_train_steps
    else:
        # Estimate based on dataset size and epochs
        image_count = len(list(Path(dataset_path).glob("*.jpg"))) + \
                     len(list(Path(dataset_path).glob("*.png"))) + \
                     len(list(Path(dataset_path).glob("*.jpeg")))
        max_train_steps = (image_count * args.num_train_epochs) // args.train_batch_size
    
    # Warmup steps (10% of total)
    warmup_steps = args.lr_warmup_steps or max(1, max_train_steps // 10)
    
    config = f"""
[model_arguments]
pretrained_model_name_or_path = "{args.pretrained_model_name}"
v2 = false
v_parameterization = false

[additional_network_arguments]
network_module = "networks.lora"
network_dim = {args.lora_rank}
network_alpha = {args.lora_alpha}
network_dropout = {args.lora_dropout}
network_args = []
network_train_unet_only = true
network_train_text_encoder_only = false

[optimizer_arguments]
optimizer_type = "{args.optimizer}"
learning_rate = {args.learning_rate}
max_grad_norm = 1.0
optimizer_args = []
lr_scheduler = "{args.lr_scheduler}"
lr_warmup_steps = {warmup_steps}

[dataset_arguments]
train_data_dir = "{dataset_path}"
resolution = "1024,1024"
batch_size = {args.train_batch_size}
enable_bucket = true
bucket_resolution_steps = 64
bucket_no_upscale = false
caption_extension = "{args.caption_extension}"
shuffle_caption = {str(args.shuffle_caption).lower()}
cache_latents = {str(args.cache_latents).lower()}
cache_latents_to_disk = {str(args.cache_latents_to_disk).lower()}
color_aug = {str(args.color_aug).lower()}
flip_aug = {str(args.flip_aug).lower()}
random_crop = false
debug_dataset = false

[training_arguments]
output_dir = "{kohya_output_dir}"
output_name = "{model_name}_lora"
save_precision = "fp16"
save_every_n_epochs = 1
save_every_n_steps = 200
save_last_n_steps = 0
save_state = false
resume = ""
train_batch_size = {args.train_batch_size}
max_train_steps = {max_train_steps}
max_train_epochs = {args.num_train_epochs}
max_data_loader_n_workers = 8
persistent_data_loader_workers = true
seed = {args.seed}
gradient_checkpointing = {str(args.gradient_checkpointing).lower()}
gradient_accumulation_steps = 1
mixed_precision = "{args.mixed_precision}"
clip_skip = {args.clip_skip}
logging_dir = "/tmp/logs"
log_with = "tensorboard"
log_prefix = "sdxl_lora"

[sample_prompt_arguments]
sample_every_n_steps = 100
sample_every_n_epochs = 0
sample_sampler = "euler_a"
sample_prompts = [
    "{args.instance_prompt}",
    "{args.instance_prompt}, high quality, detailed",
    "{args.instance_prompt}, portrait, studio lighting"
]

[dreambooth_arguments]
prior_loss_weight = {args.prior_loss_weight}
with_prior_preservation = {str(args.with_prior_preservation).lower()}
class_data_dir = "{getattr(args, 'class_data_dir', '')}"
num_class_images = {getattr(args, 'num_class_images', 200)}

[sdxl_arguments]
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = false
no_half_vae = true
"""

    # Add xformers if enabled
    if args.enable_xformers:
        config += "\n[memory_arguments]\nxformers = true\n"
    
    # Write config file
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config.strip())
    
    print(f"Generated Kohya config at: {config_path}")
    return config_path