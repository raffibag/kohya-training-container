#!/usr/bin/env python3
"""
Automated captioning and labeling for S3 datasets
Runs inside the training container with full ML dependencies
"""

import os
import json
import boto3
import yaml
from pathlib import Path
from typing import Dict, List
import tempfile
import random
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip

# Default schema embedded in the script
DEFAULT_SCHEMA = {
    "photographer_style_schema": {
        "lighting_fundamentals": [
            "natural_sunlight", "golden_hour", "blue_hour", "overcast_diffused",
            "studio_strobe", "continuous_tungsten", "mixed_lighting"
        ],
        "lighting_quality": [
            "hard_directional", "soft_diffused", "high_contrast", "low_contrast",
            "dramatic_shadows", "rim_backlighting", "chiaroscuro"
        ],
        "composition_rules": [
            "rule_of_thirds", "golden_ratio", "centered_symmetry", "leading_lines",
            "negative_space", "frame_within_frame"
        ],
        "camera_angles": [
            "eye_level_standard", "low_angle_heroic", "high_angle_down",
            "dutch_tilt_dynamic", "birds_eye_overhead", "extreme_close_detail"
        ],
        "color_treatment": [
            "natural_realistic", "warm_golden_tones", "cool_blue_tones",
            "cinematic_teal_orange", "moody_desaturated", "high_saturation_vivid"
        ]
    },
    "model_actor_schema": {
        "facial_expression": [
            "neutral", "genuine_smile", "laughing", "serious", "contemplative",
            "confident", "playful", "intense", "relaxed", "focused"
        ],
        "body_pose": [
            "standing_straight", "casual_lean", "sitting", "walking", "running",
            "arms_crossed", "hands_on_hips", "hand_to_face", "looking_over_shoulder"
        ],
        "hair_style": [
            "long_straight", "long_wavy", "short_styled", "ponytail", "bun",
            "loose_flowing", "braided", "slicked_back"
        ],
        "clothing_style": [
            "casual_wear", "formal_attire", "athletic_wear", "business_casual",
            "evening_dress", "streetwear", "minimalist", "layered_outfit"
        ]
    }
}

class ContainerDatasetProcessor:
    def __init__(self, 
                 bucket_name: str,
                 schema_path: str = None,
                 profile: str = "default",
                 region: str = "us-west-2"):
        """Initialize processor with AWS and ML models"""
        
        # AWS setup
        if profile != "default":
            session = boto3.Session(profile_name=profile)
            self.s3 = session.client('s3', region_name=region)
        else:
            self.s3 = boto3.client('s3', region_name=region)
        
        self.bucket_name = bucket_name
        
        # Initialize ML models
        print("🤖 Loading BLIP model for captioning...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        print("🤖 Loading CLIP model for labeling...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load schema - try file first, then S3, then use default
        self.schema_labels = {}
        schemas = None
        
        if schema_path:
            if schema_path.startswith('s3://'):
                # Download from S3
                print(f"📥 Downloading schema from {schema_path}")
                try:
                    parts = schema_path.replace('s3://', '').split('/', 1)
                    bucket = parts[0]
                    key = parts[1]
                    
                    response = self.s3.get_object(Bucket=bucket, Key=key)
                    schemas = yaml.safe_load(response['Body'].read())
                    print("✅ Schema loaded from S3")
                except Exception as e:
                    print(f"⚠️ Failed to load schema from S3: {e}")
            elif os.path.exists(schema_path):
                # Load from local file
                with open(schema_path, 'r') as f:
                    schemas = yaml.safe_load(f)
                print(f"✅ Schema loaded from file: {schema_path}")
        
        # Use default schema if nothing loaded
        if not schemas:
            print("📋 Using default embedded schema")
            schemas = DEFAULT_SCHEMA
        
        self.schema_labels = self._prepare_schema_labels(schemas)
        print(f"📋 Loaded {len(self.schema_labels)} schema labels")
        print("✅ All models loaded successfully!")
    
    def _prepare_schema_labels(self, schemas: Dict) -> Dict[str, str]:
        """Convert schema to CLIP-friendly prompts"""
        labels = {}
        
        # Process both photographer and model schemas
        for schema_name in ['photographer_style_schema', 'model_actor_schema']:
            if schema_name in schemas:
                schema = schemas[schema_name]
                for category, items in schema.items():
                    if isinstance(items, list):
                        for item in items:
                            # Convert label to natural language prompt
                            prompt = self._label_to_prompt(item, category)
                            labels[item] = prompt
        
        return labels
    
    def _label_to_prompt(self, label: str, category: str) -> str:
        """Convert schema label to CLIP prompt"""
        clean_label = label.replace('_', ' ')
        
        if 'lighting' in category:
            return f"a photo with {clean_label} lighting"
        elif 'pose' in category or 'geometry' in category:
            return f"a person in {clean_label} pose"
        elif 'expression' in category:
            return f"a person with {clean_label}"
        elif 'hair' in category:
            return f"a person with {clean_label} hair"
        elif 'composition' in category:
            return f"a photo using {clean_label} composition"
        elif 'color' in category:
            return f"a photo with {clean_label} color grading"
        elif 'angle' in category:
            return f"a photo taken from {clean_label}"
        elif 'clothing' in category:
            return f"a person wearing {clean_label}"
        else:
            return f"a photo showing {clean_label}"
    
    def generate_caption(self, image_path: str, trigger_word: str) -> str:
        """Generate detailed caption using BLIP"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Generate caption with BLIP
            inputs = self.blip_processor(image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                self.blip_model = self.blip_model.cuda()
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
            
            raw_caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Format into structured caption
            formatted_caption = self._format_caption(raw_caption, trigger_word)
            
            return formatted_caption
            
        except Exception as e:
            print(f"⚠️  Caption generation failed: {e}")
            return f"{trigger_word}, professional photo"
    
    def generate_labels(self, image_path: str, confidence_threshold: float = 0.3) -> Dict:
        """Generate schema labels using CLIP"""
        # Always return the expected structure
        result = {
            "labels_detected": {},
            "confidence_scores": {}
        }
        
        if not self.schema_labels:
            return result
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Prepare text prompts
            labels = list(self.schema_labels.keys())
            prompts = [self.schema_labels[label] for label in labels]
            text_tokens = clip.tokenize(prompts).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize features for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate raw cosine similarities (better for multi-label)
                similarities = (image_features @ text_features.T).cpu().numpy()[0]
            
            # Extract labels above threshold
            for i, (label, confidence) in enumerate(zip(labels, similarities)):
                result["confidence_scores"][label] = float(confidence)
                if confidence >= confidence_threshold:
                    result["labels_detected"][label] = float(confidence)
            
            return result
            
        except Exception as e:
            print(f"⚠️  Label generation failed: {e}")
            return result
    
    def _format_caption(self, raw_caption: str, trigger_word: str) -> str:
        """Format BLIP caption into structured format"""
        # Clean up BLIP output
        caption = raw_caption.strip()
        if caption.lower().startswith('a photo of'):
            caption = caption[10:].strip()
        elif caption.lower().startswith('a picture of'):
            caption = caption[12:].strip()
        
        # Start with trigger word
        parts = [trigger_word]
        
        # Extract meaningful elements
        caption_lower = caption.lower()
        
        # Look for pose/action
        pose_indicators = ['sitting', 'standing', 'looking', 'smiling', 'walking', 'posing', 'holding']
        for indicator in pose_indicators:
            if indicator in caption_lower:
                parts.append(f"{indicator}")
                break
        
        # Add expression
        if 'smiling' in caption_lower or 'smile' in caption_lower:
            parts.append('smiling expression')
        elif 'serious' in caption_lower:
            parts.append('serious expression')
        else:
            parts.append('neutral expression')
        
        # Add appearance details from original caption
        if 'hair' in caption_lower:
            words = caption.split()
            for i, word in enumerate(words):
                if 'hair' in word.lower():
                    # Get surrounding context
                    start = max(0, i-2)
                    end = min(len(words), i+2)
                    hair_desc = ' '.join(words[start:end])
                    if len(hair_desc) < 30:
                        parts.append(hair_desc.lower())
                    break
        
        # Add lighting/setting inference
        if 'bright' in caption_lower or 'outdoor' in caption_lower:
            parts.append('natural lighting')
        elif 'dark' in caption_lower or 'studio' in caption_lower:
            parts.append('studio lighting')
        else:
            parts.append('soft lighting')
        
        # Add clothing if mentioned
        clothing_words = ['wearing', 'dress', 'shirt', 'suit', 'jacket']
        for word in clothing_words:
            if word in caption_lower:
                clothing_start = caption_lower.find(word)
                clothing_part = caption[clothing_start:clothing_start+20]
                parts.append(clothing_part.strip())
                break
        
        # Join and limit length
        formatted = ', '.join(parts[:6])  # Max 6 parts
        
        return formatted
    
    def process_s3_dataset(self, 
                          s3_prefix: str,
                          trigger_word: str,
                          automated_percentage: float = 0.8,
                          max_images: int = None) -> Dict:
        """Process entire S3 dataset with captioning and labeling"""
        
        print(f"🔍 Scanning S3 path: s3://{self.bucket_name}/{s3_prefix}")
        
        # List all images
        paginator = self.s3.get_paginator('list_objects_v2')
        image_keys = []
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        # Skip if caption already exists and is recent
                        caption_key = key.replace(Path(key).suffix, '.txt')
                        try:
                            caption_obj = self.s3.head_object(Bucket=self.bucket_name, Key=caption_key)
                            # If caption exists and was created after the script's improvement, skip
                            continue
                        except:
                            # Caption doesn't exist, process this image
                            image_keys.append(key)
        
        if max_images:
            image_keys = image_keys[:max_images]
        
        # Shuffle and split
        random.shuffle(image_keys)
        num_automated = int(len(image_keys) * automated_percentage)
        automated_keys = image_keys[:num_automated]
        manual_keys = image_keys[num_automated:]
        
        print(f"📊 Found {len(image_keys)} images")
        print(f"📊 Processing {len(automated_keys)} automatically, {len(manual_keys)} reserved for manual")
        
        results = {
            "total_images": len(image_keys),
            "automated_processed": 0,
            "manual_reserved": len(manual_keys),
            "processed_details": {},
            "errors": []
        }
        
        # Process automated portion
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i, image_key in enumerate(automated_keys):
                try:
                    print(f"🖼️  Processing {i+1}/{len(automated_keys)}: {Path(image_key).name}")
                    
                    # Download image
                    filename = Path(image_key).name
                    local_path = temp_path / filename
                    self.s3.download_file(self.bucket_name, image_key, str(local_path))
                    
                    # Generate caption and labels
                    caption = self.generate_caption(str(local_path), trigger_word)
                    label_results = self.generate_labels(str(local_path))
                    
                    # Save caption to S3
                    caption_key = image_key.replace(Path(image_key).suffix, '.txt')
                    self.s3.put_object(
                        Bucket=self.bucket_name,
                        Key=caption_key,
                        Body=caption
                    )
                    
                    # Save labels to S3 if any detected
                    if label_results["labels_detected"]:
                        labels_key = image_key.replace(Path(image_key).suffix, '_labels.json')
                        self.s3.put_object(
                            Bucket=self.bucket_name,
                            Key=labels_key,
                            Body=json.dumps(label_results, indent=2)
                        )
                    
                    results["processed_details"][image_key] = {
                        "caption": caption,
                        "labels": label_results["labels_detected"],
                        "source": "blip_clip_automated"
                    }
                    
                    results["automated_processed"] += 1
                    
                    print(f"   ✅ Caption: {caption}")
                    print(f"   🏷️  Labels: {list(label_results['labels_detected'].keys())}")
                    
                except Exception as e:
                    error_msg = f"Error processing {image_key}: {str(e)}"
                    print(f"   ❌ {error_msg}")
                    results["errors"].append(error_msg)
        
        # Save results manifest
        manifest_key = f"{s3_prefix}/improved_automated_results.json"
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=manifest_key,
            Body=json.dumps(results, indent=2)
        )
        
        print(f"\n🎉 Processing complete!")
        print(f"   ✅ Processed: {results['automated_processed']} images")
        print(f"   📋 Errors: {len(results['errors'])}")
        print(f"   📄 Manifest: s3://{self.bucket_name}/{manifest_key}")
        
        return results

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated dataset processing in training container")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("prefix", help="S3 prefix/path to images") 
    parser.add_argument("trigger_word", help="Trigger word for captions")
    parser.add_argument("--schema", help="Path to schema YAML file")
    parser.add_argument("--profile", default="default", help="AWS profile")
    parser.add_argument("--max-images", type=int, help="Max images to process")
    parser.add_argument("--automated-percent", type=float, default=0.8, help="Percentage for automated processing")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ContainerDatasetProcessor(
        bucket_name=args.bucket,
        schema_path=args.schema,
        profile=args.profile
    )
    
    # Process dataset
    results = processor.process_s3_dataset(
        s3_prefix=args.prefix,
        trigger_word=args.trigger_word,
        automated_percentage=args.automated_percent,
        max_images=args.max_images
    )
    
    print(f"\n📈 Final Results:")
    print(f"   Total images: {results['total_images']}")
    print(f"   Automated: {results['automated_processed']}")
    print(f"   Manual reserved: {results['manual_reserved']}")
    print(f"   Errors: {len(results['errors'])}")

if __name__ == "__main__":
    main()