#!/usr/bin/env python3
"""
Simplified automated captioning using Hugging Face transformers only
Works with locally available packages
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile

class SimpleDatasetCaptioner:
    def __init__(self):
        """Initialize with basic BLIP model"""
        print("Loading BLIP model for image captioning...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("Model loaded successfully!")
        
    def generate_caption(self, image_path: str, trigger_word: str) -> str:
        """Generate caption for a single image"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Generate caption with BLIP
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Clean and format caption
            caption = caption.strip()
            if caption.lower().startswith('a photo of'):
                caption = caption[10:].strip()
            elif caption.lower().startswith('a picture of'):
                caption = caption[12:].strip()
            elif caption.lower().startswith('an image of'):
                caption = caption[11:].strip()
            
            # Create structured caption with trigger word
            formatted_caption = self._format_caption(caption, trigger_word)
            
            return formatted_caption
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return f"{trigger_word}, professional photo"
    
    def _format_caption(self, raw_caption: str, trigger_word: str) -> str:
        """Format raw caption into structured format"""
        parts = [trigger_word]
        
        # Extract meaningful parts from BLIP caption
        caption_lower = raw_caption.lower()
        
        # Look for pose/position keywords
        pose_keywords = ['sitting', 'standing', 'looking', 'wearing', 'holding', 'walking', 'smiling', 'posing']
        for keyword in pose_keywords:
            if keyword in caption_lower:
                # Find the phrase containing this keyword
                words = raw_caption.split()
                for i, word in enumerate(words):
                    if keyword in word.lower():
                        # Get some context around the keyword
                        start = max(0, i-1)
                        end = min(len(words), i+3)
                        phrase = ' '.join(words[start:end])
                        if len(phrase) < 50:  # Reasonable length
                            parts.append(phrase.lower())
                        break
                break
        
        # Add some descriptive elements from the caption
        if 'smile' in caption_lower or 'smiling' in caption_lower:
            parts.append('smiling expression')
        elif 'serious' in caption_lower:
            parts.append('serious expression')
        else:
            parts.append('neutral expression')
        
        # Look for clothing/appearance
        clothing_keywords = ['dress', 'shirt', 'jacket', 'suit', 'clothing', 'outfit', 'wearing']
        for keyword in clothing_keywords:
            if keyword in caption_lower:
                parts.append(f"wearing {keyword}")
                break
        
        # Add lighting inference (basic)
        if 'bright' in caption_lower or 'light' in caption_lower:
            parts.append('bright lighting')
        elif 'dark' in caption_lower:
            parts.append('dramatic lighting')
        else:
            parts.append('natural lighting')
        
        # Join parts with commas, limit to reasonable length
        formatted = ', '.join(parts[:5])  # Max 5 parts
        
        return formatted

def process_s3_images(bucket_name: str, prefix: str, trigger_word: str, max_images: int = None):
    """Process images from S3 with improved captions"""
    import boto3
    
    # Initialize S3 client
    session = boto3.Session(profile_name='raffibag')
    s3 = session.client('s3', region_name='us-west-2')
    
    # Initialize captioner
    captioner = SimpleDatasetCaptioner()
    
    # List images in S3
    paginator = s3.get_paginator('list_objects_v2')
    image_keys = []
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_keys.append(obj['Key'])
    
    if max_images:
        image_keys = image_keys[:max_images]
    
    print(f"Found {len(image_keys)} images to process")
    
    results = {
        "processed_images": {},
        "errors": []
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for i, image_key in enumerate(image_keys):
            try:
                print(f"Processing {i+1}/{len(image_keys)}: {image_key}")
                
                # Download image
                filename = image_key.split('/')[-1]
                local_path = temp_path / filename
                s3.download_file(bucket_name, image_key, str(local_path))
                
                # Generate better caption
                caption = captioner.generate_caption(str(local_path), trigger_word)
                
                # Save caption back to S3
                caption_key = image_key.replace(Path(image_key).suffix, '.txt')
                s3.put_object(
                    Bucket=bucket_name,
                    Key=caption_key,
                    Body=caption
                )
                
                results["processed_images"][image_key] = {
                    "caption": caption,
                    "source": "blip_improved"
                }
                
                print(f"  Generated: {caption}")
                
            except Exception as e:
                error_msg = f"Error processing {image_key}: {str(e)}"
                print(f"  {error_msg}")
                results["errors"].append(error_msg)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved automated captioning")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("prefix", help="S3 prefix/path")
    parser.add_argument("trigger_word", help="Trigger word for captions")
    parser.add_argument("--max-images", type=int, help="Maximum images to process")
    
    args = parser.parse_args()
    
    results = process_s3_images(
        args.bucket, 
        args.prefix, 
        args.trigger_word,
        args.max_images
    )
    
    print(f"\n=== Results ===")
    print(f"Processed: {len(results['processed_images'])}")
    print(f"Errors: {len(results['errors'])}")