#!/usr/bin/env python3
"""
Automated schema labeling using CLIP and vision models
Processes images against comprehensive_labeling_schemas.yaml attributes
"""

import yaml
import json
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import clip
import numpy as np
from transformers import pipeline

class AutomatedSchemaLabeler:
    def __init__(self, 
                 schema_path: str,
                 clip_model: str = "ViT-B/32",
                 confidence_threshold: float = 0.3):
        """
        Initialize automated labeling system
        
        Args:
            schema_path: Path to comprehensive_labeling_schemas.yaml
            clip_model: CLIP model to use
            confidence_threshold: Minimum confidence for label assignment
        """
        self.confidence_threshold = confidence_threshold
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # Load schemas
        with open(schema_path, 'r') as f:
            self.schemas = yaml.safe_load(f)
        
        # Prepare text prompts for each schema attribute
        self.label_prompts = self._prepare_label_prompts()
        
        print(f"Loaded {len(self.label_prompts)} label prompts from schema")
        
    def _prepare_label_prompts(self) -> Dict[str, Dict]:
        """Convert schema labels to CLIP-friendly text prompts"""
        prompts = {}
        
        # Process model/actor schema
        if 'model_actor_schema' in self.schemas:
            schema = self.schemas['model_actor_schema']
            prompts['model_actor'] = {}
            
            # Face geometry
            if 'face_geometry' in schema:
                for label in schema['face_geometry']:
                    prompts['model_actor'][label] = f"a photo showing {label.replace('_', ' ')}"
            
            # Facial expression  
            if 'facial_expression' in schema:
                for label in schema['facial_expression']:
                    prompts['model_actor'][label] = f"a person with {label.replace('_', ' ')}"
            
            # Hair styles
            if 'hair_styles' in schema:
                for label in schema['hair_styles']:
                    prompts['model_actor'][label] = f"a person with {label.replace('_', ' ')} hair"
            
            # Add other categories...
            for category_name, labels in schema.items():
                if isinstance(labels, list) and category_name not in ['face_geometry', 'facial_expression', 'hair_styles']:
                    for label in labels:
                        prompt = self._label_to_prompt(label, category_name)
                        prompts['model_actor'][label] = prompt
        
        # Process photographer style schema
        if 'photographer_style_schema' in self.schemas:
            schema = self.schemas['photographer_style_schema']
            prompts['photographer_style'] = {}
            
            for category_name, labels in schema.items():
                if isinstance(labels, list):
                    for label in labels:
                        prompt = self._label_to_prompt(label, category_name)
                        prompts['photographer_style'][label] = prompt
        
        return prompts
    
    def _label_to_prompt(self, label: str, category: str) -> str:
        """Convert a label to a CLIP-friendly prompt"""
        label_clean = label.replace('_', ' ')
        
        # Category-specific prompt templates
        if 'lighting' in category:
            return f"a photo with {label_clean} lighting"
        elif 'composition' in category:
            return f"a photo using {label_clean} composition"
        elif 'color' in category:
            return f"a photo with {label_clean} color treatment"
        elif 'clothing' in category or 'outfit' in category:
            return f"a person wearing {label_clean}"
        elif 'pose' in category or 'position' in category:
            return f"a person in {label_clean} pose"
        elif 'environment' in category:
            return f"a photo taken in {label_clean} environment"
        else:
            return f"a photo showing {label_clean}"
    
    def label_image(self, image_path: str, schema_type: str = "model_actor") -> Dict:
        """
        Label a single image using the specified schema
        
        Args:
            image_path: Path to image file
            schema_type: Which schema to use (model_actor or photographer_style)
            
        Returns:
            Dictionary of labels and confidence scores
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get relevant prompts
        if schema_type not in self.label_prompts:
            raise ValueError(f"Unknown schema type: {schema_type}")
        
        prompts = self.label_prompts[schema_type]
        
        # Prepare text inputs
        text_inputs = list(prompts.values())
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        
        # Calculate similarities
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarities
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Extract results
        results = {
            "labels_detected": {},
            "confidence_scores": {},
            "schema_type": schema_type,
            "total_labels_checked": len(prompts)
        }
        
        label_names = list(prompts.keys())
        similarities_np = similarities.cpu().numpy()[0]
        
        for i, (label, confidence) in enumerate(zip(label_names, similarities_np)):
            results["confidence_scores"][label] = float(confidence)
            
            # Only include labels above threshold
            if confidence >= self.confidence_threshold:
                results["labels_detected"][label] = float(confidence)
        
        print(f"Detected {len(results['labels_detected'])} labels above threshold {self.confidence_threshold}")
        
        return results
    
    def label_dataset(self, 
                     dataset_path: str, 
                     schema_type: str = "model_actor",
                     output_manifest: str = "automated_labels_manifest.json") -> Dict:
        """
        Label entire dataset using automated schema labeling
        
        Args:
            dataset_path: Path to dataset directory
            schema_type: Which schema to use
            output_manifest: Output manifest filename
            
        Returns:
            Complete labeling results
        """
        dataset_path = Path(dataset_path)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        all_images = [f for f in dataset_path.iterdir() 
                     if f.suffix.lower() in image_extensions]
        
        results = {
            "schema_type": schema_type,
            "total_images": len(all_images),
            "images_processed": 0,
            "labels_summary": {},
            "detailed_results": {}
        }
        
        # Process each image
        for i, image_path in enumerate(all_images):
            print(f"Processing {i+1}/{len(all_images)}: {image_path.name}")
            
            try:
                image_labels = self.label_image(str(image_path), schema_type)
                results["detailed_results"][image_path.name] = image_labels
                results["images_processed"] += 1
                
                # Update summary statistics
                for label in image_labels["labels_detected"]:
                    if label not in results["labels_summary"]:
                        results["labels_summary"][label] = 0
                    results["labels_summary"][label] += 1
                    
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                results["detailed_results"][image_path.name] = {
                    "error": str(e),
                    "labels_detected": {},
                    "confidence_scores": {}
                }
        
        # Calculate label frequencies
        if results["images_processed"] > 0:
            results["label_frequencies"] = {
                label: count / results["images_processed"] 
                for label, count in results["labels_summary"].items()
            }
        
        # Save manifest
        manifest_path = dataset_path / output_manifest
        with open(manifest_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAutomated labeling complete!")
        print(f"Processed: {results['images_processed']}/{results['total_images']} images")
        print(f"Most common labels: {sorted(results['labels_summary'].items(), key=lambda x: x[1], reverse=True)[:5]}")
        print(f"Manifest saved: {manifest_path}")
        
        return results

def combine_labels_and_captions(labels_manifest: str, 
                               captions_manifest: str,
                               output_manifest: str = "combined_manifest.json") -> Dict:
    """
    Combine automated labeling and captioning results
    
    Args:
        labels_manifest: Path to automated labels manifest
        captions_manifest: Path to captions manifest  
        output_manifest: Output combined manifest
        
    Returns:
        Combined results
    """
    # Load both manifests
    with open(labels_manifest, 'r') as f:
        labels_data = json.load(f)
    
    with open(captions_manifest, 'r') as f:
        captions_data = json.load(f)
    
    # Combine results
    combined = {
        "processing_type": "automated_labels_and_captions",
        "labels_data": labels_data,
        "captions_data": captions_data,
        "combined_results": {}
    }
    
    # Match images between both datasets
    for image_name in labels_data.get("detailed_results", {}):
        base_name = Path(image_name).stem
        
        # Find corresponding caption
        caption_info = None
        for caption_image, caption_data in captions_data.get("automated_captions", {}).items():
            if Path(caption_image).stem == base_name:
                caption_info = caption_data
                break
        
        combined["combined_results"][image_name] = {
            "labels": labels_data["detailed_results"][image_name],
            "caption": caption_info
        }
    
    # Save combined manifest
    with open(output_manifest, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"Combined manifest saved: {output_manifest}")
    return combined

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated schema labeling")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument("schema_path", help="Path to comprehensive_labeling_schemas.yaml")
    parser.add_argument("--schema-type", choices=["model_actor", "photographer_style"], 
                       default="model_actor", help="Which schema to use")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Confidence threshold for labels")
    
    args = parser.parse_args()
    
    # Initialize labeler
    labeler = AutomatedSchemaLabeler(
        schema_path=args.schema_path,
        confidence_threshold=args.confidence
    )
    
    # Process dataset
    results = labeler.label_dataset(
        args.dataset_path,
        schema_type=args.schema_type
    )
    
    print(f"\n=== Automated Labeling Complete ===")
    print(f"Schema: {args.schema_type}")
    print(f"Images processed: {results['images_processed']}")
    print(f"Unique labels found: {len(results['labels_summary'])}")

if __name__ == "__main__":
    main()